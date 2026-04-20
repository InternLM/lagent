"""Browser session manager for Lagent browser tools.

Manages Playwright browser sessions, tabs, element ref registries, and
artifact directories (screenshots, downloads, traces) in a thread-safe way.
"""

import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@dataclass
class BrowserTarget:
    """Represents a single browser tab/page within a session."""

    target_id: str
    page: Any  # playwright Page object
    url: str = ''
    title: str = ''

    def refresh_info(self) -> None:
        """Update url/title from the live page."""
        try:
            self.url = self.page.url
            self.title = self.page.title()
        except Exception:
            pass


@dataclass
class BrowserSession:
    """Represents a managed browser session.

    Attributes:
        session_id: unique identifier for this session.
        browser: Playwright Browser instance.
        context: Playwright BrowserContext instance.
        targets: mapping from target_id to BrowserTarget.
        active_target_id: target_id of the currently active tab.
        refs: mapping from ref string (e.g. ``"r1"``) to element info dict.
        artifact_dir: directory path for storing screenshots/downloads/traces.
    """

    session_id: str
    browser: Any  # playwright Browser
    context: Any  # playwright BrowserContext
    targets: Dict[str, 'BrowserTarget'] = field(default_factory=dict)
    active_target_id: Optional[str] = None
    refs: Dict[str, dict] = field(default_factory=dict)
    artifact_dir: Optional[str] = None

    @property
    def active_page(self) -> Optional[Any]:
        """Return the Playwright Page for the active target, or ``None``."""
        if self.active_target_id and self.active_target_id in self.targets:
            return self.targets[self.active_target_id].page
        # Fallback: first available target
        if self.targets:
            return next(iter(self.targets.values())).page
        return None

    def set_active_by_url(self, url: str) -> bool:
        """Switch the active target to the first tab whose URL matches.

        Args:
            url (str): URL (or prefix) to match.

        Returns:
            bool: ``True`` if a matching target was found and activated.
        """
        for tid, target in self.targets.items():
            target.refresh_info()
            if target.url == url or target.url.startswith(url):
                self.active_target_id = tid
                return True
        return False

    def set_active_by_index(self, index: int) -> bool:
        """Switch the active target by zero-based tab index.

        Args:
            index (int): zero-based index into :attr:`targets`.

        Returns:
            bool: ``True`` if the index was valid.
        """
        keys = list(self.targets.keys())
        if 0 <= index < len(keys):
            self.active_target_id = keys[index]
            return True
        return False

    def bind_refs(self, elements: List[dict]) -> None:
        """Register interactive elements as named refs.

        Args:
            elements (list[dict]): element info dicts produced by the
                snapshot serializer.  Each dict must contain at least a
                ``selector`` key that can be used to re-locate the element.
        """
        self.refs.clear()
        for idx, el in enumerate(elements):
            ref_id = f'r{idx + 1}'
            self.refs[ref_id] = el

    def resolve_ref(self, ref: str) -> Optional[dict]:
        """Return element info for a ref string such as ``"r1"``.

        Args:
            ref (str): ref identifier.

        Returns:
            dict | None: element info dict, or ``None`` if not found.
        """
        return self.refs.get(ref)


class BrowserSessionManager:
    """Thread-safe singleton manager for Playwright browser sessions.

    Usage::

        manager = BrowserSessionManager()
        session = manager.get_or_create_session('my-session')
        page = session.active_page
        # ... do stuff with page ...
        manager.close_session('my-session')
    """

    _instance: Optional['BrowserSessionManager'] = None
    _class_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> 'BrowserSessionManager':
        with cls._class_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._sessions: Dict[str, BrowserSession] = {}
                inst._lock = threading.Lock()
                inst._playwright = None
                inst._playwright_ctx = None
                cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_playwright(self) -> None:
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                'playwright is not installed. '
                'Install it with: pip install playwright && playwright install'
            )
        if self._playwright is None:
            self._playwright_ctx = sync_playwright()
            self._playwright = self._playwright_ctx.start()

    def _make_artifact_dir(self, session_id: str, base: Optional[str]) -> str:
        root = base or os.path.join(os.getcwd(), '.browser_artifacts')
        artifact_dir = os.path.join(root, session_id)
        os.makedirs(artifact_dir, exist_ok=True)
        return artifact_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: Optional[str] = None,
        artifact_dir: Optional[str] = None,
        browser_type: str = 'chromium',
        headless: bool = True,
        **launch_kwargs: Any,
    ) -> BrowserSession:
        """Launch a new browser and create a session.

        Args:
            session_id (str | None): identifier for the session.  A random
                UUID is used when not provided.
            artifact_dir (str | None): root directory for browser artifacts.
                Defaults to ``<cwd>/.browser_artifacts/<session_id>``.
            browser_type (str): Playwright browser type – ``'chromium'``,
                ``'firefox'``, or ``'webkit'``.  Defaults to ``'chromium'``.
            headless (bool): run the browser in headless mode.  Defaults to
                ``True``.
            **launch_kwargs: extra keyword arguments forwarded to
                ``browser_type.launch()``.

        Returns:
            BrowserSession: the newly created session.

        Raises:
            RuntimeError: if ``playwright`` is not installed, or if
                *session_id* is already in use.
        """
        with self._lock:
            self._ensure_playwright()
            session_id = session_id or str(uuid.uuid4())
            if session_id in self._sessions:
                raise RuntimeError(f"Session '{session_id}' already exists.")

            launcher = getattr(self._playwright, browser_type)
            browser: Browser = launcher.launch(headless=headless, **launch_kwargs)
            context: BrowserContext = browser.new_context()
            page: Page = context.new_page()

            target_id = str(uuid.uuid4())
            target = BrowserTarget(target_id=target_id, page=page)
            target.refresh_info()

            art_dir = self._make_artifact_dir(session_id, artifact_dir)
            session = BrowserSession(
                session_id=session_id,
                browser=browser,
                context=context,
                targets={target_id: target},
                active_target_id=target_id,
                artifact_dir=art_dir,
            )
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[BrowserSession]:
        """Return an existing session by ID, or ``None`` if not found.

        Args:
            session_id (str): session identifier.

        Returns:
            BrowserSession | None: the session object.
        """
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create_session(
        self,
        session_id: str,
        **kwargs: Any,
    ) -> BrowserSession:
        """Return an existing session or create a new one.

        Args:
            session_id (str): session identifier.
            **kwargs: forwarded to :meth:`create_session` when creating.

        Returns:
            BrowserSession: existing or newly created session.
        """
        with self._lock:
            session = self._sessions.get(session_id)
        if session is not None:
            return session
        return self.create_session(session_id=session_id, **kwargs)

    def list_sessions(self) -> List[str]:
        """Return a list of all active session IDs.

        Returns:
            list[str]: session identifiers.
        """
        with self._lock:
            return list(self._sessions.keys())

    def open_tab(self, session_id: str, url: Optional[str] = None) -> str:
        """Open a new tab in an existing session.

        Args:
            session_id (str): session identifier.
            url (str | None): optional URL to navigate the new tab to.

        Returns:
            str: the new target ID.

        Raises:
            KeyError: if *session_id* does not exist.
        """
        with self._lock:
            session = self._sessions[session_id]
            page: Page = session.context.new_page()
            if url:
                page.goto(url)
            target_id = str(uuid.uuid4())
            target = BrowserTarget(target_id=target_id, page=page)
            target.refresh_info()
            session.targets[target_id] = target
            session.active_target_id = target_id
            return target_id

    def close_tab(self, session_id: str, target_id: str) -> None:
        """Close a specific tab within a session.

        Args:
            session_id (str): session identifier.
            target_id (str): target identifier to close.

        Raises:
            KeyError: if either *session_id* or *target_id* does not exist.
        """
        with self._lock:
            session = self._sessions[session_id]
            target = session.targets.pop(target_id)
            try:
                target.page.close()
            except Exception:
                pass
            if session.active_target_id == target_id:
                session.active_target_id = next(iter(session.targets), None)

    def close_session(self, session_id: str) -> None:
        """Close a browser session and release all resources.

        Args:
            session_id (str): session identifier.  No-op if not found.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        try:
            session.browser.close()
        except Exception:
            pass

    def close_all(self) -> None:
        """Close all sessions and stop the Playwright process."""
        with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            self.close_session(sid)
        with self._lock:
            if self._playwright is not None:
                try:
                    self._playwright_ctx.stop()
                except Exception:
                    pass
                self._playwright = None
                self._playwright_ctx = None
