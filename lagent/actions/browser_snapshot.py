"""Browser snapshot action for Lagent.

Provides :class:`BrowserSnapshot`, a Lagent action that captures a
model-friendly text representation of the currently active browser page,
together with an optional screenshot artifact.  Interactive elements are
registered as *refs* (``r1``, ``r2``, …) that later browser-interaction
actions can resolve back to DOM nodes.

The :class:`AiSnapshotSerializer` helper converts raw Playwright page data
into a structured text snapshot and collects emission statistics.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

try:
    from .browser_session import PLAYWRIGHT_AVAILABLE, BrowserSession, BrowserSessionManager
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    BrowserSessionManager = None  # type: ignore[assignment,misc]
    BrowserSession = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# JavaScript snippet executed inside the browser to collect element metadata.
# ---------------------------------------------------------------------------
_COLLECT_ELEMENTS_JS = """
() => {
    const SELECTORS = [
        'a[href]',
        'button:not([disabled])',
        'input:not([disabled])',
        'select:not([disabled])',
        'textarea:not([disabled])',
        '[role="button"]:not([disabled])',
        '[role="link"]',
        '[role="checkbox"]',
        '[role="radio"]',
        '[role="menuitem"]',
        '[role="option"]',
        '[role="tab"]',
        '[role="combobox"]',
        '[contenteditable="true"]',
    ].join(', ');

    function isVisible(el) {
        if (!el) return false;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return false;
        const style = window.getComputedStyle(el);
        return style.display !== 'none'
            && style.visibility !== 'hidden'
            && parseFloat(style.opacity) > 0;
    }

    function getText(el) {
        const t = (el.innerText || el.textContent || '').trim();
        return t.slice(0, 120);
    }

    function getLabel(el) {
        const id = el.getAttribute('id');
        if (id) {
            const lbl = document.querySelector(`label[for="${id}"]`);
            if (lbl) return (lbl.innerText || '').trim();
        }
        return el.getAttribute('aria-label') || '';
    }

    const seen = new Set();
    const results = [];
    document.querySelectorAll(SELECTORS).forEach(el => {
        if (!isVisible(el) || seen.has(el)) return;
        seen.add(el);

        const tag = el.tagName.toLowerCase();
        const type = el.getAttribute('type') || '';
        const role = el.getAttribute('role') || '';
        let text = getText(el);
        if (!text) text = getLabel(el);
        if (!text) text = el.getAttribute('aria-label') || '';
        if (!text) text = el.getAttribute('placeholder') || '';
        if (!text) text = el.getAttribute('value') || '';
        if (!text && tag === 'input') text = el.getAttribute('name') || '';

        const info = {
            tag,
            type,
            role,
            text: text.slice(0, 100),
            href: el.getAttribute('href') || '',
            placeholder: el.getAttribute('placeholder') || '',
            name: el.getAttribute('name') || '',
            value: (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')
                       ? (el.value || '') : '',
            options: [],
        };

        if (tag === 'select') {
            info.options = Array.from(el.options).map(o => o.text.trim());
        }

        results.push(info);
    });
    return results;
}
"""


# ---------------------------------------------------------------------------
# Snapshot statistics dataclass
# ---------------------------------------------------------------------------

@dataclass
class SnapshotStats:
    """Statistics emitted alongside a browser snapshot.

    Attributes:
        lines: number of lines in the page-text section.
        chars: total characters in the page-text section.
        refs: number of interactive refs registered.
        interactive: number of interactive elements found on the page.
    """

    lines: int = 0
    chars: int = 0
    refs: int = 0
    interactive: int = 0


# ---------------------------------------------------------------------------
# AI snapshot serializer
# ---------------------------------------------------------------------------

class AiSnapshotSerializer:
    """Convert Playwright page content into a model-friendly text snapshot.

    The output format is::

        URL: <url>
        Title: <title>

        === INTERACTIVE ELEMENTS ===
        [r1] LINK "Home" href="/home"
        [r2] BUTTON "Submit"
        [r3] INPUT text name="q" placeholder="Search..."
        [r4] SELECT options=["Option A","Option B"]

        === PAGE TEXT ===
        <visible page text, truncated to max_text_chars>

        --- stats: lines=42 chars=1234 refs=4 interactive=4 ---

    Args:
        max_total_chars (int): hard cap on the total snapshot length
            (excluding the stats line).  Defaults to ``20000``.
        max_text_chars (int): maximum characters for the page-text section.
            Defaults to ``10000``.
        max_refs (int): maximum number of interactive element refs to emit.
            Defaults to ``100``.
    """

    def __init__(
        self,
        max_total_chars: int = 20_000,
        max_text_chars: int = 10_000,
        max_refs: int = 100,
    ) -> None:
        self.max_total_chars = max_total_chars
        self.max_text_chars = max_text_chars
        self.max_refs = max_refs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def serialize(
        self,
        page: Any,
        existing_refs: Optional[Dict[str, dict]] = None,
    ) -> Tuple[str, SnapshotStats, List[dict]]:
        """Produce a text snapshot of *page*.

        Args:
            page: a Playwright ``Page`` object.
            existing_refs (dict | None): not used directly; reserved for
                future incremental diffing.

        Returns:
            tuple: ``(snapshot_text, stats, elements)`` where *elements* is
            the list of interactive element dicts that should be bound as
            refs in the session.
        """
        url = page.url
        try:
            title = page.title()
        except Exception:
            title = ''

        elements = self._extract_interactive_elements(page)
        capped_elements = elements[: self.max_refs]

        # Build interactive section
        interactive_lines: List[str] = []
        for idx, el in enumerate(capped_elements):
            ref_id = f'r{idx + 1}'
            interactive_lines.append(self._format_element(el, ref_id))

        # Build page-text section
        try:
            raw_text = page.inner_text('body') or ''
        except Exception:
            raw_text = ''
        page_text = self._truncate(raw_text, self.max_text_chars)

        stats = SnapshotStats(
            lines=len(page_text.splitlines()),
            chars=len(page_text),
            refs=len(capped_elements),
            interactive=len(elements),
        )

        # Assemble snapshot
        parts: List[str] = [
            f'URL: {url}',
            f'Title: {title}',
            '',
        ]
        if interactive_lines:
            parts += ['=== INTERACTIVE ELEMENTS ==='] + interactive_lines + ['']
        if page_text:
            parts += ['=== PAGE TEXT ===', page_text, '']
        parts.append(
            f'--- stats: lines={stats.lines} chars={stats.chars} '
            f'refs={stats.refs} interactive={stats.interactive} ---'
        )

        snapshot = '\n'.join(parts)
        snapshot = self._truncate(snapshot, self.max_total_chars)

        return snapshot, stats, capped_elements

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_interactive_elements(self, page: Any) -> List[dict]:
        """Run :data:`_COLLECT_ELEMENTS_JS` in the page and return results."""
        try:
            return page.evaluate(_COLLECT_ELEMENTS_JS) or []
        except Exception:
            return []

    def _format_element(self, el: dict, ref_id: str) -> str:
        """Render a single element dict as a compact ref line."""
        tag = el.get('tag', '').upper()
        role = el.get('role', '').upper()
        el_type = el.get('type', '')
        text = el.get('text', '').strip()
        href = el.get('href', '')
        placeholder = el.get('placeholder', '')
        name = el.get('name', '')
        options = el.get('options', [])

        # Determine display kind
        if tag == 'A' or role == 'LINK':
            kind = 'LINK'
        elif tag == 'BUTTON' or role == 'BUTTON':
            kind = 'BUTTON'
        elif tag == 'INPUT':
            kind = f'INPUT {el_type}' if el_type else 'INPUT'
        elif tag == 'SELECT':
            kind = 'SELECT'
        elif tag == 'TEXTAREA':
            kind = 'TEXTAREA'
        else:
            kind = role or tag or 'ELEMENT'

        parts = [f'[{ref_id}]', kind]
        if text:
            parts.append(f'"{text}"')
        if href:
            parts.append(f'href="{href}"')
        if placeholder:
            parts.append(f'placeholder="{placeholder}"')
        if name:
            parts.append(f'name="{name}"')
        if options:
            opts_str = json.dumps(options[:10], ensure_ascii=False)
            parts.append(f'options={opts_str}')

        return ' '.join(parts)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Truncate *text* to *max_chars* characters deterministically."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + '\n[... truncated ...]'


# ---------------------------------------------------------------------------
# BrowserSnapshot action
# ---------------------------------------------------------------------------

class BrowserSnapshot(BaseAction):
    """Capture a model-friendly snapshot of the active browser page.

    This action connects to a managed
    :class:`~lagent.actions.browser_session.BrowserSession`, extracts visible
    text and interactive elements from the current page, registers them as
    named refs (``r1``, ``r2``, …), and optionally saves a screenshot.

    Args:
        artifact_dir (str | None): root directory for browser artifacts
            (screenshots, downloads, traces).  Defaults to
            ``<cwd>/.browser_artifacts``.
        max_total_chars (int): hard cap on snapshot length.
            Defaults to ``20000``.
        max_text_chars (int): cap on the page-text section.
            Defaults to ``10000``.
        max_refs (int): maximum interactive refs to include.
            Defaults to ``100``.
        description (dict | None): custom tool description.
        parser (Type[BaseParser]): parser class.  Defaults to
            :class:`~lagent.actions.parser.JsonParser`.

    Example::

        from lagent.actions.browser_snapshot import BrowserSnapshot
        from lagent.actions.browser_session import BrowserSessionManager

        manager = BrowserSessionManager()
        session = manager.create_session('demo')
        session.active_page.goto('https://example.com')

        snap = BrowserSnapshot()
        result = snap.run(session_id='demo')
        print(result['snapshot'])
    """

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        max_total_chars: int = 20_000,
        max_text_chars: int = 10_000,
        max_refs: int = 100,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ) -> None:
        self._artifact_dir = artifact_dir
        self._max_total_chars = max_total_chars
        self._max_text_chars = max_text_chars
        self._max_refs = max_refs
        if PLAYWRIGHT_AVAILABLE and BrowserSessionManager is not None:
            self._session_manager: Optional[BrowserSessionManager] = BrowserSessionManager()
        else:
            self._session_manager = None
        self._serializer = AiSnapshotSerializer(
            max_total_chars=max_total_chars,
            max_text_chars=max_text_chars,
            max_refs=max_refs,
        )
        super().__init__(description, parser)

    # ------------------------------------------------------------------
    # Tool API
    # ------------------------------------------------------------------

    @tool_api(explode_return=True)
    def run(
        self,
        session_id: str,
        target: Optional[str] = None,
        include_image: bool = False,
        max_total_chars: Optional[int] = None,
        max_text_chars: Optional[int] = None,
        max_refs: Optional[int] = None,
    ) -> dict:
        """Capture a snapshot of the current browser page.

        Args:
            session_id (str): browser session identifier; a new session is
                created automatically if *session_id* is not yet known.
            target (str): select a tab by zero-based index (e.g. ``"0"``) or
                by matching URL prefix.  Omit to use the currently active tab.
            include_image (bool): when ``True``, a PNG screenshot is saved to
                the session's artifact directory and its path is returned.
            max_total_chars (int): per-call override for the total snapshot
                character limit.
            max_text_chars (int): per-call override for the page-text section
                character limit.
            max_refs (int): per-call override for the maximum number of
                interactive refs.

        Returns:
            dict: snapshot result
                * snapshot: model-friendly text representation of the page
                * url: URL of the active page
                * title: title of the active page
                * stats: dict with keys ``lines``, ``chars``, ``refs``,
                  ``interactive``
                * screenshot_path: absolute path to screenshot PNG, or ``""``
                  when *include_image* is ``False``
        """
        if not PLAYWRIGHT_AVAILABLE or self._session_manager is None:
            return ActionReturn(
                args={'session_id': session_id},
                type=self.name,
                errmsg=(
                    'playwright is not installed. '
                    'Install with: pip install playwright && playwright install'
                ),
                state=ActionStatusCode.API_ERROR,
            )

        # Resolve (or create) session
        try:
            session: BrowserSession = self._session_manager.get_or_create_session(
                session_id,
                artifact_dir=self._artifact_dir,
            )
        except Exception as exc:
            return ActionReturn(
                args={'session_id': session_id},
                type=self.name,
                errmsg=f'Failed to get/create session: {exc}',
                state=ActionStatusCode.API_ERROR,
            )

        # Optionally switch active tab
        if target is not None:
            self._switch_target(session, target)

        page = session.active_page
        if page is None:
            return ActionReturn(
                args={'session_id': session_id},
                type=self.name,
                errmsg='No active page found in session.',
                state=ActionStatusCode.API_ERROR,
            )

        # Build per-call serializer (use instance defaults when not overridden)
        serializer = AiSnapshotSerializer(
            max_total_chars=max_total_chars or self._max_total_chars,
            max_text_chars=max_text_chars or self._max_text_chars,
            max_refs=max_refs or self._max_refs,
        )

        try:
            snapshot_text, stats, elements = serializer.serialize(
                page, existing_refs=session.refs
            )
        except Exception as exc:
            return ActionReturn(
                args={'session_id': session_id},
                type=self.name,
                errmsg=f'Serialization failed: {exc}',
                state=ActionStatusCode.API_ERROR,
            )

        # Bind refs into session state
        session.bind_refs(elements)

        # Optionally take a screenshot
        screenshot_path = ''
        if include_image:
            screenshot_path = self._capture_screenshot(session, page)

        # Refresh active target info
        if session.active_target_id and session.active_target_id in session.targets:
            session.targets[session.active_target_id].refresh_info()

        current_url = page.url
        try:
            current_title = page.title()
        except Exception:
            current_title = ''

        return {
            'snapshot': snapshot_text,
            'url': current_url,
            'title': current_title,
            'stats': asdict(stats),
            'screenshot_path': screenshot_path,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _switch_target(self, session: 'BrowserSession', target: str) -> None:
        """Attempt to switch the session's active tab.

        *target* is tried as a zero-based integer index first; if that fails
        it is matched against tab URLs.

        Args:
            session: the active :class:`BrowserSession`.
            target (str): tab selector.
        """
        try:
            index = int(target)
            session.set_active_by_index(index)
        except (ValueError, TypeError):
            session.set_active_by_url(target)

    def _capture_screenshot(self, session: 'BrowserSession', page: Any) -> str:
        """Save a PNG screenshot and return the absolute file path.

        Args:
            session: the active :class:`BrowserSession`.
            page: Playwright ``Page`` object.

        Returns:
            str: absolute path to the screenshot file, or ``""`` on failure.
        """
        try:
            import time
            art_dir = session.artifact_dir or os.getcwd()
            os.makedirs(art_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            path = os.path.join(art_dir, f'screenshot_{timestamp}.png')
            page.screenshot(path=path, full_page=False)
            return os.path.abspath(path)
        except Exception:
            return ''
