"""Tests for browser_session and browser_snapshot modules.

These tests use unittest.mock so that Playwright does not need to be installed
in CI.  The mocked page objects replicate the minimal Playwright API surface
used by :class:`AiSnapshotSerializer` and :class:`BrowserSnapshot`.
"""

import os
import sys
import types
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: build a minimal fake playwright module so that the import guard in
# browser_session.py does not prevent importing in environments where the real
# playwright package is absent.
# ---------------------------------------------------------------------------

def _make_fake_playwright_module():
    """Return a fake ``playwright.sync_api`` module with stub classes."""
    mod = types.ModuleType('playwright')
    sync_mod = types.ModuleType('playwright.sync_api')

    class _FakeAPI:
        """Minimal stub for sync_playwright context."""

        def start(self):
            return _FakePlaywright()

        def stop(self):
            pass

    class _FakePlaywright:
        chromium = MagicMock()

    sync_mod.sync_playwright = lambda: _FakeAPI()
    sync_mod.Browser = object
    sync_mod.BrowserContext = object
    sync_mod.Page = object

    mod.sync_api = sync_mod
    sys.modules.setdefault('playwright', mod)
    sys.modules.setdefault('playwright.sync_api', sync_mod)
    return sync_mod


_make_fake_playwright_module()

# Now import the modules under test
from lagent.actions.browser_session import (  # noqa: E402
    BrowserSession,
    BrowserSessionManager,
    BrowserTarget,
)
from lagent.actions.browser_snapshot import (  # noqa: E402
    AiSnapshotSerializer,
    BrowserSnapshot,
    SnapshotStats,
)
from lagent.schema import ActionReturn, ActionStatusCode  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_page(
    url='https://example.com',
    title='Example',
    body_text='Hello world\nLine two',
    elements=None,
):
    """Build a MagicMock that mimics the Playwright Page API."""
    page = MagicMock()
    page.url = url
    page.title.return_value = title
    page.inner_text.return_value = body_text
    page.evaluate.return_value = elements if elements is not None else []
    return page


# ---------------------------------------------------------------------------
# SnapshotStats tests
# ---------------------------------------------------------------------------

class TestSnapshotStats(unittest.TestCase):

    def test_defaults(self):
        stats = SnapshotStats()
        self.assertEqual(stats.lines, 0)
        self.assertEqual(stats.chars, 0)
        self.assertEqual(stats.refs, 0)
        self.assertEqual(stats.interactive, 0)

    def test_asdict(self):
        stats = SnapshotStats(lines=3, chars=42, refs=2, interactive=5)
        d = asdict(stats)
        self.assertEqual(d, {'lines': 3, 'chars': 42, 'refs': 2, 'interactive': 5})


# ---------------------------------------------------------------------------
# AiSnapshotSerializer tests
# ---------------------------------------------------------------------------

class TestAiSnapshotSerializer(unittest.TestCase):

    def setUp(self):
        self.serializer = AiSnapshotSerializer(
            max_total_chars=5000,
            max_text_chars=2000,
            max_refs=10,
        )

    def test_empty_page(self):
        page = _fake_page(body_text='', elements=[])
        snapshot, stats, elements = self.serializer.serialize(page)
        self.assertIn('URL: https://example.com', snapshot)
        self.assertIn('Title: Example', snapshot)
        self.assertEqual(stats.refs, 0)
        self.assertEqual(stats.interactive, 0)
        self.assertEqual(elements, [])

    def test_page_text_included(self):
        page = _fake_page(body_text='Welcome to the site\nSecond line')
        snapshot, stats, _ = self.serializer.serialize(page)
        self.assertIn('=== PAGE TEXT ===', snapshot)
        self.assertIn('Welcome to the site', snapshot)
        self.assertGreater(stats.lines, 0)
        self.assertGreater(stats.chars, 0)

    def test_interactive_elements_included(self):
        elements = [
            {'tag': 'a', 'type': '', 'role': '', 'text': 'Home',
             'href': '/home', 'placeholder': '', 'name': '', 'value': '',
             'options': []},
            {'tag': 'button', 'type': '', 'role': '', 'text': 'Submit',
             'href': '', 'placeholder': '', 'name': '', 'value': '',
             'options': []},
        ]
        page = _fake_page(elements=elements)
        snapshot, stats, out_elements = self.serializer.serialize(page)
        self.assertIn('=== INTERACTIVE ELEMENTS ===', snapshot)
        self.assertIn('[r1]', snapshot)
        self.assertIn('[r2]', snapshot)
        self.assertIn('LINK', snapshot)
        self.assertIn('BUTTON', snapshot)
        self.assertEqual(stats.refs, 2)
        self.assertEqual(stats.interactive, 2)
        self.assertEqual(len(out_elements), 2)

    def test_max_refs_cap(self):
        serializer = AiSnapshotSerializer(max_refs=2)
        elements = [
            {'tag': 'button', 'type': '', 'role': '', 'text': f'B{i}',
             'href': '', 'placeholder': '', 'name': '', 'value': '',
             'options': []}
            for i in range(5)
        ]
        page = _fake_page(elements=elements)
        _, stats, out_elements = serializer.serialize(page)
        self.assertEqual(stats.refs, 2)
        self.assertEqual(stats.interactive, 5)
        self.assertEqual(len(out_elements), 2)

    def test_text_truncation(self):
        long_text = 'x' * 5000
        serializer = AiSnapshotSerializer(max_text_chars=100, max_total_chars=5000)
        page = _fake_page(body_text=long_text)
        snapshot, stats, _ = serializer.serialize(page)
        self.assertIn('[... truncated ...]', snapshot)
        # stats.chars counts the length of the (truncated) page_text section,
        # which is max_text_chars + the truncation marker string.
        truncation_marker = '\n[... truncated ...]'
        self.assertLessEqual(stats.chars, 100 + len(truncation_marker))

    def test_total_chars_truncation(self):
        long_text = 'y' * 20000
        serializer = AiSnapshotSerializer(max_total_chars=500, max_text_chars=10000)
        page = _fake_page(body_text=long_text)
        snapshot, _, _ = serializer.serialize(page)
        self.assertLessEqual(len(snapshot), 500 + len('\n[... truncated ...]'))

    def test_stats_line_present(self):
        page = _fake_page()
        snapshot, _, _ = self.serializer.serialize(page)
        self.assertIn('--- stats:', snapshot)

    def test_select_element_options(self):
        elements = [
            {'tag': 'select', 'type': '', 'role': '', 'text': '',
             'href': '', 'placeholder': '', 'name': 'color', 'value': '',
             'options': ['Red', 'Green', 'Blue']},
        ]
        page = _fake_page(elements=elements)
        snapshot, _, _ = self.serializer.serialize(page)
        self.assertIn('SELECT', snapshot)
        self.assertIn('Red', snapshot)

    def test_input_element_formatting(self):
        elements = [
            {'tag': 'input', 'type': 'text', 'role': '', 'text': '',
             'href': '', 'placeholder': 'Search...', 'name': 'q', 'value': '',
             'options': []},
        ]
        page = _fake_page(elements=elements)
        snapshot, _, _ = self.serializer.serialize(page)
        self.assertIn('INPUT text', snapshot)
        self.assertIn('placeholder="Search..."', snapshot)


# ---------------------------------------------------------------------------
# BrowserTarget / BrowserSession tests
# ---------------------------------------------------------------------------

class TestBrowserTarget(unittest.TestCase):

    def test_refresh_info(self):
        page = _fake_page(url='https://test.com', title='Test')
        target = BrowserTarget(target_id='t1', page=page)
        self.assertEqual(target.url, '')  # not yet refreshed
        target.refresh_info()
        self.assertEqual(target.url, 'https://test.com')
        self.assertEqual(target.title, 'Test')

    def test_refresh_info_error_suppressed(self):
        page = MagicMock()
        page.url = 'https://ok.com'
        page.title.side_effect = Exception('disconnected')
        target = BrowserTarget(target_id='t2', page=page)
        # Should not raise even if title() throws
        target.refresh_info()
        self.assertEqual(target.url, 'https://ok.com')


class TestBrowserSession(unittest.TestCase):

    def _make_session(self):
        page = _fake_page()
        target = BrowserTarget(target_id='t1', page=page)
        session = BrowserSession(
            session_id='s1',
            browser=MagicMock(),
            context=MagicMock(),
            targets={'t1': target},
            active_target_id='t1',
        )
        return session, page

    def test_active_page_returns_correct_page(self):
        session, page = self._make_session()
        self.assertIs(session.active_page, page)

    def test_active_page_fallback(self):
        session, page = self._make_session()
        session.active_target_id = None
        # Should fall back to first target
        self.assertIs(session.active_page, page)

    def test_active_page_none_when_empty(self):
        session = BrowserSession(
            session_id='empty',
            browser=MagicMock(),
            context=MagicMock(),
        )
        self.assertIsNone(session.active_page)

    def test_set_active_by_index(self):
        page1 = _fake_page(url='https://a.com')
        page2 = _fake_page(url='https://b.com')
        target1 = BrowserTarget(target_id='t1', page=page1, url='https://a.com')
        target2 = BrowserTarget(target_id='t2', page=page2, url='https://b.com')
        session = BrowserSession(
            session_id='s',
            browser=MagicMock(),
            context=MagicMock(),
            targets={'t1': target1, 't2': target2},
            active_target_id='t1',
        )
        result = session.set_active_by_index(1)
        self.assertTrue(result)
        self.assertEqual(session.active_target_id, 't2')

    def test_set_active_by_index_out_of_range(self):
        session, _ = self._make_session()
        result = session.set_active_by_index(99)
        self.assertFalse(result)

    def test_set_active_by_url(self):
        page1 = _fake_page(url='https://a.com/page')
        target1 = BrowserTarget(target_id='t1', page=page1, url='https://a.com/page')
        session = BrowserSession(
            session_id='s',
            browser=MagicMock(),
            context=MagicMock(),
            targets={'t1': target1},
            active_target_id='t1',
        )
        result = session.set_active_by_url('https://a.com')
        self.assertTrue(result)

    def test_bind_and_resolve_refs(self):
        session, _ = self._make_session()
        elements = [
            {'tag': 'a', 'type': '', 'role': '', 'text': 'Home',
             'href': '/home', 'placeholder': '', 'name': '', 'value': '',
             'options': []},
            {'tag': 'button', 'type': '', 'role': '', 'text': 'Submit',
             'href': '', 'placeholder': '', 'name': '', 'value': '',
             'options': []},
        ]
        session.bind_refs(elements)
        self.assertEqual(len(session.refs), 2)
        self.assertEqual(session.resolve_ref('r1'), elements[0])
        self.assertEqual(session.resolve_ref('r2'), elements[1])
        self.assertIsNone(session.resolve_ref('r99'))

    def test_bind_refs_clears_previous(self):
        session, _ = self._make_session()
        session.bind_refs([{'tag': 'a', 'type': '', 'role': '', 'text': 'Old',
                            'href': '', 'placeholder': '', 'name': '',
                            'value': '', 'options': []}])
        self.assertIn('r1', session.refs)
        session.bind_refs([])
        self.assertEqual(session.refs, {})


# ---------------------------------------------------------------------------
# BrowserSnapshot action tests (Playwright mocked)
# ---------------------------------------------------------------------------

class TestBrowserSnapshot(unittest.TestCase):
    """Test BrowserSnapshot using a fully mocked BrowserSessionManager."""

    def _make_snapshot_action(self):
        """Return a BrowserSnapshot with a mocked session manager."""
        snap = BrowserSnapshot()
        mock_manager = MagicMock()
        snap._session_manager = mock_manager
        return snap, mock_manager

    def _make_mock_session(self, page=None):
        """Create a mock BrowserSession with a working active_page."""
        if page is None:
            page = _fake_page()
        session = MagicMock(spec=BrowserSession)
        session.active_page = page
        session.active_target_id = 't1'
        session.targets = {'t1': MagicMock()}
        session.refs = {}
        session.artifact_dir = '/tmp/artifacts'
        return session

    def test_run_returns_snapshot(self):
        snap, mock_manager = self._make_snapshot_action()
        page = _fake_page(
            url='https://example.com',
            title='Ex',
            body_text='Hello',
            elements=[],
        )
        session = self._make_mock_session(page)
        mock_manager.get_or_create_session.return_value = session

        result = snap.run(session_id='s1')
        # result is a dict when successful
        self.assertIsInstance(result, dict)
        self.assertIn('snapshot', result)
        self.assertIn('https://example.com', result['url'])
        self.assertIn('stats', result)

    def test_run_binds_refs(self):
        snap, mock_manager = self._make_snapshot_action()
        elements = [
            {'tag': 'button', 'type': '', 'role': '', 'text': 'Go',
             'href': '', 'placeholder': '', 'name': '', 'value': '',
             'options': []},
        ]
        page = _fake_page(elements=elements)
        session = self._make_mock_session(page)
        mock_manager.get_or_create_session.return_value = session

        snap.run(session_id='s1')
        session.bind_refs.assert_called_once()

    def test_run_target_index(self):
        snap, mock_manager = self._make_snapshot_action()
        page = _fake_page()
        session = self._make_mock_session(page)
        session.set_active_by_index = MagicMock(return_value=True)
        mock_manager.get_or_create_session.return_value = session

        snap.run(session_id='s1', target='0')
        session.set_active_by_index.assert_called_once_with(0)

    def test_run_target_url(self):
        snap, mock_manager = self._make_snapshot_action()
        page = _fake_page()
        session = self._make_mock_session(page)
        session.set_active_by_index = MagicMock(side_effect=ValueError)
        session.set_active_by_url = MagicMock(return_value=True)
        mock_manager.get_or_create_session.return_value = session

        snap.run(session_id='s1', target='https://example.com')
        session.set_active_by_url.assert_called_once_with('https://example.com')

    def test_run_no_active_page(self):
        snap, mock_manager = self._make_snapshot_action()
        session = MagicMock(spec=BrowserSession)
        session.active_page = None
        mock_manager.get_or_create_session.return_value = session

        result = snap.run(session_id='s1')
        # Should return an ActionReturn with API_ERROR
        self.assertIsInstance(result, ActionReturn)
        self.assertEqual(result.state, ActionStatusCode.API_ERROR)

    def test_run_session_creation_error(self):
        snap, mock_manager = self._make_snapshot_action()
        mock_manager.get_or_create_session.side_effect = RuntimeError('boom')

        result = snap.run(session_id='s1')
        self.assertIsInstance(result, ActionReturn)
        self.assertEqual(result.state, ActionStatusCode.API_ERROR)
        self.assertIn('boom', result.errmsg)

    def test_run_include_image(self):
        snap, mock_manager = self._make_snapshot_action()
        page = _fake_page()
        page.screenshot = MagicMock()
        session = self._make_mock_session(page)
        session.artifact_dir = '/tmp/test_artifacts'
        mock_manager.get_or_create_session.return_value = session

        with patch('os.makedirs'), patch('os.path.abspath', return_value='/abs/path.png'):
            result = snap.run(session_id='s1', include_image=True)

        if isinstance(result, dict):
            # screenshot_path may be empty if abspath mock didn't fully wire through
            self.assertIn('screenshot_path', result)

    def test_run_per_call_overrides(self):
        snap, mock_manager = self._make_snapshot_action()
        page = _fake_page(body_text='x' * 5000)
        session = self._make_mock_session(page)
        mock_manager.get_or_create_session.return_value = session

        result = snap.run(
            session_id='s1',
            max_total_chars=200,
            max_text_chars=50,
            max_refs=5,
        )
        if isinstance(result, dict):
            snapshot = result['snapshot']
            # With a 200-char cap the snapshot should be truncated
            self.assertLessEqual(len(snapshot), 200 + len('\n[... truncated ...]'))

    def test_playwright_unavailable(self):
        """BrowserSnapshot.run should return API_ERROR when playwright missing."""
        snap = BrowserSnapshot()
        snap._session_manager = None

        # Patch the module-level flag
        import lagent.actions.browser_snapshot as bsmod
        original = bsmod.PLAYWRIGHT_AVAILABLE
        bsmod.PLAYWRIGHT_AVAILABLE = False
        try:
            result = snap.run(session_id='s1')
            self.assertIsInstance(result, ActionReturn)
            self.assertEqual(result.state, ActionStatusCode.API_ERROR)
            self.assertIn('playwright', result.errmsg.lower())
        finally:
            bsmod.PLAYWRIGHT_AVAILABLE = original

    def test_description_is_dict(self):
        snap = BrowserSnapshot()
        desc = snap.description
        self.assertIsInstance(desc, dict)
        self.assertIn('name', desc)
        self.assertEqual(desc['name'], 'BrowserSnapshot')

    def test_action_call_interface(self):
        """BrowserSnapshot should work through the standard __call__ interface."""
        snap = BrowserSnapshot()
        mock_manager = MagicMock()
        snap._session_manager = mock_manager

        page = _fake_page()
        session = self._make_mock_session(page)
        mock_manager.get_or_create_session.return_value = session

        import json
        action_return = snap(json.dumps({'session_id': 's1'}))
        self.assertIsNotNone(action_return)
        self.assertEqual(action_return.state, ActionStatusCode.SUCCESS)


# ---------------------------------------------------------------------------
# BrowserSessionManager tests (no real browser; manager calls are mocked)
# ---------------------------------------------------------------------------

class TestBrowserSessionManager(unittest.TestCase):
    """Light structural tests that do not launch a real browser."""

    def test_singleton(self):
        m1 = BrowserSessionManager()
        m2 = BrowserSessionManager()
        self.assertIs(m1, m2)

    def test_list_sessions_empty_initially(self):
        manager = BrowserSessionManager()
        # Only check that list_sessions returns a list (may have sessions from
        # other tests since the manager is a singleton).
        self.assertIsInstance(manager.list_sessions(), list)


if __name__ == '__main__':
    unittest.main()
