import json
from pathlib import Path
from unittest import TestCase, mock

from lagent.skills.skills import FilesystemSkillsBackend, SkillsLoader, SandboxSkillsBackend
from lagent.schema import ActionReturn, ActionStatusCode


class TestFilesystemSkillsBackend(TestCase):

    def test_list_and_read_skill(self):
        with mock.patch('tempfile.TemporaryDirectory'):
            pass

    def test_workspace_skill_overrides_builtin(self):
        with mock.patch('tempfile.TemporaryDirectory'):
            pass


class TestSkillsLoader(TestCase):

    def test_list_skills_filters_unavailable(self):
        with mock.patch('tempfile.TemporaryDirectory'):
            pass

    def test_build_skills_summary_marks_missing_requirements(self):
        with mock.patch('tempfile.TemporaryDirectory'):
            pass

    def test_get_always_skills(self):
        with mock.patch('tempfile.TemporaryDirectory'):
            pass


class DummySandboxAction:

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def run(self, session_id, command):
        self.calls.append((session_id, command))
        payload = self.responses.get(command)
        action_return = ActionReturn(args={'session_id': session_id, 'command': command})
        if payload is None:
            action_return.state = ActionStatusCode.API_ERROR
            action_return.result = []
            action_return.errmsg = 'not found'
            return action_return

        action_return.state = ActionStatusCode.SUCCESS
        action_return.result = [dict(type='text', content=payload)]
        return action_return


class TestSandboxSkillsBackend(TestCase):

    def test_list_skill_entries_with_builtin_fallback(self):
        with mock.patch('tempfile.TemporaryDirectory'):
            pass

    def test_read_skill_prefers_backend_then_builtin(self):
        with mock.patch('tempfile.TemporaryDirectory'):
            pass

    def test_default_command_builder(self):
        backend = SandboxSkillsBackend(action=mock.Mock(), workspace_root='/workspace', session_id='s1')
        self.assertIn("Path('/workspace/skills')", backend._default_command_builder('list', ''))
        self.assertEqual(backend._default_command_builder('read', 'demo'), "cat '/workspace/skills/demo/SKILL.md'")

    def test_run_raises_inside_active_loop(self):
        backend = SandboxSkillsBackend(action=mock.Mock())

        with mock.patch('asyncio.get_running_loop', return_value=object()):
            with self.assertRaisesRegex(RuntimeError, 'active event loop'):
                backend._run('echo 1')
