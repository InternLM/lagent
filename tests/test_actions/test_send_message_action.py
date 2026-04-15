"""Unit tests for SendMessageAction (lagent/actions/send_message.py)."""

import os
import sys
import types

import pytest

# --- bypass circular import ---
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)
if "lagent.services" not in sys.modules:
    _pkg = types.ModuleType("lagent.services")
    _pkg.__path__ = [os.path.join(_here, "lagent", "services")]
    _pkg.__package__ = "lagent.services"
    sys.modules["lagent.services"] = _pkg

from lagent.actions.send_message import SendMessageAction, AsyncSendMessageAction
from lagent.schema import ActionStatusCode
from lagent.services.mailbox import Mailbox


def _make_action(
    agent_name: str = "agent-A",
    allowed_receivers=None,
) -> tuple[SendMessageAction, Mailbox]:
    mb = Mailbox()
    mb.register("agent-A")
    mb.register("agent-B")
    mb.register("lead")
    action = SendMessageAction(
        mailbox=mb,
        agent_name=agent_name,
        allowed_receivers=allowed_receivers,
    )
    return action, mb


# ═══════════════════════════════════════════════════════════════════════
#  SEND
# ═══════════════════════════════════════════════════════════════════════

class TestSend:
    def test_send_basic(self):
        action, mb = _make_action()
        result = action.send(to="agent-B", content="hello")
        assert result.result is not None
        assert "sent to @agent-B" in result.result[0]["content"]
        assert mb.has_messages("agent-B")

    def test_send_content_preserved(self):
        action, mb = _make_action()
        action.send(to="agent-B", content="specific message")
        msgs = mb.drain("agent-B")
        assert len(msgs) == 1
        assert msgs[0].content == "specific message"
        assert msgs[0].sender == "agent-A"
        assert msgs[0].receiver == "agent-B"

    def test_send_broadcast(self):
        action, mb = _make_action()
        result = action.send(to="*", content="hello all")
        assert "Broadcast" in result.result[0]["content"]
        assert mb.has_messages("agent-B")
        assert mb.has_messages("lead")
        assert not mb.has_messages("agent-A")  # sender excluded

    def test_send_allowed_receivers_permits(self):
        action, mb = _make_action(allowed_receivers=["lead"])
        result = action.send(to="lead", content="report")
        assert result.result is not None
        assert mb.has_messages("lead")

    def test_send_allowed_receivers_blocks(self):
        action, mb = _make_action(allowed_receivers=["lead"])
        result = action.send(to="agent-B", content="hello")
        assert result.state == ActionStatusCode.API_ERROR
        assert "Cannot send" in result.errmsg
        assert not mb.has_messages("agent-B")

    def test_send_broadcast_ignores_allowed(self):
        action, mb = _make_action(allowed_receivers=["lead"])
        result = action.send(to="*", content="hello all")
        # Broadcast is always allowed
        assert result.result is not None

    def test_send_no_restrictions(self):
        action, mb = _make_action(allowed_receivers=None)
        result = action.send(to="agent-B", content="hello")
        assert result.result is not None


# ═══════════════════════════════════════════════════════════════════════
#  CHECK_INBOX
# ═══════════════════════════════════════════════════════════════════════

class TestCheckInbox:
    def test_empty_inbox(self):
        action, mb = _make_action()
        result = action.check_inbox()
        assert "No new messages" in result.result[0]["content"]

    def test_with_messages(self):
        action, mb = _make_action()
        # Another agent sends to agent-A
        from lagent.schema import AgentMessage
        mb.send(AgentMessage(sender="agent-B", receiver="agent-A", content="hey there"))
        result = action.check_inbox()
        content = result.result[0]["content"]
        assert "1 pending" in content
        assert "@agent-B" in content
        assert "hey there" in content

    def test_check_inbox_does_not_consume(self):
        action, mb = _make_action()
        from lagent.schema import AgentMessage
        mb.send(AgentMessage(sender="agent-B", receiver="agent-A", content="hey"))
        action.check_inbox()
        assert mb.has_messages("agent-A")  # still there


# ═══════════════════════════════════════════════════════════════════════
#  TOOLKIT METADATA
# ═══════════════════════════════════════════════════════════════════════

class TestMeta:
    def test_is_toolkit(self):
        action, _ = _make_action()
        assert action.is_toolkit

    def test_has_two_apis(self):
        action, _ = _make_action()
        desc = action.description
        api_names = {api["name"] for api in desc["api_list"]}
        assert api_names == {"send", "check_inbox"}

    def test_auto_registers_agent(self):
        mb = Mailbox()
        action = SendMessageAction(mailbox=mb, agent_name="new-agent")
        assert "new-agent" in mb.agents


# ═══════════════════════════════════════════════════════════════════════
#  ASYNC VARIANT
# ═══════════════════════════════════════════════════════════════════════

class TestAsyncSendMessageAction:
    def test_instantiation(self):
        mb = Mailbox()
        action = AsyncSendMessageAction(mailbox=mb, agent_name="async-agent")
        assert action._mailbox is mb
        assert action._agent_name == "async-agent"
