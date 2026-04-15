"""Unit tests for Mailbox (lagent/services/mailbox.py)."""

import asyncio
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

# --- bypass circular import in lagent.services.__init__.py ---
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)
if "lagent.services" not in sys.modules:
    _pkg = types.ModuleType("lagent.services")
    _pkg.__path__ = [os.path.join(_here, "lagent", "services")]
    _pkg.__package__ = "lagent.services"
    sys.modules["lagent.services"] = _pkg

from lagent.schema import AgentMessage
from lagent.services.mailbox import Mailbox


def _msg(sender: str, receiver: str, content: str) -> AgentMessage:
    return AgentMessage(sender=sender, receiver=receiver, content=content)


# ═══════════════════════════════════════════════════════════════════════
#  REGISTRATION
# ═══════════════════════════════════════════════════════════════════════

class TestRegister:
    def test_register_creates_queue(self):
        mb = Mailbox()
        mb.register("agent-A")
        assert "agent-A" in mb.agents

    def test_register_idempotent(self):
        mb = Mailbox()
        mb.register("agent-A")
        mb.register("agent-A")
        assert mb.agents.count("agent-A") == 1

    def test_agents_empty_initially(self):
        mb = Mailbox()
        assert mb.agents == []


# ═══════════════════════════════════════════════════════════════════════
#  SEND
# ═══════════════════════════════════════════════════════════════════════

class TestSend:
    def test_send_basic(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "hello"))
        assert mb.has_messages("B")
        assert not mb.has_messages("A")

    def test_send_creates_queue_for_receiver(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "hello"))
        assert "B" in mb.agents

    def test_send_without_receiver_raises(self):
        mb = Mailbox()
        with pytest.raises(ValueError, match="receiver"):
            mb.send(AgentMessage(sender="A", content="oops"))

    def test_send_multiple_messages(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "msg1"))
        mb.send(_msg("A", "B", "msg2"))
        mb.send(_msg("C", "B", "msg3"))
        assert mb.message_count("B") == 3

    def test_broadcast(self):
        mb = Mailbox()
        mb.register("A")
        mb.register("B")
        mb.register("C")
        mb.send(_msg("A", "*", "hello everyone"))
        assert mb.has_messages("B")
        assert mb.has_messages("C")
        assert not mb.has_messages("A")  # sender excluded

    def test_broadcast_to_unregistered_skips(self):
        mb = Mailbox()
        mb.register("A")
        # No B or C registered
        mb.send(_msg("A", "*", "hello"))
        # Should not crash, just no recipients
        assert mb.message_count() == 0


# ═══════════════════════════════════════════════════════════════════════
#  DRAIN
# ═══════════════════════════════════════════════════════════════════════

class TestDrain:
    def test_drain_returns_messages(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "hello"))
        mb.send(_msg("A", "B", "world"))
        msgs = mb.drain("B")
        assert len(msgs) == 2
        assert msgs[0].content == "hello"
        assert msgs[1].content == "world"

    def test_drain_clears_queue(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "hello"))
        mb.drain("B")
        assert not mb.has_messages("B")
        assert mb.drain("B") == []

    def test_drain_empty_returns_empty(self):
        mb = Mailbox()
        assert mb.drain("nonexistent") == []

    def test_drain_only_affects_target(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "for B"))
        mb.send(_msg("A", "C", "for C"))
        mb.drain("B")
        assert mb.has_messages("C")


# ═══════════════════════════════════════════════════════════════════════
#  PEEK
# ═══════════════════════════════════════════════════════════════════════

class TestPeek:
    def test_peek_returns_messages(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "hello"))
        msgs = mb.peek("B")
        assert len(msgs) == 1

    def test_peek_does_not_remove(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "hello"))
        mb.peek("B")
        assert mb.has_messages("B")

    def test_peek_returns_copy(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "hello"))
        peeked = mb.peek("B")
        peeked.clear()
        assert mb.has_messages("B")  # original not affected

    def test_peek_empty(self):
        mb = Mailbox()
        assert mb.peek("nobody") == []


# ═══════════════════════════════════════════════════════════════════════
#  HAS_MESSAGES / MESSAGE_COUNT
# ═══════════════════════════════════════════════════════════════════════

class TestQueryMethods:
    def test_has_messages(self):
        mb = Mailbox()
        assert not mb.has_messages("B")
        mb.send(_msg("A", "B", "hello"))
        assert mb.has_messages("B")

    def test_message_count_per_agent(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "1"))
        mb.send(_msg("A", "B", "2"))
        mb.send(_msg("A", "C", "3"))
        assert mb.message_count("B") == 2
        assert mb.message_count("C") == 1

    def test_message_count_total(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "1"))
        mb.send(_msg("A", "C", "2"))
        assert mb.message_count() == 2


# ═══════════════════════════════════════════════════════════════════════
#  CLEAR
# ═══════════════════════════════════════════════════════════════════════

class TestClear:
    def test_clear_one_agent(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "1"))
        mb.send(_msg("A", "C", "2"))
        mb.clear("B")
        assert not mb.has_messages("B")
        assert mb.has_messages("C")

    def test_clear_all(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "1"))
        mb.send(_msg("A", "C", "2"))
        mb.clear_all()
        assert mb.message_count() == 0


# ═══════════════════════════════════════════════════════════════════════
#  WAIT_FOR_MESSAGE (async)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestWaitForMessage:
    async def test_wait_receives_signal(self):
        mb = Mailbox()
        mb.register("B")

        async def sender():
            await asyncio.sleep(0.1)
            mb.send(_msg("A", "B", "hello"))

        task = asyncio.create_task(sender())
        result = await mb.wait_for_message("B", timeout=2.0)
        assert result is True
        assert mb.has_messages("B")
        await task

    async def test_wait_timeout(self):
        mb = Mailbox()
        mb.register("B")
        result = await mb.wait_for_message("B", timeout=0.1)
        assert result is False

    async def test_wait_auto_registers(self):
        mb = Mailbox()
        # Don't explicitly register
        async def sender():
            await asyncio.sleep(0.1)
            mb.send(_msg("A", "B", "hello"))

        task = asyncio.create_task(sender())
        result = await mb.wait_for_message("B", timeout=2.0)
        assert result is True
        await task


# ═══════════════════════════════════════════════════════════════════════
#  STATE_DICT / LOAD_STATE_DICT
# ═══════════════════════════════════════════════════════════════════════

class TestStateDict:
    def test_roundtrip(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "hello"))
        mb.send(_msg("C", "B", "world"))
        mb.send(_msg("A", "D", "test"))

        state = mb.state_dict()

        mb2 = Mailbox()
        mb2.load_state_dict(state)

        assert mb2.message_count("B") == 2
        assert mb2.message_count("D") == 1
        msgs = mb2.drain("B")
        assert msgs[0].content == "hello"
        assert msgs[1].content == "world"

    def test_state_dict_format(self):
        mb = Mailbox()
        mb.send(_msg("A", "B", "hello"))
        state = mb.state_dict()
        assert state["version"] == 1
        assert "B" in state["queues"]
        assert len(state["queues"]["B"]) == 1
        assert state["queues"]["B"][0]["sender"] == "A"


# ═══════════════════════════════════════════════════════════════════════
#  FILE PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════

class TestFilePersistence:
    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mailbox.json"

            mb = Mailbox(store_path=path)
            mb.send(_msg("A", "B", "hello"))
            mb.send(_msg("C", "D", "world"))
            assert path.exists()

            mb2 = Mailbox(store_path=path)
            assert mb2.has_messages("B")
            assert mb2.has_messages("D")
            msgs = mb2.drain("B")
            assert len(msgs) == 1
            assert msgs[0].content == "hello"

    def test_persist_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mailbox.json"
            mb = Mailbox(store_path=path)
            mb.send(_msg("A", "B", "test"))

            data = json.loads(path.read_text("utf-8"))
            assert data["version"] == 1
            assert "B" in data["queues"]
            assert data["queues"]["B"][0]["content"] == "test"

    def test_drain_updates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mailbox.json"
            mb = Mailbox(store_path=path)
            mb.send(_msg("A", "B", "hello"))
            mb.drain("B")

            data = json.loads(path.read_text("utf-8"))
            assert data["queues"].get("B", []) == []
