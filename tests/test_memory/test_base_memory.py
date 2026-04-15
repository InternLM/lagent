"""Tests for lagent.memory.base_memory.Memory"""

import asyncio

from lagent.memory.base_memory import Memory
from lagent.schema import AgentMessage


async def test_add_and_len():
    m = Memory()
    m.add(AgentMessage(sender='user', content='hello'))
    m.add(AgentMessage(sender='assistant', content='hi'))
    m.add(AgentMessage(sender='user', content='how are you'))
    assert len(m.memory) == 3


async def test_delete_multi_index():
    m = Memory()
    for i in range(5):
        m.add(AgentMessage(sender='user', content=f'msg{i}'))
    m.delete([1, 3])
    remaining = [msg.content for msg in m.memory]
    assert remaining == ['msg0', 'msg2', 'msg4']


async def test_delete_single_index():
    m = Memory()
    for i in range(3):
        m.add(AgentMessage(sender='user', content=f'msg{i}'))
    m.delete(1)
    remaining = [msg.content for msg in m.memory]
    assert remaining == ['msg0', 'msg2']


async def test_reset():
    m = Memory()
    m.add(AgentMessage(sender='user', content='hello'))
    m.reset()
    assert len(m.memory) == 0


async def test_save_load_roundtrip():
    m = Memory()
    m.add(AgentMessage(sender='user', content='test'))
    saved = m.save()
    m2 = Memory()
    m2.load(saved)
    assert len(m2.memory) == 1
    assert m2.memory[0].content == 'test'


async def test_get_memory_all():
    m = Memory()
    for i in range(5):
        m.add(AgentMessage(sender='user', content=f'msg{i}'))
    assert len(m.get_memory()) == 5


async def test_get_memory_with_filter():
    m = Memory()
    m.add(AgentMessage(sender='user', content='keep'))
    m.add(AgentMessage(sender='assistant', content='skip'))
    m.add(AgentMessage(sender='user', content='keep2'))
    filtered = m.get_memory(filter_func=lambda i, msg: msg.sender == 'user')
    assert len(filtered) == 2


async def test_add_string():
    m = Memory()
    m.add('hello')
    assert len(m.memory) == 1
    assert m.memory[0].content == 'hello'


async def test_add_none():
    m = Memory()
    m.add(None)
    assert len(m.memory) == 0


async def main():
    tests = [
        test_add_and_len,
        test_delete_multi_index,
        test_delete_single_index,
        test_reset,
        test_save_load_roundtrip,
        test_get_memory_all,
        test_get_memory_with_filter,
        test_add_string,
        test_add_none,
    ]
    for test in tests:
        await test()
        print(f"  {test.__name__}: OK")
    print("  ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
