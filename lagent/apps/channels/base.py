from abc import ABC, abstractmethod
from lagent.apps.bus import MessageBus

class BaseChannel(ABC):
    """
    通讯频造的基础抽象类 (例如 CLI, WhatsApp, WeChat 等)
    每一个 Channel 都有责任生产输入，并消费属于自己的输出。
    """
    def __init__(self, bus: MessageBus, channel_name: str):
        self.bus = bus
        self.channel_name = channel_name
        self.outbound_queue = bus.subscribe_outbound(channel_name)

    @abstractmethod
    async def start(self):
        """需要通过 asyncio.gather 并发运行读写逻辑"""
        pass
