import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Callable

logger = logging.getLogger("lagent.interclaw.bus")

@dataclass
class InboundEvent:
    channel: str
    session_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OutboundEvent:
    channel: str
    session_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class MessageBus:
    """
    基于 Pub/Sub 模式的高可用内存总线。
    支持一个统一的 Inbound 队列给 Dispatcher，以及多个按 Channel 路由的 Outbound 队列。
    """
    def __init__(self):
        self.inbound_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()
        self._outbound_queues: Dict[str, asyncio.Queue[OutboundEvent]] = {}

    def subscribe_outbound(self, channel: str) -> asyncio.Queue[OutboundEvent]:
        """供各种 Channel (如 cli, whatsapp) 注册并监听自己的发件箱"""
        if channel not in self._outbound_queues:
            self._outbound_queues[channel] = asyncio.Queue()
            logger.debug(f"Channel [{channel}] subscribed to outbound bus.")
        return self._outbound_queues[channel]

    async def publish_inbound(self, event: InboundEvent) -> None:
        """供所有外部刺激 (通道、定时器) 推送消息给大脑"""
        await self.inbound_queue.put(event)

    async def consume_inbound(self) -> InboundEvent:
        """供 Dispatcher 持续消耗任务"""
        return await self.inbound_queue.get()

    async def publish_outbound(self, event: OutboundEvent) -> None:
        """供 Dispatcher 推送大脑的回复到指定通道"""
        queue = self._outbound_queues.get(event.channel)
        if queue:
            await queue.put(event)
        else:
            logger.warning(f"No subscriber for channel: {event.channel}, message dropped.")
