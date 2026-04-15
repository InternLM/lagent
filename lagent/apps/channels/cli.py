import asyncio
import sys
import logging
from .base import BaseChannel
from lagent.apps.bus import MessageBus, InboundEvent

logger = logging.getLogger("lagent.interclaw.channels.cli")

class CLIChannel(BaseChannel):
    """
    终端互动频道：真实可用的互动命令行端点。
    不仅负责把你的键盘敲击传给 Agent，也负责把 Agent 回收的话打印回控制台。
    """
    def __init__(self, bus: MessageBus):
        super().__init__(bus, channel_name="cli")

    async def _read_loop(self):
        """Producer: 将用户的回车敲击送如总线"""
        loop = asyncio.get_running_loop()
        while True:
            try:
                # 使用 run_in_executor 防止 input() 阻塞 asyncio 事件循环
                line = await loop.run_in_executor(None, input, "\n[You] ➤ ")
                text = line.strip()
                if text.lower() in ('exit', 'quit', 'q'):
                    logger.info("User requested exit.")
                    # 优雅地通知主程序终止
                    import os, signal
                    os.kill(os.getpid(), signal.SIGINT)
                    break
                
                if text:
                    event = InboundEvent(
                        channel=self.channel_name,
                        session_id="cli:local_master",
                        content=text
                    )
                    await self.bus.publish_inbound(event)
            except EOFError:
                break
            except asyncio.CancelledError:
                break

    async def _write_loop(self):
        """Consumer: 监听分配给 CLI 的 OutboundEvent 并打印"""
        while True:
            try:
                event = await self.outbound_queue.get()
                print(f"\n[Agent] ⚡ {event.content}\n", end="", flush=True)
            except asyncio.CancelledError:
                break

    async def start(self):
        logger.info("CLI Channel activated. You can start typing.")
        tasks = [
            asyncio.create_task(self._read_loop(), name="CLI_Reader"),
            asyncio.create_task(self._write_loop(), name="CLI_Writer"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
