import asyncio
import logging
from pathlib import Path
from lagent.apps.bus import MessageBus, InboundEvent

logger = logging.getLogger("lagent.interclaw.services.heartbeat")

class HeartbeatService:
    """
    状态心跳服务：定期扫描某个物理介质 (如 Markdown 文件) 
    发现未处理内容，即封装为一个内部指令并发送给大脑处理。
    """
    def __init__(self, bus: MessageBus, pulse_seconds: int = 15, file_path: str = ".interclaw_data/HEARTBEAT.md"):
        self.bus = bus
        self.pulse_seconds = pulse_seconds
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("No background tasks for now.\n")

    async def start(self):
        logger.info(f"Heartbeat Service started, pulse every {self.pulse_seconds}s for {self.file_path.name}")
        while True:
            try:
                await asyncio.sleep(self.pulse_seconds)
                
                if self.file_path.exists():
                    content = self.file_path.read_text(encoding="utf-8").strip()
                    # 避免空白文件造成无效触发，如果包含特定的关键字可以考虑唤醒
                    if content and "No background tasks" not in content and "TODO:" in content:
                        logger.info("Heartbeat found actionable items in Markdown!")
                        
                        await self.bus.publish_inbound(InboundEvent(
                            channel="heartbeat",
                            session_id="system:heartbeat",
                            content=f"请梳理处理并更新我的后台记录：\n{content}"
                        ))
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat Error: {e}")
