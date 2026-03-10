import asyncio
import json
import time
import logging
from pathlib import Path
from lagent.interclaw.bus import MessageBus, InboundEvent

logger = logging.getLogger("lagent.interclaw.services.cron")

class CronService:
    """
    真实的定时任务调度器：模仿 nanobot 实现。
    独立地扫描 json 表，一旦发现过期任务，则向推总线投放。无任何强耦合。
    """
    def __init__(self, bus: MessageBus, db_path: str = ".interclaw_data/cron_jobs.json"):
        self.bus = bus
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 初始化默认数据
        if not self.db_path.exists():
            with open(self.db_path, "w") as f:
                json.dump([], f)

    def _load_jobs(self):
        try:
            with open(self.db_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_jobs(self, jobs):
        with open(self.db_path, "w") as f:
            json.dump(jobs, f, indent=2)

    async def start(self):
        """按分钟级别的精度轮询 jobs"""
        logger.info(f"Cron Service started, watching {self.db_path}")
        while True:
            try:
                jobs = self._load_jobs()
                now = time.time()
                pending = []
                remaining = []
                
                for job in jobs:
                    if job.get("trigger_time", 0) <= now:
                        pending.append(job)
                    else:
                        remaining.append(job)
                        
                if pending:
                    # 将尚未执行的覆盖回硬盘
                    self._save_jobs(remaining)
                    # 依次触发
                    for job in pending:
                        logger.info(f"Cron FIRED: {job.get('id')}")
                        await self.bus.publish_inbound(InboundEvent(
                            channel="cron",
                            # 以单独的隔离身份执行，避免干扰正常的聊天记忆，或直接融合进用户记忆
                            session_id=job.get("session_id", "cron:system"),
                            content=f"[系统按时触发] {job.get('task_content')}"
                        ))
                
                # 每秒滴答探测一次
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cron loop failure: {e}")
                await asyncio.sleep(5)
