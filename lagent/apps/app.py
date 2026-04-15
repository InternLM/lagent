import asyncio
import logging
from typing import List

from lagent.apps.bus import MessageBus
from lagent.apps.dispatcher import InterclawDispatcher
from lagent.apps.session import SessionManager

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # 允许打印 INFO 及以上级别的日志 (如果你想看 debug 就改成 logging.DEBUG)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class InterclawApp:
    """
    Interclaw OS 的核心应用容器 (Inversion of Control Container)
    负责管理总线、分发器，以及动态注册的各种通道和后台服务。
    """
    def __init__(self, agent_engine):
        """
        :param agent_engine: 核心的 Agent 引擎（纯无状态，如 lagent 实例或 FakeLagent）
        """
        self.bus = MessageBus()
        self.session_mgr = SessionManager()
        self.dispatcher = InterclawDispatcher(self.bus, self.session_mgr, agent_engine=agent_engine)
        self._channels = []
        self._services = []

    def register_channel(self, channel_class, **kwargs):
        """
        动态注册通信通道 (如 CLI, 飞书, 微信)
        自动将 bus 注入给 channel。
        """
        channel_instance = channel_class(self.bus, **kwargs)
        self._channels.append(channel_instance)
        logger.info(f"Registered channel: {channel_instance.__class__.__name__}")
        return self

    def register_service(self, service_class, **kwargs):
        """
        动态注册后台服务 (如 Cron, Heartbeat)
        自动将 bus 注入给 service。
        """
        service_instance = service_class(self.bus, **kwargs)
        self._services.append(service_instance)
        logger.info(f"Registered service: {service_instance.__class__.__name__}")
        return self

    async def _run_all(self):
        tasks = []
        # 1. 启动核心 Dispatcher (负责消费 inbound 数据并发送给 LLM，将结果推入 outbound)
        tasks.append(asyncio.create_task(self.dispatcher.start()))
        
        # 2. 启动所有的 Channels (负责独立监听外部输入，并阻塞等待 outbound 返回结果)
        for channel in self._channels:
            tasks.append(asyncio.create_task(channel.start()))
            
        # 3. 启动后台服务 (定时任务等)
        for service in self._services:
            tasks.append(asyncio.create_task(service.start()))

        # 启动并等待所有的事件循环
        await asyncio.gather(*tasks)

    def run(self):
        """启动整个 Agent OS 框架的入口"""
        logger.info("Starting Interclaw OS...")
        try:
            asyncio.run(self._run_all())
        except KeyboardInterrupt:
            logger.info("Interclaw OS interrupted by user. Shutting down...")
            # 这里可以补充优雅退出的逻辑
