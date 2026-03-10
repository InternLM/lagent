import asyncio
import logging
from .bus import MessageBus, OutboundEvent
from .session import SessionManager

logger = logging.getLogger("lagent.interclaw.dispatcher")

class InterclawDispatcher:
    """
    核心调度网关。
    将异步事件总线 (MessageBus) 与同步/异步的大模型计算 (Lagent) 完美缝合。
    """
    def __init__(self, bus: MessageBus, session_manager: SessionManager, agent_engine=None):
        self.bus = bus
        self.session_manager = session_manager
        self._running = False
        self.agent_engine = agent_engine

    async def start(self) -> None:
        self._running = True
        logger.info("Interclaw Dispatcher initialized. Ready to orchestrate the Agents.")
        
        while self._running:
            try:
                event = await self.bus.consume_inbound()
                logger.info(f"==> Inbound Hit | Channel:[{event.channel}] Session:[{event.session_id}]")
                
                # 1. 挂载持久化记忆
                state = self.session_manager.load_state(event.session_id)
                
                # 2. 从“休眠仓”唤醒 Agent (纯粹无状态)
                agent = self.agent_engine
                # self.agent_engine.load_state_dict(state)
                
                # 3. 运行思考
                reply = await agent(event.content)
                
                # 4. 把变更后的神经节点冷冻回冰柜
                # self.session_manager.save_state(event.session_id, agent.state_dict())
                
                # 5. 返还结果给各自负责分发的人
                if reply:
                    await self.bus.publish_outbound(OutboundEvent(
                        channel=event.channel,
                        session_id=event.session_id,
                        content=reply.content
                    ))
                    
            except asyncio.CancelledError:
                logger.info("Dispatcher task cancelled properly.")
                break
            except Exception as e:
                logger.error(f"Dispatcher internal error: {e}", exc_info=True)

    def stop(self):
        self._running = False
