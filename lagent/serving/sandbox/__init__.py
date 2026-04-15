"""lagent.serving.sandbox — sandbox deployment via Unix socket daemon.

Server side (runs inside sandbox):
    - :class:`BaseDaemon` — socket server + protocol
    - :class:`ActionDaemon` — Level 1: action execution
    - :class:`AgentDaemon` — Level 2: full agent

Client side:
    - :class:`SandboxAgent` — drop-in for AsyncAgent (runs outside sandbox)
    - ``SandboxActionExecutor`` — see ``lagent.actions.sandbox_executor``
    - ``HybridActionExecutor`` — see ``lagent.actions.hybrid_executor``
"""

from .agent import SandboxAgent
from .daemon import ActionDaemon, AgentDaemon, BaseDaemon, async_lagent_call, lagent_call

__all__ = [
    "BaseDaemon",
    "ActionDaemon",
    "AgentDaemon",
    "SandboxAgent",
    "lagent_call",
    "async_lagent_call",
]
