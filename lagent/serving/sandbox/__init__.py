"""lagent.serving.sandbox — sandbox deployment via Unix socket daemon.

Server side (runs inside sandbox):
    - :class:`BaseDaemon` — socket server + protocol
    - :class:`ActionDaemon` — action execution
    - :class:`SkillsDaemon` — skills loading
    - :class:`AgentDaemon` — full agent

Client side:
    - :class:`SandboxAgent` — drop-in for AsyncAgent
    - ``SandboxActionExecutor`` — see ``lagent.actions.sandbox_executor``
    - ``SandboxSkillsLoader`` — see ``lagent.skills.sandbox_skills``
"""

__all__ = [
    "BaseDaemon",
    "ActionDaemon",
    "SkillsDaemon",
    "AgentDaemon",
    "SandboxAgent",
    "lagent_call",
    "async_lagent_call",
]


def __getattr__(name):
    if name == "SandboxAgent":
        from .agent import SandboxAgent

        return SandboxAgent
    if name in {
        "BaseDaemon",
        "ActionDaemon",
        "SkillsDaemon",
        "AgentDaemon",
        "lagent_call",
        "async_lagent_call",
    }:
        from .daemon import ActionDaemon, AgentDaemon, BaseDaemon, SkillsDaemon, async_lagent_call, lagent_call

        values = {
            "BaseDaemon": BaseDaemon,
            "ActionDaemon": ActionDaemon,
            "SkillsDaemon": SkillsDaemon,
            "AgentDaemon": AgentDaemon,
            "lagent_call": lagent_call,
            "async_lagent_call": async_lagent_call,
        }
        return values[name]
    raise AttributeError(name)
