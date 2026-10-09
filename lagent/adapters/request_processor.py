"""Request/response hooks for :class:`SessionClient`.

Processors run inside the proxy boundary, before a request is serialized and
forwarded.  The processed request is therefore also the request recorded for
training; processors must never rewrite a trace after the model call.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProxyRequestContext:
    """Stable metadata for one request observed by the proxy."""

    session_id: str
    request_index: int
    provider: str
    path: str


class ProxyRequestProcessor:
    """No-op base class for optional, stateful proxy request processors."""

    def before_forward(
        self,
        request: dict[str, Any],
        context: ProxyRequestContext,
    ) -> dict[str, Any]:
        del context
        return request

    def after_response(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        context: ProxyRequestContext,
    ) -> None:
        del request, response, context

    def get_stats(self) -> dict[str, Any]:
        return {}

    def reset(self) -> None:
        pass
