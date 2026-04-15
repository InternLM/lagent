"""lagent.serving — agent deployment backends.

Each sub-package provides a server + client pair for a specific transport:

- ``http``    — HTTP API (Starlette/uvicorn)
- ``ray``     — Ray Serve
- ``sandbox`` — Unix socket daemon via bash channel
"""
