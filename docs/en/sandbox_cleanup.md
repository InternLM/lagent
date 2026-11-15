# DBox provider and allocation tracking

This layer builds on Lagent #385 (`fix/session-request-provenance-v1`). It retains
that PR's Mini-SWE adapter, request-ID propagation and `SessionClient` timeout
interface. DBox lifecycle and transport support are prerequisites imported from
the existing sandbox implementation, rather than changes to the session adapter.

`DBoxProvider.create(..., on_allocated=callback)` calls the asynchronous callback
as soon as it receives an allocated ID, before readiness polling. XTuner uses the
callback to persist ownership even if polling fails or the worker later exits.
Callback failure enters provider rollback. Failed rollback retains the owned ID.

Only validation or authentication failure before the create POST is sent proves
`allocation_state="not_allocated"`. A lost response after sending the POST remains
unknown and must not trigger blind create retries or a successful cleanup result.

Trusted credential files accept plain AK:SK, JSON `aksk` / `api_key`, or JSON
`access_key_id` / `secret_key`. Credentials stay in management requests and are not
forwarded to task commands or serialized into XTuner resource journals.

The provider manages exact IDs and exposes lifecycle GET separately from command
execution. It does not scan projects or run a background recycler. XTuner owns
durable journals, exit recovery, concurrency limits and retry budgets. Recovery
requires a factory rebuilding the same endpoint, project and identity.

The accompanying shared process monitor distinguishes an exited process from an
unavailable probe, and the network helper uses the same resolved addresses for
endpoint pinning and egress policy. Existing Gateway mode remains available.
