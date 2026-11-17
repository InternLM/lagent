# DBox provider and allocation tracking

This layer builds on Lagent #385 (`fix/session-request-provenance-v1`). It retains
that PR's Mini-SWE adapter, request-ID propagation and `SessionClient` timeout
interface. DBox lifecycle and transport support are prerequisites imported from
the existing sandbox implementation, rather than changes to the session adapter.

`DBoxProvider.create(..., on_allocated=callback)` and `GatewayProvider.create`
call the asynchronous callback as soon as they receive an allocated ID, before
readiness polling. XTuner uses the
callback to persist ownership even if polling fails or the worker later exits.
Callback failure enters provider rollback. Failed rollback retains the owned ID.

The shared `providers.lifecycle` module defines this ordering for both HTTP
adapters: local validation, one allocation request, ownership notification,
platform readiness, and client construction. Readiness rejects terminal states
and limits both polling and status requests to one budget. Each adapter retains
its own request format and rollback cancellation behavior; the shared allocation
scope preserves the original exception if rollback also fails. `SandboxProvider`
documents the async contract, with typed callback and normalized status fields.

Both providers explicitly expose `supports_deletion_confirmation=True`: their
`get()` interface supplies lifecycle status, including a normalized `deleted`
result for HTTP 404. The owning Pool can confirm deletion independently of
whether the provider performs an additional confirmation inside `delete()`.

Only validation or authentication failure before the create POST is sent proves
`allocation_state="not_allocated"`. A lost response after sending the POST remains
unknown and must not trigger blind create retries or a successful cleanup result.

Trusted credential files accept plain AK:SK, JSON `aksk` / `api_key`, or JSON
`access_key_id` / `secret_key`. Credentials stay in management requests and are not
forwarded to task commands or serialized into XTuner resource journals.
Both HTTP adapters use the same trusted-file parser and reread the file for each
allocation. Direct DBox also rereads it whenever authenticating another request.
DBox retains an execd token only on its returned client;
provider ownership tracking does not keep a second token registry.

DBox file uploads require an absolute POSIX destination; relative paths fail
before transport. Small and large uploads use a private staging file beside the
resolved destination and replace it atomically after all bytes are written.
Existing destination mode, owner and group are restored before replacement; a
metadata failure leaves the previous file intact. Newly created files are 0600,
so provisioners must explicitly grant workload users any required access.
Destination symlinks are resolved once per upload, so changing a symlink during a
chunked transfer cannot redirect its later writes or cleanup. Uploads up to
24,000 bytes still use one execution request; larger uploads send bounded chunks
without changing these destination rules. This works with the existing mounted
remote-exec executable and requires no new upload-server endpoint.

The provider manages exact IDs and exposes lifecycle GET separately from command
execution. It does not scan projects or run a background recycler. XTuner owns
durable journals, exit recovery, concurrency limits and retry budgets. Recovery
requires a factory rebuilding the same endpoint, project and identity.

The accompanying shared process monitor distinguishes an exited process from an
unavailable probe. Execd streams with an integer `execution_complete.exit_code`
terminate at that event and require zero for success. Legacy completion events
without an exit code drain to HTTP EOF, including late output and errors. Both
forms share one bounded transport wall deadline; legacy streams without EOF fail
at that deadline. Existing Gateway mode remains available.
`DBoxSandboxClient.configure_network` accepts peer clients belonging to the same
provider. `get_diagnostics` exposes only state, inspect, and event GET requests.
Recipes can use these interfaces without accessing provider identity or private
HTTP methods.
