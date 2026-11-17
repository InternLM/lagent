"""Gateway allocation notifications must survive readiness and rollback."""

import asyncio
import unittest

import httpx

from lagent.serving.sandbox.providers.gateway import GatewayProvider


class GatewayAllocationTests(unittest.IsolatedAsyncioTestCase):
    async def provider(self, handler):
        provider = GatewayProvider("http://gateway.test", user="test-user", project="test-project")
        await provider._client.aclose()
        provider._client = httpx.AsyncClient(base_url=provider.gateway_url, transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(provider.aclose)
        return provider

    async def test_pending_allocation_is_reported_before_readiness_and_cancellation(self):
        recorded, requests = [], []
        allocated = asyncio.Event()

        def handler(request):
            requests.append(request.method)
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json={"ok": True, "env": {"env_id": "pending", "status": "pending", "url": "http://sandbox.test"}},
                )
            return httpx.Response(404)

        provider = await self.provider(handler)

        async def on_allocated(sandbox_id):
            self.assertIn(sandbox_id, provider.owned_env_ids)
            self.assertEqual(requests, ["POST"])
            recorded.append(sandbox_id)
            allocated.set()

        task = asyncio.create_task(
            provider.create("image", cluster_name="dbox", key="test-ak:test-sk", on_allocated=on_allocated)
        )
        await asyncio.wait_for(allocated.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(recorded, ["pending"])
        self.assertFalse(provider.owned_env_ids)
        self.assertEqual(requests, ["POST", "DELETE", "GET"])

    async def test_successful_rollback_retains_allocation_evidence_for_caller(self):
        recorded, requests = [], []

        def handler(request):
            requests.append(request.method)
            if request.method == "POST":
                return httpx.Response(200, json={"ok": True, "env": {"env_id": "failed", "status": "failed"}})
            return httpx.Response(204 if request.method == "DELETE" else 404)

        provider = await self.provider(handler)

        async def on_allocated(sandbox_id):
            recorded.append(sandbox_id)

        with self.assertRaisesRegex(RuntimeError, "terminal state"):
            await provider.create("image", cluster_name="dbox", key="test-ak:test-sk", on_allocated=on_allocated)
        self.assertEqual(recorded, ["failed"])
        self.assertFalse(provider.owned_env_ids)
        self.assertEqual(requests, ["POST", "DELETE", "GET"])

    async def test_callback_failure_rolls_back_without_forwarding_callback(self):
        requests = []

        def handler(request):
            requests.append(request.method)
            self.assertNotIn(b"on_allocated", request.content)
            if request.method == "POST":
                return httpx.Response(
                    200, json={"ok": True, "env": {"env_id": "legacy", "url": "http://sandbox.test"}}
                )
            return httpx.Response(404)

        provider = await self.provider(handler)

        async def on_allocated(sandbox_id):
            raise OSError("journal unavailable")

        with self.assertRaisesRegex(OSError, "journal unavailable"):
            await provider.create("image", on_allocated=on_allocated)
        self.assertEqual(requests, ["POST", "DELETE"])
        self.assertFalse(provider.owned_env_ids)

    async def test_pre_request_validation_and_credential_errors_are_not_allocated(self):
        provider = await self.provider(lambda request: self.fail("no HTTP request expected"))
        for options in (
            {"key": "incomplete"},
            {"key": "test-ak:test-sk", "ttl_seconds": 1},
            {"credential_file": "/does-not-exist/review-credential"},
        ):
            with self.subTest(options=list(options)):
                with self.assertRaises((ValueError, OSError)) as error:
                    await provider.create("image", cluster_name="dbox", **options)
                self.assertEqual(error.exception.allocation_state, "not_allocated")

    async def test_sent_post_timeout_remains_ambiguous(self):
        requests = []

        def handler(request):
            requests.append(request.method)
            raise httpx.ReadTimeout("response lost", request=request)

        provider = await self.provider(handler)
        with self.assertRaisesRegex(RuntimeError, "ambiguous") as error:
            await provider.create("image", cluster_name="dbox", key="test-ak:test-sk")
        self.assertNotEqual(getattr(error.exception, "allocation_state", None), "not_allocated")
        self.assertEqual(requests, ["POST"])
