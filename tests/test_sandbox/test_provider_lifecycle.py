"""Exercise the same allocation contract through both real HTTP adapters."""

import asyncio
import json
import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from lagent.serving.sandbox.providers import DBoxProvider, GatewayProvider, SandboxClient, SandboxProvider
from lagent.serving.sandbox.providers.lifecycle import wait_ready


class ProviderLifecycleContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.credential = Path(self.tmp.name) / 'credential'
        self.credential.write_text('test-access:test-secret')

    async def provider(self, adapter, handler, **kwargs):
        if adapter == 'direct':
            provider = DBoxProvider(
                'http://provider.test',
                project='test-project',
                credential_file=str(self.credential),
                poll_interval=0.001,
                **kwargs,
            )
        else:
            provider = GatewayProvider(
                'http://provider.test',
                user='test-user',
                project='test-project',
                credential_file=str(self.credential),
                **kwargs,
            )
        await provider._client.aclose()
        provider._client = httpx.AsyncClient(base_url='http://provider.test', transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(provider.aclose)
        self.assertIsInstance(provider, SandboxProvider)
        return provider

    @staticmethod
    def response(adapter, status):
        if adapter == 'direct':
            return httpx.Response(201, json={'id': 'allocated', 'state': status})
        return httpx.Response(
            200, json={'ok': True, 'env': {'env_id': 'allocated', 'status': status, 'url': 'http://sandbox.test'}}
        )

    async def test_trusted_credential_formats_and_rotation_use_public_create(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                requests = []

                def handler(request):
                    requests.append(request)
                    return self.response(adapter, 'ready')

                provider = await self.provider(adapter, handler)
                key = 'test-access:test-secret'
                for serialized, expected in (
                    (key, key),
                    (json.dumps({'aksk': key}), key),
                    (json.dumps({'api_key': key}), key),
                    (json.dumps({'access_key_id': 'test-access', 'secret_key': 'test-secret'}), key),
                    ('rotated-access:rotated-secret', 'rotated-access:rotated-secret'),
                ):
                    self.credential.write_text(serialized)
                    client, _ = await provider.create('image', cluster_name='dbox')
                    self.addAsyncCleanup(client.aclose)
                    request = requests[-1]
                    self.assertEqual(request.method, 'POST')
                    if adapter == 'direct':
                        actual = request.headers['OPEN-SANDBOX-API-KEY']
                        self.assertNotIn(expected, request.content.decode())
                    else:
                        actual = json.loads(request.content)['extra_params']['aksk']
                    self.assertEqual(actual, expected)
                    self.assertNotIn(expected, repr(client))
                    self.assertNotIn(expected, str(client._client.headers))
                self.assertEqual(len(requests), 5)

    async def test_invalid_credential_files_fail_before_post_without_echo(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                provider = await self.provider(adapter, lambda request: self.fail('no POST expected'))
                for content in (
                    'private-incomplete',
                    '{"api_key":"private-incomplete"}',
                    '{"access_key_id":"private-access"}',
                    '{"access_key_id":"private-access","secret_key":"private secret"}',
                    '{"aksk": "private-malformed-json"',
                ):
                    self.credential.write_text(content)
                    with self.assertRaises(ValueError) as error:
                        await provider.create('image', cluster_name='dbox')
                    self.assertEqual(error.exception.allocation_state, 'not_allocated')
                    self.assertNotIn('private', ''.join(traceback.format_exception(error.exception)))
                self.credential.unlink()
                with self.assertRaises(ValueError) as error:
                    await provider.create('image', cluster_name='dbox')
                self.assertEqual(error.exception.allocation_state, 'not_allocated')
                self.assertNotIn(str(self.credential), ''.join(traceback.format_exception(error.exception)))

    async def test_ownership_and_callback_precede_readiness_get(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                requests, recorded = [], []

                def handler(request):
                    requests.append(request.method)
                    self.assertNotIn(b'on_allocated', request.content)
                    if request.method == 'POST':
                        return self.response(adapter, 'pending')
                    self.assertEqual(recorded, ['allocated'])
                    return self.response(adapter, 'ready')

                provider = await self.provider(adapter, handler)

                async def allocated(sandbox_id):
                    self.assertEqual(requests, ['POST'])
                    self.assertIn(sandbox_id, provider.owned_env_ids)
                    recorded.append(sandbox_id)

                client, sandbox_id = await provider.create('image', cluster_name='dbox', on_allocated=allocated)
                self.addAsyncCleanup(client.aclose)
                self.assertEqual(sandbox_id, 'allocated')
                self.assertEqual(requests, ['POST', 'GET'])
                self.assertEqual(await client.get_lifecycle_state(), 'ready')
                self.assertEqual(requests, ['POST', 'GET', 'GET'])

    async def test_callback_failure_preserves_primary_error_and_retryable_ownership(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                requests = []

                def handler(request):
                    requests.append(request.method)
                    if request.method == 'POST':
                        return self.response(adapter, 'pending')
                    return httpx.Response(503)

                provider = await self.provider(adapter, handler)
                primary = OSError('journal unavailable')

                async def allocated(sandbox_id):
                    raise primary

                with self.assertRaises(OSError) as error:
                    await provider.create('image', cluster_name='dbox', on_allocated=allocated)
                self.assertIs(error.exception, primary)
                self.assertEqual(provider.owned_env_ids, {'allocated'})
                self.assertEqual(requests, ['POST', 'DELETE'])

    async def test_expired_allocation_keeps_caller_evidence_after_confirmed_rollback(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                requests, recorded = [], []

                def handler(request):
                    requests.append(request.method)
                    if request.method == 'POST':
                        return self.response(adapter, 'expired')
                    return httpx.Response(204 if request.method == 'DELETE' else 404)

                provider = await self.provider(adapter, handler)

                async def allocated(sandbox_id):
                    recorded.append(sandbox_id)

                with self.assertRaisesRegex(RuntimeError, 'terminal state'):
                    await provider.create('image', cluster_name='dbox', on_allocated=allocated)
                self.assertEqual(recorded, ['allocated'])
                self.assertFalse(provider.owned_env_ids)
                self.assertEqual(requests, ['POST', 'DELETE', 'GET'])

    async def test_invalid_input_is_not_allocated_but_sent_post_timeout_is_unknown(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                requests = []

                def handler(request):
                    requests.append(request.method)
                    raise httpx.ReadTimeout('response lost', request=request)

                provider = await self.provider(adapter, handler)
                with self.assertRaises(ValueError) as error:
                    await provider.create('image', ttl_seconds=1, cluster_name='dbox')
                self.assertEqual(error.exception.allocation_state, 'not_allocated')
                self.assertEqual(requests, [])
                with self.assertRaisesRegex(RuntimeError, 'ambiguous') as error:
                    await provider.create('image', cluster_name='dbox')
                self.assertNotEqual(getattr(error.exception, 'allocation_state', None), 'not_allocated')
                self.assertEqual(requests, ['POST'])

    async def test_cancel_after_notification_rolls_back_and_propagates_cancellation(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                recorded, requests = [], []
                published = asyncio.Event()

                def handler(request):
                    requests.append(request.method)
                    if request.method == 'POST':
                        return self.response(adapter, 'pending')
                    return httpx.Response(204 if request.method == 'DELETE' else 404)

                provider = await self.provider(adapter, handler)

                async def allocated(sandbox_id):
                    recorded.append(sandbox_id)
                    published.set()
                    await asyncio.Event().wait()

                task = asyncio.create_task(provider.create('image', cluster_name='dbox', on_allocated=allocated))
                await asyncio.wait_for(published.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(recorded, ['allocated'])
                self.assertFalse(provider.owned_env_ids)
                self.assertEqual(requests, ['POST', 'DELETE', 'GET'])

    async def test_cancellation_during_rollback_keeps_adapter_transport_and_primary_error(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                deleting, release_delete, deleted = (asyncio.Event() for _ in range(3))

                async def handler(request):
                    if request.method == 'POST':
                        return self.response(adapter, 'pending')
                    if request.method == 'DELETE':
                        deleting.set()
                        await release_delete.wait()
                        return httpx.Response(204)
                    deleted.set()
                    return httpx.Response(404)

                provider = await self.provider(adapter, handler)
                primary = OSError('journal unavailable')

                async def allocated(sandbox_id):
                    raise primary

                task = asyncio.create_task(provider.create('image', cluster_name='dbox', on_allocated=allocated))
                await asyncio.wait_for(deleting.wait(), timeout=1)
                task.cancel()
                if adapter == 'gateway':
                    # Gateway keeps its shielded delete running independently.
                    with self.assertRaises(OSError) as error:
                        await asyncio.wait_for(task, timeout=1)
                    self.assertIs(error.exception, primary)
                    self.assertEqual(provider.owned_env_ids, {'allocated'})
                else:
                    # Direct DBox waits for the in-flight delete even when
                    # cancellation arrives during rollback.
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                release_delete.set()
                await asyncio.wait_for(deleted.wait(), timeout=1)
                if adapter == 'direct':
                    with self.assertRaises(OSError) as error:
                        await task
                    self.assertIs(error.exception, primary)
                self.assertFalse(provider.owned_env_ids)

    async def test_readiness_budget_shorter_than_poll_interval_still_rolls_back(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                requests = []

                def handler(request):
                    requests.append(request.method)
                    if request.method == 'POST':
                        return self.response(adapter, 'pending')
                    return httpx.Response(204 if request.method == 'DELETE' else 404)

                provider = await self.provider(adapter, handler, ready_timeout=0.0001)
                with self.assertRaisesRegex(TimeoutError, 'readiness timed out'):
                    await asyncio.wait_for(provider.create('image', cluster_name='dbox'), timeout=1)
                self.assertFalse(provider.owned_env_ids)
                self.assertEqual(requests, ['POST', 'DELETE', 'GET'])

    async def test_repeated_cancellation_does_not_cancel_inflight_delete(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                published, deleting, release_delete, deleted = (asyncio.Event() for _ in range(4))

                async def handler(request):
                    if request.method == 'POST':
                        return self.response(adapter, 'pending')
                    if request.method == 'DELETE':
                        deleting.set()
                        await release_delete.wait()
                        return httpx.Response(204)
                    deleted.set()
                    return httpx.Response(404)

                provider = await self.provider(adapter, handler)

                async def allocated(sandbox_id):
                    published.set()
                    await asyncio.Event().wait()

                task = asyncio.create_task(provider.create('image', cluster_name='dbox', on_allocated=allocated))
                await asyncio.wait_for(published.wait(), timeout=1)
                task.cancel()
                await asyncio.wait_for(deleting.wait(), timeout=1)
                task.cancel()
                if adapter == 'gateway':
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=1)
                else:
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                self.assertEqual(provider.owned_env_ids, {'allocated'})
                release_delete.set()
                await asyncio.wait_for(deleted.wait(), timeout=1)
                if adapter == 'direct':
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.assertFalse(provider.owned_env_ids)

    async def test_status_404_is_deletion_proof_but_failed_status_is_not(self):
        for adapter in ('direct', 'gateway'):
            with self.subTest(adapter=adapter):
                responses = [self.response(adapter, 'failed'), httpx.Response(404)]
                provider = await self.provider(adapter, lambda request: responses.pop(0))
                self.assertTrue(provider.supports_deletion_confirmation)
                self.assertEqual((await provider.get('allocated'))['env']['status'], 'failed')
                self.assertEqual((await provider.get('allocated'))['env']['status'], 'deleted')


class ClientLifecycleObservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_observation_preserves_legacy_clients_and_normalizes_status(self):
        provider = SimpleNamespace(get=AsyncMock(return_value={'env': {'status': 'READY'}}))
        for sandbox_id, lifecycle_provider in ((None, None), ('owned', None), (None, provider)):
            with self.subTest(sandbox_id=sandbox_id, provider=lifecycle_provider is not None):
                async with SandboxClient(
                    'http://sandbox.test', sandbox_id=sandbox_id, lifecycle_provider=lifecycle_provider
                ) as client:
                    self.assertIsNone(await client.get_lifecycle_state())
        provider.get.assert_not_awaited()
        async with SandboxClient('http://sandbox.test', sandbox_id='owned', lifecycle_provider=provider) as client:
            self.assertEqual(await client.get_lifecycle_state(), 'ready')
            provider.get.assert_awaited_once_with('owned')
            provider.get.return_value = {'status': 123}
            self.assertIsNone(await client.get_lifecycle_state())


class ReadinessBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_hanging_status_request_is_cancelled_at_total_readiness_budget(self):
        cancelled = asyncio.Event()

        async def get_status(sandbox_id):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with self.assertRaisesRegex(TimeoutError, 'readiness timed out'):
            await asyncio.wait_for(
                wait_ready('allocated', {'status': 'pending'}, get_status, timeout=0.5, poll_interval=0.001), timeout=5
            )
        self.assertTrue(cancelled.is_set())
