"""DBox lifecycle contract tests using real HTTP clients and MockTransport."""

import asyncio
import base64
import json
import subprocess
from pathlib import Path
import tempfile
import traceback
import unittest
from uuid import UUID

import httpx

from lagent.serving.sandbox.providers import DBoxProvider, SandboxClient


class DBoxProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.credential = Path(self.tmp.name) / "credential"
        self.secret = "fake-test-access-key:fake-test-secret-key"
        self.credential.write_text(self.secret)
        self.requests = []

    async def provider(self, handler, **kwargs):
        provider = DBoxProvider(
            "https://dbox.test", project="test-project", credential_file=str(self.credential),
            poll_interval=0.001, **kwargs,
        )
        await provider._client.aclose()

        async def record(request):
            self.requests.append(request)
            result = handler(request)
            return await result if asyncio.iscoroutine(result) else result

        provider._client = httpx.AsyncClient(base_url=provider.api_url, transport=httpx.MockTransport(record))
        self.addAsyncCleanup(provider.aclose)
        return provider

    async def test_native_payload_keeps_images_and_pair_group_without_credentials(self):
        count = 0

        def handler(request):
            nonlocal count
            count += 1
            return httpx.Response(202, json={"sandboxID": f"cprod2-sandbox-{count}", "status": {"state": "Running"}})

        provider = await self.provider(handler)
        for image in ("registry.test/task:level1-task-server", "registry.test/task:level1"):
            client, env_id = await provider.create(image_tag=image, ttl_seconds=21600,
                                                  resources={"cpu": 2, "memory": 4}, cluster_name="dbox",
                                                  user="test-user")
            self.addAsyncCleanup(client.aclose)
            self.assertIsInstance(client, SandboxClient)
            self.assertIn(env_id, provider.owned_env_ids)
            self.assertNotIn(DBoxProvider._AUTH_HEADER, client._client.headers)
        first, second = [json.loads(request.content) for request in self.requests]
        self.assertEqual(first["image"]["uri"], "registry.test/task:level1-task-server")
        self.assertEqual(second["image"]["uri"], "registry.test/task:level1")
        self.assertEqual(first["extensions"]["sandboxGroup"], second["extensions"]["sandboxGroup"])
        self.assertEqual(first["extensions"]["sandboxGroup"], str(UUID(provider.sandbox_group)))
        self.assertEqual(first["extensions"], {
            "project": "test-project", "sandboxGroup": provider.sandbox_group,
            "runtimeClassName": "sysbox-runc", "disableProjectDataMount": "true",
        })
        self.assertEqual(first["resourceRequests"], {"cpu": "2", "memory": "4Gi"})
        self.assertEqual(first["timeout"], 21600)
        self.assertNotIn("user", first)
        self.assertNotIn("user", first["env"])
        self.assertEqual(first["entrypoint"], ["/mnt/llm-ai-infra/tools/remote-exec-server", "--host", "127.0.0.1", "--port", "18000"])
        self.assertEqual(first["volumes"], [{"name": "shared", "mountPath": "/mnt/llm-ai-infra/",
                                             "subPath": "/mnt/llm-ai-infra", "readOnly": True}])
        self.assertEqual([p["containerPort"] for p in first["ports"]], [44772])
        self.assertNotEqual(first["env"]["EXECD_ACCESS_TOKEN"], second["env"]["EXECD_ACCESS_TOKEN"])
        self.assertEqual(first["networkPolicy"]["defaultAction"], "deny")
        for request in self.requests:
            self.assertEqual(request.headers[DBoxProvider._AUTH_HEADER], self.secret)
            self.assertNotIn(self.secret, request.content.decode())
            self.assertEqual(request.url.path, "/v1/sandboxes")
        self.assertNotIn(DBoxProvider._AUTH_HEADER, provider._client.headers)

    async def test_group_is_unique_per_provider_and_explicit_uuid_is_preserved(self):
        handler = lambda request: httpx.Response(200, json={})
        first = await self.provider(handler)
        second = await self.provider(handler)
        explicit = await self.provider(handler, sandbox_group=first.sandbox_group)
        self.assertNotEqual(first.sandbox_group, second.sandbox_group)
        self.assertEqual(first.sandbox_group, explicit.sandbox_group)

    async def test_pending_and_nested_status_normalize_without_raw_response(self):
        states = [
            {"sandboxID": "cprod2-pair", "status": {"state": "Pending"}},
            {"id": "cprod2-pair", "state": "running", "extensions": {"clusterID": "cprod2"},
             "env": {"unsafe": self.secret}, "sensitive": self.secret},
        ]
        provider = await self.provider(lambda request: httpx.Response(202 if request.method == "POST" else 200,
                                                                      json=states.pop(0)))
        client, env_id = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        self.assertEqual(env_id, "cprod2-pair")
        self.assertEqual(client.url, "https://dbox.test/v1/sandboxes/cprod2-pair/proxy/44772")
        info = provider._normalize({"state": "Running", "extensions": {"clusterID": "cprod2"},
                                   "env": {"unsafe": self.secret}}, env_id)
        self.assertEqual(info["env"]["status"], "ready")
        self.assertEqual(info["env"]["cluster"], "cprod2")
        self.assertNotIn(self.secret, json.dumps(info))
        self.assertEqual([r.method for r in self.requests], ["POST", "GET"])

    async def test_data_plane_preserves_proxy_path_and_does_not_send_iam(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={"id": "cprod2-data", "state": "running"}))
        client, _ = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        await client._client.aclose()
        data_requests = []

        def data_handler(request):
            data_requests.append(request)
            self.assertEqual(request.headers[DBoxProvider._AUTH_HEADER], self.secret)
            self.assertEqual(request.headers["X-EXECD-ACCESS-TOKEN"], provider._execd_tokens["cprod2-data"])
            command = json.loads(request.content)["command"]
            self.assertNotIn(self.secret, command)
            self.assertNotIn(provider._execd_tokens["cprod2-data"], command)
            result = {"ok": True, "return_code": 0}
            if "/download" in command:
                result["content_b64"] = base64.b64encode(b"content").decode()
            encoded = base64.b64encode(json.dumps(result).encode()).decode()
            events = [{"type": "stdout", "text": "DBOX_RESPONSE:" + encoded}, {"type": "execution_complete"}]
            return httpx.Response(200, text="\n\n".join(json.dumps(e) for e in events))

        client._client = httpx.AsyncClient(base_url=client.url, transport=httpx.MockTransport(data_handler))
        self.assertTrue((await client.health_check())["ok"])
        self.assertEqual((await client.execute("echo test"))["return_code"], 0)
        await client.upload_bytes("/tmp/check", b"content")
        self.assertEqual(await client.download_file("/tmp/check"), b"content")
        self.assertEqual([r.url.path for r in data_requests],
                         ["/v1/sandboxes/cprod2-data/proxy/44772/command"] * 4)

    async def test_peer_policy_rejects_other_trials_and_verifies_platform_response(self):
        def handler(request):
            policy = json.loads(request.content)
            return httpx.Response(200, json={"status": "ok", "policy": policy})

        provider = await self.provider(handler)
        provider.owned_env_ids.update({"cprod2-main", "cprod2-server"})
        result = await provider.configure_peer("cprod2-main", "cprod2-server", external_targets=["model.test"])
        self.assertEqual(result["policy"]["egress"], [
            {"action": "allow", "target": "sandbox:cprod2-server"},
            {"action": "allow", "target": "model.test"},
        ])
        with self.assertRaises(ValueError):
            await provider.configure_peer("cprod2-main", "cprod2-other-trial")
        with self.assertRaises(ValueError):
            await provider.configure_peer("cprod2-main", "cprod2-server", external_targets=["http://unsafe/"])

    async def test_large_binary_upload_is_complete_and_failed_upload_keeps_destination(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={"id": "cprod2-data", "state": "running"}))
        client, _ = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        target = Path(self.tmp.name) / "large.bin"
        commands = []
        fail_chunk = False

        async def execute(command, **kwargs):
            commands.append(command)
            if fail_chunk and "base64.b64decode" in command:
                return {"return_code": 1}
            result = subprocess.run(command, shell=True, capture_output=True, timeout=10)
            return {"return_code": result.returncode}

        client.execute = execute
        content = bytes(range(256)) * 300
        await client.upload_bytes(str(target), content)
        self.assertEqual(target.read_bytes(), content)
        fail_chunk = True
        with self.assertRaisesRegex(RuntimeError, "chunk upload failed"):
            await client.upload_bytes(str(target), content + b"more")
        self.assertEqual(target.read_bytes(), content)

    async def test_execd_success_without_response_or_with_stderr_is_not_command_success(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={"id": "cprod2-data", "state": "running"}))
        client, _ = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        await client._client.aclose()
        client._client = httpx.AsyncClient(base_url=client.url, transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=json.dumps({"type": "execution_complete"}))))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            await client.execute("true")

    async def test_execd_http_status_survives_redaction_and_is_retryable_for_pid_polling(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={"id": "cprod2-data", "state": "running"}))
        client, _ = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        await client._client.aclose()
        client._client = httpx.AsyncClient(base_url=client.url, transport=httpx.MockTransport(
            lambda request: httpx.Response(503, text=self.secret)))
        with self.assertRaises(httpx.HTTPError) as error:
            await client.is_pid_running(123)
        self.assertIn("HTTP 503", str(error.exception))
        self.assertNotIn(self.secret, str(error.exception))

    async def test_terminal_creation_rolls_back_and_confirms_404(self):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(202, json={"id": "cprod2-failed", "state": "failed"})
            if request.method == "DELETE":
                return httpx.Response(204)
            return httpx.Response(404)

        provider = await self.provider(handler)
        with self.assertRaisesRegex(RuntimeError, "terminal state"):
            await provider.create(image_tag="task:main")
        self.assertFalse(provider.owned_env_ids)
        self.assertEqual([r.method for r in self.requests], ["POST", "DELETE", "GET"])

    async def test_platform_timeout_rolls_back(self):
        deleted = False

        def handler(request):
            nonlocal deleted
            if request.method == "DELETE":
                deleted = True
                return httpx.Response(204)
            if deleted:
                return httpx.Response(404)
            return httpx.Response(202 if request.method == "POST" else 200,
                                  json={"id": "cprod2-pending", "state": "pending"})

        provider = await self.provider(handler, ready_timeout=0.002)
        with self.assertRaisesRegex(TimeoutError, "readiness"):
            await provider.create(image_tag="task:main")
        self.assertFalse(provider.owned_env_ids)
        self.assertEqual(sum(r.method == "POST" for r in self.requests), 1)

    async def test_cancelled_readiness_rolls_back_before_propagating(self):
        querying = asyncio.Event()
        deleted = False

        async def handler(request):
            nonlocal deleted
            if request.method == "POST":
                return httpx.Response(202, json={"id": "cprod2-cancel", "state": "pending"})
            if request.method == "DELETE":
                deleted = True
                return httpx.Response(204)
            if deleted:
                return httpx.Response(404)
            querying.set()
            await asyncio.Future()

        provider = await self.provider(handler)
        task = asyncio.create_task(provider.create(image_tag="task:main"))
        await asyncio.wait_for(querying.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(deleted)
        self.assertFalse(provider.owned_env_ids)

    async def test_failed_rollback_retains_ownership_for_pool_cleanup(self):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(202, json={"id": "cprod2-retry-delete", "state": "failed"})
            return httpx.Response(503, text=self.secret)

        provider = await self.provider(handler)
        with self.assertRaisesRegex(RuntimeError, "terminal state"):
            await provider.create(image_tag="task:main")
        self.assertEqual(provider.owned_env_ids, {"cprod2-retry-delete"})

    async def test_second_creation_failure_preserves_first_lease(self):
        count = 0

        def handler(request):
            nonlocal count
            count += 1
            if count == 1:
                return httpx.Response(200, json={"id": "cprod2-server", "state": "running"})
            return httpx.Response(503, text=self.secret)

        provider = await self.provider(handler)
        client, _ = await provider.create(image_tag="task:server")
        self.addAsyncCleanup(client.aclose)
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            await provider.create(image_tag="task:main")
        self.assertEqual(provider.owned_env_ids, {"cprod2-server"})
        self.assertEqual([r.method for r in self.requests], ["POST", "POST"])

    async def test_ambiguous_create_is_not_retried_or_logged_with_secret(self):
        def handler(request):
            raise httpx.ReadTimeout(self.secret, request=request)

        provider = await self.provider(handler)
        with self.assertRaisesRegex(RuntimeError, "allocation may be ambiguous") as caught:
            await provider.create(image_tag="task:main")
        rendered = "".join(traceback.format_exception(caught.exception))
        self.assertNotIn(self.secret, rendered)
        self.assertIsNone(caught.exception.__cause__)
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(provider.owned_env_ids)

    async def test_http_failure_does_not_expose_body(self):
        provider = await self.provider(lambda request: httpx.Response(401, text=self.secret))
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
            await provider.create(image_tag="task:main")
        self.assertNotIn(self.secret, "".join(traceback.format_exception(caught.exception)))

    async def test_429_classifies_without_exposing_message_or_retrying_create(self):
        for code in ("QUOTA_EXCEEDED", "RATE_LIMITED", self.secret):
            with self.subTest(code=code):
                provider = await self.provider(lambda request: httpx.Response(
                    429, json={"error": {"code": code, "message": self.secret}}))
                before = len(self.requests)
                with self.assertRaisesRegex(RuntimeError, "HTTP 429") as caught:
                    await provider.create(image_tag="task:main")
                rendered = "".join(traceback.format_exception(caught.exception))
                self.assertNotIn(self.secret, rendered)
                if code != self.secret:
                    self.assertIn(code, rendered)
                self.assertEqual(len(self.requests), before + 1)
                self.assertFalse(provider.owned_env_ids)

    async def test_malformed_response_does_not_expose_body(self):
        provider = await self.provider(lambda request: httpx.Response(202, text=self.secret))
        with self.assertRaisesRegex(RuntimeError, "invalid JSON") as caught:
            await provider.create(image_tag="task:main")
        self.assertNotIn(self.secret, "".join(traceback.format_exception(caught.exception)))

    async def test_deleted_status_confirmation(self):
        responses = [httpx.Response(202), httpx.Response(200, json={"state": "deleting"}),
                     httpx.Response(200, json={"status": {"state": "Deleted"}})]
        provider = await self.provider(lambda request: responses.pop(0))
        provider.owned_env_ids.add("cprod2-delete")
        await provider.delete("cprod2-delete")
        self.assertFalse(provider.owned_env_ids)
        self.assertEqual([r.method for r in self.requests], ["DELETE", "GET", "GET"])

    async def test_delete_404_is_idempotent(self):
        provider = await self.provider(lambda request: httpx.Response(404))
        provider.owned_env_ids.add("cprod2-gone")
        await provider.delete("cprod2-gone")
        await provider.delete("cprod2-gone")
        self.assertFalse(provider.owned_env_ids)
        self.assertEqual([r.method for r in self.requests], ["DELETE", "DELETE"])

    async def test_unconfirmed_delete_retains_ownership(self):
        provider = await self.provider(lambda request: httpx.Response(202 if request.method == "DELETE" else 200,
                                                                      json={"state": "running"}), delete_timeout=0.002)
        provider.owned_env_ids.add("cprod2-unconfirmed")
        with self.assertRaisesRegex(TimeoutError, "deletion was not confirmed"):
            await provider.delete("cprod2-unconfirmed")
        self.assertEqual(provider.owned_env_ids, {"cprod2-unconfirmed"})

    async def test_credential_formats_read_per_request(self):
        provider = await self.provider(lambda request: httpx.Response(404))
        for serialized in (self.secret, json.dumps({"aksk": self.secret}), json.dumps({"api_key": self.secret})):
            self.credential.write_text(serialized)
            await provider.get("cprod2-read")
            self.assertEqual(self.requests[-1].headers[DBoxProvider._AUTH_HEADER], self.secret)
        self.credential.write_text("replacement-access:replacement-secret")
        await provider.get("cprod2-read")
        self.assertEqual(self.requests[-1].headers[DBoxProvider._AUTH_HEADER], "replacement-access:replacement-secret")

    async def test_invalid_credentials_fail_before_http_without_echo(self):
        provider = await self.provider(lambda request: httpx.Response(404))
        for invalid in ("ak-only", ":sk", "ak:", "ak:sk:more", "ak:sk\nline", json.dumps({"aksk": 12}), "[]"):
            with self.subTest(invalid=invalid):
                self.credential.write_text(invalid)
                with self.assertRaisesRegex(ValueError, "trusted credential") as caught:
                    await provider.get("cprod2-invalid")
                self.assertNotIn(invalid, str(caught.exception))
        self.assertFalse(self.requests)

    async def test_rejects_unsupported_or_unsafe_create_inputs_before_http(self):
        provider = await self.provider(lambda request: httpx.Response(404))
        cases = [
            {"ttl_seconds": 0}, {"ttl_seconds": True}, {"ttl_seconds": 86401},
            {"resources": {"cpu": 2, "memory": "4Gi"}}, {"resources": {"cpu": -1, "memory": 4}},
            {"resources": {"cpu": True, "memory": 4}}, {"resources": {"cpu": 2}},
            {"cluster_name": "gateway"}, {"env_vars": {"SECRET": self.secret}},
            {"key": self.secret}, {"extra_params": {"sandboxGroup": "override"}},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                await provider.create(image_tag="task:main", **kwargs)
        self.assertFalse(self.requests)

    async def test_rejects_path_injection_ids_before_http(self):
        provider = await self.provider(lambda request: httpx.Response(404))
        for value in ("../elsewhere", "id?token=bad", "id/other", self.secret, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                await provider.delete(value)
        self.assertFalse(self.requests)

    async def test_network_policy_declares_protocol_and_rejects_changed_allowlist(self):
        async def handler(request):
            policy = json.loads(request.content)
            self.assertEqual(policy['ingress']['rules'][0]['protocol'], 'tcp')
            return httpx.Response(200, json={'status': 'ok', 'policy': policy})
        provider = await self.provider(handler)
        provider.owned_env_ids.update({'main', 'peer'})
        result = await provider.configure_network('main', peer_ids=['peer'], external_targets=['10.1.2.3/32'])
        self.assertEqual(result['policy']['egress'], [
            {'action': 'allow', 'target': 'sandbox:peer'}, {'action': 'allow', 'target': '10.1.2.3/32'}])
        result = await provider.configure_network('main', external_targets=['model.test'])
        self.assertNotIn('except', result['policy']['ingress']['rules'][0])
        await provider._client.aclose()
        def wrong_policy(request):
            policy = json.loads(request.content)
            policy['egress'].append({'action': 'allow', 'target': '0.0.0.0/0'})
            return httpx.Response(200, json={'status': 'ok', 'policy': policy})
        provider._client = httpx.AsyncClient(base_url=provider.api_url, transport=httpx.MockTransport(wrong_policy))
        with self.assertRaisesRegex(RuntimeError, 'confirm'):
            await provider.configure_network('main', external_targets=['10.1.2.3/32'])

    async def test_network_policy_rejects_cidrs_wider_than_one_host(self):
        provider = await self.provider(lambda request: httpx.Response(500))
        provider.owned_env_ids.add('main')
        for target in ['0.0.0.0/0', '10.0.0.0/8', '10.1.2.0/24', '10.1.2.2/31']:
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, '/32'):
                await provider.configure_network('main', external_targets=[target])
        self.assertFalse(self.requests)


if __name__ == "__main__":
    unittest.main()
