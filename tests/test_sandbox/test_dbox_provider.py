"""DBox lifecycle contract tests using real HTTP clients and MockTransport."""

import asyncio
import base64
import errno
import json
import os
import stat
import subprocess
import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
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
            "https://dbox.test",
            project="test-project",
            credential_file=str(self.credential),
            poll_interval=0.001,
            **kwargs,
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
            client, env_id = await provider.create(
                image_tag=image,
                ttl_seconds=21600,
                resources={"cpu": 2, "memory": 4},
                cluster_name="dbox",
                user="test-user",
            )
            self.addAsyncCleanup(client.aclose)
            self.assertIsInstance(client, SandboxClient)
            self.assertIn(env_id, provider.owned_env_ids)
            self.assertNotIn(DBoxProvider._AUTH_HEADER, client._client.headers)
        first, second = [json.loads(request.content) for request in self.requests]
        self.assertEqual(first["image"]["uri"], "registry.test/task:level1-task-server")
        self.assertEqual(second["image"]["uri"], "registry.test/task:level1")
        self.assertEqual(first["extensions"]["sandboxGroup"], second["extensions"]["sandboxGroup"])
        self.assertEqual(first["extensions"]["sandboxGroup"], str(UUID(provider.sandbox_group)))
        self.assertEqual(
            first["extensions"],
            {
                "project": "test-project",
                "sandboxGroup": provider.sandbox_group,
                "runtimeClassName": "sysbox-runc",
                "disableProjectDataMount": "true",
            },
        )
        self.assertEqual(first["resourceRequests"], {"cpu": "2", "memory": "4Gi"})
        self.assertEqual(first["timeout"], 21600)
        self.assertNotIn("user", first)
        self.assertNotIn("user", first["env"])
        self.assertEqual(
            first["entrypoint"],
            ["/mnt/llm-ai-infra/tools/remote-exec-server", "--host", "127.0.0.1", "--port", "18000"],
        )
        self.assertEqual(
            first["volumes"],
            [{"name": "shared", "mountPath": "/mnt/llm-ai-infra/", "subPath": "/mnt/llm-ai-infra", "readOnly": True}],
        )
        self.assertEqual([p["containerPort"] for p in first["ports"]], [44772])
        self.assertNotEqual(first["env"]["EXECD_ACCESS_TOKEN"], second["env"]["EXECD_ACCESS_TOKEN"])
        self.assertEqual(first["networkPolicy"]["defaultAction"], "deny")
        for request in self.requests:
            self.assertEqual(request.headers[DBoxProvider._AUTH_HEADER], self.secret)
            self.assertNotIn(self.secret, request.content.decode())
            self.assertEqual(request.url.path, "/v1/sandboxes")
        self.assertNotIn(DBoxProvider._AUTH_HEADER, provider._client.headers)

    async def test_group_is_unique_per_provider_and_explicit_uuid_is_preserved(self):
        def handler(request):
            return httpx.Response(200, json={})

        first = await self.provider(handler)
        second = await self.provider(handler)
        explicit = await self.provider(handler, sandbox_group=first.sandbox_group)
        self.assertNotEqual(first.sandbox_group, second.sandbox_group)
        self.assertEqual(first.sandbox_group, explicit.sandbox_group)

    async def test_allocation_is_reported_before_readiness_wait(self):
        recorded = []

        def handler(request):
            if request.method == "POST":
                return httpx.Response(201, json={"id": "cprod2-pending", "state": "pending"})
            self.assertEqual(recorded, ["cprod2-pending"])
            return httpx.Response(200, json={"id": "cprod2-pending", "state": "running"})

        provider = await self.provider(handler)

        async def on_allocated(env_id):
            self.assertIn(env_id, provider.owned_env_ids)
            self.assertEqual([r.method for r in self.requests], ["POST"])
            recorded.append(env_id)

        client, env_id = await provider.create(image_tag="task:main", on_allocated=on_allocated)
        self.addAsyncCleanup(client.aclose)
        self.assertEqual(recorded, [env_id])

    async def test_allocation_callback_failure_rolls_back_without_readiness(self):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(201, json={"id": "cprod2-pending", "state": "pending"})
            self.assertEqual(request.method, "DELETE")
            return httpx.Response(404)

        provider = await self.provider(handler)

        async def on_allocated(env_id):
            raise OSError("journal write failed")

        with self.assertRaisesRegex(OSError, "journal write failed"):
            await provider.create(image_tag="task:main", on_allocated=on_allocated)
        self.assertEqual([r.method for r in self.requests], ["POST", "DELETE"])
        self.assertFalse(provider.owned_env_ids)

    async def test_input_validation_failure_proves_no_allocation(self):
        provider = await self.provider(lambda request: self.fail("must not send request"))
        with self.assertRaises(ValueError) as error:
            await provider.create(image_tag="bad image")
        self.assertEqual(error.exception.allocation_state, "not_allocated")
        self.assertEqual(self.requests, [])

    async def test_pending_and_nested_status_normalize_without_raw_response(self):
        states = [
            {"sandboxID": "cprod2-pair", "status": {"state": "Pending"}},
            {
                "id": "cprod2-pair",
                "state": "running",
                "extensions": {"clusterID": "cprod2"},
                "env": {"unsafe": self.secret},
                "sensitive": self.secret,
            },
        ]
        provider = await self.provider(
            lambda request: httpx.Response(202 if request.method == "POST" else 200, json=states.pop(0))
        )
        client, env_id = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        self.assertEqual(env_id, "cprod2-pair")
        self.assertEqual(client.url, "https://dbox.test/v1/sandboxes/cprod2-pair/proxy/44772")
        info = provider._normalize(
            {"state": "Running", "extensions": {"clusterID": "cprod2"}, "env": {"unsafe": self.secret}}, env_id
        )
        self.assertEqual(info["env"]["status"], "ready")
        self.assertEqual(info["env"]["cluster"], "cprod2")
        self.assertNotIn(self.secret, json.dumps(info))
        self.assertEqual([r.method for r in self.requests], ["POST", "GET"])

    async def test_data_plane_preserves_proxy_path_and_does_not_send_iam(self):
        allocated_tokens = []

        def allocate(request):
            allocated_tokens.append(json.loads(request.content)['env']['EXECD_ACCESS_TOKEN'])
            return httpx.Response(200, json={"id": "cprod2-data", "state": "running"})

        provider = await self.provider(allocate)
        client, _ = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        await client._client.aclose()
        data_requests = []

        def data_handler(request):
            data_requests.append(request)
            self.assertEqual(request.headers[DBoxProvider._AUTH_HEADER], self.secret)
            self.assertEqual(request.headers["X-EXECD-ACCESS-TOKEN"], allocated_tokens[0])
            command = json.loads(request.content)["command"]
            self.assertNotIn(self.secret, command)
            self.assertNotIn(allocated_tokens[0], command)
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
        self.assertEqual([r.url.path for r in data_requests], ["/v1/sandboxes/cprod2-data/proxy/44772/command"] * 4)

    async def test_peer_policy_rejects_other_trials_and_verifies_platform_response(self):
        def handler(request):
            policy = json.loads(request.content)
            return httpx.Response(200, json={"status": "ok", "policy": policy})

        provider = await self.provider(handler)
        provider.owned_env_ids.update({"cprod2-main", "cprod2-server"})
        result = await provider.configure_peer("cprod2-main", "cprod2-server", external_targets=["model.test"])
        self.assertEqual(
            result["policy"]["egress"],
            [
                {"action": "allow", "target": "sandbox:cprod2-server"},
                {"action": "allow", "target": "model.test"},
            ],
        )
        with self.assertRaises(ValueError):
            await provider.configure_peer("cprod2-main", "cprod2-other-trial")
        with self.assertRaises(ValueError):
            await provider.configure_peer("cprod2-main", "cprod2-server", external_targets=["http://unsafe/"])

    async def test_client_network_interface_checks_provider_before_changing_policy(self):
        allocations = iter(("main", "server"))

        def handler(request):
            if request.method == "POST":
                return httpx.Response(201, json={"id": next(allocations), "state": "ready"})
            return httpx.Response(200, json={"status": "ok", "policy": json.loads(request.content)})

        provider = await self.provider(handler)
        main, _ = await provider.create("image")
        server, _ = await provider.create("image")
        self.addAsyncCleanup(main.aclose)
        self.addAsyncCleanup(server.aclose)
        result = await main.configure_network(peer_clients=[server], external_targets=["model.test"])
        self.assertEqual(self.requests[-1].url.path, "/v1/sandboxes/main/networkpolicy")
        self.assertEqual(
            result["policy"]["egress"],
            [
                {"action": "allow", "target": "sandbox:server"},
                {"action": "allow", "target": "model.test"},
            ],
        )
        other = await self.provider(lambda request: httpx.Response(201, json={"id": "server", "state": "ready"}))
        foreign, _ = await other.create("image")
        self.addAsyncCleanup(foreign.aclose)
        requests_before = len(self.requests)
        with self.assertRaisesRegex(ValueError, "same provider"):
            await main.configure_network(peer_clients=[foreign])
        self.assertEqual(len(self.requests), requests_before)

    async def test_client_diagnostics_exposes_only_supported_readonly_routes(self):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(201, json={"id": "diagnostics", "state": "ready"})
            return httpx.Response(200, json={"status": "ok"})

        provider = await self.provider(handler)
        client, _ = await provider.create("image")
        self.addAsyncCleanup(client.aclose)
        for kind, suffix in (("state", ""), ("inspect", "/inspect"), ("events", "/events")):
            with self.subTest(kind=kind):
                response = await client.get_diagnostics(kind, timeout=0.5)
                self.assertEqual(response.json(), {"status": "ok"})
                request = self.requests[-1]
                self.assertEqual(request.method, "GET")
                self.assertEqual(request.url.path, "/v1/sandboxes/diagnostics" + suffix)
                self.assertEqual(request.extensions["timeout"]["read"], 0.5)
        requests_before = len(self.requests)
        with self.assertRaisesRegex(ValueError, "Diagnostic kind"):
            await client.get_diagnostics("../networkpolicy")
        self.assertEqual(len(self.requests), requests_before)

    async def test_large_binary_upload_is_complete_and_failed_upload_keeps_destination(self):
        provider = await self.provider(
            lambda request: httpx.Response(200, json={"id": "cprod2-data", "state": "running"})
        )
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
            return {"return_code": result.returncode, 'stdout': result.stdout.decode()}

        client.execute = execute
        content = bytes(range(256)) * 300
        await client.upload_bytes(str(target), content)
        self.assertEqual(target.read_bytes(), content)
        fail_chunk = True
        with self.assertRaisesRegex(RuntimeError, "chunk upload failed"):
            await client.upload_bytes(str(target), content + b"more")
        self.assertEqual(target.read_bytes(), content)

    async def test_large_upload_creates_parent_and_commits_in_target_directory(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={"id": "upload", "state": "running"}))
        client, _ = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        target = Path(self.tmp.name) / "new directory" / "nested" / "payload.bin"
        content = b"x" * 24001
        commits = []

        async def execute(command, **kwargs):
            if "os.replace(" in command:
                staging = list(target.parent.glob(".dbox-transfer-*"))
                self.assertEqual(len(staging), 1)
                self.assertEqual(staging[0].read_bytes(), content)
                self.assertEqual(staging[0].stat().st_mode & 0o777, 0o600)
                self.assertFalse(target.exists())
                commits.append(staging[0])
            result = subprocess.run(command, shell=True, capture_output=True, timeout=10)
            return {"return_code": result.returncode, 'stdout': result.stdout.decode()}

        client.execute = execute
        result = await client.upload_bytes(str(target), content)
        self.assertEqual(target.read_bytes(), content)
        self.assertEqual(result, {"ok": True, "target_path": str(target), "size": len(content)})
        self.assertEqual(len(commits), 1)
        self.assertFalse(list(target.parent.glob(".dbox-transfer-*")))

    async def test_upload_sizes_share_absolute_path_metadata_and_private_new_file_rules(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={'id': 'upload', 'state': 'ready'}))
        client, _ = await provider.create('image')
        self.addAsyncCleanup(client.aclose)
        commands = []

        async def execute(command, **kwargs):
            commands.append(command)
            result = subprocess.run(command, shell=True, capture_output=True, cwd=kwargs['cwd'], timeout=10)
            return {
                'return_code': result.returncode,
                'stdout': result.stdout.decode(),
                'stderr': result.stderr.decode(),
            }

        client.execute = execute
        target = Path(self.tmp.name) / 'nested directory' / 'payload'
        for size in (0, 24000, 24001):
            with self.subTest(size=size):
                content = b'x' * size
                for relative in ('payload', '../payload', ''):
                    before = len(commands)
                    with self.assertRaisesRegex(ValueError, 'absolute POSIX'):
                        await client.upload_bytes(relative, content)
                    self.assertEqual(len(commands), before)
                before = len(commands)
                await client.upload_bytes(str(target), content)
                if size <= 24000:
                    self.assertEqual(len(commands) - before, 1)
                self.assertEqual(target.read_bytes(), content)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
                # Preserve a different owner/group where the test process has
                # permission, rather than only checking its own default IDs.
                if os.geteuid() == 0:
                    try:
                        os.chown(target, 12345, 12346)
                    except OSError as exc:
                        # A root process may still have only UID 0 mapped in
                        # its user namespace or lack CAP_CHOWN.
                        if exc.errno not in {errno.EINVAL, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
                            raise
                target.chmod(0o751)
                metadata = target.stat()
                await client.upload_bytes(str(target), content + b'!')
                current = target.stat()
                self.assertEqual(stat.S_IMODE(current.st_mode), 0o751)
                self.assertEqual((current.st_uid, current.st_gid), (metadata.st_uid, metadata.st_gid))
                self.assertEqual(target.read_bytes(), content + b'!')
                self.assertFalse(list(target.parent.glob('.dbox-transfer-*')))
                target.unlink()

    async def test_upload_metadata_failure_keeps_existing_destination_and_removes_staging(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={'id': 'upload', 'state': 'ready'}))
        client, _ = await provider.create('image')
        self.addAsyncCleanup(client.aclose)
        target = Path(self.tmp.name) / 'existing'

        async def execute(command, **kwargs):
            # Inject a metadata syscall failure in the actual emitted commit
            # program, before its atomic replace, at both size paths.
            command = command.replace(
                'os.chmod(staging, stat.S_IMODE(metadata.st_mode))', 'raise OSError("metadata unavailable")'
            )
            result = subprocess.run(command, shell=True, capture_output=True, cwd=kwargs['cwd'], timeout=10)
            return {'return_code': result.returncode, 'stdout': result.stdout.decode()}

        client.execute = execute
        for size in (24000, 24001):
            with self.subTest(size=size):
                target.write_bytes(b'old-content')
                target.chmod(0o751)
                with self.assertRaisesRegex(RuntimeError, 'upload.*failed'):
                    await client.upload_bytes(str(target), b'x' * size)
                self.assertEqual(target.read_bytes(), b'old-content')
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o751)
                self.assertFalse(list(target.parent.glob('.dbox-transfer-*')))

    async def test_large_upload_freezes_resolved_symlink_destination_before_chunks(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={'id': 'upload', 'state': 'ready'}))
        client, _ = await provider.create('image')
        self.addAsyncCleanup(client.aclose)
        first, second, link = (Path(self.tmp.name) / name for name in ('first', 'second', 'link'))
        first.write_bytes(b'first-old')
        second.write_bytes(b'second-old')
        link.symlink_to(first)

        async def execute(command, **kwargs):
            result = subprocess.run(command, shell=True, capture_output=True, cwd=kwargs['cwd'], timeout=10)
            if 'print(json.dumps' in command:
                link.unlink()
                link.symlink_to(second)
            return {'return_code': result.returncode, 'stdout': result.stdout.decode()}

        client.execute = execute
        content = b'x' * 24001
        await client.upload_bytes(str(link), content)
        self.assertEqual(first.read_bytes(), content)
        self.assertEqual(second.read_bytes(), b'second-old')
        self.assertEqual(link.resolve(), second)
        self.assertFalse(list(Path(self.tmp.name).glob('.dbox-transfer-*')))

    async def test_small_upload_cleanup_failure_preserves_primary_metadata_error(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={'id': 'upload', 'state': 'ready'}))
        client, _ = await provider.create('image')
        self.addAsyncCleanup(client.aclose)
        target = Path(self.tmp.name) / 'existing'
        target.write_bytes(b'old-content')
        errors = []

        async def execute(command, **kwargs):
            command = command.replace(
                'os.chmod(staging, stat.S_IMODE(metadata.st_mode))', 'raise OSError("primary metadata unavailable")'
            )
            command = command.replace('try: os.unlink(staging)', 'try: raise OSError("secondary cleanup unavailable")')
            result = subprocess.run(command, shell=True, capture_output=True, cwd=kwargs['cwd'], timeout=10)
            errors.append(result.stderr.decode())
            return {'return_code': result.returncode}

        client.execute = execute
        with self.assertRaisesRegex(RuntimeError, 'atomic upload failed'):
            await client.upload_bytes(str(target), b'new-content')
        self.assertEqual(len(errors), 1)
        self.assertIn('primary metadata unavailable', errors[0])
        self.assertNotIn('secondary cleanup unavailable', errors[0])
        self.assertEqual(target.read_bytes(), b'old-content')

    async def test_large_upload_does_not_replace_primary_failure_with_cleanup_error(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={"id": "upload", "state": "running"}))
        client, _ = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        for primary in (OSError("chunk write failed"), asyncio.CancelledError()):
            with self.subTest(primary=type(primary).__name__):
                calls = []

                async def execute(command, **kwargs):
                    calls.append(command)
                    if 'os.unlink(staging)' in command:
                        raise RuntimeError("cleanup unavailable")
                    if "base64.b64decode" in command:
                        raise primary
                    return {
                        'return_code': 0,
                        'stdout': json.dumps(
                            {'target': '/tmp/unused-test-target', 'staging': '/tmp/.dbox-transfer-fixed'}
                        ),
                    }

                client.execute = execute
                with patch(
                    'lagent.serving.sandbox.providers.dbox_client.uuid4', return_value=SimpleNamespace(hex='fixed')
                ):
                    with self.assertRaises(type(primary)) as error:
                        await client.upload_bytes("/tmp/unused-test-target", b"x" * 24001)
                self.assertIs(error.exception, primary)
                self.assertIn('os.unlink(staging)', calls[-1])

    async def test_failed_staging_does_not_write_chunks_or_commit(self):
        provider = await self.provider(lambda request: httpx.Response(200, json={"id": "upload", "state": "running"}))
        client, _ = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        commands = []

        async def execute(command, **kwargs):
            commands.append(command)
            return {"return_code": 1}

        client.execute = execute
        with self.assertRaisesRegex(RuntimeError, "staging failed"):
            await client.upload_bytes("/tmp/unused-test-target", b"x" * 24001)
        self.assertEqual(len(commands), 1)

    async def test_execd_success_without_response_or_with_stderr_is_not_command_success(self):
        provider = await self.provider(
            lambda request: httpx.Response(200, json={"id": "cprod2-data", "state": "running"})
        )
        client, _ = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url=client.url,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=json.dumps({"type": "execution_complete"}))
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            await client.execute("true")

    async def test_execd_http_status_survives_redaction_and_is_retryable_for_pid_polling(self):
        provider = await self.provider(
            lambda request: httpx.Response(200, json={"id": "cprod2-data", "state": "running"})
        )
        client, _ = await provider.create(image_tag="task:main")
        self.addAsyncCleanup(client.aclose)
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url=client.url, transport=httpx.MockTransport(lambda request: httpx.Response(503, text=self.secret))
        )
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
            return httpx.Response(
                202 if request.method == "POST" else 200, json={"id": "cprod2-pending", "state": "pending"}
            )

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
                provider = await self.provider(
                    lambda request: httpx.Response(429, json={"error": {"code": code, "message": self.secret}})
                )
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
        responses = [
            httpx.Response(202),
            httpx.Response(200, json={"state": "deleting"}),
            httpx.Response(200, json={"status": {"state": "Deleted"}}),
        ]
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
        provider = await self.provider(
            lambda request: httpx.Response(202 if request.method == "DELETE" else 200, json={"state": "running"}),
            delete_timeout=0.002,
        )
        provider.owned_env_ids.add("cprod2-unconfirmed")
        with self.assertRaisesRegex(TimeoutError, "deletion was not confirmed"):
            await provider.delete("cprod2-unconfirmed")
        self.assertEqual(provider.owned_env_ids, {"cprod2-unconfirmed"})

    async def test_credential_formats_read_per_request(self):
        provider = await self.provider(lambda request: httpx.Response(404))
        for serialized in (
            self.secret,
            json.dumps({"aksk": self.secret}),
            json.dumps({"api_key": self.secret}),
            json.dumps(dict(zip(("access_key_id", "secret_key"), self.secret.split(":")))),
        ):
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

    async def test_json_pair_create_keeps_credentials_out_of_sandbox(self):
        self.credential.write_text(json.dumps(dict(zip(("access_key_id", "secret_key"), self.secret.split(":")))))
        provider = await self.provider(
            lambda request: httpx.Response(202, json={"id": "cprod2-json", "state": "running"})
        )
        client, _ = await provider.create("image")
        self.addAsyncCleanup(client.aclose)
        self.assertEqual(self.requests[0].headers[DBoxProvider._AUTH_HEADER], self.secret)
        self.assertNotIn(self.secret, self.requests[0].content.decode())
        self.assertNotIn(DBoxProvider._AUTH_HEADER, client._client.headers)

    async def test_credential_failure_before_create_is_not_allocated(self):
        provider = await self.provider(lambda request: httpx.Response(404))
        for fields in (
            {"access_key_id": "private-ak"},
            {"secret_key": "private-sk"},
            {"access_key_id": "private:ak", "secret_key": "private-sk"},
            {"access_key_id": "private-ak", "secret_key": 123},
            {"access_key_id": "private-ak", "secret_key": "private sk"},
        ):
            self.credential.write_text(json.dumps(fields))
            with self.assertRaises(ValueError) as caught:
                await provider.create("image")
            self.assertEqual(caught.exception.allocation_state, "not_allocated")
            self.assertNotIn("private", ''.join(traceback.format_exception(caught.exception)))
        self.credential.unlink()
        with self.assertRaises(ValueError) as caught:
            await provider.create("image")
        self.assertEqual(caught.exception.allocation_state, "not_allocated")
        self.assertFalse(self.requests)

    async def test_get_credential_failure_and_sent_post_remain_uncertain(self):
        provider = await self.provider(lambda request: httpx.Response(404))
        self.credential.write_text("incomplete")
        with self.assertRaises(ValueError) as caught:
            await provider.get("cprod2-existing")
        self.assertFalse(hasattr(caught.exception, "allocation_state"))
        self.credential.write_text(self.secret)

        def timeout(request):
            raise httpx.ReadTimeout("lost POST response", request=request)

        provider = await self.provider(timeout)
        with self.assertRaises(RuntimeError) as caught:
            await provider.create("image")
        self.assertNotEqual(getattr(caught.exception, "allocation_state", None), "not_allocated")
        self.assertEqual(len(self.requests), 1)

    async def test_rejects_unsupported_or_unsafe_create_inputs_before_http(self):
        provider = await self.provider(lambda request: httpx.Response(404))
        cases = [
            {"ttl_seconds": 0},
            {"ttl_seconds": True},
            {"ttl_seconds": 86401},
            {"resources": {"cpu": 2, "memory": "4Gi"}},
            {"resources": {"cpu": -1, "memory": 4}},
            {"resources": {"cpu": True, "memory": 4}},
            {"resources": {"cpu": 2}},
            {"cluster_name": "gateway"},
            {"env_vars": {"SECRET": self.secret}},
            {"key": self.secret},
            {"extra_params": {"sandboxGroup": "override"}},
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
        self.assertEqual(
            result['policy']['egress'],
            [{'action': 'allow', 'target': 'sandbox:peer'}, {'action': 'allow', 'target': '10.1.2.3/32'}],
        )
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
