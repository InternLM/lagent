"""Monitoring must distinguish process exit from transport and lifecycle failure."""
import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from lagent.serving.sandbox.providers.dbox_client import DBoxSandboxClient, DBoxTransportError


class ProcessTransportTests(unittest.IsolatedAsyncioTestCase):
    async def client(self, handler):
        client = DBoxSandboxClient('http://dbox.test/v1/sandboxes/owned/proxy/44772/',
            auth_headers=lambda:{'OPEN-SANDBOX-API-KEY':'private-ak:private-sk'},
            execd_token='private-execd-token', sandbox_id='owned',
            provider=SimpleNamespace(get=AsyncMock(return_value={'env':{'status':'failed'}})))
        await client._client.aclose()
        client._client=httpx.AsyncClient(base_url=client.url,transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_structured_409_preserves_safe_code_and_phase_not_body_or_credentials(self):
        client=await self.client(lambda r:httpx.Response(409,json={'detail':{
            'code':'SANDBOX_NOT_RUNNING','message':'private-ak:private-sk private-execd-token'}}))
        with self.assertRaises(DBoxTransportError) as error:
            await client.is_pid_running(123)
        exc=error.exception
        self.assertEqual((exc.status_code,exc.error_code,exc.phase),(409,'SANDBOX_NOT_RUNNING','execd_proxy'))
        self.assertNotIn('private',str(exc))
        self.assertEqual(await client.get_lifecycle_state(),'failed')

    async def test_unknown_error_body_and_code_are_not_logged(self):
        client=await self.client(lambda r:httpx.Response(503,json={'code':'private-execd-token','message':'private-ak:private-sk'}))
        with self.assertRaises(DBoxTransportError) as error:
            await client.is_pid_running(123)
        self.assertIsNone(error.exception.error_code)
        self.assertNotIn('private',str(error.exception))

    async def test_completed_execution_does_not_wait_for_http_eof_and_closes_stream(self):
        class Stream(httpx.AsyncByteStream):
            closed = False
            async def __aiter__(self):
                payload = base64.b64encode(json.dumps({'return_code':0}).encode()).decode()
                yield (json.dumps({'type':'stdout','text':'DBOX_RESPONSE:'+payload})+'\n'+
                       json.dumps({'type':'execution_complete'})+'\n').encode()
                raise httpx.ReadTimeout('private response URL')
            async def aclose(self): self.closed = True
        stream = Stream()
        client = await self.client(lambda r:httpx.Response(200,stream=stream))
        self.assertEqual((await client.execute('true'))['return_code'],0)
        self.assertTrue(stream.closed)

    async def test_read_timeout_before_and_after_response_headers_has_distinct_evidence(self):
        def headers_timeout(request): raise httpx.ReadTimeout('private request URL')
        client = await self.client(headers_timeout)
        with self.assertRaises(DBoxTransportError) as error:
            await client.is_pid_running(123)
        exc = error.exception
        self.assertEqual(exc.phase,'execd_response_headers')
        self.assertEqual(exc.transport_error_type,'ReadTimeout')
        self.assertFalse(exc.response_received)
        self.assertIsNone(exc.status_code)
        self.assertGreaterEqual(exc.elapsed_seconds,0)
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"type":"stdout","text":"partial"}\n'
                raise httpx.ReadTimeout('private stream URL')
        client = await self.client(lambda r:httpx.Response(200,stream=Stream()))
        with self.assertRaises(DBoxTransportError) as error:
            await client.is_pid_running(123)
        exc = error.exception
        self.assertEqual((exc.status_code,exc.phase,exc.transport_error_type),
                         (200,'execd_response_stream','ReadTimeout'))
        self.assertTrue(exc.response_received)
        self.assertNotIn('private',str(exc))

    async def test_known_proxy_failure_code_is_preserved_without_free_form_response(self):
        client = await self.client(lambda r:httpx.Response(503,json={'detail':{
            'code':'PROXY_POOL_EXHAUSTED','message':'private-ak:private-sk'}}))
        with self.assertRaises(DBoxTransportError) as error:
            await client.is_pid_running(123)
        self.assertEqual(error.exception.error_code,'PROXY_POOL_EXHAUSTED')
        self.assertEqual(error.exception.phase,'execd_proxy')
        self.assertNotIn('private',str(error.exception))

    async def test_503_is_preserved_when_its_diagnostic_body_times_out(self):
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                raise httpx.ReadTimeout('private response URL')
                yield b''
        client = await self.client(lambda r:httpx.Response(503,stream=Stream()))
        with self.assertRaises(DBoxTransportError) as error:
            await client.is_pid_running(123)
        self.assertEqual(error.exception.status_code,503)
        self.assertEqual(error.exception.phase,'execd_proxy')
        self.assertIsNone(error.exception.error_code)

    async def test_malformed_probe_result_is_never_interpreted_as_process_exit(self):
        client=await self.client(lambda r:httpx.Response(200))
        for text in ('', 'DBOX_PID:', 'DBOX_PID:2', 'DBOX_PID:137', 'unexpected'):
            with self.subTest(text=text):
                client._execd_stdout=AsyncMock(return_value=text)
                with self.assertRaisesRegex(DBoxTransportError,'valid process state'):
                    await client.is_pid_running(123)

    async def test_live_and_exited_probe_result_and_no_kill_zero_zombie_test(self):
        client=await self.client(lambda r:httpx.Response(200))
        client._execd_stdout=AsyncMock(side_effect=['DBOX_PID:0','DBOX_PID:1'])
        self.assertTrue(await client.is_pid_running(123))
        self.assertFalse(await client.is_pid_running(123))
        command=client._execd_stdout.call_args.args[0]
        self.assertIn('/proc/123/stat',command)
        self.assertNotIn('kill -0',command)

    async def test_process_probe_bypasses_the_remote_exec_queue(self):
        # The probe is one execd command; it never goes through remote-exec,
        # which also runs the agent's tool commands.
        requests=[]
        def handler(request):
            requests.append(json.loads(request.content)['command'])
            return httpx.Response(200,text=json.dumps({'type':'stdout','text':'DBOX_PID:0'})+'\n'
                                  +json.dumps({'type':'execution_complete'})+'\n')
        client=await self.client(handler)
        client.execute=AsyncMock(side_effect=AssertionError('remote-exec used'))
        self.assertTrue(await client.is_pid_running(123))
        self.assertEqual(len(requests),1)
        self.assertIn('/proc/123/stat',requests[0])
        self.assertNotIn('127.0.0.1:18000',requests[0])

    async def test_stalled_process_probe_is_a_bounded_transport_error(self):
        client=await self.client(lambda r:httpx.Response(200))
        async def stalled(*args): await asyncio.sleep(1)
        client._execd_stdout=stalled
        client.process_probe_deadline_sec=.01
        with self.assertRaises(DBoxTransportError) as error:
            await client.is_pid_running(123)
        self.assertEqual((error.exception.phase,error.exception.transport_error_type),('process_probe','TimeoutError'))

    async def test_mutating_commands_are_never_replayed_after_transport_failure(self):
        calls=[]
        def handler(request):
            calls.append(request)
            return httpx.Response(503)
        client=await self.client(handler)
        with self.assertRaises(DBoxTransportError): await client.execute('increment-a-counter')
        self.assertEqual(len(calls),1)

    @unittest.skipUnless(hasattr(os,'fork') and Path('/proc').exists(),'requires Linux procfs')
    async def test_real_zombie_is_exited_even_though_kill_zero_succeeds(self):
        client=await self.client(lambda r:httpx.Response(200))
        async def execd_stdout(command,timeout):
            # The real execd runs the command through a shell.
            result=subprocess.run(['/bin/sh','-c',command],capture_output=True,text=True,timeout=5)
            return result.stdout.strip()
        client._execd_stdout=execd_stdout
        pid=os.fork()
        if pid==0: os._exit(0)
        try:
            for _ in range(100):
                if Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]=='Z': break
                time.sleep(.001)
            else: self.fail('child did not enter zombie state')
            os.kill(pid,0)
            self.assertFalse(await client.is_pid_running(pid))
        finally: os.waitpid(pid,0)


if __name__=='__main__': unittest.main()
