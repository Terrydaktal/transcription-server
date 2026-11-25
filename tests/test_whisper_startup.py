import os
import select
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from whisper_service import ServiceStartupError, main, reserve_service_endpoint


class ServiceStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        environment = patch.dict(os.environ, {"XDG_RUNTIME_DIR": self.temp.name})
        environment.start()
        self.addCleanup(environment.stop)

    def test_duplicate_is_refused_even_before_first_server_listens(self):
        with reserve_service_endpoint("127.0.0.1", 0) as first:
            port = first.getsockname()[1]
            self.assertFalse(first.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN))
            with self.assertRaisesRegex(ServiceStartupError, "already starting or running"):
                with reserve_service_endpoint("127.0.0.1", port):
                    self.fail("Duplicate acquired the endpoint")
            self.assertFalse(first.get_inheritable())
            first.listen()
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                accepted, _ = first.accept()
                accepted.close()

    def test_host_alias_does_not_bypass_startup_lock(self):
        with reserve_service_endpoint("0.0.0.0", 0) as first:
            with self.assertRaises(ServiceStartupError):
                with reserve_service_endpoint("localhost", first.getsockname()[1]):
                    self.fail("Host alias bypassed the lock")

    def test_existing_listener_is_detected_without_a_lock_file(self):
        with socket.socket() as existing:
            existing.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            existing.bind(("127.0.0.1", 0))
            existing.listen()
            port = existing.getsockname()[1]
            with self.assertRaisesRegex(ServiceStartupError, "already in use"):
                with reserve_service_endpoint("127.0.0.1", port):
                    self.fail("Occupied port was accepted")
            self.assertTrue(existing.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN))

    def test_other_ports_are_allowed_for_intentional_separate_services(self):
        with reserve_service_endpoint("127.0.0.1", 0) as first:
            with reserve_service_endpoint("127.0.0.1", 0) as second:
                self.assertNotEqual(first.getsockname()[1], second.getsockname()[1])

    def test_model_load_failure_releases_port_and_lock(self):
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            with reserve_service_endpoint("127.0.0.1", 0) as first:
                port = first.getsockname()[1]
                raise RuntimeError("model failed")
        self.assertEqual(first.fileno(), -1)
        # The stale PID is deliberately left on disk; flock ownership is authoritative.
        self.assertEqual(len(list(Path(self.temp.name).rglob("*.lock"))), 1)
        with reserve_service_endpoint("127.0.0.1", port):
            pass

    def test_crashed_process_does_not_leave_a_stale_lock(self):
        code = (
            "from whisper_service import reserve_service_endpoint\n"
            "with reserve_service_endpoint('127.0.0.1', 0) as sock:\n"
            "    print(sock.getsockname()[1], flush=True)\n"
            "    input()\n"
        )
        with subprocess.Popen(
            [sys.executable, "-u", "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as child:
            try:
                ready, _, _ = select.select([child.stdout], [], [], 10)
                self.assertTrue(ready, "Child failed to claim its endpoint")
                port = int(child.stdout.readline())
                with self.assertRaisesRegex(ServiceStartupError, f"PID {child.pid}"):
                    with reserve_service_endpoint("127.0.0.1", port):
                        self.fail("Second process acquired the lock")
            finally:
                child.kill()
                child.communicate(timeout=5)
        with reserve_service_endpoint("127.0.0.1", port):
            pass

    def test_ipv6_endpoint_is_guarded(self):
        if not socket.has_ipv6:
            self.skipTest("IPv6 unavailable")
        with socket.socket(socket.AF_INET6) as probe:
            try:
                probe.bind(("::1", 0))
            except OSError:
                self.skipTest("IPv6 loopback unavailable")
        with reserve_service_endpoint("::1", 0) as first:
            with self.assertRaises(ServiceStartupError):
                with reserve_service_endpoint("::1", first.getsockname()[1]):
                    self.fail("Duplicate IPv6 endpoint was accepted")

    def test_real_http_server_survives_duplicate_and_releases_endpoint_on_shutdown(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        # Exercise the real CLI startup and Uvicorn socket hand-off with a tiny
        # replacement app: never load a model or touch the user's audio queue.
        code = f"""
from types import SimpleNamespace
from fastapi import FastAPI
import whisper_service as service
service.parse_args = lambda: None
service.build_runtime_config = lambda args: SimpleNamespace(
    host='127.0.0.1', port={port}, http_access_log=False)
service.report_microphone_startup_status = lambda: None
service.configure_app = lambda config: print('TEST_MODEL_LOADED', flush=True)
service.app = FastAPI()
@service.app.get('/ready')
async def ready():
    return {{'ok': True}}
raise SystemExit(service.main())
"""
        command = [sys.executable, "-u", "-c", code]
        cwd = Path(__file__).resolve().parents[1]
        url = f"http://127.0.0.1:{port}/ready"
        with subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as child:
            try:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        with urllib.request.urlopen(url, timeout=1) as response:
                            self.assertEqual(response.status, 200)
                        break
                    except urllib.error.URLError:
                        self.assertIsNone(child.poll(), "Fixture server exited during startup")
                        self.assertLess(time.monotonic(), deadline, "HTTP startup timed out")
                        time.sleep(0.02)
                duplicate = subprocess.run(
                    command, cwd=cwd, capture_output=True, text=True, timeout=10
                )
                self.assertEqual(duplicate.returncode, 1, duplicate.stderr)
                self.assertIn("already in use", duplicate.stderr)
                self.assertNotIn("TEST_MODEL_LOADED", duplicate.stdout)
                with urllib.request.urlopen(url, timeout=1) as response:
                    self.assertEqual(response.status, 200)
            finally:
                child.terminate()
                try:
                    output, errors = child.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.communicate(timeout=5)
                    raise
        self.assertIn("TEST_MODEL_LOADED", output)
        self.assertIn("Application shutdown complete", errors)
        with reserve_service_endpoint("127.0.0.1", port):
            pass

    def run_main(self, port, configure=None):
        with ExitStack() as mocks:
            config = SimpleNamespace(host="127.0.0.1", port=port, http_access_log=False)
            mocks.enter_context(patch("whisper_service.dictConfig"))
            mocks.enter_context(patch("whisper_service.parse_args"))
            mocks.enter_context(patch("whisper_service.build_runtime_config", return_value=config))
            load = mocks.enter_context(patch("whisper_service.configure_app", side_effect=configure))
            microphone = mocks.enter_context(patch("whisper_service.report_microphone_startup_status"))
            log = mocks.enter_context(patch("whisper_service.print"))
            mocks.enter_context(patch("uvicorn.Config"))
            server = mocks.enter_context(patch("uvicorn.Server"))
            server.return_value.started = True
            result = main()
            return result, load, microphone, log, server.return_value

    def test_main_exits_before_model_loading_or_startup_side_effects_on_duplicate(self):
        with reserve_service_endpoint("127.0.0.1", 0) as first:
            result, load, microphone, log, server = self.run_main(first.getsockname()[1])
            self.assertEqual(result, 1)
            load.assert_not_called()
            microphone.assert_not_called()
            server.run.assert_not_called()
            self.assertIn("WARNING: Whisper is already", log.call_args.args[0])

    def test_main_refuses_older_or_unrelated_server_before_loading_model(self):
        with socket.socket() as existing:
            existing.bind(("127.0.0.1", 0))
            existing.listen()
            result, load, _, log, server = self.run_main(existing.getsockname()[1])
            self.assertEqual(result, 1)
            load.assert_not_called()
            server.run.assert_not_called()
            self.assertIn("already in use", log.call_args.args[0])

    def test_main_holds_reservation_during_model_loading_and_passes_socket_to_server(self):
        def load(config):
            with self.assertRaises(ServiceStartupError):
                with reserve_service_endpoint(config.host, config.port):
                    self.fail("Endpoint was unguarded during model loading")

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        result, model, _, _, server = self.run_main(port, configure=load)
        self.assertEqual(result, 0)
        model.assert_called_once()
        server.run.assert_called_once()
        [reserved] = server.run.call_args.kwargs["sockets"]
        self.assertEqual(reserved.fileno(), -1)
        with reserve_service_endpoint("127.0.0.1", port):
            pass


if __name__ == "__main__":
    unittest.main()
