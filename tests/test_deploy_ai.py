import contextlib
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import Mock, call, patch


command = runpy.run_path(str(Path(__file__).resolve().parents[1] / "deploy-ai"))
activate = command["activate"]


class DeployActivationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.old = self.root / "releases" / ("a" * 40)
        self.new = self.root / "releases" / ("b" * 40)
        self.old.mkdir(parents=True)
        self.new.mkdir()
        self.current = self.root / "current"
        self.current.symlink_to(self.old)

        self.active = Mock(return_value=True)
        self.ready = Mock()
        self.systemctl = Mock()
        self.install_unit = Mock()
        self.enterContext(patch.dict(activate.__globals__, {
            "is_active": self.active,
            "wait_ready": self.ready,
            "systemctl": self.systemctl,
            "install_unit": self.install_unit,
            "port_in_use": Mock(return_value=False),
        }))
        self.enterContext(contextlib.redirect_stdout(None))

    def test_active_deploy_waits_for_idle_before_advancing_current(self):
        def check(revision, _seconds, *, idle=False):
            self.assertEqual(self.current.resolve(), self.old if idle else self.new)
            self.assertEqual(revision, (self.old if idle else self.new).name)

        self.ready.side_effect = check
        activate(self.root, self.new, start=False, seconds=5)
        self.assertEqual(self.ready.call_args_list, [
            call(self.old.name, 5, idle=True), call(self.new.name, 5),
        ])
        self.systemctl.assert_called_once_with("restart", command["UNIT"])
        self.assertEqual(self.current.resolve(), self.new)

    def test_busy_server_keeps_current_revision(self):
        self.ready.side_effect = RuntimeError("still busy")
        with self.assertRaisesRegex(RuntimeError, "busy"):
            activate(self.root, self.new, start=False, seconds=5)
        self.assertEqual(self.current.resolve(), self.old)
        self.systemctl.assert_not_called()

    def test_failed_health_check_restarts_previous_revision(self):
        self.ready.side_effect = [None, RuntimeError("unhealthy candidate"), None]
        with self.assertRaisesRegex(RuntimeError, "previous selection was restored"):
            activate(self.root, self.new, start=False, seconds=5)
        self.assertEqual(self.current.resolve(), self.old)
        self.assertEqual(self.systemctl.call_args_list, [call("restart", command["UNIT"])] * 2)
        self.assertEqual(self.ready.call_args_list[-1], call(self.old.name, 5))

    def test_stopped_server_stages_release_without_starting(self):
        self.active.return_value = False
        activate(self.root, self.new, start=False, seconds=5)
        self.assertEqual(self.current.resolve(), self.new)
        self.systemctl.assert_not_called()
        self.ready.assert_not_called()


if __name__ == "__main__":
    unittest.main()
