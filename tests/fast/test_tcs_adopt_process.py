import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.tcs_async_rl.adopt_process import AdoptedProcess, adopt_helpers, process_snapshot


class AdoptProcessTests(unittest.TestCase):
    def test_process_wait_timeout_and_reused_identity_do_not_signal_another_process(self):
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]) as child:
            snapshot = process_snapshot(child.pid)
            adopted = AdoptedProcess(snapshot)
            self.assertIsNone(adopted.poll())
            with self.assertRaises(subprocess.TimeoutExpired):
                adopted.wait(timeout=0)
            with patch("experiments.tcs_async_rl.adopt_process.os.kill") as kill:
                changed = AdoptedProcess({**snapshot, "start_ticks": snapshot["start_ticks"] + 1})
                self.assertIsNotNone(changed.poll())
                changed.terminate()
                changed.kill()
                kill.assert_not_called()
            adopted.terminate()
            adopted.wait(timeout=5)
            child.wait(timeout=5)

    def test_adoption_rejects_wrong_session_live_parent_and_changed_helper(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            processes = {
                1: {"pid": 1, "start_ticks": 1, "state": "Z", "args": ["supervisor"]},
                2: {"pid": 2, "start_ticks": 2, "state": "Z", "args": ["guard"]},
                3: {
                    "pid": 3,
                    "start_ticks": 3,
                    "state": "S",
                    "args": [
                        "python",
                        "-m",
                        "experiments.tcs_async_rl.reward_service",
                        str(root / "judging"),
                    ],
                },
                4: {
                    "pid": 4,
                    "start_ticks": 4,
                    "state": "S",
                    "args": ["python", "-m", "experiments.tcs_async_rl.reward_tunnel", str(root)],
                },
            }
            manifest = {
                "job_id": "123",
                "session_uuid": "session",
                "directory": str(root),
                "supervisor": processes[1],
                "retired_guard": processes[2],
                "helpers": [processes[3], processes[4]],
            }
            path = root / "adoption.json"
            path.write_text(json.dumps(manifest))
            with patch(
                "experiments.tcs_async_rl.adopt_process.process_snapshot",
                side_effect=lambda pid: processes[pid],
            ):
                helpers = adopt_helpers(path, directory=root, job_id="123", session_uuid="session")
                self.assertEqual([h.pid for h in helpers], [3, 4])
                with self.assertRaises(ValueError):
                    adopt_helpers(path, directory=root, job_id="123", session_uuid="other")
                processes[1] = {**processes[1], "state": "S"}
                with self.assertRaises(ValueError):
                    adopt_helpers(path, directory=root, job_id="123", session_uuid="session")
                processes[1] = {**processes[1], "state": "Z"}
                processes[3] = {**processes[3], "args": ["another process"]}
                with self.assertRaises(ValueError):
                    adopt_helpers(path, directory=root, job_id="123", session_uuid="session")


if __name__ == "__main__":
    unittest.main()
