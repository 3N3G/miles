import subprocess
import unittest
from unittest.mock import patch

from experiments.tcs_async_rl.reward_tunnel import allocation, ssh_command


class RewardTunnelTests(unittest.TestCase):
    def test_foreign_job_and_multiple_nodes_are_rejected(self):
        for row in (
            "other|tcs-rl-smoke-v1|RUNNING|k007-003",
            "geney|other-project|RUNNING|k007-003",
            "geney|tcs-rl-smoke-v1|RUNNING|k007-[003-004]",
        ):
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=row)):
                with self.assertRaises(ValueError):
                    allocation("123", "tcs-rl-smoke-v1", "geney")

    def test_only_the_own_running_node_is_returned(self):
        for state, node, expected in (("PENDING", "", None), ("RUNNING", "k007-003", ("geney", "k007-003"))):
            row = f"geney|tcs-rl-smoke-v1|{state}|{node}"
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=row)):
                self.assertEqual(allocation("123", "tcs-rl-smoke-v1", "geney"), expected)
        with patch("subprocess.run") as mocked:
            with self.assertRaises(ValueError):
                allocation("123; unwanted", "tcs-rl-smoke-v1", "geney")
            mocked.assert_not_called()

    def test_forward_stays_on_loopback_and_keeps_host_verification(self):
        argv = ssh_command(("geney", "k007-003"), forward_port=18491)
        self.assertIn("127.0.0.1:18491:127.0.0.1:18491", argv)
        self.assertIn("HostName=k007-003", argv)
        self.assertIn("StrictHostKeyChecking=accept-new", argv)
        self.assertNotIn("StrictHostKeyChecking=no", argv)
        self.assertNotIn("-A", argv)
        self.assertEqual(argv[-1], "amd")


if __name__ == "__main__":
    unittest.main()
