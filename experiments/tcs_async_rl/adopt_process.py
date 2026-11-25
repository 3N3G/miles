"""Reattach only explicitly recorded Linux helpers from this session's supervisor.

This retains the running scorer and its in-flight paid requests while replacing
the monitor. Process start times and command lines guard against PID reuse.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path


def process_snapshot(pid):
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "start_ticks": int(stat[19]),
        "state": stat[0],
        "args": (root / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
    }


class AdoptedProcess:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.pid, self.args = snapshot["pid"], snapshot["args"]

    def poll(self):
        try:
            actual = process_snapshot(self.pid)
        except FileNotFoundError:
            return 1
        return (
            None
            if actual["start_ticks"] == self.snapshot["start_ticks"]
            and actual["args"] == self.args
            and actual["state"] != "Z"
            else 1
        )

    def terminate(self):
        if self.poll() is None:
            os.kill(self.pid, signal.SIGTERM)

    def kill(self):
        if self.poll() is None:
            os.kill(self.pid, signal.SIGKILL)

    def wait(self, timeout=None):
        start = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - start >= timeout:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.1)
        return 1


def adopt_helpers(manifest_path, *, directory, job_id, session_uuid):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["job_id"] != job_id
        or manifest["session_uuid"] != session_uuid
        or manifest["directory"] != str(directory)
        or AdoptedProcess(manifest["supervisor"]).poll() is None
        or AdoptedProcess(manifest["retired_guard"]).poll() is None
    ):
        raise ValueError("Adoption requires this session's stopped supervisor and guard")
    helpers = [AdoptedProcess(item) for item in manifest["helpers"]]
    for helper, module, folder in zip(
        helpers,
        ("reward_service", "reward_tunnel"),
        (directory / "judging", directory),
        strict=True,
    ):
        expected = f"experiments.tcs_async_rl.{module}"
        if helper.poll() is not None or expected not in helper.args or str(folder) not in helper.args:
            raise ValueError("Recorded helper identity changed before adoption")
    return helpers
