"""Read GPU and filesystem telemetry only while an approved own job is running."""

import argparse
import datetime
import fcntl
import json
import shlex
import subprocess
import time
from pathlib import Path

from experiments.tcs_async_rl.reward_tunnel import allocation, ssh_command
from experiments.tcs_async_rl.watch_group import ACTIVE, remote_state


def collect(target, work_dir):
    code = (
        "import json,shutil,subprocess; "
        "r=subprocess.run(['rocm-smi','--showmeminfo','vram','--showuse','--showpower','--showtemp','--json'],"
        "capture_output=True,text=True,timeout=30); "
        "r.check_returncode(); "
        f"print(json.dumps(dict(gpus=json.loads(r.stdout),shared_free_bytes=shutil.disk_usage({str(work_dir)!r}).free,"
        "tmp_free_bytes=shutil.disk_usage('/tmp').free,shm_free_bytes=shutil.disk_usage('/dev/shm').free)))"
    )
    result = subprocess.run(
        ssh_command(target) + ["python3 -c " + shlex.quote(code)],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    cli = parser.parse_args()
    with (cli.run_dir / "gpu-telemetry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            row = {"at_utc": datetime.datetime.now(datetime.UTC).isoformat(), "job_id": cli.job_id}
            try:
                state = remote_state(cli.job_id)
                row["scheduler"] = state
                if state["state"] not in ACTIVE | {"UNKNOWN"}:
                    break
                target = allocation(cli.job_id, cli.job_name, cli.user)
                if target is not None:
                    row.update(collect(target, cli.work_dir))
                    row["node"] = target[1]
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                row["error_type"] = type(error).__name__
            with (cli.run_dir / "gpu-telemetry.jsonl").open("a") as output:
                output.write(json.dumps(row, allow_nan=False) + "\n")
            time.sleep(60)


if __name__ == "__main__":
    main()
