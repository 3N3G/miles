"""Read GPU utilization inside this session's existing AMD allocation.

Use one overlapping read-only Slurm step, not a new allocation. The host's
sysfs files avoid depending on a particular container's management binaries.
"""

import argparse
import datetime
import json
import shlex
import subprocess
from pathlib import Path

PROBE = """
import json,pathlib
fields=('gpu_busy_percent','mem_info_vram_total','mem_info_vram_used')
rows=[]
for path in pathlib.Path('/sys/class/drm').glob('card[0-9]*/device'):
    if all((path/name).exists() for name in fields):
        rows.append({'device':path.parent.name,**{name:int((path/name).read_text()) for name in fields}})
print(json.dumps(sorted(rows,key=lambda row:int(row['device'][4:]))))
"""


def sample(directory, session_uuid):
    submission = json.loads((directory / "submission.json").read_text())
    if submission["session_uuid"] != session_uuid:
        raise ValueError("Only inspect this session's recorded allocation")
    job = submission["job_id"]
    if not job.isdigit():
        raise ValueError("Expected a recorded Slurm job ID")
    argv = [
        "srun",
        f"--jobid={job}",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--immediate=10",
        "python3",
        "-c",
        PROBE,
    ]
    result = subprocess.run(["ssh", "amd", shlex.join(argv)], capture_output=True, text=True, timeout=40)
    result.check_returncode()
    devices = json.loads(result.stdout)
    if len(devices) != 8:
        raise ValueError("Expected telemetry from all eight allocation GPUs")
    row = {"at_utc": datetime.datetime.now(datetime.UTC).isoformat(), "job_id": job, "devices": devices}
    with (directory / "gpu-telemetry.jsonl").open("a") as output:
        output.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--session-uuid", required=True)
    args = parser.parse_args()
    print(json.dumps(sample(args.run_directory, args.session_uuid)))
