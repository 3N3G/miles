"""Submit the authorized pilot from the VM and immediately record its receipt."""

import argparse
import datetime
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="amd")
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    data = Path(args.data_root)
    out = data / "runs" / args.run_name
    python = ["python", "-u", "src/pilot.py", "--items", "evidence/pilot_items.json", "--out", str(out)]
    command = ["apptainer", "exec", "--cleanenv", "--rocm", "--bind", "/work1:/work1", "--env", f"PYTHONPATH={data / 'runtime'},HF_HOME={data / 'hf'},OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4,TOKENIZERS_PARALLELISM=false", args.image, *python]
    script = "#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(args.remote_repo) + "\n" + shlex.join(command) + "\n"
    sbatch = ["sbatch", "--parsable", "--nodes=1", "--gpus=1", "--cpus-per-task=8", "--partition=mi3501x", "--account=aviralkumar", "--qos=alloc_aviralkumar_09082026_09302027", "--time=01:00:00", "--job-name=inkling-state-pilot", "--output", str(out) + "-%j.log"]
    remote = "mkdir -p " + shlex.quote(str(out)) + " && cd " + shlex.quote(args.remote_repo) + " && git pull --rebase && " + shlex.join(sbatch)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, remote], input=script, capture_output=True, text=True)
    print(result.stdout, end="")
    print(result.stderr, end="")
    lines = result.stdout.strip().splitlines()
    job_id = lines[-1].split(";")[0] if result.returncode == 0 and lines else None
    if job_id is not None and not job_id.isdigit():
        raise RuntimeError("Unrecognized scheduler receipt")
    receipt = {"at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cluster": "hpcfund", "job_id": job_id, "submission_returncode": result.returncode, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "run_name": args.run_name, "output": str(out), "stdout": result.stdout, "stderr": result.stderr, "script": script}
    with Path("jobs.jsonl").open("a") as f:
        f.write(json.dumps(receipt) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
