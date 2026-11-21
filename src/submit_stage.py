"""Submit a recorded research stage without modifying another project's runtime."""

import argparse
import datetime
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--minutes", type=int, required=True)
    parser.add_argument("--host", default="amd")
    parser.add_argument("--root", default="/work1/aviralkumar/geney/inkling-interp")
    parser.add_argument("--data", default="/work1/aviralkumar/geney/inkling-interp-data")
    parser.add_argument("--image", default="/work1/aviralkumar/geney/containers/vllm-rocm-v0.29.0.sif")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or not 1 <= args.minutes <= 60:
        raise ValueError("Provide a command and a runtime of 1..60 minutes")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    command = ["apptainer", "exec", "--cleanenv", "--rocm", "--bind", "/work1:/work1", "--env", f"PYTHONPATH={args.data}/runtime,HF_HOME={args.data}/hf,OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4,TOKENIZERS_PARALLELISM=false", args.image, *args.command]
    script = "#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(args.root) + "\n" + shlex.join(command) + "\n"
    sbatch = ["sbatch", "--parsable", "--nodes=1", "--cpus-per-task=8", "--partition=mi3501x", "--account=aviralkumar", "--qos=alloc_aviralkumar_09082026_09302027", f"--time=00:{args.minutes:02}:00" if args.minutes < 60 else "--time=01:00:00", f"--job-name=inkling-{args.name}", f"--output={args.data}/runs/{args.name}-%j.log"]
    remote = "cd " + shlex.quote(args.root) + " && git pull --rebase && test \"$(git rev-parse HEAD)\" = " + shlex.quote(commit) + " && " + shlex.join(sbatch)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, remote], input=script, text=True, capture_output=True)
    job = result.stdout.strip().splitlines()[-1].split(";")[0] if result.returncode == 0 else None
    if job is not None and not job.isdigit():
        raise RuntimeError("Unrecognized scheduler receipt")
    row = {"at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cluster": "hpcfund", "job_id": job, "submission_returncode": result.returncode, "code_commit": commit, "run_name": args.name, "plan": args.plan, "script": script, "stdout": result.stdout, "stderr": result.stderr}
    with Path("jobs.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({"job_id": job, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
