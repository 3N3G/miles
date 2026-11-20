"""Durable VM monitor: import, provisionally grade, analyze, and upload one own job.

This never submits, cancels, or modifies a Slurm job. Run after committing the
monitor and grading code. All provider credentials remain in the VM client.
"""

import argparse
import datetime
import fcntl
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"}


def remote_state(job_id):
    if not job_id.isdigit():
        raise ValueError("Job ID must be numeric")
    result = subprocess.run(
        ["ssh", "amd", f"sacct -X -j {job_id} --noheader --parsable2 --format=State,ExitCode,Elapsed,Start,End"],
        capture_output=True,
        text=True,
        timeout=40,
        check=True,
    )
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return {"state": "UNKNOWN"}
    state, exit_code, elapsed, start, end = rows[0][:5]
    return {
        "state": state.split()[0],
        "exit_code": exit_code,
        "elapsed": elapsed,
        "start_cluster_time": start,
        "end_cluster_time": end,
    }


def _run(argv, path):
    with path.open("w") as output:
        result = subprocess.run(argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def _record_completion(job_id, state, run_dir, hf_url, repo, continue_to_rl=False):
    event = {
        "recorded_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "cluster": "hpcfund",
        "job_id": job_id,
        "status": state["state"],
        "scheduler": state,
        "local_results": str(run_dir),
        "hf_results": hf_url,
    }
    grades = run_dir / "group-grades.json"
    if grades.exists():
        result = json.loads(grades.read_text())
        event.update(
            scored_responses=result["scored_responses"],
            all_zero=result["all_zero"],
            uniform_scores=result["uniform_scores"],
        )
    with (repo / "experiments/tcs_async_rl/jobs.jsonl").open("a") as output:
        output.write(json.dumps(event) + "\n")
    current = json.loads(
        subprocess.check_output(["workstate", "--repo", str(repo), "get", "tcs-async-rl-recon"], text=True)
    )
    jobs = [{**job, "status": state["state"]} if job["id"] == job_id else job for job in current["jobs"]]
    detail = (
        f"{event['scored_responses']} provisional group grades saved."
        if grades.exists()
        else "No group grades exist; inspect generation and stage logs."
    )
    succeeded = state["state"] == "COMPLETED" and state.get("grading_returncode") == 0 and hf_url is not None
    update = {
        "state": "needs-you" if succeeded and not continue_to_rl else "running",
        "summary": f"AMD job {job_id} ended {state['state']}. {detail} Human calibration remains pending.",
        "nextAction": (
            "Verify nonconstant grades, then submit and debug the authorized bounded RL runtime check."
            if succeeded and continue_to_rl
            else (
                "Review the saved group and provisional grades before selecting a learning run."
                if succeeded
                else "Diagnose the recorded inference, grading, or upload failure under the approved repair scope. Preserve all old inputs and results."
            )
        ),
        "jobs": jobs,
    }
    if hf_url:
        update["sources"] = current["sources"] + [
            {"url": hf_url, "label": f"Private results for AMD inference job {job_id}"}
        ]
    subprocess.run(
        [
            "workstate",
            "--repo",
            str(repo),
            "put",
            "tcs-async-rl-recon",
            "--expected-revision",
            str(current["revision"]),
            "--actor",
            "codex",
            "--change",
            f"Imported terminal state and results for approved AMD job {job_id}",
        ],
        input=json.dumps(update),
        text=True,
        check=True,
    )
    paths = ["experiments/tcs_async_rl/jobs.jsonl", ".workstreams/tcs-async-rl-recon.json", "STATUS.md"]
    subprocess.run(["git", "add", *paths], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "--only", "-m", f"Record AMD group job {job_id} completion", "--", *paths],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "push"], cwd=repo, check=True)


def watch(*, job_id, remote_run_dir, run_dir, poll_seconds, state_repo, continue_to_rl=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            state = remote_state(job_id)
            state["observed_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
            (run_dir / "watcher-state.json").write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps(state), flush=True)
        except (subprocess.SubprocessError, OSError) as error:
            print(f"Scheduler check failed: {type(error).__name__}", flush=True)
            time.sleep(poll_seconds)
            continue
        if state["state"] in ACTIVE or state["state"] == "UNKNOWN":
            time.sleep(poll_seconds)
            continue
        break
    subprocess.run(["rsync", "-a", f"amd:{remote_run_dir}/", f"{run_dir}/"], check=True)
    if (run_dir / "responses.jsonl").exists() and (run_dir / "responses.jsonl").stat().st_size:
        grade_code = _run(
            [
                "python3",
                "-m",
                "experiments.tcs_async_rl.grade_group",
                "--run-dir",
                str(run_dir),
                "--input-file",
                str(run_dir / "input.json"),
            ],
            run_dir / "grading.log",
        )
        state["grading_returncode"] = grade_code
        _run(
            [
                "uv",
                "run",
                "--no-project",
                "--with",
                "matplotlib",
                "python",
                "-m",
                "experiments.tcs_async_rl.analyze_group",
                str(run_dir),
            ],
            run_dir / "analysis.log",
        )
    upload = subprocess.run(["push-run", "miles", str(run_dir)], capture_output=True, text=True, check=False)
    (run_dir / "upload.log").write_text(upload.stdout + upload.stderr)
    urls = [line for line in upload.stdout.splitlines() if line.startswith("https://huggingface.co/datasets/")]
    hf_url = urls[-1] if upload.returncode == 0 and urls else None
    _record_completion(job_id, state, run_dir, hf_url, state_repo, continue_to_rl=continue_to_rl)
    (run_dir / "watcher-complete.json").write_text(
        json.dumps({"scheduler": state, "hf_results": hf_url}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--state-repo", type=Path, default=ROOT)
    parser.add_argument(
        "--continue-to-rl", action="store_true", help="Keep the workstream active for an already approved RL run"
    )
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watch(
            job_id=args.job_id,
            remote_run_dir=args.remote_run_dir,
            run_dir=args.run_dir,
            poll_seconds=args.poll_seconds,
            state_repo=args.state_repo,
            continue_to_rl=args.continue_to_rl,
        )


if __name__ == "__main__":
    main()
