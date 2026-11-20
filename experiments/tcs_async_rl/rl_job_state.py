"""Persist this workstream's own RL job events through its existing state helper."""

import datetime
import json
import subprocess


def record(repo, event, *, summary, next_action, needs_user=False):
    registry = "experiments/tcs_async_rl/jobs.jsonl"
    event = {"recorded_at_utc": datetime.datetime.now(datetime.UTC).isoformat(), **event}
    with (repo / registry).open("a") as output:
        output.write(json.dumps(event) + "\n")
    current = json.loads(
        subprocess.check_output(["workstate", "--repo", str(repo), "get", "tcs-async-rl-recon"], text=True)
    )
    update = {"summary": summary, "nextAction": next_action, "state": "needs-you" if needs_user else "running"}
    if event.get("job_id"):
        job = {"id": event["job_id"], "cluster": "hpcfund", "status": event["status"]}
        update["jobs"] = [job, *[old for old in current["jobs"] if old["id"] != job["id"]]]
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
            summary,
        ],
        input=json.dumps(update),
        text=True,
        check=True,
    )
    paths = [registry, ".workstreams/tcs-async-rl-recon.json", "STATUS.md"]
    subprocess.run(["git", "add", *paths], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--only", "-m", summary, "--", *paths], cwd=repo, check=True)
    pushed = subprocess.run(["git", "push"], cwd=repo, check=False)
    if pushed.returncode:
        subprocess.run(["git", "pull", "--rebase"], cwd=repo, check=True)
        subprocess.run(["git", "push"], cwd=repo, check=True)
