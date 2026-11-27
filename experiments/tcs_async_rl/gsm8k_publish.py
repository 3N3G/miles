"""Upload a terminal GSM8K run and verify local and previously evicted artifacts."""

import argparse
import collections
import json
from pathlib import Path

from experiments.tcs_async_rl.gsm8k_artifacts import write_json
from experiments.tcs_async_rl.hf_artifacts import verify_files
from experiments.tcs_async_rl.rl_archive import read_index, upload_run
from experiments.tcs_async_rl.rl_artifacts import file_sha256


def publish(directory):
    # Hub authentication remains in the VM SDK. No credentials enter artifacts.
    from huggingface_hub import HfApi

    monitor = json.loads((directory / "monitor-complete.json").read_text())
    if monitor["scheduler"]["state"] != "COMPLETED":
        raise ValueError("This finalizer requires successful Slurm completion")
    if not json.loads((directory / "wandb-full-verification.json").read_text())["passed"]:
        raise ValueError("Verify every W&B scalar before final publication")
    local = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(directory)
        if {".cache", "cache", "__pycache__"} & set(relative.parts) or path.suffix == ".lock":
            continue
        if path.name in {"final-upload.log", "final-publication.json"}:
            continue  # These receipts describe this upload and are written later.
        if path.suffix in {".pt", ".pth", ".bin", ".safetensors", ".tar", ".sqsh"} or "checkpoints" in relative.parts:
            raise ValueError("Native tensors and checkpoints must stay on AMD")
        local[str(relative)] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    receipt = upload_run(directory, log_name="final-upload.log")
    repo = receipt.split("/datasets/", 1)[1].rsplit("/commit/", 1)[0]
    revision = receipt.rsplit("/", 1)[1]
    prefix = f"miles/{directory.name}/"
    api = HfApi()
    remote_files = list(api.list_repo_tree(repo, path_in_repo=prefix.rstrip("/"), recursive=True, repo_type="dataset", revision=revision))
    remote = {item.path: item for item in remote_files}
    verify_files(directory, local, remote_files)
    archived = read_index(directory)
    old_commits = collections.defaultdict(list)
    for name, entry in archived.items():
        item = remote.get(prefix + name)
        if item is None or item.size != entry["bytes"]:
            raise ValueError(f"Final archive lost an evicted artifact: {name}")
        if item.lfs is not None:
            if item.lfs.sha256 != entry["sha256"]:
                raise ValueError(f"Final archive changed an evicted artifact: {name}")
        else:
            # Its prior immutable object was already checked against local SHA256
            # before eviction. Identical Git blob IDs preserve those exact bytes.
            old_revision = entry["hf_commit_url"].rsplit("/", 1)[1]
            old_commits[old_revision].append(name)
    for old_revision, names in old_commits.items():
        for start in range(0, len(names), 128):
            paths = [prefix + name for name in names[start : start + 128]]
            old = api.get_paths_info(repo, paths, repo_type="dataset", revision=old_revision)
            if {item.path for item in old} != set(paths):
                raise ValueError("Prior immutable archive objects are incomplete")
            for item in old:
                if remote[item.path].blob_id != item.blob_id:
                    raise ValueError(f"Final archive changed a verified Git object: {item.path}")
    all_entries = {**archived, **local}
    result = {"commit_url": receipt, "path": prefix.rstrip("/"), "verified": True, "verified_files": len(all_entries), "verified_bytes": sum(x["bytes"] for x in all_entries.values()), "local_files": len(local), "previously_evicted_files": len(archived), "checkpoints_uploaded": False}
    write_json(directory / "final-publication.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    print(json.dumps(publish(parser.parse_args().directory), indent=2))
