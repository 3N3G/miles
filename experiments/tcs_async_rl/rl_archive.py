"""Bound VM storage by evicting immutable trace copies only after successful private upload.

The original artifacts stay on AMD. Hugging Face upload does not delete older
remote files. Each evicted copy records its bytes, checksum and immutable commit.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes, file_sha256

TRACE_BATCH_BYTES = 64 * 2**20
VM_RESERVE_BYTES = 2**30
VM_STARTUP_RESERVE_BYTES = 5 * 2**30


def require_vm_startup_space(directory):
    """Reject a new GPU submission before the VM lacks room for monitoring."""
    free = shutil.disk_usage(directory).free
    if free < VM_STARTUP_RESERVE_BYTES:
        raise RuntimeError(
            f"VM monitoring needs at least {VM_STARTUP_RESERVE_BYTES} free bytes before submission; "
            f"found {free}. Free verified cache space before allocating GPUs."
        )
    return free


def read_index(directory):
    path = directory / "archive-index.json"
    return json.loads(path.read_text()) if path.exists() else {}


def trace_paths(directory):
    return [
        p
        for p in (directory / "rollout_data").rglob("*.gz")
        if p.is_file() and not p.is_symlink() and p.name.endswith((".jsonl.gz", ".generated.json.gz"))
    ]


def upload_run(directory, *, log_name="upload.log", invoke=subprocess.run):
    """Upload without deleting files, requiring an immutable private receipt."""
    upload_cache = directory / ".cache/xet-upload"
    if upload_cache.is_symlink() or directory.resolve() not in upload_cache.resolve().parents:
        raise ValueError("Upload cache must belong to this run")
    try:
        result = invoke(
            ["push-run", "miles", str(directory)],
            capture_output=True,
            text=True,
            timeout=1800,
            env={**os.environ, "HF_XET_CACHE": str(upload_cache)},
        )
    finally:
        if upload_cache.is_dir() and not upload_cache.is_symlink():
            shutil.rmtree(upload_cache)
    (directory / log_name).write_text(result.stdout + result.stderr)
    result.check_returncode()
    urls = re.findall(r"https://huggingface\.co/datasets/[^\s]+/commit/[0-9a-f]{40}", result.stdout + result.stderr)
    if not urls:
        raise RuntimeError("Upload has no immutable receipt; retain every local trace")
    return urls[-1]


def archive_chunks(directory, *, invoke=subprocess.run):
    paths = trace_paths(directory)
    if not paths:
        return None
    entries = {str(p.relative_to(directory)): {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in paths}
    receipt = upload_run(directory, log_name="incremental-upload.log", invoke=invoke)
    index = read_index(directory)
    for name, entry in entries.items():
        index[name] = {**entry, "hf_commit_url": receipt}
    _atomic_bytes(directory / "archive-index.json", (json.dumps(index, indent=2) + "\n").encode())
    for path in paths:
        if file_sha256(path) == entries[str(path.relative_to(directory))]["sha256"]:
            path.unlink()
    return receipt


def exclusion_file(directory):
    index = read_index(directory)
    paths = []
    for name in index:
        relative = Path(name)
        if relative.parts[0] != "rollout_data" or ".." in relative.parts or relative.is_absolute():
            raise ValueError("Archive index contains an invalid relative path")
        paths.append("/" + name)
    output = directory / "archived-traces.exclude"
    output.write_text("\n".join(paths) + "\n")
    return output


def copy_trace_batch(remote_directory, directory):
    """Copy at most 64 MiB of new immutable traces, retaining one GiB VM margin."""
    code = (
        "import json; from pathlib import Path; "
        f"root=Path({str(remote_directory)!r}); "
        "print(json.dumps([(str(p.relative_to(root)),p.stat().st_size) "
        "for p in (root/'rollout_data').rglob('*.gz') if p.is_file() and not p.is_symlink()]))"
    )
    result = subprocess.run(
        ["ssh", "amd", "python3 -c " + shlex.quote(code)], capture_output=True, text=True, check=True, timeout=60
    )
    archived = read_index(directory)
    local = {str(p.relative_to(directory)): p.stat().st_size for p in trace_paths(directory)}
    pending = [(name, size) for name, size in json.loads(result.stdout) if name not in archived and name not in local]
    free = shutil.disk_usage(directory).free
    allowance = max(0, min(TRACE_BATCH_BYTES - sum(local.values()), free - VM_RESERVE_BYTES))
    selected, total = [], 0
    for name, size in sorted(pending, key=lambda x: (not x[0].endswith(".jsonl.gz"), x[0])):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "rollout_data":
            raise ValueError("Remote trace escaped its run directory")
        if total + size <= allowance:
            selected.append(name)
            total += size
    if pending and not selected and not local:
        raise RuntimeError(
            f"Trace transfer deferred: free_bytes={free}, reserve_bytes={VM_RESERVE_BYTES}, "
            f"pending_files={len(pending)}. AMD originals and unarchived VM files remain intact."
        )
    listing = directory / "trace-transfer.list"
    listing.write_text("\n".join(selected) + "\n")
    if selected:
        subprocess.run(
            ["rsync", "-a", "--files-from", str(listing), f"amd:{remote_directory}/", f"{directory}/"],
            check=True,
            timeout=180,
        )
    return len(pending) - len(selected)
