"""Verify uploaded artifacts against immutable Hub hashes before VM eviction."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def verify_files(directory, entries, remote_files):
    remote = {item.path: item for item in remote_files}
    prefix = f"miles/{directory.name}/"
    checked = []
    for name, expected in entries.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact escaped its run directory")
        path = directory / relative
        item = remote.get(prefix + name)
        if item is None or item.size != expected["bytes"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Uploaded artifact is missing or differs in size: {name}")
        sha = hashlib.sha256()
        blob = hashlib.sha1(f"blob {expected['bytes']}\0".encode())
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                sha.update(block)
                blob.update(block)
        if sha.hexdigest() != expected["sha256"]:
            raise ValueError(f"Local artifact changed after upload: {name}")
        if item.lfs is not None:
            if item.lfs.sha256 != expected["sha256"]:
                raise ValueError(f"Uploaded artifact SHA256 differs: {name}")
        elif item.blob_id != blob.hexdigest():
            raise ValueError(f"Uploaded artifact Git blob differs: {name}")
        checked.append({"path": prefix + name, **expected})
    return checked


def main():
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--commit-url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    match = re.fullmatch(r"https://huggingface\.co/datasets/([^/]+/[^/]+)/commit/([0-9a-f]{40})", args.commit_url)
    if match is None:
        raise ValueError("Verification requires an immutable dataset commit")
    repo, revision = match.groups()
    entries = json.loads(args.manifest.read_text())
    paths = [f"miles/{args.directory.name}/{name}" for name in entries]
    api = HfApi()
    remote = []
    for start in range(0, len(paths), 128):
        remote.extend(api.get_paths_info(repo, paths[start : start + 128], repo_type="dataset", revision=revision))
    checked = verify_files(args.directory, entries, remote)
    output = args.directory / "archive-readbacks" / f"{revision}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"repo": repo, "revision": revision, "verified": checked}, indent=2) + "\n")
    print(json.dumps({"repo": repo, "revision": revision, "verified_files": len(checked)}))


if __name__ == "__main__":
    main()
