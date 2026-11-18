"""VM-only credential loading and receipt-preserving OpenAI judge requests.

The user authorized this local loader on 2026-11-18. It never exports credentials
to the cluster, command arguments, saved requests, or result artifacts.
"""

import argparse
import json
import os
import re
import shlex
import urllib.error
import urllib.request
from pathlib import Path

PROVIDER_VARIABLES = {"OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"}


def load_credentials(path):
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError("Credential file must be owned by this user and must not be a symlink")
    credentials = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"(?:export\s+)?([A-Z_]+)\s*=(.*)", line.strip())
        if match is None or match[1] not in PROVIDER_VARIABLES:
            continue
        parts = shlex.split(match[2], comments=True)
        if len(parts) != 1 or not parts[0] or any(char in parts[0] for char in "$`\n\r"):
            raise ValueError("Unsupported provider assignment")
        if match[1] in credentials:
            raise ValueError("Duplicate provider assignment")
        credentials[match[1]] = parts[0]
    return credentials


def metadata_check(credentials):
    checks = []
    probes = [
        ("OPENAI_API_KEY", "https://api.openai.com/v1/models/gpt-6-luna", "Authorization"),
        ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon", "x-goog-api-key"),
    ]
    for variable, url, header in probes:
        if variable not in credentials:
            checks.append({"variable": variable, "status": "absent"})
            continue
        value = f"Bearer {credentials[variable]}" if header == "Authorization" else credentials[variable]
        request = urllib.request.Request(url, headers={header: value})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                checks.append({"variable": variable, "http_status": response.status})
        except urllib.error.HTTPError as error:
            checks.append({"variable": variable, "http_status": error.code})
        except (urllib.error.URLError, TimeoutError) as error:
            checks.append({"variable": variable, "error_type": type(error).__name__})
    return checks


def request_chat(payload, *, key, endpoint="https://api.openai.com/v1/chat/completions", timeout=600):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
            return {
                "http_status": response.status,
                "body": {name: body.get(name) for name in ("id", "model", "choices", "usage")},
                "request_id": response.headers.get("x-request-id"),
            }
    except urllib.error.HTTPError as error:
        # Error bodies can echo inputs. Never log exception text or request headers.
        return {"http_status": error.code, "request_id": error.headers.get("x-request-id")}
    except (urllib.error.URLError, TimeoutError) as error:
        return {"error_type": type(error).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".config/api-keys.env")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = metadata_check(load_credentials(args.key_file))
    record = {
        "host": "VM",
        "scope": "Metadata GET only. No credential values or response bodies retained.",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
