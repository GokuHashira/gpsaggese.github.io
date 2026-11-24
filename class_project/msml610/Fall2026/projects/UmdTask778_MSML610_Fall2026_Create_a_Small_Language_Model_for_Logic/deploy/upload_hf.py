#!/usr/bin/env python
"""
Upload a release folder (model) or the demo app (Space) to the Hugging Face Hub.

Run on a LOGIN node: compute nodes cannot reach the Hub's file servers. Uses the
token stored by `huggingface-cli login` (never passed on the command line and
never printed). Repositories are created **private** unless `--public` is given.

Examples:

> python deploy/upload_hf.py model models/release/slm-math-358m-sft --repo-id <user>/slm-math-358m-sft
> python deploy/upload_hf.py space deploy/space --repo-id <user>/slm-math-demo
"""

import argparse
import os
from typing import List, Optional

import huggingface_hub

_REQUIRED = {
    "model": ("config.json", "model.safetensors", "tokenizer.json", "README.md"),
    "space": ("app.py", "requirements.txt", "README.md"),
}


def main(argv: Optional[List[str]] = None) -> str:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=("model", "space"))
    ap.add_argument("folder")
    ap.add_argument("--repo-id", required=True, help="<user or org>/<name>")
    ap.add_argument("--public", action="store_true", help="create the repo public (default private)")
    ap.add_argument("--message", default="Upload")
    a = ap.parse_args(argv)
    missing = [f for f in _REQUIRED[a.kind] if not os.path.exists(os.path.join(a.folder, f))]
    if missing:
        raise SystemExit(f"{a.folder} is missing {missing}")
    api = huggingface_hub.HfApi()
    user = api.whoami()["name"]  # fails early if not logged in; prints only the user name
    print(f"logged in as {user}; uploading {a.kind} {a.folder} -> {a.repo_id} "
          f"({'public' if a.public else 'private'})")
    api.create_repo(a.repo_id, repo_type=a.kind, private=not a.public, exist_ok=True,
                    **({"space_sdk": "gradio"} if a.kind == "space" else {}))
    api.upload_folder(repo_id=a.repo_id, repo_type=a.kind, folder_path=a.folder,
                      commit_message=a.message, ignore_patterns=["__pycache__/*", "*.pyc"])
    url = f"https://huggingface.co/{'spaces/' if a.kind == 'space' else ''}{a.repo_id}"
    print(f"done: {url}")
    return url


if __name__ == "__main__":
    main()
