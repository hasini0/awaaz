"""Publish the app as a Hugging Face Space (Docker, free CPU).

    hf auth login                      # once: paste a token with "write" access
    python scripts/deploy_space.py     # creates/updates https://huggingface.co/spaces/<you>/awaaz

Uploads exactly the files tracked by git (so no virtualenv, local database or MuRIL weights), in one commit.
Without MuRIL the Space runs the TF-IDF + e5 ensemble. Spaces need a settings header on README.md;
it is added to the uploaded copy only, so the GitHub README stays clean.
"""
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_HEADER = """---
title: Awaaz
emoji: 🎙️
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: Multilingual voice civic grievance chatbot
---

"""


def main(name: str = "awaaz"):
    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(ROOT / f)) for f in files if f != "README.md"]
    readme = SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8")
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))

    api.create_commit(repo_id, operations=ops, commit_message="Deploy Awaaz", repo_type="space")
    print(f"Uploaded {len(ops)} files. Building now (about 10-15 min):\n  https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main(*sys.argv[1:])
