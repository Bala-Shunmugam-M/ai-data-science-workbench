"""
deploy.huggingface.push_space
=============================

PURPOSE
-------
Publish the workbench to a Hugging Face Space.

WHAT IT UPLOADS
---------------
The application and its pipeline - ``src/``, ``config/``, ``app/``,
``webapi/``, ``main.py``, ``requirements.txt`` and the committed sample CSVs.

Deliberately NOT uploaded:
  * ``webapp/`` beyond its sample data - the Next.js console is a separate
    surface and its node_modules is half a gigabyte.
  * ``workspaces/`` - the Space regenerates its own demo projects at build
    time, so shipping local runs would only make the upload slow and the
    published results ambiguous about where they came from.
  * ``.git``, caches, and build output.

The Space's ``Dockerfile`` and ``README.md`` live in this directory but must
land at the Space's ROOT, because that is where Hugging Face looks for them.
The repository's own README is a different document and is not published here:
HF requires YAML frontmatter that the project README must not carry.

AUTHENTICATION
--------------
Requires a Hugging Face token with WRITE permission, supplied by the user:

    setx HF_TOKEN "hf_..."        (Windows, new shell afterwards)
    export HF_TOKEN=hf_...        (bash)

Create one at https://huggingface.co/settings/tokens - choose "Write".
This script never prints the token.

USAGE
-----
    python deploy/huggingface/push_space.py [--name ai-data-science-workbench]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent

# Only these paths are published. An allow-list rather than an ignore-list: a
# new directory of local experiments should not silently start being uploaded.
INCLUDE = [
    "src", "config", "app", "webapi",
    "main.py", "requirements.txt", "LICENSE",
]
SAMPLES = "webapp/public/samples"


def build_allow_patterns() -> list[str]:
    patterns: list[str] = []
    for item in INCLUDE:
        path = PROJECT_ROOT / item
        patterns.append(f"{item}/**" if path.is_dir() else item)
    patterns.append(f"{SAMPLES}/*.csv")
    patterns.append("deploy/huggingface/bootstrap.py")
    return patterns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="ai-data-science-workbench",
                        help="Space name under your account")
    parser.add_argument("--private", action="store_true",
                        help="create the Space privately")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("No HF_TOKEN in the environment.\n"
              "Create a WRITE token at https://huggingface.co/settings/tokens\n"
              "then set it and re-run:\n"
              '    setx HF_TOKEN "hf_..."   (Windows - open a new shell)\n'
              "    export HF_TOKEN=hf_...   (bash)")
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.name}"
    print(f"Target Space: {repo_id}")

    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)

    # Dockerfile and README must sit at the Space root, so they are uploaded
    # individually rather than as part of the folder sweep.
    for source, dest in ((HERE / "Dockerfile", "Dockerfile"),
                         (HERE / "README.md", "README.md")):
        api.upload_file(path_or_fileobj=str(source), path_in_repo=dest,
                        repo_id=repo_id, repo_type="space")
        print(f"  uploaded {dest}")

    api.upload_folder(
        folder_path=str(PROJECT_ROOT),
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=build_allow_patterns(),
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        commit_message="Deploy the AI Data Science Workbench",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nPushed. The Space builds automatically - the first build takes a\n"
          f"few minutes because it installs dependencies and runs the pipeline\n"
          f"to generate the demo projects.\n\n    {url}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
