"""Explicit installation of pinned local model artifacts, never a provider call."""

import json
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    manifest = json.loads(Path("model_manifest.json").read_text())
    for name, revision in manifest["models"].items():
        snapshot_download(
            name, revision=revision, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "1_Pooling/*"]
        )
        print(f"Cached {name}@{revision}")


if __name__ == "__main__":
    main()
