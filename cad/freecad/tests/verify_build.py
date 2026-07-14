from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_run(run_dir: Path) -> int:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("manifest status is not complete")
    for item in manifest["artifacts"]:
        path = run_dir / item["relative_path"]
        if not path.is_file():
            raise ValueError(f"missing artifact: {path}")
        if digest(path) != item["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
    return len(manifest["artifacts"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        artifact_count = verify_run(args.run_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"BUILD_VERIFY_PASS artifacts={artifact_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
