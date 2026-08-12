# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Prepare and validate a LeRobot Bi-Flexiv record for RECAP.

The Flexiv recorder already exports LeRobot v3.  This command therefore does
not rewrite the data: it safely extracts a zip (when requested), validates the
three camera streams and 20-D state/action fields, and writes a small manifest
used to document the RECAP run.

Example:
    python examples/offline_rl/prepare_flexiv_recap.py \
        --input-zip /path/to/stack-cubes-flexiv.zip \
        --output-dir /data/stack-cubes-flexiv
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


REQUIRED_VIEWS = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _find_dataset_root(root: Path) -> Path:
    candidates = [root, *[p for p in root.iterdir() if p.is_dir()]]
    for candidate in candidates:
        if (candidate / "meta" / "info.json").exists():
            return candidate
    raise FileNotFoundError(f"No LeRobot dataset found below {root}")


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        base = destination.resolve()
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe zip member: {member.filename}")
        zf.extractall(destination)
    return _find_dataset_root(destination)


def _validate_dataset(dataset: Path) -> dict:
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    missing = [name for name in REQUIRED_VIEWS if name not in features]
    if missing:
        raise ValueError(f"Missing camera features: {missing}")
    for key in ("observation.state", "action"):
        shape = features.get(key, {}).get("shape")
        if shape != [20]:
            raise ValueError(f"{key} must have shape [20], got {shape}")
    for view in REQUIRED_VIEWS:
        video_path = dataset / "videos" / view
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video directory: {video_path}")
    if not (dataset / "data").exists() or not (dataset / "meta").exists():
        raise ValueError("Dataset must contain data/ and meta/ directories")
    return {
        "dataset": str(dataset.resolve()),
        "format": "lerobot_v3",
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "views": list(REQUIRED_VIEWS),
        "state_dim": 20,
        "action_dim": 20,
        "recap_dataset_type": "sft",
        "note": (
            "RECAP compute_returns uses successful-demo semantics for type=sft. "
            "Use type=rollout only after adding per-episode is_success labels."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-zip", type=Path)
    source.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.input_zip:
        if not args.input_zip.exists():
            raise FileNotFoundError(args.input_zip)
        if args.output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {args.output_dir}"
            )
        dataset = _safe_extract(args.input_zip, args.output_dir)
        # If the zip contains a single top-level dataset directory, keep that
        # directory as the LeRobot root.  The manifest points to it explicitly.
    else:
        dataset = args.dataset_dir.resolve()
        if not dataset.exists():
            raise FileNotFoundError(dataset)

    manifest = _validate_dataset(dataset)
    # Keep the manifest outside LeRobot's reserved metadata files.  RECAP and
    # LeRobot versions differ in how strictly they validate meta/ contents.
    manifest_path = dataset / "recap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"RECAP dataset ready: {dataset}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
