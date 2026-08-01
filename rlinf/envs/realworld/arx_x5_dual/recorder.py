# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""LeRobot episode recording for ARX X5 dual-arm evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import packaging.version

from rlinf.utils.eval_events import emit_event
from rlinf.utils.logging import get_logger

_STATE_NAMES = (
    [f"left_joint_{index}.pos" for index in range(1, 7)]
    + ["left_gripper.pos"]
    + [f"right_joint_{index}.pos" for index in range(1, 7)]
    + ["right_gripper.pos"]
)
_CAMERA_KEYS = {
    "base_0_rgb": "cam_high",
    "left_wrist_0_rgb": "cam_left_wrist",
    "right_wrist_0_rgb": "cam_right_wrist",
}


def _features(image_height: int, image_width: int, use_videos: bool) -> dict:
    image_dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": _STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": _STATE_NAMES,
        },
    }
    for camera_name in _CAMERA_KEYS.values():
        features[f"observation.images.{camera_name}"] = {
            "dtype": image_dtype,
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _canonicalize_features(features: Mapping[str, Any]) -> dict:
    """Reduce features to the semantic fields used for resume validation.

    ``info.json`` stores DEFAULT_FEATURES (``timestamp``/``frame_index``/...) and
    encoder-injected ``info`` blocks that the in-code definition does not have,
    so we only compare ``dtype``/``shape``/``names`` for the recording keys.
    JSON round-trips convert tuples to lists, so normalize for a stable compare.
    """

    return {
        key: json.loads(
            json.dumps(
                {field: value.get(field) for field in ("dtype", "shape", "names")},
                sort_keys=True,
                default=list,
            )
        )
        for key, value in features.items()
    }


def _check_features_match(existing: Mapping[str, Any], requested: Mapping) -> None:
    existing_c = _canonicalize_features(existing)
    requested_c = _canonicalize_features(requested)
    mismatched = sorted(
        key for key in set(requested_c) if existing_c.get(key) != requested_c.get(key)
    )
    if not mismatched:
        return
    raise ValueError(
        "ARX LeRobot resume requested but dataset features mismatch; "
        f"mismatched keys: {mismatched}. Existing dataset: "
        f"{json.dumps({k: existing_c.get(k) for k in mismatched}, sort_keys=True)}, "
        "requested: "
        f"{json.dumps({k: requested_c.get(k) for k in mismatched}, sort_keys=True)}."
    )


def _check_integrity(root: Path) -> None:
    """Run LeRobot's local integrity check on an existing dataset directory.

    Missing ``meta/`` files, parquet files, or videos make the loading fail
    before any appending happens; unreadable parquet content surfaces when
    the dataset table is materialized. On Hub fallback, abort instead: this
    runs on the robot node before recording, and silently downloading episodes
    is never the right action there.
    """

    try:
        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
        from lerobot.common.datasets.utils import (
            EPISODES_PATH,
            EPISODES_STATS_PATH,
            INFO_PATH,
            STATS_PATH,
            TASKS_PATH,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ARX recording requires RLinf's pinned LeRobot installation. "
            "Re-run requirements/install.sh for openpi + arx_x5_dual."
        ) from exc

    # 1. Meta files required by LeRobotDatasetMetadata.load_metadata().
    # Check presence first: otherwise a missing meta file triggers a Hub
    # download attempt inside load_metadata(), which must never happen on
    # the robot node during resume.
    info_path = root / INFO_PATH
    if not info_path.is_file():
        raise FileNotFoundError(
            f"ARX LeRobot resume integrity check: dataset at {root} is missing "
            f"{INFO_PATH}; it looks incomplete or corrupted. Move or delete it "
            "before resuming, or point recording.repo_id/root elsewhere."
        )
    info = json.loads(info_path.read_text())
    meta_paths = [TASKS_PATH, EPISODES_PATH]
    meta_paths.append(
        STATS_PATH
        if packaging.version.parse(info["codebase_version"])
        < packaging.version.parse("v2.1")
        else EPISODES_STATS_PATH
    )
    missing_meta = [path for path in meta_paths if not (root / path).is_file()]
    if missing_meta:
        raise FileNotFoundError(
            f"ARX LeRobot resume integrity check: dataset at {root} is missing "
            f"meta files {missing_meta}; it looks incomplete or corrupted. Move "
            "or delete it before resuming, or point recording.repo_id/root "
            "elsewhere."
        )

    # All meta files are on disk, so loading stays fully local.
    meta = LeRobotDatasetMetadata(repo_id="local/check", root=root)

    # 2. Per-episode parquet and video files required by LeRobotDataset.
    file_paths = [
        meta.get_data_file_path(ep_idx) for ep_idx in range(meta.total_episodes)
    ]
    file_paths += [
        meta.get_video_file_path(ep_idx, vid_key)
        for vid_key in meta.video_keys
        for ep_idx in range(meta.total_episodes)
    ]
    missing_files = [path for path in file_paths if not (root / path).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"ARX LeRobot resume integrity check: dataset at {root} is missing "
            f"{len(missing_files)} data/video files, e.g. {missing_files[:5]}; "
            "it looks incomplete or corrupted. Move or delete it before "
            "resuming, or point recording.repo_id/root elsewhere."
        )

    # 3. Load from local disk; this parses the parquet files and runs
    # LeRobot's timestamp/fps consistency check. Guard against Hub downloads.
    dataset = LeRobotDataset(repo_id="local/check", root=root, download_videos=False)
    if dataset.revision != meta.revision:  # a Hub fallback happened
        raise RuntimeError(
            f"ARX LeRobot resume integrity check: dataset at {root} triggered "
            "a Hub fallback, refusing to download on the robot node. Move or "
            "delete the local dataset before resuming, or point "
            "recording.repo_id/root elsewhere."
        )


class ArxX5DualLeRobotRecorder:
    """Buffer ARX inference frames and explicitly save or discard episodes."""

    def __init__(
        self,
        *,
        repo_id: str,
        task: str,
        fps: int,
        image_height: int,
        image_width: int,
        root: str | None = None,
        use_videos: bool = True,
        image_writer_threads: int = 4,
        image_writer_processes: int = 0,
        resume: bool = False,
    ) -> None:
        try:
            from lerobot.common.constants import HF_LEROBOT_HOME
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "ARX recording requires RLinf's pinned LeRobot installation. "
                "Re-run requirements/install.sh for openpi + arx_x5_dual."
            ) from exc

        self._logger = get_logger()
        self._task = task
        self._frame_count = 0
        self._closed = False
        dataset_root = (
            Path(root).expanduser() if root else Path(HF_LEROBOT_HOME) / repo_id
        )
        features = _features(image_height, image_width, use_videos)
        if resume and (dataset_root / "meta" / "info.json").is_file():
            _check_integrity(dataset_root)
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_root,
                download_videos=False,
            )
            _check_features_match(dataset.meta.features, features)
            if image_writer_processes or image_writer_threads:
                dataset.start_image_writer(image_writer_processes, image_writer_threads)
            dataset.episode_buffer = dataset.create_episode_buffer()
            self._logger.info(
                "Resuming ARX LeRobot recording: repo_id=%s, root=%s, "
                "existing_episodes=%d",
                repo_id,
                dataset_root,
                dataset.meta.total_episodes,
            )
        elif resume and dataset_root.exists() and any(dataset_root.iterdir()):
            # Non-empty directory without meta/info.json: run the integrity
            # check so a merely unusual layout (e.g. renamed meta dir) can be
            # diagnosed precisely; genuinely broken datasets raise here with
            # the list of missing pieces.
            _check_integrity(dataset_root)
            raise RuntimeError(
                "ARX LeRobot resume requested but the dataset at "
                f"{dataset_root} is missing meta/info.json while its data "
                "files pass the integrity check; it looks incomplete or "
                "corrupted. Move or delete it before resuming, or point "
                "recording.repo_id/root elsewhere."
            )
        else:
            dir_exists = dataset_root.exists()
            if not resume and dir_exists:
                raise FileExistsError(
                    f"ARX LeRobot dataset directory {dataset_root} already exists. "
                    "Set recording.resume=true to append to the existing dataset, or "
                    "move/delete the directory to record from scratch."
                )
            if resume:
                if dir_exists:
                    # Empty leftover directory: LeRobotDataset.create() calls
                    # mkdir(exist_ok=False), so remove the empty dir to avoid a
                    # FileExistsError.
                    dataset_root.rmdir()
                self._logger.info(
                    "ARX LeRobot resume requested but no existing dataset at %s; "
                    "creating a new one.",
                    dataset_root,
                )
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=dataset_root,
                robot_type="bi_arx5",
                fps=fps,
                features=features,
                use_videos=use_videos,
                image_writer_threads=image_writer_threads,
                image_writer_processes=image_writer_processes,
            )
        self._dataset = dataset
        self._logger.info(
            "ARX LeRobot recording enabled: repo_id=%s, root=%s, task=%r, fps=%d",
            repo_id,
            dataset_root,
            task,
            fps,
        )

    def add_frame(
        self,
        observation: Mapping[str, Any],
        action: np.ndarray,
    ) -> None:
        """Add one pre-action observation and its executed absolute action."""

        if self._frame_count == 0:
            emit_event("recording_start", "recorder", task=self._task)
        state = observation["state"]["joint_position"]
        frames = observation["frames"]
        frame = {
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": self._task,
        }
        for source_key, dataset_key in _CAMERA_KEYS.items():
            frame[f"observation.images.{dataset_key}"] = np.asarray(
                frames[source_key], dtype=np.uint8
            )
        self._dataset.add_frame(frame)
        self._frame_count += 1

    def save_episode(self) -> None:
        """Commit the current episode to the LeRobot dataset."""

        if self._frame_count == 0:
            self._logger.warning("ARX recording episode is empty; nothing was saved.")
            return
        frame_count = self._frame_count
        emit_event("recording_save_start", "recorder", frames=frame_count)
        self._dataset.save_episode()
        self._frame_count = 0
        emit_event("recording_saved", "recorder", frames=frame_count)
        self._logger.info("Saved ARX LeRobot episode with %d frames.", frame_count)

    def discard_episode(self, reason: str) -> None:
        """Drop buffered metadata and temporary images for the current episode."""

        if self._frame_count == 0:
            return
        frame_count = self._frame_count
        image_writer = getattr(self._dataset, "image_writer", None)
        if image_writer is not None:
            image_writer.wait_until_done()
        self._dataset.clear_episode_buffer()
        self._frame_count = 0
        emit_event(
            "recording_discarded", "recorder", frames=frame_count, reason=reason
        )
        self._logger.info(
            "Discarded ARX LeRobot episode with %d frames (%s).",
            frame_count,
            reason,
        )

    def close(self) -> None:
        """Discard an unfinished episode and stop asynchronous image writers."""

        if self._closed:
            return
        self._closed = True
        self.discard_episode("environment closed before save")
        image_writer = getattr(self._dataset, "image_writer", None)
        if image_writer is not None:
            image_writer.wait_until_done()
            image_writer.stop()
