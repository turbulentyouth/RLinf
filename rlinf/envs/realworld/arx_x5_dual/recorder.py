# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""LeRobot episode recording for ARX X5 dual-arm evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

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
        else:
            dir_exists = dataset_root.exists()
            non_empty = dir_exists and any(dataset_root.iterdir())
            if resume and non_empty:
                raise RuntimeError(
                    "ARX LeRobot resume requested but the dataset at "
                    f"{dataset_root} is missing meta/info.json while its directory "
                    "is non-empty; it looks incomplete or corrupted. Move or delete "
                    "it before resuming, or point recording.repo_id/root elsewhere."
                )
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
        self._dataset.save_episode()
        self._frame_count = 0
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
