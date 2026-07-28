# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""LeRobot episode recording for ARX X5 dual-arm evaluation."""

from __future__ import annotations

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
    ) -> None:
        try:
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
        self._dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=Path(root).expanduser() if root else None,
            robot_type="bi_arx5",
            fps=fps,
            features=_features(image_height, image_width, use_videos),
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )
        self._logger.info(
            "ARX LeRobot recording enabled: repo_id=%s, root=%s, task=%r, fps=%d",
            repo_id,
            root or "<LeRobot default>",
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
