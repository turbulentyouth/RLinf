# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LeRobot episode recording for Flexiv Rizon4 dual-arm evaluation.

Uses the lerobot-xense fork (``lerobot.datasets.*``, dataset format v3.0) so
recorded datasets are byte-compatible with xense-openpi bi_flexiv training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rlinf.utils.eval_events import emit_event
from rlinf.utils.logging import get_logger

# 与 xense-openpi bi_flexiv 数据集（如 Xense/pack_6_cosmetic_bottles_into_carton）
# 完全一致的 20D 顺序：
# [left_tcp.x/y/z/r1-r6 (9), right_tcp.x/y/z/r1-r6 (9), left_gripper, right_gripper]
_TCP_NAMES = ["x", "y", "z"] + [f"r{index}" for index in range(1, 7)]
_STATE_NAMES = (
    [f"left_tcp.{name}" for name in _TCP_NAMES]
    + [f"right_tcp.{name}" for name in _TCP_NAMES]
    + ["left_gripper.pos", "right_gripper.pos"]
)
# Env frames 键与 xense LeRobot 数据集相机键一致，不做重命名。
_CAMERA_KEYS = {
    "head": "head",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
}


def _features(image_height: int, image_width: int, use_videos: bool) -> dict:
    image_dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (20,),
            "names": _STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (20,),
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
        "Flexiv LeRobot resume requested but dataset features mismatch; "
        f"mismatched keys: {mismatched}. Existing dataset: "
        f"{json.dumps({k: existing_c.get(k) for k in mismatched}, sort_keys=True)}, "
        "requested: "
        f"{json.dumps({k: requested_c.get(k) for k in mismatched}, sort_keys=True)}."
    )


class BiFlexivDualLeRobotRecorder:
    """Buffer Flexiv inference frames and explicitly save or discard episodes.

    Resume semantics mirror ``lerobot-record --resume``: opening an existing
    dataset with ``LeRobotDataset(repo_id, root=...)`` validates the codebase
    version and continues appending from ``total_episodes``. A dataset path
    that does not exist locally raises before any Hub download can happen on
    the robot node.
    """

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
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.utils.constants import HF_LEROBOT_HOME
        except ImportError as exc:
            raise RuntimeError(
                "Flexiv recording requires the lerobot-xense fork "
                "(lerobot.datasets.*). Re-run requirements/install.sh for "
                "openpi + bi_flexiv."
            ) from exc

        self._logger = get_logger()
        self._task = task
        self._frame_count = 0
        self._closed = False
        dataset_root = (
            Path(root).expanduser() if root else Path(HF_LEROBOT_HOME) / repo_id
        )
        features = _features(image_height, image_width, use_videos)
        if resume and dataset_root.is_dir() and any(dataset_root.iterdir()):
            # LeRobotDataset.__init__ runs check_version_compatibility against
            # the local meta; a Hub fallback never happens when root exists.
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_root,
                download_videos=False,
            )
            _check_features_match(dataset.meta.features, features)
            if image_writer_processes or image_writer_threads:
                dataset.start_image_writer(image_writer_processes, image_writer_threads)
            self._logger.info(
                "Resuming Flexiv LeRobot recording: repo_id=%s, root=%s, "
                "existing_episodes=%d",
                repo_id,
                dataset_root,
                dataset.meta.total_episodes,
            )
        else:
            if dataset_root.exists():
                if not resume:
                    raise FileExistsError(
                        f"Flexiv LeRobot dataset directory {dataset_root} already "
                        "exists. Set recording.resume=true to append to the "
                        "existing dataset, or move/delete the directory to "
                        "record from scratch."
                    )
                # Empty leftover directory (or missing meta): LeRobotDataset.
                # create() calls mkdir(exist_ok=False), so remove the empty
                # dir to avoid a FileExistsError. A non-empty dir without
                # valid meta fails loudly inside create() instead of being
                # silently overwritten.
                if not any(dataset_root.iterdir()):
                    dataset_root.rmdir()
                    self._logger.info(
                        "Flexiv LeRobot resume requested but no existing dataset "
                        "at %s; creating a new one.",
                        dataset_root,
                    )
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=dataset_root,
                robot_type="bi_flexiv_rizon4_rt",
                fps=fps,
                features=features,
                use_videos=use_videos,
                image_writer_threads=image_writer_threads,
                image_writer_processes=image_writer_processes,
            )
        self._dataset = dataset
        self._logger.info(
            "Flexiv LeRobot recording enabled: repo_id=%s, root=%s, task=%r, fps=%d",
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
        """Add one pre-action observation and its executed 20D action."""

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
            self._logger.warning(
                "Flexiv recording episode is empty; nothing was saved."
            )
            return
        frame_count = self._frame_count
        emit_event("recording_save_start", "recorder", frames=frame_count)
        self._dataset.save_episode()
        self._frame_count = 0
        emit_event("recording_saved", "recorder", frames=frame_count)
        self._logger.info("Saved Flexiv LeRobot episode with %d frames.", frame_count)

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
        emit_event("recording_discarded", "recorder", frames=frame_count, reason=reason)
        self._logger.info(
            "Discarded Flexiv LeRobot episode with %d frames (%s).",
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
