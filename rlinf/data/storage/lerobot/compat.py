# Copyright 2025 The RLinf Authors.
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

"""Shims over the lerobot releases RLinf supports.

Three things move around across the versions RLinf runs against:

===================  ==============  ==================================
lerobot              dataset format  ``add_frame``
===================  ==============  ==================================
< 0.2                v2.1            ``add_frame(frame)``
0.3.x                v2.1            ``add_frame(frame, task, ...)``
>= 0.4               v3.0            ``add_frame(frame)``
===================  ==============  ==================================

Because 0.4 reverted the 0.3.x signature, dispatch on the signature itself —
a version comparison gets 0.4+ backwards. The v2.1 -> v3.0 format change is
what removed ``LeRobotDataset.episode_data_index``.
"""

import functools
import inspect
import json
import re
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "add_frame_to_dataset",
    "episode_boundaries",
    "is_lerobot_v3_dataset",
    "load_local_lerobot_dataset",
    "load_local_lerobot_metadata",
]


@functools.lru_cache(maxsize=None)
def _add_frame_takes_task(add_frame: Callable[..., Any]) -> bool:
    """Whether ``LeRobotDataset.add_frame`` takes the task as its own argument.

    Only lerobot 0.3.x does: it exposes ``add_frame(frame, task,
    timestamp=None)`` and rejects a ``"task"`` key inside *frame* as an unknown
    feature. Both older (< 0.2) and newer (>= 0.4) releases expose
    ``add_frame(frame)`` and read the task from ``frame["task"]``.

    Args:
        add_frame: The unbound ``add_frame`` function of a dataset class.

    Returns:
        ``True`` for the lerobot 0.3.x signature, ``False`` otherwise.
    """
    try:
        sig = inspect.signature(add_frame)
    except (TypeError, ValueError):  # C-implemented or otherwise opaque
        return False
    return "task" in sig.parameters


def add_frame_to_dataset(dataset: Any, frame_data: dict[str, Any]) -> None:
    """Add one frame to a ``LeRobotDataset``, whatever lerobot version is installed.

    *frame_data* is never mutated: callers reuse these dicts after the write —
    the DAgger worker hands the same frames to the in-memory training store and
    to a deferred archive buffer — and lerobot 0.4+ pops ``"task"`` out of the
    frame it is given.

    Args:
        dataset: The ``LeRobotDataset`` to append to.
        frame_data: The frame fields, including a ``"task"`` entry.

    Raises:
        ValueError: If *frame_data* carries no ``"task"`` entry, which every
            lerobot version requires.
    """
    task = frame_data.get("task")
    if task is None:
        raise ValueError(
            f"Frame is missing the required 'task' field; got keys {sorted(frame_data)}."
        )
    frame = dict(frame_data)
    if _add_frame_takes_task(type(dataset).add_frame):
        # lerobot 0.3.x validates the frame against the feature schema, which
        # has no ``task`` entry, so the task must be passed separately.
        del frame["task"]
        dataset.add_frame(frame, task=task)
    else:
        dataset.add_frame(frame)


def episode_boundaries(dataset: Any) -> tuple[list[int], list[int]]:
    """Return each episode's ``[from, to)`` frame range in the flat frame index.

    Dataset format v2.1 exposes ``LeRobotDataset.episode_data_index``, a dict of
    two tensors. v3.0 (lerobot >= 0.4) dropped it in favour of
    ``dataset_from_index`` / ``dataset_to_index`` columns on
    ``dataset.meta.episodes``.

    Both are ordered by the dataset's own episode list, so entry ``i`` refers to
    the same episode either way — including when the dataset was opened with an
    ``episodes=[...]`` subset.

    Args:
        dataset: An open ``LeRobotDataset``.

    Returns:
        ``(starts, ends)``, one entry per episode.

    Raises:
        RuntimeError: If neither layout is present, which means the installed
            lerobot is neither v2.1 nor v3.0 shaped.
    """
    index = getattr(dataset, "episode_data_index", None)
    if index is not None:  # v2.1
        return [int(x) for x in index["from"]], [int(x) for x in index["to"]]

    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if episodes is not None:  # v3.0
        try:
            return (
                [int(x) for x in episodes["dataset_from_index"]],
                [int(x) for x in episodes["dataset_to_index"]],
            )
        except (KeyError, TypeError):
            pass

    raise RuntimeError(
        "Cannot determine episode boundaries: the dataset exposes neither "
        "`episode_data_index` (lerobot < 0.4) nor `meta.episodes` with "
        "`dataset_from_index` / `dataset_to_index` columns (lerobot >= 0.4). "
        f"Got a {type(dataset).__name__} from lerobot "
        f"{_installed_lerobot_version()}."
    )


def is_lerobot_v3_dataset(root: str | Path) -> bool:
    """Return whether a local dataset uses the LeRobot v3 file layout.

    The codebase version is authoritative. The ``tasks.parquet`` check keeps
    the detector useful for early v3 exporters that omitted or misreported the
    version string.
    """
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            version = str(json.load(f).get("codebase_version", ""))
        if version.lower().startswith("v3"):
            return True
    return (root / "meta" / "tasks.parquet").exists()


def _lerobot_version_tuple() -> tuple[int, ...]:
    """Return the installed LeRobot release as a numeric tuple."""
    try:
        version = importlib_metadata.version("lerobot")
    except importlib_metadata.PackageNotFoundError:
        try:
            import lerobot

            version = getattr(lerobot, "__version__", "0")
        except ImportError:
            return (0,)
    match = re.match(r"^(\d+(?:\.\d+)*)", str(version))
    return tuple(int(part) for part in match.group(1).split(".")) if match else (0,)


def _require_v3_capable_lerobot(root: Path) -> None:
    """Fail before LeRobot performs a misleading Hugging Face Hub fallback."""
    if is_lerobot_v3_dataset(root) and _lerobot_version_tuple() < (0, 4):
        raise RuntimeError(
            f"LeRobot v3 dataset detected at {root}, but the installed lerobot "
            f"version is {_installed_lerobot_version()}. LeRobot 0.3.x only "
            "understands v2.1 metadata and looks for meta/tasks.jsonl. Install "
            "RLinf's pinned v3-compatible LeRobot with: "
            "uv pip install --force-reinstall --no-deps 'lerobot @ "
            "git+https://github.com/huggingface/lerobot.git@v0.4.4'."
        )


def load_local_lerobot_metadata(root: str | Path) -> Any:
    """Load metadata from a local LeRobot dataset without Hub fallback."""
    root = Path(root).expanduser().resolve()
    _require_v3_capable_lerobot(root)
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(root.name, root=root)


def load_local_lerobot_dataset(
    root: str | Path,
    *,
    delta_timestamps: dict[str, list[float]] | None = None,
    download_videos: bool = False,
    metadata: Any | None = None,
) -> tuple[Any, Any]:
    """Load a local LeRobot dataset and its metadata across v2.1/v3 releases."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root).expanduser().resolve()
    meta = metadata if metadata is not None else load_local_lerobot_metadata(root)
    dataset = LeRobotDataset(
        root.name,
        root=root,
        delta_timestamps=delta_timestamps,
        download_videos=download_videos,
    )
    return meta, dataset


def _installed_lerobot_version() -> str:
    try:
        import lerobot

        return getattr(lerobot, "__version__", "(unknown version)")
    except ImportError:
        return "(not installed)"
