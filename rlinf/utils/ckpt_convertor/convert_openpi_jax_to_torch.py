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

"""Detect and prepare OpenPI checkpoints for PyTorch inference.

An evaluation config can point ``checkpoint_conversion.checkpoint_path`` to
either an RLinf-compatible PyTorch checkpoint or an OpenPI JAX Orbax
checkpoint. PyTorch checkpoints are used directly. JAX checkpoints are
converted once, while complete converted outputs are reused on later launches.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

CheckpointFormat = Literal["jax", "pytorch"]
Precision = Literal["float32", "bfloat16"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenPiCheckpointConfig:
    """Resolved OpenPI checkpoint preparation settings."""

    checkpoint_path: str
    converted_checkpoint_dir: str | None
    config_name: str
    precision: Precision


def _as_local_path(checkpoint_path: str | Path) -> Path:
    """Resolve a local checkpoint path used for automatic format detection."""

    raw_path = str(checkpoint_path)
    if "://" in raw_path:
        raise ValueError(
            "Automatic checkpoint detection currently requires a local path, "
            f"got {raw_path}."
        )
    return Path(raw_path).expanduser().resolve()


def _is_nonempty_file(path: Path) -> bool:
    """Return whether a regular file exists and contains data."""

    return path.is_file() and path.stat().st_size > 0


def is_pytorch_checkpoint(checkpoint_path: str | Path) -> bool:
    """Return whether a path contains an RLinf-loadable PyTorch checkpoint."""

    path = _as_local_path(checkpoint_path)
    if path.is_file():
        return path.suffix in {".bin", ".pt", ".pth", ".safetensors"} and (
            path.stat().st_size > 0
        )
    if not path.is_dir():
        return False

    known_files = (
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "model_state_dict" / "full_weights.pt",
        path / "actor" / "model_state_dict" / "full_weights.pt",
    )
    if any(_is_nonempty_file(candidate) for candidate in known_files):
        return True
    return any(_is_nonempty_file(candidate) for candidate in path.glob("*.safetensors"))


def is_jax_checkpoint(checkpoint_path: str | Path) -> bool:
    """Return whether a directory has the expected OpenPI Orbax structure."""

    path = _as_local_path(checkpoint_path)
    if not path.is_dir():
        return False
    params_dir = path / "params"
    params_markers = (
        params_dir / "_METADATA",
        params_dir / "_sharding",
        params_dir / "manifest.ocdbt",
    )
    return params_dir.is_dir() and (
        (path / "_CHECKPOINT_METADATA").is_file()
        or any(marker.exists() for marker in params_markers)
    )


def detect_checkpoint_format(checkpoint_path: str | Path) -> CheckpointFormat:
    """Detect whether a local OpenPI checkpoint is PyTorch or JAX Orbax."""

    path = _as_local_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if is_pytorch_checkpoint(path):
        return "pytorch"
    if is_jax_checkpoint(path):
        return "jax"
    raise ValueError(
        "Could not detect checkpoint format. Expected PyTorch weights "
        "(*.safetensors, *.pt, or *.bin) or an OpenPI Orbax params directory; "
        f"got {path}."
    )


def is_complete_converted_checkpoint(checkpoint_dir: str | Path) -> bool:
    """Check that an automatically converted OpenPI checkpoint is complete."""

    path = _as_local_path(checkpoint_dir)
    weights_path = path / "model.safetensors"
    config_path = path / "config.json"
    if not _is_nonempty_file(weights_path) or not _is_nonempty_file(config_path):
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required_keys = {
        "action_dim",
        "action_horizon",
        "paligemma_variant",
        "action_expert_variant",
        "precision",
    }
    return isinstance(config, dict) and required_keys.issubset(config)


def _resolve_converted_checkpoint_dir(config: OpenPiCheckpointConfig) -> Path:
    """Resolve the configured output or derive ``<JAX path>_torch``."""

    if config.converted_checkpoint_dir:
        return _as_local_path(config.converted_checkpoint_dir)
    source = _as_local_path(config.checkpoint_path)
    return source.with_name(f"{source.name}_torch")


@contextmanager
def _conversion_lock(output_dir: Path):
    """Serialize conversion attempts targeting the same output directory."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_id = hashlib.sha256(str(output_dir).encode("utf-8")).hexdigest()[:24]
    lock_path = Path(tempfile.gettempdir()) / f"rlinf-openpi-{lock_id}.conversion.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _remove_incomplete_output(output_dir: Path) -> None:
    """Remove an incomplete conversion target before retrying conversion."""

    if output_dir.is_symlink() or output_dir.is_file():
        output_dir.unlink()
    elif output_dir.is_dir():
        shutil.rmtree(output_dir)


def _convert_jax_checkpoint(config: OpenPiCheckpointConfig, output_dir: Path) -> None:
    """Run the existing OpenPI JAX-to-PyTorch parameter conversion."""

    from openpi.models import pi0_config

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python import (
        convert_pi0_checkpoint,
    )

    # Resolve the model config from RLinf's own registry; the OpenPI package
    # does not know RLinf-specific configs such as pi05_arx_x5_dual.
    model_config = get_openpi_config(
        config.config_name, model_path=config.checkpoint_path
    ).model
    if not isinstance(model_config, pi0_config.Pi0Config):
        raise ValueError(f"OpenPI config {config.config_name} is not a Pi0Config.")
    convert_pi0_checkpoint(
        checkpoint_dir=config.checkpoint_path,
        precision=config.precision,
        output_path=str(output_dir),
        model_config=model_config,
    )


def ensure_torch_checkpoint(config: OpenPiCheckpointConfig) -> str:
    """Return a PyTorch checkpoint, converting a JAX source when necessary."""

    source_path = _as_local_path(config.checkpoint_path)
    checkpoint_format = detect_checkpoint_format(source_path)
    if checkpoint_format == "pytorch":
        logger.info("Using PyTorch checkpoint directly: %s", source_path)
        # OpenPI's model loader receives a directory, even when a single
        # safetensors/pt file was used for format detection.
        return str(source_path.parent if source_path.is_file() else source_path)

    output_dir = _resolve_converted_checkpoint_dir(config)
    if output_dir == source_path:
        raise ValueError("JAX source and PyTorch output directories must differ.")

    with _conversion_lock(output_dir):
        if is_complete_converted_checkpoint(output_dir):
            logger.info("Using existing converted PyTorch checkpoint: %s", output_dir)
            return str(output_dir)

        if output_dir.exists() or output_dir.is_symlink():
            logger.warning("Removing incomplete conversion output: %s", output_dir)
            _remove_incomplete_output(output_dir)

        logger.info(
            "Converting OpenPI JAX checkpoint %s -> %s", source_path, output_dir
        )
        _convert_jax_checkpoint(config, output_dir)
        if not is_complete_converted_checkpoint(output_dir):
            raise RuntimeError(
                "OpenPI checkpoint conversion finished without a complete "
                f"PyTorch output at {output_dir}."
            )
        logger.info("OpenPI checkpoint conversion completed: %s", output_dir)
    return str(output_dir)


def _conversion_config_from_cfg(cfg: DictConfig) -> OpenPiCheckpointConfig:
    """Resolve ``checkpoint_conversion`` from a composed RLinf config."""

    section = OmegaConf.select(cfg, "checkpoint_conversion")
    if section is None:
        raise KeyError("Config does not define checkpoint_conversion.")
    resolved = OmegaConf.to_container(section, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("checkpoint_conversion must be a mapping.")

    required = ("checkpoint_path", "config_name", "precision")
    missing = [key for key in required if not resolved.get(key)]
    if missing:
        raise ValueError(
            "checkpoint_conversion has empty required settings: " + ", ".join(missing)
        )

    precision = str(resolved["precision"])
    if precision not in ("float32", "bfloat16"):
        raise ValueError(
            "checkpoint_conversion.precision must be float32 or bfloat16, "
            f"got {precision}."
        )

    output_dir = resolved.get("converted_checkpoint_dir")
    return OpenPiCheckpointConfig(
        checkpoint_path=str(resolved["checkpoint_path"]),
        converted_checkpoint_dir=str(output_dir) if output_dir else None,
        config_name=str(resolved["config_name"]),
        precision=cast(Precision, precision),
    )


def prepare_openpi_checkpoint(cfg: DictConfig) -> DictConfig:
    """Prepare an optional configured checkpoint before workers are launched."""

    section = OmegaConf.select(cfg, "checkpoint_conversion")
    if section is None or not bool(section.get("enabled", True)):
        return cfg

    checkpoint_path = ensure_torch_checkpoint(_conversion_config_from_cfg(cfg))
    with open_dict(cfg):
        cfg.checkpoint_conversion.resolved_checkpoint_dir = checkpoint_path
        cfg.rollout.model.model_path = checkpoint_path
    return cfg


def load_conversion_config(config_path: str | Path) -> OpenPiCheckpointConfig:
    """Compose an RLinf evaluation YAML and resolve conversion settings."""

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {config_path}")

    repo_root = Path(__file__).resolve().parents[3]
    os.environ.setdefault("EMBODIED_PATH", str(repo_root / "examples" / "embodiment"))
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.1", config_dir=str(config_path.parent)):
        cfg = compose(config_name=config_path.stem)
    return _conversion_config_from_cfg(cfg)


def main() -> None:
    """Prepare the checkpoint configured by an RLinf evaluation YAML."""

    parser = argparse.ArgumentParser(
        description="Detect and prepare an OpenPI checkpoint for PyTorch inference."
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to an evaluation YAML containing checkpoint_conversion.",
    )
    args = parser.parse_args()
    checkpoint_path = ensure_torch_checkpoint(load_conversion_config(args.config_path))
    print(f"PyTorch checkpoint ready: {checkpoint_path}")


if __name__ == "__main__":
    main()
