#!/usr/bin/env python3
"""Convert an RLinf OpenPI checkpoint to an xense-openpi checkpoint.

The CFG-model SFT checkpoint used by ``pi05_bi_flexiv`` already stores the
OpenPI PyTorch parameter names (``paligemma_with_expert.*``).  xense-openpi
expects the same tensors in ``model.safetensors`` and the normalization asset
under ``assets/<asset-id>/norm_stats.json``.

This script deliberately does not convert OpenPI_RLinf/bare-Pi0 checkpoints.
Those checkpoints need a matching OpenPI PyTorch reference model to restore
the action-expert head; use the repository's ``sft2deploy`` converter for that
case.

Usage::

    .venv/bin/python rlinf/utils/ckpt_convertor/convert_rlinf_full_weights_to_xense_openpi.py \
        --full-weights /path/to/full_weights.pt \
        --norm-stats /path/to/norm_stats.json \
        --output /path/to/xense_checkpoint

The resulting directory can be passed as the local checkpoint directory to
the xense-openpi policy server together with its matching config, for example
``pi05_bi_flexiv``.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
from collections.abc import Mapping


_OPENPI_PREFIX = "paligemma_with_expert."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-weights",
        required=True,
        type=pathlib.Path,
        help="RLinf consolidated full_weights.pt file.",
    )
    parser.add_argument(
        "--norm-stats",
        required=True,
        type=pathlib.Path,
        help="Input norm_stats.json file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="Output xense-openpi checkpoint directory.",
    )
    parser.add_argument(
        "--asset-id",
        default="bi_flexiv",
        help="xense-openpi asset id; defaults to bi_flexiv.",
    )
    return parser.parse_args()


def _load_state_dict(path: pathlib.Path) -> Mapping[str, object]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"full_weights.pt not found: {path}")

    # mmap keeps the 8+ GB checkpoint from being copied during deserialization.
    state_dict = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Expected a state-dict mapping in {path}, got {type(state_dict)!r}"
        )
    return state_dict


def _normalize_keys(state_dict: Mapping[str, object]) -> dict[str, object]:
    """Remove harmless wrapper prefixes while preserving OpenPI parameter names."""
    prefixes = (
        "_fsdp_wrapped_module.",
        "_orig_mod.",
        "module.",
        "model.",
    )
    normalized: dict[str, object] = {}
    for key, value in state_dict.items():
        normalized_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    changed = True
                    break
        if normalized_key in normalized:
            raise ValueError(f"Duplicate parameter key after normalization: {normalized_key}")
        normalized[normalized_key] = value
    return normalized


def convert(
    full_weights: pathlib.Path,
    norm_stats: pathlib.Path,
    output: pathlib.Path,
    asset_id: str = "bi_flexiv",
) -> pathlib.Path:
    """Write an xense-openpi checkpoint and return its model.safetensors path."""
    import safetensors.torch
    import torch

    if not norm_stats.is_file():
        raise FileNotFoundError(f"norm_stats.json not found: {norm_stats}")

    raw_state_dict = _load_state_dict(full_weights)
    state_dict = _normalize_keys(raw_state_dict)
    openpi_keys = [key for key in state_dict if key.startswith(_OPENPI_PREFIX)]
    if not openpi_keys:
        raise ValueError(
            "The checkpoint does not contain paligemma_with_expert.* keys. "
            "It is likely an OpenPI_RLinf/bare-Pi0 checkpoint; use the existing "
            "sft2deploy converter with a matching OpenPI PyTorch reference model."
        )

    tensors: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint entry {key!r} is not a tensor")
        # Safetensors requires dense CPU tensors and does not preserve shared
        # storage aliases. Cloning only when needed keeps the conversion light.
        tensor = value.detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensors[key] = tensor

    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.safetensors"
    safetensors.torch.save_file(tensors, str(model_path))

    stats_path = output / "assets" / asset_id / "norm_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(norm_stats, stats_path)

    print(f"Wrote model: {model_path}")
    print(f"Wrote norm stats: {stats_path}")
    print(f"Tensor count: {len(tensors)}")
    print(f"Model size: {model_path.stat().st_size / 1024**3:.3f} GiB")
    return model_path


if __name__ == "__main__":
    args = _parse_args()
    convert(args.full_weights, args.norm_stats, args.output, args.asset_id)
