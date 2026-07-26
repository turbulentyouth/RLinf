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

"""Tests for config-driven OpenPI checkpoint preparation."""

import json
from pathlib import Path

from omegaconf import OmegaConf

import rlinf.utils.ckpt_convertor.convert_openpi_jax_to_torch as converter


def _config(checkpoint_path: Path, output_path: Path | None = None):
    return OmegaConf.create(
        {
            "checkpoint_conversion": {
                "enabled": True,
                "checkpoint_path": str(checkpoint_path),
                "converted_checkpoint_dir": (
                    str(output_path) if output_path is not None else None
                ),
                "resolved_checkpoint_dir": None,
                "config_name": "pi05_arx_x5_dual",
                "precision": "bfloat16",
            },
            "rollout": {"model": {"model_path": str(checkpoint_path)}},
        }
    )


def _write_complete_output(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "model.safetensors").write_bytes(b"weights")
    (output_path / "config.json").write_text(
        json.dumps(
            {
                "action_dim": 14,
                "action_horizon": 1,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "precision": "bfloat16",
            }
        ),
        encoding="utf-8",
    )


def test_prepare_uses_pytorch_checkpoint_directly(tmp_path):
    """PyTorch input is returned without importing or running conversion."""

    checkpoint_path = tmp_path / "torch_checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "model.safetensors").write_bytes(b"weights")
    cfg = _config(checkpoint_path)

    result = converter.prepare_openpi_checkpoint(cfg)

    assert result.rollout.model.model_path == str(checkpoint_path)
    assert result.checkpoint_conversion.resolved_checkpoint_dir == str(checkpoint_path)


def test_prepare_converts_jax_once_and_reuses_complete_output(tmp_path, monkeypatch):
    """JAX input is converted once; a complete output is reused afterward."""

    checkpoint_path = tmp_path / "30000"
    (checkpoint_path / "params").mkdir(parents=True)
    (checkpoint_path / "_CHECKPOINT_METADATA").write_text("jax", encoding="utf-8")
    output_path = tmp_path / "30000_torch"
    output_path.mkdir()
    (output_path / "partial").write_text("incomplete", encoding="utf-8")
    cfg = _config(checkpoint_path, output_path)
    calls: list[tuple[str, str]] = []

    def fake_convert(config, output_dir):
        calls.append((config.checkpoint_path, str(output_dir)))
        _write_complete_output(output_dir)

    monkeypatch.setattr(converter, "_convert_jax_checkpoint", fake_convert)

    result = converter.prepare_openpi_checkpoint(cfg)
    second_result = converter.prepare_openpi_checkpoint(
        _config(checkpoint_path, output_path)
    )

    assert calls == [(str(checkpoint_path), str(output_path))]
    assert result.rollout.model.model_path == str(output_path)
    assert second_result.rollout.model.model_path == str(output_path)


def test_load_arx_openpi_conversion_config(monkeypatch):
    """Resolve the automatic preparation settings from the shared ARX config."""

    repo_path = Path(__file__).resolve().parents[2]
    monkeypatch.delenv("ARX_PI05_TORCH_CHECKPOINT", raising=False)

    conversion = converter.load_conversion_config(
        repo_path / "evaluations" / "realworld" / "realworld_eval_arx_x5_dual_pi05.yaml"
    )

    assert conversion.checkpoint_path == "/home/li/hubo/RLinf/30000_torch"
    assert conversion.converted_checkpoint_dir is None
    assert conversion.config_name == "pi05_arx_x5_dual"
    assert conversion.precision == "bfloat16"
