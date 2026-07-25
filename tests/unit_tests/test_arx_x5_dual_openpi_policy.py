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

"""ARX 三相机 observation 与 OpenPI π0.5 转换器测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

np = pytest.importorskip("numpy")
pytest.importorskip("openpi")

from rlinf.models.embodiment.openpi.policies.arx_x5_dual_policy import (  # noqa: E402
    ArxX5DualInputs,
    ArxX5DualOutputs,
)


def test_openpi_input_maps_head_and_two_wrist_cameras():
    """确认推理时堆叠的两个额外视角被拆成左右腕相机。"""

    head = np.full((8, 10, 3), 1, dtype=np.uint8)
    left_wrist = np.full((8, 10, 3), 2, dtype=np.uint8)
    right_wrist = np.full((8, 10, 3), 3, dtype=np.uint8)
    state = np.arange(14, dtype=np.float32)

    transform = ArxX5DualInputs(action_dim=32)
    result = transform(
        {
            "observation/image": head,
            "observation/extra_view_image": np.stack([left_wrist, right_wrist], axis=0),
            "observation/state": state,
            "prompt": "把物体放入盒中",
        }
    )

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], head)
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], left_wrist)
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], right_wrist)
    np.testing.assert_array_equal(result["state"][:14], state)
    np.testing.assert_array_equal(result["state"][14:], np.zeros(18))
    assert result["prompt"] == "把物体放入盒中"


def test_openpi_output_keeps_first_14_absolute_action_dimensions():
    """确认 π0.5 补齐维度会被移除，14 维绝对动作顺序保持不变。"""

    model_actions = np.arange(25 * 32, dtype=np.float32).reshape(25, 32)
    result = ArxX5DualOutputs()({"actions": model_actions})

    assert result["actions"].shape == (25, 14)
    np.testing.assert_array_equal(result["actions"], model_actions[:, :14])


def test_realworld_eval_config_is_rollout_only(monkeypatch):
    """确认真机评估配置不会创建训练 actor 或 dummy 机械臂。"""

    repo_path = Path(__file__).resolve().parents[2]
    config_dir = repo_path / "evaluations" / "realworld"
    environment = {
        "EMBODIED_PATH": str(repo_path / "examples" / "embodiment"),
        "ROBOT_RLINF_PYTHON": "/opt/rlinf/.venv/bin/python",
        "ARX_PI05_CHECKPOINT": "/models/arx-pi05",
        "ARX_NORM_STATS_PATH": "/models/arx-pi05/assets",
        "ARX_SFT_REPO_ID": "arx/sft",
        "ARX_HEAD_CAMERA_SERIAL": "head",
        "ARX_LEFT_CAMERA_SERIAL": "left",
        "ARX_RIGHT_CAMERA_SERIAL": "right",
        "ARX_TASK_DESCRIPTION": "pick up the object",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="realworld_eval_arx_x5_dual_pi05")
    OmegaConf.resolve(cfg)

    assert cfg.runner.task_type == "embodied_eval"
    assert cfg.runner.only_eval is True
    assert "actor" not in cfg
    assert "critic" not in cfg
    assert "reward" not in cfg
    assert "train" not in cfg.env
    assert cfg.env.eval.override_cfg.is_dummy is False
    assert cfg.rollout.model.model_type == "openpi"
    assert cfg.rollout.model.openpi.num_images_in_input == 3
    assert cfg.rollout.model.action_dim == 14
    assert cfg.cluster.component_placement.rollout.node_group == "inference"
    assert cfg.cluster.component_placement.env.node_group == "robot"


@pytest.mark.parametrize(
    ("config_name", "expected_mode"),
    [
        ("realworld_eval_arx_x5_dual_pi05_dryrun_hardware", "hardware"),
        ("realworld_eval_arx_x5_dual_pi05_dryrun_distributed", "distributed"),
    ],
)
def test_two_machine_dry_run_configs(monkeypatch, config_name, expected_mode):
    """确认两种 dry-run 都保留两机推理拓扑并使用互斥硬件模式。"""

    repo_path = Path(__file__).resolve().parents[2]
    config_dir = repo_path / "evaluations" / "realworld"
    common_environment = {
        "EMBODIED_PATH": str(repo_path / "examples" / "embodiment"),
        "ROBOT_RLINF_PYTHON": "/opt/rlinf/.venv/bin/python",
        "ARX_PI05_CHECKPOINT": "/models/arx-pi05",
        "ARX_NORM_STATS_PATH": "/models/arx-pi05/assets",
        "ARX_SFT_REPO_ID": "arx/sft",
    }
    for key, value in common_environment.items():
        monkeypatch.setenv(key, value)

    if expected_mode == "hardware":
        monkeypatch.setenv("ARX_HEAD_CAMERA_SERIAL", "head")
        monkeypatch.setenv("ARX_LEFT_CAMERA_SERIAL", "left")
        monkeypatch.setenv("ARX_RIGHT_CAMERA_SERIAL", "right")
        monkeypatch.setenv("ARX_TASK_DESCRIPTION", "pick up the object")
    else:
        for key in (
            "ARX_HEAD_CAMERA_SERIAL",
            "ARX_LEFT_CAMERA_SERIAL",
            "ARX_RIGHT_CAMERA_SERIAL",
            "ARX_TASK_DESCRIPTION",
        ):
            monkeypatch.delenv(key, raising=False)

    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)

    assert cfg.cluster.num_nodes == 2
    assert cfg.runner.only_eval is True
    assert cfg.env.eval.max_steps_per_rollout_epoch == 5
    assert cfg.env.eval.override_cfg.dry_run is (expected_mode == "hardware")
    assert cfg.env.eval.override_cfg.is_dummy is (expected_mode == "distributed")
    if expected_mode == "distributed":
        assert cfg.env.eval.override_cfg.head_camera_serial is None
        assert cfg.env.eval.override_cfg.left_wrist_camera_serial is None
        assert cfg.env.eval.override_cfg.right_wrist_camera_serial is None
