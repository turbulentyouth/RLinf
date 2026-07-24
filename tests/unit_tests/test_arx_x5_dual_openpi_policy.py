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

import pytest

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
            "observation/extra_view_image": np.stack(
                [left_wrist, right_wrist], axis=0
            ),
            "observation/state": state,
            "prompt": "把物体放入盒中",
        }
    )

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], head)
    np.testing.assert_array_equal(
        result["image"]["left_wrist_0_rgb"], left_wrist
    )
    np.testing.assert_array_equal(
        result["image"]["right_wrist_0_rgb"], right_wrist
    )
    np.testing.assert_array_equal(result["state"][:14], state)
    np.testing.assert_array_equal(result["state"][14:], np.zeros(18))
    assert result["prompt"] == "把物体放入盒中"


def test_openpi_output_keeps_first_14_absolute_action_dimensions():
    """确认 π0.5 补齐维度会被移除，14 维绝对动作顺序保持不变。"""

    model_actions = np.arange(25 * 32, dtype=np.float32).reshape(25, 32)
    result = ArxX5DualOutputs()({"actions": model_actions})

    assert result["actions"].shape == (25, 14)
    np.testing.assert_array_equal(result["actions"], model_actions[:, :14])
