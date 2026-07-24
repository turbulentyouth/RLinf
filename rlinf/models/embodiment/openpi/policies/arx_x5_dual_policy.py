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

"""ARX X5 双臂与 OpenPI π0.5 之间的输入输出转换。"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

_ARX_ACTION_DIM = 14


def _parse_rgb_image(image: np.ndarray) -> np.ndarray:
    """把 RLinf/LeRobot 图像统一转换成 OpenPI 使用的 HWC uint8。

    Args:
        image: ``HWC`` 或 ``CHW`` 图像。浮点图像约定数值位于 ``[0, 1]``。

    Returns:
        形状为 ``(H, W, 3)``、类型为 ``uint8`` 的 RGB 图像。
    """

    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"ARX RGB 图像必须是 3 维，实际形状为 {image.shape}。")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"ARX RGB 图像最后一维必须是 3，实际为 {image.shape}。")
    return np.ascontiguousarray(image, dtype=np.uint8)


def _extract_three_views(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从训练或推理数据中提取头部、左腕和右腕图像。

    推理时，``RealWorldEnv`` 会提供一个主图 ``observation/image`` 和按顺序
    堆叠的 ``observation/extra_view_image``；训练时，LeRobot 数据经过 repack
    后会提供两个分开的 ``extra_view_image-0/1`` 字段。

    Args:
        data: 单个样本的 OpenPI data 字典。

    Returns:
        ``(head, left_wrist, right_wrist)`` 三张 HWC uint8 图像。
    """

    head = _parse_rgb_image(data["observation/image"])
    stacked = data.get("observation/extra_view_image")
    if stacked is not None:
        stacked = np.asarray(stacked)
        if stacked.ndim != 4 or stacked.shape[0] != 2:
            raise ValueError(
                "ARX 推理 observation/extra_view_image 必须按"
                f"[左腕, 右腕]堆叠成 (2,H,W,C)，实际为 {stacked.shape}。"
            )
        left_wrist = _parse_rgb_image(stacked[0])
        right_wrist = _parse_rgb_image(stacked[1])
    else:
        left_wrist = _parse_rgb_image(data["observation/extra_view_image-0"])
        right_wrist = _parse_rgb_image(data["observation/extra_view_image-1"])
    return head, left_wrist, right_wrist


@dataclasses.dataclass(frozen=True)
class ArxX5DualInputs(transforms.DataTransformFn):
    """把 ARX observation 转换成 π0.5 模型输入。

    输入状态必须严格遵循：
    ``[左 q1..q6, 左夹爪, 右 q1..q6, 右夹爪]``。

    两个夹爪维度使用 ARX SDK 的物理夹爪宽度，不能在训练数据中归一化成
    ``[0, 1]`` 后又把模型输出直接当作米制宽度下发。数值归一化只应由
    OpenPI 的 ``norm_stats`` 负责，并在输出阶段自动反归一化。

    Args:
        action_dim: π0.5 内部动作维度。OpenPI 通常会使用比真实机器人更大的
           统一维度，因此状态和训练动作会在尾部补零到这个长度。
        model_type: OpenPI 模型类型。当前注册配置传入 ``PI05``。

    Effects:
        三路相机分别映射到 π0.5 的 ``base_0_rgb``、
        ``left_wrist_0_rgb`` 和 ``right_wrist_0_rgb``；14 维状态保持原顺序，
        不做 ALOHA 关节翻转，也不把绝对关节动作转换成 delta。
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        """转换一个 ARX 样本。

        Args:
            data: 包含三路图像、14 维状态、可选动作和语言 prompt 的字典。

        Returns:
            OpenPI 标准输入，包括 ``state``、``image``、``image_mask``、
            可选 ``actions`` 和 ``prompt``。

        Raises:
            ValueError: 状态或训练动作不是 14 维。
        """

        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (_ARX_ACTION_DIM,):
            raise ValueError(
                f"ARX state 必须是 ({_ARX_ACTION_DIM},)，实际为 {state.shape}。"
            )
        head, left_wrist, right_wrist = _extract_three_views(data)

        inputs = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != _ARX_ACTION_DIM:
                raise ValueError(
                    "ARX 训练 actions 必须是 (action_horizon, 14)，"
                    f"实际为 {actions.shape}。"
                )
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class ArxX5DualOutputs(transforms.DataTransformFn):
    """把 π0.5 输出恢复成 ARX Env 接受的 14 维绝对动作。

    π0.5 内部可能输出补零后的统一动作维度。本转换器只保留前 14 维，并且不
    做关节翻转、delta 累加或夹爪归一化，因此返回值可以直接传给
    ``ArxX5DualEnv.step(action)``。
    """

    def __call__(self, data: dict) -> dict:
        """截取模型动作的 ARX 有效维度。

        Args:
            data: 包含 ``actions`` 的 OpenPI 输出字典，形状通常为
                ``(action_horizon, model_action_dim)``。

        Returns:
            ``{"actions": absolute_actions}``，其中动作形状为
            ``(action_horizon, 14)``。
        """

        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < _ARX_ACTION_DIM:
            raise ValueError(
                "π0.5 输出 actions 必须是二维动作块，且最后一维至少为 "
                f"{_ARX_ACTION_DIM}；实际为 {actions.shape}。"
            )
        return {"actions": actions[:, :_ARX_ACTION_DIM]}
