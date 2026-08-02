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

"""Flexiv Rizon4 双臂 TCP（6D 旋转）数据配置。"""

from __future__ import annotations

import dataclasses
import pathlib

import einops
import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi import transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

_BI_FLEXIV_ACTION_DIM = 20


def _parse_rgb_image(image: np.ndarray) -> np.ndarray:
    """把 RLinf/LeRobot 图像统一转换成 xense-openpi 期望的 CHW uint8。

    ``openpi.policies.bi_flexiv_policy.BiFlexivInputs`` 内部的
    ``_decode_bi_flexiv`` 会对每张图无条件执行 ``c h w -> h w c``，
    因此喂给它的图像必须是 ``CHW``。本函数接受 ``HWC`` 或 ``CHW``、
    ``uint8`` 或 ``[0,1]`` 浮点图像，统一输出 ``CHW uint8``。

    Args:
        image: ``HWC`` 或 ``CHW`` 图像。浮点图像约定数值位于 ``[0, 1]``。

    Returns:
        形状为 ``(3, H, W)``、类型为 ``uint8`` 的 RGB 图像。
    """

    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Flexiv RGB 图像必须是 3 维，实际形状为 {image.shape}。")
    if image.shape[-1] == 3:
        image = einops.rearrange(image, "h w c -> c h w")
    elif image.shape[0] != 3:
        raise ValueError(f"Flexiv RGB 图像没有 3 通道维，实际为 {image.shape}。")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image, dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class BiFlexivPolicyInputs(transforms.DataTransformFn):
    """把 RLinf/LeRobot 样本转换成 xense-openpi BiFlexivInputs 的输入格式。

    两种来源共用同一条转换链（输入都是 repack 之后的**扁平字典**）：

    - 训练数据（xense LeRobot 格式）经 ``RepackTransform`` 后提供
      ``observation.images.{head,left_wrist,right_wrist}``、``observation.state``、
      ``action`` 和 ``task``（repack 结构见 ``BiFlexivDataConfig.create``，
      与 xense-openpi 的 ``LeRobotBiFlexivDataConfig`` 一致）；
    - 真机推理时 RLinf ``RealWorldEnv`` 提供主图 ``observation/image``、
      按 ``[左腕, 右腕]`` 堆叠的 ``observation/extra_view_image``、
      ``observation/state`` 和 ``prompt``。

    输出键名与 ``openpi.policies.bi_flexiv_policy.BiFlexivInputs`` 的约定
    完全一致：``images`` 字典（CHW uint8，BiFlexivInputs 内部会再转成
    HWC）、``state``、可选 ``actions``、可选 ``prompt``。
    """

    def __call__(self, data: dict) -> dict:
        # RepackTransform 使用 '/' 作为路径分隔符；训练样本在到达本转换器
        # 之前已经被展平成扁平键，因此这里用扁平键读取。
        if "observation/state" in data:
            state = np.asarray(data["observation/state"], dtype=np.float32)
        else:
            state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (_BI_FLEXIV_ACTION_DIM,):
            raise ValueError(
                f"Flexiv state 必须是 ({_BI_FLEXIV_ACTION_DIM},)，实际为 {state.shape}。"
            )

        if "observation/image" in data:
            head = _parse_rgb_image(data["observation/image"])
            stacked = np.asarray(data["observation/extra_view_image"])
            if stacked.ndim != 4 or stacked.shape[0] != 2:
                raise ValueError(
                    "Flexiv 推理 observation/extra_view_image 必须按"
                    f"[左腕, 右腕]堆叠成 (2,H,W,C)，实际为 {stacked.shape}。"
                )
            left_wrist = _parse_rgb_image(stacked[0])
            right_wrist = _parse_rgb_image(stacked[1])
        elif "observation/images/head" in data:
            head = _parse_rgb_image(data["observation/images/head"])
            left_wrist = _parse_rgb_image(data["observation/images/left_wrist"])
            right_wrist = _parse_rgb_image(data["observation/images/right_wrist"])
        else:
            images = data["images"]
            head = _parse_rgb_image(images["head"])
            left_wrist = _parse_rgb_image(images["left_wrist"])
            right_wrist = _parse_rgb_image(images["right_wrist"])

        inputs = {
            "state": state,
            "images": {
                "head": head,
                "left_wrist": left_wrist,
                "right_wrist": right_wrist,
            },
        }

        if "action" in data:
            actions = np.asarray(data["action"], dtype=np.float32)
        elif "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
        else:
            actions = None
        if actions is not None:
            if actions.ndim != 2 or actions.shape[-1] != _BI_FLEXIV_ACTION_DIM:
                raise ValueError(
                    "Flexiv 训练 actions 必须是 (action_horizon, 20)，"
                    f"实际为 {actions.shape}。"
                )
            inputs["actions"] = actions

        prompt = data.get("prompt", data.get("task"))
        if prompt is not None:
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class BiFlexivDataConfig(DataConfigFactory):
    """把 Flexiv LeRobot 数据和在线 observation 接到同一套 π0.5 转换链。

    与 xense-openpi 的 ``LeRobotBiFlexivDataConfig`` 对齐：policy 输入输出
    转换复用 ``openpi.policies.bi_flexiv_policy`` 中的
    ``BiFlexivInputs``/``BiFlexivOutputs``；``use_delta_cartesian_actions``
    为 True 时训练端用 ``DeltaActions`` 把绝对 TCP 动作转为增量（夹爪两维
    保持绝对），推理端用 ``AbsoluteActions`` 把模型输出的增量还原为绝对
    TCP 位姿，再传给 ``BiFlexivDualEnv.step(action)``。

    Attributes:
        use_delta_cartesian_actions: 是否对前 18 维 TCP 使用增量动作。
            xense 官方 bi_flexiv 训练配置均为 True。
        default_prompt: 当数据集中没有任务文本时注入的默认指令。
    """

    use_delta_cartesian_actions: bool = True
    default_prompt: str | None = None

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> DataConfig:
        """创建 OpenPI 使用的 repack、数据转换和模型转换。

        Args:
            assets_dirs: OpenPI 归一化统计和其他资产的根目录。
            model_config: 当前 π0.5 模型结构配置。

        Returns:
            完整 ``DataConfig``。训练和推理都会复用其中相同的状态顺序、
            三相机映射和 20 维 TCP 动作输出规则。
        """

        # 延迟导入避免在只使用其他 dataconfig 时加载多余的 policy 模块。
        from openpi.policies import bi_flexiv_policy

        # xense LeRobot 数据集列名 -> BiFlexivPolicyInputs 期望的键，与
        # xense-openpi 的 LeRobotBiFlexivDataConfig.repack_transforms 一致。
        # RepackTransform 的映射方向是 {新键: 数据集旧键}；真机推理路径不经过
        # repack（openpi_action_model 直接提供 observation/image 等键）。
        repack_transforms = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "head": "observation.images.head",
                            "left_wrist": "observation.images.left_wrist",
                            "right_wrist": "observation.images.right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[BiFlexivPolicyInputs(), bi_flexiv_policy.BiFlexivInputs()],
            outputs=[bi_flexiv_policy.BiFlexivOutputs()],
        )
        if self.use_delta_cartesian_actions:
            # 双臂笛卡尔：18 维 TCP（左 0-8 + 右 9-17，全部增量）+ 2 维夹爪
            # （绝对）。数据集顺序：
            # [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)]
            delta_action_mask = _transforms.make_bool_mask(18, -1, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
