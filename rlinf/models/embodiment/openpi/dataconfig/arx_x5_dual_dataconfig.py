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

"""ARX X5 双臂绝对关节位置数据配置。"""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import arx_x5_dual_policy


@dataclasses.dataclass(frozen=True)
class ArxX5DualDataConfig(DataConfigFactory):
    """把 ARX LeRobot 数据和在线 observation 接到同一套 π0.5 转换链。

    当前配置明确采用绝对关节动作：数据集中的 14 维 ``actions`` 会直接归一化
    并送入 π0.5，不增加 ``DeltaActions``；推理输出反归一化后也直接返回绝对
    关节位置。这样 SFT 数据、π0.5 checkpoint 和 Env.step 的动作含义完全一致。

    Attributes:
        default_prompt: 当数据集中没有任务文本时注入的默认指令。
    """

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
            model_config: 当前 π0.5 模型结构配置，用于获得内部动作维度。

        Returns:
            完整 ``DataConfig``。训练和推理都会复用其中相同的状态顺序、
            三相机映射和 14 维绝对动作输出规则。
        """

        # CollectEpisode 导出的 LeRobot 帧使用 image、extra_view_image-0/1、
        # state、actions 和 task。这里把它们改名为 policy transform 使用的键。
        repack_transforms = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/extra_view_image-0": "extra_view_image-0",
                        "observation/extra_view_image-1": "extra_view_image-1",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                arx_x5_dual_policy.ArxX5DualInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[arx_x5_dual_policy.ArxX5DualOutputs()],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
        )
