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
"""Data config for bi-Flexiv dual-arm (Rizon4) SFT.

The dataset ``Xense/stack-cubes-flexiv-0813`` stores *absolute* TCP rot6d
actions (``action[t] ≈ state[t+1]``), so we convert absolute → SE(3)
body-frame delta for training (pi0/pi05 train on deltas) and recover absolute
actions at inference. Component-wise subtraction would break on the 6D
rotation, hence ``RigidBodyDeltaActions`` / ``RigidBodyAbsoluteActions``.
"""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import bi_flexiv_policy
from rlinf.models.embodiment.openpi.transforms import (
    BI_FLEXIV_ROT6D_LAYOUT,
    RigidBodyAbsoluteActions,
    RigidBodyDeltaActions,
)


@dataclasses.dataclass(frozen=True)
class BiFlexivDataConfig(DataConfigFactory):
    default_prompt: str | None = None

    # Dataset stores absolute TCP rot6d actions; pi0/pi05 train on deltas, so
    # convert abs → SE(3) body-frame delta at train, and back at inference.
    extra_delta_transform: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
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
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[bi_flexiv_policy.BiFlexivInputs()],
            outputs=[bi_flexiv_policy.BiFlexivOutputs()],
        )

        if self.extra_delta_transform:
            data_transforms = data_transforms.push(
                inputs=[RigidBodyDeltaActions(BI_FLEXIV_ROT6D_LAYOUT)],
                outputs=[RigidBodyAbsoluteActions(BI_FLEXIV_ROT6D_LAYOUT)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
