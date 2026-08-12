# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""OpenPI data configuration for LeRobot Bi-Flexiv records."""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import bi_flexiv_policy


@dataclasses.dataclass(frozen=True)
class BiFlexivDataConfig(DataConfigFactory):
    """Map LeRobot head/wrist videos to the OpenPI three-view interface."""

    default_prompt: str | None = None

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
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                bi_flexiv_policy.BiFlexivInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[bi_flexiv_policy.BiFlexivOutputs()],
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
