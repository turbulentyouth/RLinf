# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""OpenPI transforms for the LeRobot Bi-Flexiv TCP dataset."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


BI_FLEXIV_ACTION_DIM = 20


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Bi-Flexiv image must be 3-D, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Bi-Flexiv image must have 3 channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(image, dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class BiFlexivInputs(transforms.DataTransformFn):
    """Convert three camera views and 20-D TCP state/action to OpenPI inputs."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != BI_FLEXIV_ACTION_DIM:
            raise ValueError(
                "Bi-Flexiv state must have 20 values, "
                f"got {state.shape}"
            )
        raw_images = data.get("images")
        if raw_images is None:
            raw_images = {
                "head": data["observation/images/head"],
                "left_wrist": data["observation/images/left_wrist"],
                "right_wrist": data["observation/images/right_wrist"],
            }
        images = {
            "base_0_rgb": _parse_image(raw_images["head"]),
            "left_wrist_0_rgb": _parse_image(raw_images["left_wrist"]),
            "right_wrist_0_rgb": _parse_image(raw_images["right_wrist"]),
        }
        result = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": images,
            "image_mask": {key: np.True_ for key in images},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != BI_FLEXIV_ACTION_DIM:
                raise ValueError(
                    "Bi-Flexiv action must have 20 values, "
                    f"got {actions.shape}"
                )
            result["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )
        return result


@dataclasses.dataclass(frozen=True)
class BiFlexivOutputs(transforms.DataTransformFn):
    """Trim padded OpenPI actions back to the 20-D Flexiv action space."""

    output_action_dim: int = BI_FLEXIV_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}


@dataclasses.dataclass(frozen=True)
class BiFlexivValueInputs(transforms.DataTransformFn):
    """Convert Bi-Flexiv observations to the RECAP value-model view format."""

    action_dim: int = BI_FLEXIV_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        images = data.get("images")
        if not isinstance(images, dict):
            raise ValueError("Bi-Flexiv value input is missing the images mapping")

        def get_view(name: str) -> np.ndarray:
            if name not in images:
                raise KeyError(f"Missing Bi-Flexiv camera view: {name}")
            return _parse_image(images[name])

        state = np.asarray(data["state"], dtype=np.float32)
        result = {
            "images": {
                "base_0_rgb": get_view("head"),
                "left_wrist_0_rgb": get_view("left_wrist"),
                "right_wrist_0_rgb": get_view("right_wrist"),
            },
            "image_masks": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": transforms.pad_to_dim(state, self.action_dim),
        }
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )
        return result
