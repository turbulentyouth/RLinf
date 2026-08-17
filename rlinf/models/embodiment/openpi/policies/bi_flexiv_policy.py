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
"""Policy transforms for the bi-Flexiv dual-arm (Rizon4) LeRobot dataset.

State / action layout (20 dims), matching the columns of
``Xense/stack-cubes-flexiv-0813``:

- ``left_tcp.{x, y, z, r1..r6}`` (9 dims)
- ``right_tcp.{x, y, z, r1..r6}`` (9 dims)
- ``left_gripper.pos`` (1 dim)
- ``right_gripper.pos`` (1 dim)

The rotation is stored as a 6D representation (the first two columns of the
rotation matrix, flattened). This representation is continuous (no ±180°
discontinuities like Euler angles) and has no double-cover ambiguity like
quaternions, so no special encoding/decoding is required.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np
from openpi import transforms

# Dimensionality of the bi-Flexiv state and action vectors.
ACTION_DIM = 20


def make_bi_flexiv_example() -> dict:
    """Creates a random input example for the bi-Flexiv policy."""
    return {
        "state": np.ones((ACTION_DIM,)),
        "images": {
            "head": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Convert an image to uint8 ``[H, W, C]``.

    LeRobot stores images as float32 ``[C, H, W]``; the runtime passes uint8
    ``[H, W, C]``. Normalize both to the layout expected by the model.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _decode_bi_flexiv(data: dict) -> dict:
    """Normalize observations to the runtime ``state`` / ``images`` layout.

    Accepts either the runtime layout (``state`` plus an ``images`` dict keyed
    by ``head`` / ``left_wrist`` / ``right_wrist``) or the value-model
    observation layout (``observation/state`` plus ``observation/images.*``).
    """
    if "observation/state" in data:
        data["images"] = {
            "head": _parse_image(data["observation/images/head"]),
            "left_wrist": _parse_image(data["observation/images/left_wrist"]),
            "right_wrist": _parse_image(data["observation/images/right_wrist"]),
        }
        data["state"] = np.asarray(data["observation/state"])
    else:
        images = data["images"]
        data["images"] = {name: _parse_image(img) for name, img in images.items()}
        data["state"] = np.asarray(data["state"])
    return data


@dataclasses.dataclass(frozen=True)
class BiFlexivInputs(transforms.DataTransformFn):
    """Inputs for the bi-Flexiv policy.

    Expected inputs:

    - images: ``dict[name, img]`` where ``img`` is ``[C, H, W]`` and ``name``
      must be in ``EXPECTED_CAMERAS`` (the value-model ``observation/images.*``
      layout is also accepted and decoded first).
    - state: ``[20]``
    - actions: ``[action_horizon, 20]`` (training only)

    The three cameras are mapped onto pi0 / pi05's image slots: ``head`` is the
    base view and the two wrist cameras become the left / right wrist views.
    """

    # The expected camera names. All input cameras must be in this set. Missing
    # cameras are replaced with black images and the corresponding
    # ``image_mask`` is set to ``False``.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "head",
        "left_wrist",
        "right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_bi_flexiv(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Expected images to contain {self.EXPECTED_CAMERAS}, "
                f"got {tuple(in_images)}"
            )

        # Assume that the head image always exists.
        head_image = in_images["head"]

        images = {
            "base_0_rgb": head_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "left_wrist",
            "right_wrist_0_rgb": "right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(head_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training. No conversion is needed:
        # the 6D rotation representation is already continuous.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BiFlexivOutputs(transforms.DataTransformFn):
    """Outputs for the bi-Flexiv policy."""

    output_action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        # Return only the 20 real action dims (the model may emit padded dims).
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
