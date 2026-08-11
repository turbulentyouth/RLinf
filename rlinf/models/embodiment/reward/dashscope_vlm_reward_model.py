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

"""DashScope-compatible VLM reward model for embodied RL."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    get_reward_parser,
)


def _to_data_url(frame: Any, extra_view_index: int = 0) -> str:
    """Convert an RGB frame to a JPEG data URL accepted by the API."""
    if isinstance(frame, Image.Image):
        image = frame.convert("RGB")
    else:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        array = np.asarray(frame)
        if array.ndim == 4:
            if not 0 <= extra_view_index < array.shape[0]:
                raise IndexError(
                    f"extra_view_index={extra_view_index} is out of range for "
                    f"frame shape {array.shape}"
                )
            array = array[extra_view_index]
        if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {
            1,
            3,
            4,
        }:
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3 or array.shape[-1] not in {1, 3, 4}:
            raise ValueError(f"Expected an image frame, got shape {array.shape}")
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if (
            np.issubdtype(array.dtype, np.floating)
            and array.size > 0
            and float(array.max()) <= 1.0
        ):
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array[..., :3], mode="RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_message_text(response: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible response payload."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Cloud Qwen response does not contain choices")
    message = choices[0].get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )
    return str(content)


class DashScopeHistoryVLMRewardModel(BaseRewardModel):
    """Call a DashScope-compatible Qwen-VL endpoint for history rewards.

    The model consumes RLinf's ``history_input`` structure and returns one scalar
    reward per environment. It deliberately uses the standard library HTTP client
    so the reward path does not add another mandatory Python dependency.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.api_key_env = str(cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise ValueError(
                f"Set {self.api_key_env} before starting the cloud Qwen reward worker."
            )

        self.api_url = str(
            cfg.get(
                "api_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            )
        )
        self.api_model = str(cfg.get("api_model", ""))
        if not self.api_model:
            raise ValueError("reward.model.api_model must be set for cloud Qwen reward")

        self.api_timeout = float(cfg.get("api_timeout", 30.0))
        self.api_max_retries = max(0, int(cfg.get("api_max_retries", 2)))
        self.api_retry_backoff = max(0.0, float(cfg.get("api_retry_backoff", 0.5)))
        self.api_max_tokens = int(cfg.get("api_max_tokens", 16))
        self.api_temperature = float(cfg.get("api_temperature", 0.0))
        self.extra_view_index = int(cfg.get("extra_view_index", 0))
        self.interval_reward = float(cfg.get("interval_reward", 0.0))
        self.gt_success_bonus = float(cfg.get("gt_success_bonus", 0.0))
        self.reward_parser = get_reward_parser(
            cfg.get("reward_parser_name", "qwentrend_reward_parser")
        )(**cfg.get("reward_parser_params", {}))

    def forward(self, input_data: torch.Tensor, labels=None) -> dict[str, Any]:
        """Cloud reward is inference-only and has no differentiable forward pass."""
        raise NotImplementedError(
            "DashScopeHistoryVLMRewardModel is an inference-only reward model."
        )

    def _build_prompt(self, task: str, frame_count: int) -> str:
        return (
            f"You are currently performing the task: {task}. "
            f"You are given two synchronized {frame_count}-frame camera sequences "
            "from the same robot action window. The first sequence is the main "
            "view and the second is a selected wrist view. Judge whether the action "
            "trend is positive, negative, or unclear. Answer with exactly one word: "
            "positive, negative, or unclear."
        )

    def _build_content(
        self, task: str, main_frames: list[Any], extra_frames: list[Any]
    ) -> list[dict[str, Any]]:
        frame_count = min(len(main_frames), len(extra_frames))
        if frame_count == 0:
            return []
        content: list[dict[str, Any]] = [
            {"type": "text", "text": self._build_prompt(task, frame_count)}
        ]
        for frame_idx in range(frame_count):
            content.extend(
                [
                    {"type": "text", "text": f"Main view frame {frame_idx + 1}."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _to_data_url(main_frames[frame_idx]),
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Selected wrist view frame {frame_idx + 1}.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _to_data_url(
                                extra_frames[frame_idx], self.extra_view_index
                            ),
                        },
                    },
                ]
            )
        return content

    def _call_api(self, content: list[dict[str, Any]]) -> str:
        payload = {
            "model": self.api_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.api_temperature,
            "max_tokens": self.api_max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            self.api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.api_max_retries + 1):
            try:
                # The endpoint is explicitly configured by the user.
                with urllib_request.urlopen(  # noqa: S310
                    request, timeout=self.api_timeout
                ) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                return _extract_message_text(response_payload)
            except (
                urllib_error.HTTPError,
                urllib_error.URLError,
                TimeoutError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt == self.api_max_retries:
                    break
                time.sleep(self.api_retry_backoff * (attempt + 1))
        raise RuntimeError("Cloud Qwen reward request failed") from last_error

    def _infer_batch_size(self, history_input: dict[str, Any]) -> int:
        for history_buffer in history_input.values():
            for env_sequences in history_buffer.values():
                return len(env_sequences)
        return int(self.cfg.get("num_envs", 0))

    @torch.no_grad()
    def compute_reward(self, reward_input: dict[str, Any]) -> torch.Tensor:
        """Call Qwen for each valid history window and return scalar rewards."""
        history_input = reward_input.get("history_input", {})
        batch_size = self._infer_batch_size(history_input)
        rewards: list[float] = []
        observations = {
            key: value for key, value in reward_input.items() if key != "history_input"
        }
        task_descriptions = observations.get("task_descriptions", [])
        if isinstance(task_descriptions, str):
            task_descriptions = [task_descriptions] * batch_size
        history_window = history_input.get("history_window", {})
        main_batch = history_window.get("main_images", [[] for _ in range(batch_size)])
        extra_batch = history_window.get(
            "extra_view_images", [[] for _ in range(batch_size)]
        )

        for env_idx in range(batch_size):
            main_frames = main_batch[env_idx]
            extra_frames = extra_batch[env_idx]
            frame_count = min(len(main_frames), len(extra_frames))
            if frame_count == 0:
                rewards.append(self.interval_reward)
                continue
            task = str(
                task_descriptions[env_idx]
                if env_idx < len(task_descriptions)
                else self.cfg.get("default_task_description", "")
            )
            output_text = self._call_api(
                self._build_content(task, main_frames, extra_frames)
            )
            rewards.append(float(self.reward_parser.parse_rewards([output_text])[0]))

        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        return self._apply_gt_success_bonus(reward_tensor, observations)

    def _apply_gt_success_bonus(
        self, rewards: torch.Tensor, reward_input: dict[str, Any]
    ) -> torch.Tensor:
        """Apply an optional independent success bonus without exposing API data."""
        if self.gt_success_bonus == 0.0:
            return rewards
        env_infos = reward_input.get("env_infos", {})
        success = None
        for info_dict in (
            env_infos,
            env_infos.get("episode", {}) if isinstance(env_infos, dict) else {},
            env_infos.get("final_info", {}) if isinstance(env_infos, dict) else {},
        ):
            if isinstance(info_dict, dict):
                for key in ("success", "success_at_end", "success_once"):
                    if key in info_dict:
                        success = torch.as_tensor(info_dict[key]).reshape(-1).bool()
                        break
            if success is not None:
                break
        if success is None or success.shape[0] != rewards.shape[0]:
            return rewards
        return rewards + success.to(rewards.dtype) * self.gt_success_bonus
