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

import base64
import json
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlinf.models.embodiment.reward.dashscope_vlm_reward_model import (
    DashScopeHistoryVLMRewardModel,
    _extract_message_text,
    _to_data_url,
)


def _model_config():
    return OmegaConf.create(
        {
            "api_key_env": "TEST_QWEN_API_KEY",
            "api_url": "https://example.test/v1/chat/completions",
            "api_model": "test-qwen-vl",
            "num_envs": 1,
            "reward_parser_name": "qwentrend_reward_parser",
            "reward_parser_params": {
                "positive_reward": 1.0,
                "negative_reward": -0.2,
                "unclear_reward": 0.0,
            },
            "extra_view_index": 1,
        }
    )


def test_to_data_url_accepts_batched_extra_views():
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)

    data_url = _to_data_url(frames, extra_view_index=1)

    assert data_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1])[:2] == b"\xff\xd8"


def test_to_data_url_rejects_unknown_view_index():
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)

    with pytest.raises(IndexError, match="extra_view_index=2"):
        _to_data_url(frames, extra_view_index=2)


def test_extract_message_text_supports_text_blocks():
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "The trend is "},
                        {"type": "text", "text": "positive"},
                    ]
                }
            }
        ]
    }

    assert _extract_message_text(response) == "The trend is positive"


def test_compute_reward_posts_one_request_per_valid_environment(monkeypatch):
    monkeypatch.setenv("TEST_QWEN_API_KEY", "test-secret")
    model = DashScopeHistoryVLMRewardModel(_model_config())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist_frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    reward_input = {
        "task_descriptions": ["Place the cube in the box."],
        "history_input": {
            "history_window": {
                "main_images": [[frame] * 5],
                "extra_view_images": [[wrist_frames] * 5],
            }
        },
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "positive"}}]}
            ).encode("utf-8")

    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    module = "rlinf.models.embodiment.reward.dashscope_vlm_reward_model"
    with patch(f"{module}.urllib_request.urlopen", side_effect=fake_urlopen):
        rewards = model.compute_reward(reward_input)

    assert rewards.tolist() == [1.0]
    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url == "https://example.test/v1/chat/completions"
    assert timeout == 30.0
    assert request.headers["Authorization"] == "Bearer test-secret"
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["model"] == "test-qwen-vl"
    assert payload["messages"][0]["content"][0]["type"] == "text"
    assert sum(
        block.get("type") == "image_url"
        for block in payload["messages"][0]["content"]
    ) == 10


def test_compute_reward_uses_interval_reward_without_history(monkeypatch):
    monkeypatch.setenv("TEST_QWEN_API_KEY", "test-secret")
    cfg = _model_config()
    cfg.interval_reward = 0.25
    model = DashScopeHistoryVLMRewardModel(cfg)

    module = "rlinf.models.embodiment.reward.dashscope_vlm_reward_model"
    with patch(f"{module}.urllib_request.urlopen") as urlopen:
        rewards = model.compute_reward({"history_input": {}})

    assert rewards.tolist() == [0.25]
    urlopen.assert_not_called()
