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

"""Tests for continuous embodied evaluation and manual stopping."""

import pytest
from omegaconf import OmegaConf

from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner


class _Handle:
    def __init__(self, results=None, interrupt_once=False):
        self.results = results
        self.interrupt_once = interrupt_once
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1
        if self.interrupt_once and self.wait_calls == 1:
            raise KeyboardInterrupt
        return self.results


class _EnvGroup:
    def __init__(self, evaluate_handle=None):
        self.evaluate_handle = evaluate_handle
        self.close_calls = 0
        self.stop_calls = 0

    def evaluate(self, **kwargs):
        del kwargs
        return self.evaluate_handle

    def request_eval_stop(self):
        self.stop_calls += 1
        return _Handle()

    def close_envs(self):
        self.close_calls += 1
        return _Handle()


class _RolloutGroup:
    def __init__(self, evaluate_handle):
        self.evaluate_handle = evaluate_handle
        self.stop_calls = 0

    def evaluate(self, **kwargs):
        del kwargs
        return self.evaluate_handle

    def request_eval_stop(self):
        self.stop_calls += 1
        return _Handle()


class _MetricLogger:
    def __init__(self):
        self.logs = []
        self.finished = False

    def log(self, step, data):
        self.logs.append((step, data))

    def finish(self):
        self.finished = True


class _Logger:
    def info(self, *args):
        del args

    def warning(self, *args):
        del args


def _runner():
    runner = EmbodiedEvalRunner.__new__(EmbodiedEvalRunner)
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "continuous_eval": {"enabled": True, "max_cycles": None},
                "enable_decoupled_mode": False,
            }
        }
    )
    runner.metric_logger = _MetricLogger()
    runner.logger = _Logger()
    runner.env_channel = object()
    runner.rollout_channel = object()
    return runner


def test_continuous_run_stops_on_keyboard_interrupt_and_closes_envs():
    """Ctrl+C stops the loop after completed cycles and closes hardware envs."""

    runner = _runner()
    runner.env = _EnvGroup()
    outcomes = iter([{"episode_len": 5}, {"episode_len": 5}, KeyboardInterrupt()])

    def evaluate():
        result = next(outcomes)
        if isinstance(result, BaseException):
            raise result
        return result

    runner.evaluate = evaluate
    runner.run()

    assert [step for step, _ in runner.metric_logger.logs] == [0, 1]
    assert runner.metric_logger.finished is True
    assert runner.env.close_calls == 1


def test_evaluate_requests_cooperative_stop_after_manual_interrupt():
    """An interrupted wait stops both workers before draining their handles."""

    runner = _runner()
    env_handle = _Handle(results=[{}], interrupt_once=True)
    rollout_handle = _Handle()
    runner.env = _EnvGroup(evaluate_handle=env_handle)
    runner.rollout = _RolloutGroup(evaluate_handle=rollout_handle)

    with pytest.raises(KeyboardInterrupt):
        runner.evaluate()

    assert runner.env.stop_calls == 1
    assert runner.rollout.stop_calls == 1
    assert env_handle.wait_calls == 2
    assert rollout_handle.wait_calls == 1
