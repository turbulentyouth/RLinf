# Copyright 2026 The RLinf Authors.

from collections import deque

import gymnasium as gym

from rlinf.envs.realworld.common.wrappers import keyboard_eval_control_wrapper as module


class _Listener:
    events = deque()

    def __init__(self):
        pass

    def pop_pressed_keys(self):
        if not self.events:
            return []
        return self.events.popleft()


class _Env(gym.Env):
    def __init__(self, *, truncated=False):
        self.truncated = truncated
        self.reset_calls = 0

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.reset_calls += 1
        return {"obs": self.reset_calls}, {}

    def step(self, action):
        del action
        return {"obs": self.reset_calls}, 0.0, False, self.truncated, {}


def _wrapper(monkeypatch, env, **kwargs):
    monkeypatch.setattr(module, "KeyboardListener", _Listener)
    return module.KeyboardEvalControlWrapper(
        env,
        start_keys=(),
        success_keys=(),
        failure_keys=(),
        interrupt_keys=("Key.left", "Key.right"),
        reset_wait_seconds=kwargs.pop("reset_wait_seconds", 0.0),
        continue_key="Key.right",
        preserve_env_done=True,
        **kwargs,
    )


def test_left_arrow_interrupts_current_episode(monkeypatch):
    wrapper = _wrapper(monkeypatch, _Env())
    _Listener.events = deque([[], ["Key.left"]])
    wrapper.reset()

    _, _, terminated, truncated, info = wrapper.step(None)

    assert terminated is False
    assert truncated is True
    assert info["eval_result"] == "interrupted"


def test_right_arrow_skips_reset_wait(monkeypatch):
    wrapper = _wrapper(monkeypatch, _Env(), reset_wait_seconds=60.0)
    _Listener.events = deque([[], ["Key.right"]])

    obs, _ = wrapper.reset()

    assert obs == {"obs": 1}
    assert wrapper._running is True


def test_underlying_max_step_truncation_is_preserved(monkeypatch):
    wrapper = _wrapper(monkeypatch, _Env(truncated=True))
    _Listener.events = deque([[], []])
    wrapper.reset()

    _, _, _, truncated, info = wrapper.step(None)

    assert truncated is True
    assert info["eval_result"] is None
