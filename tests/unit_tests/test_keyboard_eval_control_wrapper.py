# Copyright 2026 The RLinf Authors.

from collections import deque

import gymnasium as gym

from rlinf.envs.realworld.arx_x5_dual.tasks import _episode_reward_kwargs
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


class _Recorder:
    def __init__(self):
        self.frames = 0
        self.saved = 0
        self.discarded: list[str] = []

    def add_frame(self, observation, action):
        del observation, action
        self.frames += 1

    def save_episode(self):
        self.saved += 1

    def discard_episode(self, reason):
        self.discarded.append(reason)


def _wrapper(monkeypatch, env, **kwargs):
    monkeypatch.setattr(module, "KeyboardListener", _Listener)
    return module.KeyboardEvalControlWrapper(
        env,
        start_keys=kwargs.pop("start_keys", ()),
        success_keys=kwargs.pop("success_keys", ()),
        failure_keys=kwargs.pop("failure_keys", ()),
        interrupt_keys=kwargs.pop(
            "interrupt_keys", ("Key.left", "Key.right")
        ),
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


def test_esc_requests_graceful_exit_instead_of_raising(monkeypatch):
    wrapper = _wrapper(monkeypatch, _Env(), exit_keys=("Key.esc",))
    _Listener.events = deque([[], ["Key.esc"]])
    wrapper.reset()

    _, _, _, truncated, info = wrapper.step(None)

    assert truncated is True
    assert info["eval_result"] == "exit"
    assert wrapper.exit_requested is True
    assert wrapper._running is False


def test_esc_during_reset_wait_exits_immediately(monkeypatch):
    wrapper = _wrapper(
        monkeypatch, _Env(), reset_wait_seconds=60.0, exit_keys=("Key.esc",)
    )
    _Listener.events = deque([[], ["Key.esc"]])

    wrapper.reset()

    assert wrapper.exit_requested is True
    assert wrapper._running is False


def test_pending_exit_skips_reset_wait_entirely(monkeypatch):
    wrapper = _wrapper(
        monkeypatch, _Env(), reset_wait_seconds=60.0, exit_keys=("Key.esc",)
    )
    # 1st reset: clear [] then right-arrow ends the wait; step: ESC latches
    # the exit flag; 2nd reset: clear [] then the wait must return instantly.
    _Listener.events = deque([[], ["Key.right"], ["Key.esc"], []])
    wrapper.reset()
    wrapper.step(None)

    wrapper.reset()

    assert wrapper.exit_requested is True
    assert wrapper._running is False


def test_normalized_step_penalty_is_emitted_while_running(monkeypatch):
    wrapper = _wrapper(
        monkeypatch,
        _Env(),
        step_reward=-1.0 / 1800,
        success_reward=0.0,
        failure_reward=-1.0,
        timeout_reward=-1.0,
    )
    _Listener.events = deque([[], []])
    wrapper.reset()

    _, reward, terminated, truncated, info = wrapper.step(None)

    assert reward == -1.0 / 1800
    assert terminated is False
    assert truncated is False
    assert info["success"] is False
    assert info["terminal_reason"] == "running"
    assert info["episode_step"] == 1


def test_success_key_uses_zero_reward_and_saves_episode(monkeypatch):
    recorder = _Recorder()
    wrapper = _wrapper(
        monkeypatch,
        _Env(),
        success_keys=("Key.right",),
        save_keys=("Key.right",),
        episode_recorder=recorder,
        step_reward=-1.0 / 1800,
        success_reward=0.0,
        failure_reward=-1.0,
        timeout_reward=-1.0,
    )
    _Listener.events = deque([[], ["Key.right"]])
    wrapper.reset()

    _, reward, terminated, truncated, info = wrapper.step(None)

    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["terminal_reason"] == "success"
    assert recorder.frames == 1
    assert recorder.saved == 1

    _, padded_reward, padded_terminated, padded_truncated, padded_info = wrapper.step(
        None
    )
    assert padded_reward == 0.0
    assert padded_terminated is False
    assert padded_truncated is False
    assert padded_info["success"] is True
    assert padded_info["episode_step"] == 1
    assert recorder.frames == 1


def test_failure_key_uses_negative_reward_and_discards_episode(monkeypatch):
    recorder = _Recorder()
    wrapper = _wrapper(
        monkeypatch,
        _Env(),
        failure_keys=("Key.left",),
        discard_keys=("Key.left",),
        episode_recorder=recorder,
        step_reward=-1.0 / 1800,
        success_reward=0.0,
        failure_reward=-1.0,
        timeout_reward=-1.0,
    )
    _Listener.events = deque([[], ["Key.left"]])
    wrapper.reset()

    _, reward, terminated, truncated, info = wrapper.step(None)

    assert reward == -1.0
    assert terminated is True
    assert truncated is False
    assert info["success"] is False
    assert info["terminal_reason"] == "failure"
    assert recorder.discarded == ["failure key pressed"]


def test_timeout_uses_failure_reward_and_discards_episode(monkeypatch):
    recorder = _Recorder()
    wrapper = _wrapper(
        monkeypatch,
        _Env(truncated=True),
        episode_recorder=recorder,
        step_reward=-1.0 / 1800,
        success_reward=0.0,
        failure_reward=-1.0,
        timeout_reward=-1.0,
    )
    _Listener.events = deque([[], []])
    wrapper.reset()

    _, reward, terminated, truncated, info = wrapper.step(None)

    assert reward == -1.0
    assert terminated is False
    assert truncated is True
    assert info["success"] is False
    assert info["terminal_reason"] == "timeout"
    assert recorder.discarded == ["episode timed out"]


def test_arx_reward_config_derives_step_penalty_from_episode_horizon():
    kwargs = _episode_reward_kwargs(
        {
            "reward": {
                "enabled": True,
                "mode": "normalized_step_penalty",
                "success_keys": ["Key.right"],
                "failure_keys": ["Key.left"],
            }
        },
        {"max_episode_steps": 1800},
    )

    assert kwargs["step_reward"] == -1.0 / 1800
    assert kwargs["success_reward"] == 0.0
    assert kwargs["failure_reward"] == -1.0
    assert kwargs["timeout_reward"] == -1.0
    assert kwargs["success_keys"] == ("Key.right",)
    assert kwargs["failure_keys"] == ("Key.left",)
