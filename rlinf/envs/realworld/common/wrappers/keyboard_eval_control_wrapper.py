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
"""Foot-pedal-gated wrapper for autonomous policy eval.

By default, ``a`` starts a rollout from idle, ``c`` ends with reward 1
(success), and ``b`` ends with reward 0 (failure). Reward values and keys are
configurable so real-world tasks can use normalized step penalties and terminal
human labels. On end, the wrapper returns ``terminated=True`` and lets the
outer ``auto_reset`` drive home.
``exit_keys`` (e.g. ESC) do not raise: they set :attr:`exit_requested`,
truncate the current episode, and let the env worker/runner wind down
gracefully (close_envs homes the arms before the driver exits).
"""

import math
import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.utils.eval_events import emit_event


class KeyboardEvalControlWrapper(gym.Wrapper):
    """Foot-pedal-gated start/stop for autonomous policy eval rollouts."""

    IDLE_POLL_S = 0.05
    PEDAL_DEBOUNCE_S = 0.2
    WAIT_HEARTBEAT_S = 10.0

    def __init__(
        self,
        env: gym.Env,
        *,
        start_keys: tuple[str, ...] = ("a",),
        success_keys: tuple[str, ...] = ("c",),
        failure_keys: tuple[str, ...] = ("b",),
        interrupt_keys: tuple[str, ...] = (),
        reset_wait_seconds: float | None = None,
        continue_key: str | None = None,
        preserve_env_done: bool = False,
        episode_recorder: Any | None = None,
        save_keys: tuple[str, ...] = (),
        discard_keys: tuple[str, ...] = (),
        exit_keys: tuple[str, ...] = (),
        step_reward: float | None = None,
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
        timeout_reward: float | None = None,
    ):
        super().__init__(env)
        if reset_wait_seconds is not None and reset_wait_seconds < 0:
            raise ValueError("reset_wait_seconds must be non-negative or null.")
        for name, value in (
            ("step_reward", step_reward),
            ("success_reward", success_reward),
            ("failure_reward", failure_reward),
            ("timeout_reward", timeout_reward),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or null.")
        self.listener = KeyboardListener()
        self.start_keys = set(start_keys)
        self.success_keys = set(success_keys)
        self.failure_keys = set(failure_keys)
        self.interrupt_keys = set(interrupt_keys)
        self.reset_wait_seconds = reset_wait_seconds
        self.continue_key = continue_key
        self.preserve_env_done = preserve_env_done
        self.episode_recorder = episode_recorder
        self.save_keys = set(save_keys)
        self.discard_keys = set(discard_keys)
        self.exit_keys = set(exit_keys)
        self.step_reward = step_reward
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.timeout_reward = timeout_reward
        self._running = False
        self._exit_requested = False
        self._last_obs: Any = None
        self._last_press_ts: dict[str, float] = {}
        self._episode_steps = 0
        self._terminal_result: str | None = None

    @property
    def exit_requested(self) -> bool:
        """Whether an exit key was pressed (request for a graceful eval stop).

        Read by the env worker between chunks; once set, the evaluation loop
        winds down and the runner closes the envs (homing the arms) instead
        of the old behavior of raising KeyboardInterrupt inside the actor.
        """
        return self._exit_requested

    def reset(self, *, seed=None, options=None):
        self._last_press_ts.clear()
        self._episode_steps = 0
        self._terminal_result = None
        self.listener.pop_pressed_keys()
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        if self.reset_wait_seconds is not None:
            return self._wait_after_reset(obs, info)
        # Block until the operator has arranged the scene and presses 'a'.
        # This is intentional (the arms are homed and idle); log on entry and
        # emit a periodic heartbeat so the wait is not mistaken for a hang.
        self._log_info(
            "Arms homed and idle. Arrange the scene, then press pedal 'a' "
            "to start the next rollout (Ctrl-C to abort)."
        )
        emit_event("idle_wait_start", "wrapper")
        last_heartbeat = time.monotonic()
        while True:
            time.sleep(self.IDLE_POLL_S)
            now = time.monotonic()
            if now - last_heartbeat >= self.WAIT_HEARTBEAT_S:
                last_heartbeat = now
                self._log_info("Still waiting for pedal 'a' to start the rollout...")
            for key in self.listener.pop_pressed_keys():
                if key in self.exit_keys:
                    self._exit_requested = True
                    self._log_info("Exit key pressed; winding down evaluation.")
                    emit_event("exit_key", "wrapper", key=key, phase="idle_wait")
                    emit_event("idle_wait_end", "wrapper", reason="exit_key")
                    return obs, info
                if key in self.start_keys:
                    self._running = True
                    self._log_info("Pedal 'a' pressed; starting rollout.")
                    emit_event("idle_wait_end", "wrapper", reason="start_key", key=key)
                    return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not self._running:
            # Idle: hold the last target. ARX episode control must pad the rest
            # of the current action chunk without delaying the reset.
            if self.reset_wait_seconds is None:
                time.sleep(self.IDLE_POLL_S)
            for key in self.listener.pop_pressed_keys():
                now = time.monotonic()
                if (
                    now - self._last_press_ts.get(key, -math.inf)
                    < self.PEDAL_DEBOUNCE_S
                ):
                    continue
                self._last_press_ts[key] = now
                if key in self.start_keys:
                    self._running = True
                    return self._idle_response(event="start")
            return self._idle_response(event=None)

        # Running: forward to the wrapped env.
        pre_step_obs = self._last_obs
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self._episode_steps += 1

        if self.step_reward is not None:
            reward = float(self.step_reward)

        if self.episode_recorder is not None:
            recorded_action = info.get("executed_action", action)
            self.episode_recorder.add_frame(pre_step_obs, recorded_action)

        if not self.preserve_env_done:
            terminated = False
            truncated = False

        result: str | None = None
        for key in self.listener.pop_pressed_keys():
            now = time.monotonic()
            if now - self._last_press_ts.get(key, -math.inf) < self.PEDAL_DEBOUNCE_S:
                continue
            self._last_press_ts[key] = now
            if key in self.exit_keys:
                if self.episode_recorder is not None:
                    self.episode_recorder.discard_episode("exit key pressed")
                self._running = False
                self._exit_requested = True
                truncated = True
                result = "exit"
                emit_event("exit_key", "wrapper", key=key)
                break
            if key in self.interrupt_keys:
                truncated = True
                result = "interrupted"
                self._running = False
                if self.episode_recorder is not None:
                    if key in self.save_keys:
                        self.episode_recorder.save_episode()
                        result = "saved"
                    elif key in self.discard_keys:
                        self.episode_recorder.discard_episode("discard key pressed")
                        result = "discarded"
                break
            if key in self.success_keys:
                terminated = True
                truncated = False
                reward = self.success_reward
                result = "success"
                self._running = False
                if self.episode_recorder is not None and key in self.save_keys:
                    self.episode_recorder.save_episode()
                emit_event("episode_result_key", "wrapper", key=key, result="success")
                break
            if key in self.failure_keys:
                terminated = True
                truncated = False
                reward = self.failure_reward
                result = "failure"
                self._running = False
                if self.episode_recorder is not None and key in self.discard_keys:
                    self.episode_recorder.discard_episode("failure key pressed")
                emit_event("episode_result_key", "wrapper", key=key, result="failure")
                break

        if result is None and truncated and self.timeout_reward is not None:
            reward = float(self.timeout_reward)
            result = "timeout"
            self._running = False
            if self.episode_recorder is not None:
                self.episode_recorder.discard_episode("episode timed out")
            emit_event(
                "episode_timeout",
                "wrapper",
                episode_steps=self._episode_steps,
                reward=reward,
            )

        if result is not None:
            self._terminal_result = result

        info["eval_phase"] = "rec" if self._running else "pre"
        info["eval_result"] = result
        if result is not None or self.step_reward is not None:
            info["success"] = result == "success"
        info["terminal_reason"] = result or "running"
        info["episode_step"] = self._episode_steps
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the optional recorder before releasing the wrapped environment."""

        try:
            if self.episode_recorder is not None:
                self.episode_recorder.close()
        finally:
            self.env.close()

    def _wait_after_reset(self, obs, info):
        """Wait for the configured delay or an early right-arrow press."""

        if self._exit_requested:
            return obs, info
        wait_seconds = float(self.reset_wait_seconds or 0.0)
        deadline = time.monotonic() + wait_seconds
        self._log_info(
            "Arms returned to start position; waiting %.1f seconds. "
            "Press %s to start early.",
            wait_seconds,
            self.continue_key,
        )
        emit_event("reset_wait_start", "wrapper", seconds=wait_seconds)
        while time.monotonic() < deadline:
            for key in self.listener.pop_pressed_keys():
                if key in self.exit_keys:
                    self._exit_requested = True
                    self._log_info("Exit key pressed; winding down evaluation.")
                    emit_event("exit_key", "wrapper", key=key, phase="reset_wait")
                    emit_event("reset_wait_end", "wrapper", reason="exit_key")
                    return obs, info
                if self.continue_key is not None and key == self.continue_key:
                    self._running = True
                    self._log_info("Continue key pressed; starting next episode.")
                    emit_event(
                        "reset_wait_end", "wrapper", reason="continue_key", key=key
                    )
                    return obs, info
            time.sleep(min(self.IDLE_POLL_S, max(0.0, deadline - time.monotonic())))
        self._running = True
        self._log_info("Reset wait completed; starting next episode.")
        emit_event("reset_wait_end", "wrapper", reason="timeout")
        return obs, info

    def _idle_response(self, event: str | None):
        info = {
            "eval_phase": "pre",
            "eval_event": event,
            "eval_result": self._terminal_result,
            "success": self._terminal_result == "success",
            "terminal_reason": self._terminal_result or "idle",
            "episode_step": self._episode_steps,
        }
        return self._last_obs, 0.0, False, False, info

    def _log_info(self, message: str, *args) -> None:
        logger = getattr(self._base_env(), "_logger", None)
        if logger is not None:
            logger.info(message, *args)

    def _base_env(self):
        return getattr(self.env, "unwrapped", self.env)
