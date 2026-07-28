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

Pedal: ``a`` starts a rollout from idle; ``c`` ends with reward=1
("success"); ``b`` ends with reward=0 ("failure"). On end, returns
``terminated=True`` and lets the outer ``auto_reset`` drive home.
"""

import math
import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


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
    ):
        super().__init__(env)
        if reset_wait_seconds is not None and reset_wait_seconds < 0:
            raise ValueError("reset_wait_seconds must be non-negative or null.")
        self.listener = KeyboardListener()
        self.start_keys = set(start_keys)
        self.success_keys = set(success_keys)
        self.failure_keys = set(failure_keys)
        self.interrupt_keys = set(interrupt_keys)
        self.reset_wait_seconds = reset_wait_seconds
        self.continue_key = continue_key
        self.preserve_env_done = preserve_env_done
        self._running = False
        self._last_obs: Any = None
        self._last_press_ts: dict[str, float] = {}

    def reset(self, *, seed=None, options=None):
        self._last_press_ts.clear()
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
        last_heartbeat = time.monotonic()
        while True:
            time.sleep(self.IDLE_POLL_S)
            now = time.monotonic()
            if now - last_heartbeat >= self.WAIT_HEARTBEAT_S:
                last_heartbeat = now
                self._log_info("Still waiting for pedal 'a' to start the rollout...")
            for key in self.listener.pop_pressed_keys():
                if key in self.start_keys:
                    self._running = True
                    self._log_info("Pedal 'a' pressed; starting rollout.")
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
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        if not self.preserve_env_done:
            terminated = False
            truncated = False

        result: str | None = None
        for key in self.listener.pop_pressed_keys():
            now = time.monotonic()
            if now - self._last_press_ts.get(key, -math.inf) < self.PEDAL_DEBOUNCE_S:
                continue
            self._last_press_ts[key] = now
            if key in self.interrupt_keys:
                truncated = True
                result = "interrupted"
                self._running = False
                break
            if key in self.success_keys:
                terminated = True
                reward = 1.0
                result = "success"
                self._running = False
                break
            if key in self.failure_keys:
                terminated = True
                reward = 0.0
                result = "failure"
                self._running = False
                break

        info["eval_phase"] = "rec" if self._running else "pre"
        info["eval_result"] = result
        return obs, reward, terminated, truncated, info

    def _wait_after_reset(self, obs, info):
        """Wait for the configured delay or an early right-arrow press."""

        wait_seconds = float(self.reset_wait_seconds or 0.0)
        deadline = time.monotonic() + wait_seconds
        self._log_info(
            "Arms returned to start position; waiting %.1f seconds. "
            "Press %s to start early.",
            wait_seconds,
            self.continue_key,
        )
        while time.monotonic() < deadline:
            for key in self.listener.pop_pressed_keys():
                if self.continue_key is not None and key == self.continue_key:
                    self._running = True
                    self._log_info("Continue key pressed; starting next episode.")
                    return obs, info
            time.sleep(min(self.IDLE_POLL_S, max(0.0, deadline - time.monotonic())))
        self._running = True
        self._log_info("Reset wait completed; starting next episode.")
        return obs, info

    def _idle_response(self, event: str | None):
        info = {"eval_phase": "pre", "eval_event": event, "eval_result": None}
        return self._last_obs, 0.0, False, False, info

    def _log_info(self, message: str, *args) -> None:
        logger = getattr(self._base_env(), "_logger", None)
        if logger is not None:
            logger.info(message, *args)

    def _base_env(self):
        return getattr(self.env, "unwrapped", self.env)
