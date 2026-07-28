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

"""Tests for reference-aligned ARX X5 dual-arm control and shutdown."""

from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.realworld.arx_x5_dual.arx_x5_dual_controller import (
    ArxX5JointController,
)
from rlinf.envs.realworld.arx_x5_dual.arx_x5_dual_env import (
    ArxX5DualEnv,
    ArxX5DualPositionConfig,
)
from rlinf.envs.realworld.arx_x5_dual.arx_x5_dual_robot_state import (
    ArxX5ArmState,
)


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warning(self, *args, **kwargs):
        del args, kwargs

    def error(self, *args, **kwargs):
        del args, kwargs


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += max(0.0, duration)


class _FakeController:
    def __init__(self, side, events):
        self.side = side
        self.events = events
        self.state = ArxX5ArmState()
        self.commands = []
        self.control_mode = "position"

    def read_state(self):
        return ArxX5ArmState(
            joint_position=self.state.joint_position.copy(),
            gripper_position=self.state.gripper_position,
        )

    def hold_current_position(self):
        self.events.append(f"{self.side}:hold")

    def set_to_normal_position_control(self, **kwargs):
        del kwargs
        self.control_mode = "position"
        self.events.append(f"{self.side}:position")

    def set_to_gravity_compensation(self):
        self.control_mode = "gravity_compensation"
        self.events.append(f"{self.side}:gravity")

    def set_to_damping(self):
        self.events.append(f"{self.side}:damping")

    def send_absolute_joint_position(self, joints, gripper):
        command = np.concatenate([np.asarray(joints), [gripper]])
        self.commands.append(command)
        self.state.joint_position = np.asarray(joints).copy()
        self.state.gripper_position = float(gripper)

    def close(self):
        self.events.append(f"{self.side}:close")


class _SdkJointState:
    def __init__(self, dof):
        self._position = np.zeros(dof, dtype=np.float64)
        self.gripper_pos = 0.0

    def pos(self):
        return self._position


class _SdkGain:
    def __init__(self, dof):
        self._kp = np.zeros(dof, dtype=np.float64)
        self._kd = np.zeros(dof, dtype=np.float64)
        self.gripper_kp = 0.0
        self.gripper_kd = 0.0

    def kp(self):
        return self._kp

    def kd(self):
        return self._kd


class _SdkArm:
    def __init__(self, robot_config, controller_config, interface_name):
        del interface_name
        self.robot_config = robot_config
        self.controller_config = controller_config
        self.state = _SdkJointState(robot_config.joint_dof)
        self.state.pos()[:] = np.arange(robot_config.joint_dof) / 10
        self.state.gripper_pos = 0.7
        self.gain = _SdkGain(robot_config.joint_dof)
        self.events = []
        self.commands = []

    def get_joint_state(self):
        return self.state

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.events.append("gain")

    def set_joint_cmd(self, command):
        self.commands.append(command)

    def reset_to_home(self):
        self.events.append("reset_home")

    def set_to_gravity_compensation(self):
        self.events.append("gravity")

    def set_to_damping(self):
        self.events.append("damping")


class _Factory:
    def __init__(self, value):
        self.value = value

    def get_config(self, *args):
        del args
        return self.value


class _FactoryType:
    instance = None

    @classmethod
    def get_instance(cls):
        return cls.instance


class _FakeSdk:
    JointState = _SdkJointState
    Arx5JointController = _SdkArm

    class RobotConfigFactory(_FactoryType):
        pass

    class ControllerConfigFactory(_FactoryType):
        pass


def _make_fake_sdk():
    robot_config = SimpleNamespace(
        joint_dof=6,
        joint_pos_min=np.full(6, -2.0),
        joint_pos_max=np.full(6, 2.0),
        gripper_width=1.57,
    )
    controller_config = SimpleNamespace(
        controller_dt=0.01,
        default_preview_time=0.0,
        background_send_recv=False,
        gravity_compensation=False,
        shutdown_to_passive=False,
        default_kp=np.arange(1.0, 7.0),
        default_kd=np.arange(2.0, 8.0),
        default_gripper_kp=3.0,
        default_gripper_kd=4.0,
    )
    _FakeSdk.RobotConfigFactory.instance = _Factory(robot_config)
    _FakeSdk.ControllerConfigFactory.instance = _Factory(controller_config)
    return _FakeSdk, controller_config


def _make_motion_env(events, *, interpolation_dt=0.25):
    env = ArxX5DualEnv.__new__(ArxX5DualEnv)
    env._left_controller = _FakeController("left", events)
    env._right_controller = _FakeController("right", events)
    env._left_state = ArxX5ArmState()
    env._right_state = ArxX5ArmState()
    env.config = SimpleNamespace(
        interpolation_controller_dt=interpolation_dt,
        kp_scale=0.5,
        kd_scale=1.5,
    )
    env._logger = _Logger()
    return env


def test_controller_matches_reference_mode_and_command_buffer_behavior():
    """The wrapper preserves SDK defaults and follows BiARX5 mode bookkeeping."""

    sdk, controller_config = _make_fake_sdk()
    controller = ArxX5JointController(
        model="X5",
        interface_name="can1",
        controller_dt=0.002,
        preview_time=0.03,
        sdk_module=sdk,
    )
    arm = controller._controller

    assert controller.control_mode == "gravity_compensation"
    assert controller_config.controller_dt == 0.002
    assert controller_config.default_preview_time == 0.03
    assert controller_config.background_send_recv is True
    assert controller_config.gravity_compensation is False
    assert controller_config.shutdown_to_passive is False

    controller.set_to_gravity_compensation()
    assert arm.events == []
    controller.reset_to_home()
    controller.hold_current_position()
    controller.set_to_normal_position_control()
    controller.send_absolute_joint_position(np.ones(6), 1.0)
    controller.send_absolute_joint_position(np.full(6, 0.5), 0.5)

    assert arm.events == ["reset_home", "gain"]
    np.testing.assert_allclose(arm.gain.kp(), controller_config.default_kp * 0.5)
    np.testing.assert_allclose(arm.gain.kd(), controller_config.default_kd * 1.5)
    assert arm.commands[1] is arm.commands[2]

    controller.set_to_gravity_compensation()
    assert arm.events[-1] == "gravity"


def test_policy_action_has_no_per_step_delta_limit():
    """A safe absolute target is not constrained relative to the previous state."""

    env = ArxX5DualEnv({"is_dummy": True, "step_frequency": 1000.0})
    try:
        env.reset()
        action = np.ones(14, dtype=np.float32)
        _, _, _, _, info = env.step(action)

        np.testing.assert_allclose(info["executed_action"], action)
        np.testing.assert_allclose(env._compose_absolute_joint_state(), action)
    finally:
        env.close()


def test_unsafe_policy_action_raises_without_clipping():
    """Reference fixed limits reject an unsafe action instead of clipping it."""

    env = ArxX5DualEnv({"is_dummy": True, "step_frequency": 1000.0})
    try:
        action = np.zeros(14, dtype=np.float32)
        action[0] = 3.0
        with pytest.raises(ValueError, match="Left joint 1 out of range"):
            env.step(action)
    finally:
        env.close()


def test_hardware_dry_run_intercepts_action_before_safety_check(monkeypatch):
    """The reference dry-run wrapper bypasses the real environment action path."""

    from rlinf.envs.realworld.arx_x5_dual import arx_x5_dual_env as env_module

    monkeypatch.setattr(env_module.time, "sleep", lambda duration: None)
    env = ArxX5DualEnv.__new__(ArxX5DualEnv)
    env.config = SimpleNamespace(
        is_dummy=False, dry_run=True, step_frequency=30.0, max_num_steps=10
    )
    env._logger = _Logger()
    env._num_steps = 0
    env._left_controller = _FakeController("left", [])
    env._right_controller = _FakeController("right", [])
    env._read_robot_state = lambda: None
    env._get_observation = lambda: {}
    env._check_action_safety = lambda action: pytest.fail(
        "hardware dry-run must not call the robot action safety path"
    )

    action = np.zeros(14, dtype=np.float32)
    action[0] = 3.0
    _, _, _, _, info = env.step(action)

    assert info["action_sent"] is False
    assert env._left_controller.commands == []
    assert env._right_controller.commands == []


def test_reset_runs_smooth_go_start_every_time():
    """Every real-hardware reset invokes smooth_go_start."""

    env = ArxX5DualEnv.__new__(ArxX5DualEnv)
    env.config = SimpleNamespace(is_dummy=False)
    env._num_steps = 9
    calls = []
    env._smooth_go_start = lambda: calls.append("start")
    env._read_robot_state = lambda: calls.append("read")
    env._get_observation = lambda: {}

    env.reset()
    env.reset()

    assert calls == ["start", "read", "start", "read"]


def test_move_to_position_uses_synchronized_ease_in_out_quad(monkeypatch):
    """Both arms share the reference quadratic easing in one command loop."""

    from rlinf.envs.realworld.arx_x5_dual import arx_x5_dual_env as env_module

    clock = _Clock()
    monkeypatch.setattr(env_module.time, "sleep", clock.sleep)

    events = []
    env = _make_motion_env(events)
    target = ArxX5DualPositionConfig(
        left_joints=[0.8] * 6,
        right_joints=[-0.8] * 6,
        left_gripper=0.8,
        right_gripper=1.2,
        duration=1.0,
    )

    env._move_to_position(target, position_name="test position")

    assert events == [
        "left:hold",
        "right:hold",
        "left:position",
        "right:position",
    ]
    assert len(env._left_controller.commands) == 4
    assert len(env._right_controller.commands) == 4

    expected_alpha = np.array([0.125, 0.5, 0.875, 1.0])
    left_progress = np.array(
        [command[0] / 0.8 for command in env._left_controller.commands]
    )
    right_progress = np.array(
        [command[0] / -0.8 for command in env._right_controller.commands]
    )
    np.testing.assert_allclose(left_progress, expected_alpha)
    np.testing.assert_allclose(right_progress, expected_alpha)
    assert clock.now == pytest.approx(1.0)


def test_zero_duration_position_is_sent_directly(monkeypatch):
    """The reference trajectory helper sends non-positive durations directly."""

    from rlinf.envs.realworld.arx_x5_dual import arx_x5_dual_env as env_module

    monkeypatch.setattr(env_module.time, "sleep", lambda duration: None)
    env = _make_motion_env([])
    target = ArxX5DualPositionConfig(
        left_joints=[0.3] * 6,
        right_joints=[-0.3] * 6,
        left_gripper=0.4,
        right_gripper=0.5,
        duration=0.0,
    )

    env._move_to_position(target, position_name="direct")

    assert len(env._left_controller.commands) == 1
    assert len(env._right_controller.commands) == 1
    np.testing.assert_allclose(env._left_controller.commands[0], [0.3] * 6 + [0.4])
    np.testing.assert_allclose(env._right_controller.commands[0], [-0.3] * 6 + [0.5])


def test_smooth_home_switches_to_gravity_compensation():
    """Home trajectory ends in the same gravity-compensation mode as BiARX5."""

    events = []
    env = _make_motion_env(events)
    env.config.home_position = ArxX5DualPositionConfig(duration=None)
    env._move_to_position = lambda cfg, position_name: events.append(position_name)

    env._smooth_go_home()

    assert events == ["Home position", "left:gravity", "right:gravity"]


def test_close_homes_then_damps_closes_cameras_and_releases_arms(monkeypatch):
    """Shutdown follows BiARX5 disconnect ordering and remains idempotent."""

    from rlinf.envs.realworld.arx_x5_dual import arx_x5_dual_env as env_module

    monkeypatch.setattr(env_module.time, "sleep", lambda duration: None)
    events = []
    env = ArxX5DualEnv.__new__(ArxX5DualEnv)
    env._closed = False
    env.config = SimpleNamespace(is_dummy=False)
    env._logger = _Logger()
    env._smooth_go_home = lambda: events.append("home")
    env._left_controller = _FakeController("left", events)
    env._right_controller = _FakeController("right", events)
    env._cameras = [SimpleNamespace(close=lambda: events.append("camera:close"))]

    env.close()
    env.close()

    assert events == [
        "home",
        "left:damping",
        "right:damping",
        "camera:close",
        "left:close",
        "right:close",
    ]


def test_close_forces_damping_when_home_is_interrupted(monkeypatch):
    """Ctrl+C during shutdown still damps and releases both arms."""

    from rlinf.envs.realworld.arx_x5_dual import arx_x5_dual_env as env_module

    monkeypatch.setattr(env_module.time, "sleep", lambda duration: None)
    events = []
    env = ArxX5DualEnv.__new__(ArxX5DualEnv)
    env._closed = False
    env.config = SimpleNamespace(is_dummy=False)
    env._logger = _Logger()

    def _interrupt_home():
        events.append("home")
        raise KeyboardInterrupt

    env._smooth_go_home = _interrupt_home
    env._left_controller = _FakeController("left", events)
    env._right_controller = _FakeController("right", events)
    env._cameras = [SimpleNamespace(close=lambda: events.append("camera:close"))]

    env.close()

    assert events == [
        "home",
        "left:damping",
        "right:damping",
        "camera:close",
        "left:close",
        "right:close",
    ]
