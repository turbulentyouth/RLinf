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

"""ARX X5 双臂环境的无硬件单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("gymnasium")


def _import_arx_classes(monkeypatch):
    """导入 ARX 类，同时禁止 RealWorldEnv 的全局 ROS 清理副作用。

    Args:
        monkeypatch: pytest 提供的临时补丁工具。

    Returns:
        ``(ArxX5JointController, ArxX5DualEnv)`` 两个待测试类。

    Notes:
        ``rlinf.envs.realworld`` 在首次导入时会扫描并清理 ROS 进程。单元测试
        不需要这个节点级初始化，所以在导入前把进程列表替换为空，避免开发者
        在机器人电脑上运行测试时影响已有 ROS 进程。
    """

    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "process_iter", lambda: [])

    from rlinf.envs.realworld.arx_x5_dual.arx_x5_dual_controller import (
        ArxX5JointController,
    )
    from rlinf.envs.realworld.arx_x5_dual.arx_x5_dual_env import ArxX5DualEnv

    return ArxX5JointController, ArxX5DualEnv


class _FakeJointState:
    """模拟 ARX SDK 的 JointState，保存关节和夹爪字段。"""

    def __init__(self, joint_dof: int):
        self.timestamp = 1.0
        self._position = np.zeros(joint_dof, dtype=np.float64)
        self._velocity = np.zeros(joint_dof, dtype=np.float64)
        self._torque = np.zeros(joint_dof, dtype=np.float64)
        self.gripper_pos = 0.0
        self.gripper_vel = 0.0
        self.gripper_torque = 0.0

    def pos(self):
        """返回可写的关节位置数组，模拟 pybind 引用。"""

        return self._position

    def vel(self):
        """返回关节速度数组。"""

        return self._velocity

    def torque(self):
        """返回关节力矩数组。"""

        return self._torque


class _FakeRobotConfigFactory:
    """为每个测试控制器生成独立的 X5 配置。"""

    @classmethod
    def get_instance(cls):
        """模拟 SDK 单例工厂接口。"""

        return cls()

    def get_config(self, model: str):
        """返回测试需要的六关节范围、速度上限和夹爪宽度。"""

        assert model == "X5"
        return SimpleNamespace(
            joint_dof=6,
            joint_pos_min=np.full(6, -2.0, dtype=np.float64),
            joint_pos_max=np.full(6, 2.0, dtype=np.float64),
            joint_vel_max=np.ones(6, dtype=np.float64),
            gripper_width=0.088,
        )


class _FakeControllerConfigFactory:
    """生成 SDK 控制器线程配置。"""

    @classmethod
    def get_instance(cls):
        """模拟 SDK 单例工厂接口。"""

        return cls()

    def get_config(self, controller_type: str, joint_dof: int):
        """返回可由被测控制器修改的配置对象。"""

        assert controller_type == "joint_controller"
        assert joint_dof == 6
        return SimpleNamespace(
            controller_dt=0.002,
            default_preview_time=0.1,
            background_send_recv=False,
            shutdown_to_passive=False,
        )


class _FakeArx5JointController:
    """记录被测封装最终发送给 SDK 的命令。"""

    instances = []

    def __init__(self, robot_config, controller_config, interface_name):
        self.robot_config = robot_config
        self.controller_config = controller_config
        self.interface_name = interface_name
        self.state = _FakeJointState(robot_config.joint_dof)
        self.last_command = None
        self.damping_enabled = False
        self.__class__.instances.append(self)

    def get_joint_state(self):
        """返回当前模拟关节反馈。"""

        return self.state

    def get_eef_state(self):
        """返回当前模拟末端位姿反馈。"""

        return SimpleNamespace(pose_6d=lambda: np.zeros(6, dtype=np.float64))

    def set_joint_cmd(self, command):
        """保存绝对关节命令，供断言检查。"""

        self.last_command = command

    def set_to_damping(self):
        """记录控制器已经进入阻尼模式。"""

        self.damping_enabled = True


class _FakeCamera:
    """返回固定 BGR 图像，用于验证三路相机到 RGB observation 的转换。"""

    def __init__(self, name: str, bgr_color: tuple[int, int, int]):
        self.name = name
        self._frame = np.zeros((6, 8, 3), dtype=np.uint8)
        self._frame[...] = bgr_color
        self.closed = False

    def get_frame(self, timeout: float):
        """忽略超时时间并返回一张独立的固定图像。"""

        assert timeout > 0
        return self._frame.copy()

    def close(self):
        """记录相机资源已关闭。"""

        self.closed = True


def _fake_sdk():
    """组装与当前封装所用 API 一致的最小 ARX SDK。"""

    _FakeArx5JointController.instances.clear()
    return SimpleNamespace(
        RobotConfigFactory=_FakeRobotConfigFactory,
        ControllerConfigFactory=_FakeControllerConfigFactory,
        Arx5JointController=_FakeArx5JointController,
        JointState=_FakeJointState,
    )


def test_joint_controller_sends_absolute_joint_command(monkeypatch):
    """确认 6 关节和夹爪绝对位置原样写入 ARX JointState。"""

    ArxX5JointController, _ = _import_arx_classes(monkeypatch)
    controller = ArxX5JointController(
        model="X5",
        interface_name="can0",
        joint_velocity_limit_scale=0.2,
        sdk_module=_fake_sdk(),
    )

    target = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    controller.send_absolute_joint_position(target, 0.04)
    sdk_controller = _FakeArx5JointController.instances[-1]

    np.testing.assert_allclose(sdk_controller.last_command.pos(), target)
    assert sdk_controller.last_command.gripper_pos == pytest.approx(0.04)
    np.testing.assert_allclose(
        sdk_controller.robot_config.joint_vel_max,
        np.full(6, 0.2),
    )
    state = controller.read_state()
    assert state.joint_position.shape == (6,)
    assert state.eef_pose_6d.shape == (6,)

    with pytest.raises(ValueError, match="超出 ARX SDK 硬件范围"):
        controller.send_absolute_joint_position(np.full(6, 3.0), 0.04)

    controller.close()
    assert sdk_controller.damping_enabled


def test_dummy_env_returns_three_images_and_executes_14d_action(monkeypatch):
    """确认三路图像、14 维状态和 14 维绝对动作组成完整闭环。"""

    _, ArxX5DualEnv = _import_arx_classes(monkeypatch)
    env = ArxX5DualEnv(
        override_cfg={
            "is_dummy": True,
            "image_height": 8,
            "image_width": 10,
            "step_frequency": 1_000_000.0,
            "max_joint_delta_per_step": 0.1,
            "max_num_steps": 2,
            "task_description": "把物体放入盒中",
        }
    )

    observation, info = env.reset()
    assert info == {}
    assert env.task_description == "把物体放入盒中"
    assert set(observation["frames"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    for image in observation["frames"].values():
        assert image.shape == (8, 10, 3)
        assert image.dtype == np.uint8
    assert observation["state"]["joint_position"].shape == (14,)
    assert env.observation_space.contains(observation)

    requested = np.array(
        [0.5] * 6 + [0.04] + [0.5] * 6 + [0.05],
        dtype=np.float64,
    )
    observation, reward, terminated, truncated, info = env.step(requested)

    expected_state = np.array(
        [0.1] * 6 + [0.04] + [0.1] * 6 + [0.05],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        observation["state"]["joint_position"], expected_state
    )
    np.testing.assert_allclose(info["requested_action"], requested)
    np.testing.assert_allclose(info["executed_action"], expected_state)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert env.action_space.contains(expected_state)

    env.close()


def test_camera_frames_are_mapped_to_three_rgb_views(monkeypatch):
    """确认头部、左腕、右腕图像保留名称，并从 BGR 转成 RGB。"""

    _, ArxX5DualEnv = _import_arx_classes(monkeypatch)
    env = ArxX5DualEnv(
        override_cfg={
            "is_dummy": True,
            "image_height": 4,
            "image_width": 4,
        }
    )
    env._cameras = [
        _FakeCamera("base_0_rgb", (10, 20, 30)),
        _FakeCamera("left_wrist_0_rgb", (40, 50, 60)),
        _FakeCamera("right_wrist_0_rgb", (70, 80, 90)),
    ]

    frames = env._read_camera_frames()

    assert set(frames) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    np.testing.assert_array_equal(frames["base_0_rgb"][0, 0], [30, 20, 10])
    np.testing.assert_array_equal(
        frames["left_wrist_0_rgb"][0, 0], [60, 50, 40]
    )
    np.testing.assert_array_equal(
        frames["right_wrist_0_rgb"][0, 0], [90, 80, 70]
    )
    assert all(frame.shape == (4, 4, 3) for frame in frames.values())

    env.close()
