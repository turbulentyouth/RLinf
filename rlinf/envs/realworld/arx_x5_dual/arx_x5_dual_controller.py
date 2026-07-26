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

"""ARX X5 绝对关节位置控制器封装。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.utils.logging import get_logger

from .arx_x5_dual_robot_state import ArxX5ArmState


def _ensure_ament_prefix_path() -> None:
    """确保 ARX5 conda 环境在 ``AMENT_PREFIX_PATH`` 中。

    ``pyarx`` 在导入时会通过 ``ament_index_python`` 定位 ROS 2 资源。如果
    运行前没有手动 source conda 环境，导入会失败。测试脚本
    ``toolkits/realworld_check/test_arx_x5_dual.py`` 采用同样的修复方式：把
    ``REPO_ROOT/.venv/arx5-conda-env`` 添加到 ``AMENT_PREFIX_PATH`` 头部。
    """

    repo_root = Path(__file__).resolve().parents[4]
    arx5_prefix = str(repo_root / ".venv" / "arx5-conda-env")
    entries = [
        path
        for path in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
        if path
    ]
    if arx5_prefix not in entries:
        os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join([arx5_prefix, *entries])


class ArxX5JointController:
    """把 ``pyarx`` 封装成 RLinf 环境可调用的单臂控制接口。

    ARX SDK 自己会在 C++ 后台线程中以高频率进行 CAN 收发。RLinf 环境只需要
    以较低频率更新绝对关节目标。每次更新的目标会被写入 SDK 插值器，然后由
    SDK 后台控制线程连续发送给机械臂。

    这个类不负责左右臂动作的拼接。一个实例只连接一条机械臂；双臂环境会创建
    两个实例，并在同一个 ``env.step()`` 中依次更新左右臂目标。

    控制结构参考 ``toolkits/realworld_check/test_arx_x5_dual.py``：500 Hz SDK
    后台通信、50 Hz 命令插值、30 ms 命令预览。控制器构造后处于阻尼/被动模式；
    环境必须调用 ``enable_position_control`` 按参考比例设置位置环增益
    （``kp_scale=0.5``、``kd_scale=1.5``）并写入当前位置作为插值起点。

    Args:
        model: ARX 机器人型号。当前任务使用 ``"X5"``。型号填写错误可能导致
            危险运动，因此不会在代码中自动猜测型号。
        interface_name: 当前机械臂使用的 Linux CAN 接口，例如 ``"can0"``。
        controller_dt: SDK 后台控制周期，单位为秒。ARX 官方默认值是
            ``0.002``，也就是 500 Hz。
        preview_time: 新关节目标的插值预览时间，单位为秒。大于零时，SDK 会
            在当前命令和新目标之间插值，避免直接跳到目标位置。参考项目使用
            ``0.03``。
        joint_velocity_limit_scale: 对 SDK 原始关节速度上限进行缩放。比如
            ``0.2`` 表示只允许使用官方速度上限的 20%。SDK 官方建议在部署
            初期主动降低速度限制。
        sdk_module: 仅供无硬件单元测试注入假的 ``pyarx`` 模块。正常运行时
            必须保持为 ``None``，届时函数会延迟导入真实 SDK。

    Raises:
        ValueError: 参数非法，或者 SDK 返回的机械臂不是 6 自由度。
        ModuleNotFoundError: 机器人节点没有安装 ``pyarx``。
    """

    JOINT_DOF = 6

    @staticmethod
    def _copy_finite_vector(
        name: str,
        value: Any,
        expected_dim: int,
    ) -> np.ndarray:
        """复制并检查 ARX SDK 返回的一维向量。

        Args:
            name: 字段名称，只用于生成明确的异常信息。
            value: SDK 返回的数组或可转换成数组的对象。
            expected_dim: 该字段应该包含的元素数量。

        Returns:
            形状为 ``(expected_dim,)`` 的独立 ``float64`` 数组。返回副本可以
            避免 SDK 后台线程更新内部缓冲区时改变 Env 已经保存的状态。

        Raises:
            RuntimeError: SDK 返回的元素数量不对，或者包含 NaN/Inf。
        """

        vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
        if vector.shape != (expected_dim,):
            raise RuntimeError(
                f"ARX SDK 字段 {name!r} 应为 ({expected_dim},)，实际为 {vector.shape}。"
            )
        if not np.all(np.isfinite(vector)):
            raise RuntimeError(f"ARX SDK 字段 {name!r} 包含 NaN 或 Inf。")
        return vector

    def __init__(
        self,
        model: str,
        interface_name: str,
        controller_dt: float = 0.002,
        preview_time: float = 0.03,
        joint_velocity_limit_scale: float = 0.2,
        sdk_module: Any | None = None,
    ) -> None:
        if controller_dt <= 0:
            raise ValueError("controller_dt 必须大于 0。")
        if preview_time <= 0:
            raise ValueError("preview_time 必须大于 0，避免绝对关节目标瞬间跳变。")
        if not 0 < joint_velocity_limit_scale <= 1:
            raise ValueError("joint_velocity_limit_scale 必须位于 (0, 1]。")

        self._logger = get_logger()
        self.model = model
        self.interface_name = interface_name
        self._closed = False

        if sdk_module is None:
            # 延迟导入非常重要：GPU 服务器不需要安装 ARX SDK。只有被 Ray
            # 放置到机器人电脑上的 Env Worker 创建该对象时才导入硬件依赖。
            _ensure_ament_prefix_path()
            import pyarx as sdk_module

        self._arx5 = sdk_module
        robot_config = sdk_module.RobotConfigFactory.get_instance().get_config(model)
        if int(robot_config.joint_dof) != self.JOINT_DOF:
            raise ValueError(
                f"ARX 型号 {model!r} 返回 {robot_config.joint_dof} 个关节，"
                f"但当前双臂 π0.5 接口固定要求每臂 {self.JOINT_DOF} 个关节。"
            )

        controller_config = (
            sdk_module.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", robot_config.joint_dof
            )
        )

        # 官方文档要求替换整个 numpy 数组，而不是对 joint_vel_max 的单个
        # 元素原地赋值。缩小速度上限可以降低早期模型输出异常时的风险。
        robot_config.joint_vel_max = (
            np.asarray(robot_config.joint_vel_max, dtype=np.float64)
            * joint_velocity_limit_scale
        )
        controller_config.controller_dt = float(controller_dt)
        controller_config.default_preview_time = float(preview_time)
        controller_config.background_send_recv = True
        controller_config.shutdown_to_passive = True

        self._robot_config = robot_config
        self._controller_config = controller_config
        self._controller = sdk_module.Arx5JointController(
            robot_config,
            controller_config,
            interface_name,
        )

    def enable_position_control(
        self, kp_scale: float = 0.5, kd_scale: float = 1.5
    ) -> None:
        """按参考流程启用位置控制。

        与 ``toolkits/realworld_check/test_arx_x5_dual.py`` 保持一致：

        1. 读取当前关节/夹爪状态；
        2. 把当前位置写入 SDK 命令缓冲区，让控制器从当前姿态开始插值；
        3. 等待 ``max(preview_time, controller_dt * 2)``，让后台线程收到命令；
        4. 按 ``kp_scale`` / ``kd_scale`` 设置位置环增益。

        控制器构造后默认处于阻尼/被动模式， Env 必须在发送第一条动作前调用
        本函数，否则首次 ``set_joint_cmd`` 可能从零命令跳变到目标位置。

        Args:
            kp_scale: 对 SDK 默认关节 ``kp`` 的缩放比例。
            kd_scale: 对 SDK 默认关节 ``kd`` 的缩放比例。
        """

        if self._closed:
            raise RuntimeError(f"ARX 控制器 {self.interface_name!r} 已经关闭。")
        if kp_scale <= 0:
            raise ValueError("kp_scale 必须大于 0。")
        if kd_scale <= 0:
            raise ValueError("kd_scale 必须大于 0。")

        state = self._controller.get_joint_state()
        command = self._arx5.JointState(self.JOINT_DOF)
        command.pos()[:] = self._copy_finite_vector(
            "current_joint_position", state.pos(), self.JOINT_DOF
        )
        command.gripper_pos = float(
            self._copy_finite_vector(
                "current_gripper_position", [state.gripper_pos], 1
            )[0]
        )
        self._controller.set_joint_cmd(command)
        time.sleep(
            max(
                self._controller_config.default_preview_time,
                self._controller_config.controller_dt * 2,
            )
        )

        config = self._controller.get_controller_config()
        gain = self._arx5.Gain(self.JOINT_DOF)
        gain.kp()[:] = np.asarray(config.default_kp, dtype=np.float64) * kp_scale
        gain.kd()[:] = np.asarray(config.default_kd, dtype=np.float64) * kd_scale
        gain.gripper_kp = config.default_gripper_kp
        gain.gripper_kd = config.default_gripper_kd
        self._controller.set_gain(gain)

    @property
    def joint_position_low(self) -> np.ndarray:
        """返回 SDK 配置中的 6 维关节位置下限副本。"""

        return np.asarray(self._robot_config.joint_pos_min, dtype=np.float64).copy()

    @property
    def joint_position_high(self) -> np.ndarray:
        """返回 SDK 配置中的 6 维关节位置上限副本。"""

        return np.asarray(self._robot_config.joint_pos_max, dtype=np.float64).copy()

    @property
    def gripper_position_low(self) -> float:
        """返回绝对夹爪命令的最小值。

        ARX ``JointState.gripper_pos`` 使用逻辑夹爪宽度，闭合端为 0。
        ``gripper_open_readout`` 是底层电机读数方向，不应直接作为策略动作。
        """

        return 0.0

    @property
    def gripper_position_high(self) -> float:
        """返回绝对夹爪命令的最大值，也就是机械臂配置中的夹爪宽度。"""

        return float(self._robot_config.gripper_width)

    @property
    def joint_velocity_limit(self) -> np.ndarray:
        """返回已经应用安全缩放后的六关节速度上限。"""

        return np.asarray(self._robot_config.joint_vel_max, dtype=np.float64).copy()

    @property
    def gripper_velocity_limit(self) -> float:
        """返回 SDK 配置中的夹爪速度上限。"""

        return float(self._robot_config.gripper_vel_max)

    def read_state(self) -> ArxX5ArmState:
        """读取当前关节、夹爪和末端位姿反馈。

        SDK 已开启后台收发线程，因此这里不能再调用 ``recv_once()``。函数只
        读取后台线程维护的最新状态，并立即复制 numpy 数组，避免下一次 SDK
        更新覆盖调用方仍在使用的内存。

        Returns:
            当前单臂状态。关节数组均为独立副本，可以安全地保存在环境中。
        """

        if self._closed:
            raise RuntimeError(f"ARX 控制器 {self.interface_name!r} 已经关闭。")

        joint_state = self._controller.get_joint_state()
        eef_state = self._controller.get_eef_state()
        timestamp = float(joint_state.timestamp)
        controller_timestamp = float(self._controller.get_timestamp())
        gripper_position = float(joint_state.gripper_pos)
        gripper_velocity = float(joint_state.gripper_vel)
        gripper_torque = float(joint_state.gripper_torque)
        scalar_values = np.asarray(
            [
                timestamp,
                controller_timestamp,
                gripper_position,
                gripper_velocity,
                gripper_torque,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalar_values)):
            raise RuntimeError("ARX SDK 返回的时间戳或夹爪状态包含 NaN/Inf。")

        return ArxX5ArmState(
            timestamp=timestamp,
            controller_timestamp=controller_timestamp,
            joint_position=self._copy_finite_vector(
                "joint_position", joint_state.pos(), self.JOINT_DOF
            ),
            joint_velocity=self._copy_finite_vector(
                "joint_velocity", joint_state.vel(), self.JOINT_DOF
            ),
            joint_torque=self._copy_finite_vector(
                "joint_torque", joint_state.torque(), self.JOINT_DOF
            ),
            gripper_position=gripper_position,
            gripper_velocity=gripper_velocity,
            gripper_torque=gripper_torque,
            eef_pose_6d=self._copy_finite_vector("eef_pose_6d", eef_state.pose_6d(), 6),
        )

    def send_absolute_joint_position(
        self,
        joint_position: np.ndarray,
        gripper_position: float,
    ) -> None:
        """向一条机械臂写入绝对关节位置和绝对夹爪位置。

        Args:
            joint_position: 形状为 ``(6,)`` 的绝对关节角，单位为弧度。
                这些数值已经由 Env 做过安全边界缩进和单步变化限制；本函数
                仍会再次检查 SDK 原始硬件范围，作为最后一道防线。
            gripper_position: 绝对夹爪宽度，合法范围是
                ``[0, robot_config.gripper_width]``。

        Effects:
            构造 ARX SDK 的 ``JointState`` 命令并调用 ``set_joint_cmd``。
            该调用只更新 SDK 内部目标；真正的 CAN 下发由 SDK 后台线程完成。

        Raises:
            ValueError: 输入形状错误、包含 NaN/Inf 或超出硬件范围。
            RuntimeError: 控制器已经关闭。
        """

        if self._closed:
            raise RuntimeError(f"ARX 控制器 {self.interface_name!r} 已经关闭。")

        joint_position = np.asarray(joint_position, dtype=np.float64)
        if joint_position.shape != (self.JOINT_DOF,):
            raise ValueError(
                f"joint_position 必须是 ({self.JOINT_DOF},)，"
                f"实际收到 {joint_position.shape}。"
            )
        if not np.all(np.isfinite(joint_position)) or not np.isfinite(gripper_position):
            raise ValueError("绝对关节命令不能包含 NaN 或 Inf。")

        low = self.joint_position_low
        high = self.joint_position_high
        if np.any(joint_position < low) or np.any(joint_position > high):
            raise ValueError(
                "绝对关节命令超出 ARX SDK 硬件范围："
                f"command={joint_position}, low={low}, high={high}。"
            )
        if not (
            self.gripper_position_low <= gripper_position <= self.gripper_position_high
        ):
            raise ValueError(
                "绝对夹爪命令超出范围："
                f"command={gripper_position}, range="
                f"[{self.gripper_position_low}, {self.gripper_position_high}]。"
            )

        command = self._arx5.JointState(self.JOINT_DOF)
        command.pos()[:] = joint_position
        command.gripper_pos = float(gripper_position)
        self._controller.set_joint_cmd(command)

    def set_to_damping(self) -> None:
        """请求 SDK 将机械臂切换到阻尼模式。

        当左右臂任一命令发送失败时，Env 会对两条机械臂都调用这个函数，避免
        一条手臂继续执行旧目标、另一条手臂已经停止的不对称状态。
        """

        if not self._closed:
            self._controller.set_to_damping()

    def close(self) -> None:
        """安全关闭控制器，并让底层对象结束后台通信线程。

        函数可以重复调用。第一次调用会先切换阻尼模式，然后释放 pybind
        控制器对象；SDK 配置的 ``shutdown_to_passive=True`` 会在析构阶段继续
        保证机械臂进入被动状态。
        """

        if self._closed:
            return
        try:
            self._controller.set_to_damping()
        except Exception as exc:  # 硬件断线时也必须继续释放对象。
            self._logger.warning(
                "关闭 ARX 控制器 %s 时切换阻尼模式失败：%s",
                self.interface_name,
                exc,
            )
        finally:
            self._closed = True
            self._controller = None
