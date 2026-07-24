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

"""连接 ARX X5 双臂、三路相机和 π0.5 的 Gym 真机环境。"""

from __future__ import annotations

import copy
import queue
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .arx_x5_dual_controller import ArxX5JointController
from .arx_x5_dual_robot_state import ArxX5ArmState

# ARX 官方 X5 配置中的关节范围。真实运行时会重新从 SDK 读取范围；这些值只在
# is_dummy=True 的无硬件测试中使用，使 dummy 环境和真机环境拥有相同动作语义。
_X5_JOINT_POSITION_LOW = np.array(
    [-3.14, -0.05, -0.1, -1.6, -1.57, -2.0], dtype=np.float64
)
_X5_JOINT_POSITION_HIGH = np.array(
    [2.618, 3.5, 3.2, 1.55, 1.57, 2.0], dtype=np.float64
)
_X5_GRIPPER_WIDTH = 0.088

# 三个名称同时承担两层职责：
# 1. 它们是 Env 原始 observation["frames"] 中的键；
# 2. OpenPI 适配器会把它们分别放进 base / left wrist / right wrist 图像槽位。
_HEAD_CAMERA_NAME = "base_0_rgb"
_LEFT_WRIST_CAMERA_NAME = "left_wrist_0_rgb"
_RIGHT_WRIST_CAMERA_NAME = "right_wrist_0_rgb"
_CAMERA_NAMES = (
    _HEAD_CAMERA_NAME,
    _LEFT_WRIST_CAMERA_NAME,
    _RIGHT_WRIST_CAMERA_NAME,
)


@dataclass
class ArxX5DualRobotConfig:
    """ARX X5 双臂绝对关节位置环境配置。

    这个配置只描述当前第一版闭环需要的内容：两条机械臂、三个 RGB 相机、
    14 维绝对关节动作以及基本安全限制。遥操作、自动回零和任务奖励不属于
    当前版本，后续需要时再独立增加，避免把动作链路和任务逻辑混在一起。

    Attributes:
        robot_model: 左右臂共同使用的 ARX 型号，当前应为 ``"X5"``。
        left_interface: 左臂 Linux CAN 接口，例如 ``"can0"``。
        right_interface: 右臂 Linux CAN 接口，例如 ``"can1"``。
        head_camera_serial: 头部相机序列号。该相机作为 π0.5 主视角。
        left_wrist_camera_serial: 左腕相机序列号。
        right_wrist_camera_serial: 右腕相机序列号。
        camera_type: 三路相机共同使用的 RLinf 相机后端，默认
            ``"realsense"``。如果三路相机型号不同，可以分别覆盖下面三个字段。
        head_camera_type: 头部相机后端；为空时回退到 ``camera_type``。
        left_wrist_camera_type: 左腕相机后端；为空时回退到 ``camera_type``。
        right_wrist_camera_type: 右腕相机后端；为空时回退到 ``camera_type``。
        image_height: 返回给 π0.5 的图像高度。
        image_width: 返回给 π0.5 的图像宽度。
        camera_frame_timeout: 单次等待相机帧的最长时间，单位为秒。
        controller_dt: ARX SDK 后台 CAN 控制周期，默认 0.002 秒（500 Hz）。
        controller_warmup_time: 创建控制器后等待首批真实反馈的时间，单位为秒。
        command_preview_time: SDK 对新绝对关节目标做插值的预览时间。
        joint_velocity_limit_scale: SDK 原始关节速度上限缩放比例。
        joint_limit_margin_ratio: 从关节硬限位两端各缩进的比例。例如 0.05
            表示只使用中间 90% 的关节范围。
        max_joint_delta_per_step: 每个 Env step 允许的最大关节变化，单位弧度。
            π0.5 输出仍然是绝对位置；这里仅将过大的单步变化截断成安全目标。
        step_frequency: RLinf Env 的目标执行频率。它低于 SDK 后台控制频率。
        max_num_steps: 一条 rollout 最多执行多少个 Env step。
        task_description: 发送给 π0.5 的语言任务指令。
        is_dummy: 为 True 时不连接机械臂和相机，使用零图像和内存状态测试链路。
    """

    robot_model: str = "X5"
    left_interface: str = "can0"
    right_interface: str = "can1"

    head_camera_serial: Optional[str] = None
    left_wrist_camera_serial: Optional[str] = None
    right_wrist_camera_serial: Optional[str] = None
    camera_type: str = "realsense"
    head_camera_type: Optional[str] = None
    left_wrist_camera_type: Optional[str] = None
    right_wrist_camera_type: Optional[str] = None
    image_height: int = 224
    image_width: int = 224
    camera_frame_timeout: float = 0.5

    controller_dt: float = 0.002
    controller_warmup_time: float = 0.5
    command_preview_time: float = 0.1
    joint_velocity_limit_scale: float = 0.2
    joint_limit_margin_ratio: float = 0.05
    max_joint_delta_per_step: float = 0.1
    step_frequency: float = 10.0
    max_num_steps: int = 300
    task_description: str = ""
    is_dummy: bool = False

    def __post_init__(self) -> None:
        """检查配置是否能形成明确且安全的双臂控制协议。"""

        if self.left_interface == self.right_interface:
            raise ValueError("左右机械臂必须使用不同的 CAN 接口。")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height 和 image_width 必须大于 0。")
        if self.camera_frame_timeout <= 0:
            raise ValueError("camera_frame_timeout 必须大于 0。")
        if self.controller_dt <= 0:
            raise ValueError("controller_dt 必须大于 0。")
        if self.controller_warmup_time < 0:
            raise ValueError("controller_warmup_time 不能小于 0。")
        if self.command_preview_time <= 0:
            raise ValueError("command_preview_time 必须大于 0。")
        if not 0 < self.joint_velocity_limit_scale <= 1:
            raise ValueError("joint_velocity_limit_scale 必须位于 (0, 1]。")
        if not 0 <= self.joint_limit_margin_ratio < 0.5:
            raise ValueError("joint_limit_margin_ratio 必须位于 [0, 0.5)。")
        if self.max_joint_delta_per_step <= 0:
            raise ValueError("max_joint_delta_per_step 必须大于 0。")
        if self.step_frequency <= 0:
            raise ValueError("step_frequency 必须大于 0。")
        if self.max_num_steps <= 0:
            raise ValueError("max_num_steps 必须大于 0。")

        if not self.is_dummy:
            missing = [
                name
                for name, serial in (
                    ("head_camera_serial", self.head_camera_serial),
                    ("left_wrist_camera_serial", self.left_wrist_camera_serial),
                    ("right_wrist_camera_serial", self.right_wrist_camera_serial),
                )
                if not serial
            ]
            if missing:
                raise ValueError(
                    "真实 ARX π0.5 环境要求配置头部、左腕和右腕三路相机。"
                    f"缺少字段：{missing}。"
                )


class ArxX5DualEnv(gym.Env):
    """ARX X5 双臂绝对关节位置真机环境。

    数据流如下：

    ``三路 RGB 图像 + 14 维当前状态 -> observation -> π0.5``

    ``π0.5 的 14 维绝对动作 -> step(action) -> ARX set_joint_cmd -> 双臂``

    14 维状态和动作使用完全相同的顺序：

    ``[左 q1..q6, 左夹爪, 右 q1..q6, 右夹爪]``。

    这种一致性非常重要。SFT 数据、归一化统计、π0.5 输出转换和真机 Env
    只要有一处顺序不同，就可能把某个关节目标发送给错误的机械臂。
    """

    metadata = {"render_modes": []}
    ACTION_DIM = 14
    JOINT_DOF_PER_ARM = 6

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo] = None,
        hardware_info: Any = None,
        env_idx: int = 0,
    ) -> None:
        """创建双臂控制器、三路相机和 Gym 空间。

        Args:
            override_cfg: YAML 中 ``env.*.override_cfg`` 解析后的字典。
            worker_info: RLinf Env Worker 信息。当前第一版不额外启动控制器
                Worker，但保留参数以符合 ``RealWorldEnv`` 的 Gym 工厂接口。
            hardware_info: 调度器硬件信息。当前 CAN 接口和相机序列号直接从
                YAML 读取，因此该参数暂未使用。
            env_idx: 当前环境编号，用于日志定位。

        Effects:
            真实模式下会立即创建两个 ARX SDK 后台线程并打开三路相机；dummy
            模式下只创建内存状态和零图像，不触碰任何硬件。
        """

        del hardware_info
        self.config = ArxX5DualRobotConfig(**override_cfg)
        self.env_idx = env_idx
        self._logger = get_logger()
        self._task_description = self.config.task_description
        self._num_steps = 0
        self.node_rank = worker_info.cluster_node_rank if worker_info else 0
        self.worker_rank = worker_info.rank if worker_info else 0
        self._closed = False

        self._left_state = ArxX5ArmState()
        self._right_state = ArxX5ArmState()
        self._last_camera_frame: dict[str, np.ndarray] = {}
        self._cameras: list[BaseCamera] = []

        if self.config.is_dummy:
            self._set_dummy_limits()
        else:
            self._setup_hardware()

        self._init_action_observation_spaces()

        if not self.config.is_dummy:
            self._open_cameras()
            self._read_robot_state()

    @property
    def task_description(self) -> str:
        """返回发送给 π0.5 和写入 LeRobot 数据集的语言任务指令。"""

        return self._task_description

    def _set_dummy_limits(self) -> None:
        """为无硬件测试设置与 X5 真机一致的默认动作范围。"""

        self._joint_position_low = np.stack([_X5_JOINT_POSITION_LOW] * 2)
        self._joint_position_high = np.stack([_X5_JOINT_POSITION_HIGH] * 2)
        self._gripper_position_low = np.zeros(2, dtype=np.float64)
        self._gripper_position_high = np.full(
            2, _X5_GRIPPER_WIDTH, dtype=np.float64
        )

    def _setup_hardware(self) -> None:
        """连接左右机械臂，并从 SDK 获取真实关节和夹爪范围。

        关节安全边界不是简单地将上下限乘一个系数，而是从完整范围的两端各
        缩进 ``joint_limit_margin_ratio``。这种算法对负数下限和正数上限都成立。
        """

        self._left_controller = ArxX5JointController(
            model=self.config.robot_model,
            interface_name=self.config.left_interface,
            controller_dt=self.config.controller_dt,
            preview_time=self.config.command_preview_time,
            joint_velocity_limit_scale=self.config.joint_velocity_limit_scale,
        )
        try:
            self._right_controller = ArxX5JointController(
                model=self.config.robot_model,
                interface_name=self.config.right_interface,
                controller_dt=self.config.controller_dt,
                preview_time=self.config.command_preview_time,
                joint_velocity_limit_scale=self.config.joint_velocity_limit_scale,
            )
        except Exception:
            self._left_controller.close()
            raise

        raw_low = np.stack(
            [
                self._left_controller.joint_position_low,
                self._right_controller.joint_position_low,
            ]
        )
        raw_high = np.stack(
            [
                self._left_controller.joint_position_high,
                self._right_controller.joint_position_high,
            ]
        )
        margin = (raw_high - raw_low) * self.config.joint_limit_margin_ratio
        self._joint_position_low = raw_low + margin
        self._joint_position_high = raw_high - margin
        self._gripper_position_low = np.array(
            [
                self._left_controller.gripper_position_low,
                self._right_controller.gripper_position_low,
            ],
            dtype=np.float64,
        )
        self._gripper_position_high = np.array(
            [
                self._left_controller.gripper_position_high,
                self._right_controller.gripper_position_high,
            ],
            dtype=np.float64,
        )

        # ARX SDK 的 CAN 收发运行在后台线程中。显式等待首批反馈，避免 Env
        # 把控制器刚创建时的默认零数组误当成机械臂真实起始关节位置。
        time.sleep(self.config.controller_warmup_time)

    def _init_action_observation_spaces(self) -> None:
        """定义 π0.5 动作与 Env observation 的固定形状。

        Effects:
            ``action_space`` 被设置为 14 维绝对位置范围；``observation_space``
            包含一个 14 维状态和三张 ``HWC uint8`` RGB 图像。
        """

        action_low = np.concatenate(
            [
                self._joint_position_low[0],
                self._gripper_position_low[0:1],
                self._joint_position_low[1],
                self._gripper_position_low[1:2],
            ]
        ).astype(np.float32)
        action_high = np.concatenate(
            [
                self._joint_position_high[0],
                self._gripper_position_high[0:1],
                self._joint_position_high[1],
                self._gripper_position_high[1:2],
            ]
        ).astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)

        frame_space = {
            name: gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.config.image_height, self.config.image_width, 3),
                dtype=np.uint8,
            )
            for name in _CAMERA_NAMES
        }
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        # 名称使用 joint_position，但内容刻意包含两个夹爪，使
                        # RealWorldEnv 拼接后仍然保持唯一、明确的 14 维顺序。
                        "joint_position": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.ACTION_DIM,),
                            dtype=np.float32,
                        )
                    }
                ),
                "frames": gym.spaces.Dict(frame_space),
            }
        )

    def _camera_specs(self) -> list[CameraInfo]:
        """将三个物理相机序列号映射到固定的 π0.5 图像名称。"""

        default_type = self.config.camera_type
        return [
            CameraInfo(
                name=_HEAD_CAMERA_NAME,
                serial_number=str(self.config.head_camera_serial),
                camera_type=self.config.head_camera_type or default_type,
            ),
            CameraInfo(
                name=_LEFT_WRIST_CAMERA_NAME,
                serial_number=str(self.config.left_wrist_camera_serial),
                camera_type=self.config.left_wrist_camera_type or default_type,
            ),
            CameraInfo(
                name=_RIGHT_WRIST_CAMERA_NAME,
                serial_number=str(self.config.right_wrist_camera_serial),
                camera_type=self.config.right_wrist_camera_type or default_type,
            ),
        ]

    def _open_cameras(self) -> None:
        """打开头部、左腕和右腕三路相机。

        Effects:
            三个相机对象按 ``头部、左腕、右腕`` 顺序保存。若任意一路创建或
            启动失败，会关闭此前已经打开的相机和两个机械臂控制器，避免构造
            Env 失败后仍残留相机线程或 CAN 后台线程。

        Raises:
            Exception: 保留并重新抛出相机后端产生的原始异常。
        """

        try:
            for camera_info in self._camera_specs():
                camera = create_camera(camera_info)
                # 先保存再 open，使 open() 中途失败时 close() 也能释放设备。
                self._cameras.append(camera)
                camera.open()
        except Exception:
            self.close()
            raise

    def _read_camera_frames(self) -> dict[str, np.ndarray]:
        """同步读取三路相机，并转换成 π0.5 所需的 RGB 图像。

        Returns:
            包含 ``base_0_rgb``、``left_wrist_0_rgb`` 和
            ``right_wrist_0_rgb`` 的字典。每张图像形状为
            ``(image_height, image_width, 3)``，类型为 ``uint8``。

        Raises:
            RuntimeError: 某相机在产生第一张有效图像之前就超时。

        Notes:
            RLinf 相机后端输出 BGR；函数在 resize 后通过 ``[..., ::-1]``
            转成 RGB。如果运行中偶发超时，会使用该相机上一张成功图像，避免
            单路相机抖动直接中断整个 rollout。
        """

        frames: dict[str, np.ndarray] = {}
        for camera in self._cameras:
            name = camera.name
            try:
                frame = camera.get_frame(timeout=self.config.camera_frame_timeout)
                self._last_camera_frame[name] = frame
            except queue.Empty:
                frame = self._last_camera_frame.get(name)
                if frame is None:
                    raise RuntimeError(
                        f"相机 {name!r} 读取超时，且没有可回退的历史图像。"
                    )
                self._logger.warning("相机 %s 读取超时，本 step 使用上一帧。", name)

            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[-1] < 3:
                raise RuntimeError(
                    f"相机 {name!r} 必须返回 HWC 彩色图像，实际为 {frame.shape}。"
                )
            # 当前 π0.5 只使用 RGB；如果相机后端额外附带深度通道，仅取前三个
            # BGR 通道，避免 ``[..., ::-1]`` 把深度误当成颜色。
            frame = frame[..., :3]

            height, width = frame.shape[:2]
            crop_size = min(height, width)
            start_x = (width - crop_size) // 2
            start_y = (height - crop_size) // 2
            square = frame[
                start_y : start_y + crop_size,
                start_x : start_x + crop_size,
            ]
            resized = cv2.resize(
                square,
                (self.config.image_width, self.config.image_height),
                interpolation=cv2.INTER_AREA,
            )
            frames[name] = np.ascontiguousarray(resized[..., ::-1], dtype=np.uint8)
        return frames

    def _read_robot_state(self) -> None:
        """从两个 ARX SDK 后台控制器读取最新状态并缓存到 Env。"""

        self._left_state = self._left_controller.read_state()
        self._right_state = self._right_controller.read_state()

    def _compose_absolute_joint_state(self) -> np.ndarray:
        """按照和 action 完全一致的顺序生成 14 维当前状态。

        Returns:
            ``float32`` 数组：
            ``[左 q1..q6, 左夹爪, 右 q1..q6, 右夹爪]``。
        """

        return np.concatenate(
            [
                self._left_state.joint_position,
                np.array([self._left_state.gripper_position]),
                self._right_state.joint_position,
                np.array([self._right_state.gripper_position]),
            ]
        ).astype(np.float32)

    def _get_observation(self) -> dict[str, Any]:
        """生成 RLinf RealWorldEnv 能包装并发送给 π0.5 的 observation。

        Returns:
            字典包含：

            - ``state/joint_position``：14 维绝对关节和夹爪状态；
            - ``frames/base_0_rgb``：头部 RGB 图像；
            - ``frames/left_wrist_0_rgb``：左腕 RGB 图像；
            - ``frames/right_wrist_0_rgb``：右腕 RGB 图像。

        Effects:
            无硬件模式返回全零图像；真实模式每次调用都会读取三路最新图像。
            外层 ``RealWorldEnv._wrap_obs`` 会把头部图像变成 ``main_images``，
            两路腕部图像堆叠成 ``extra_view_images``，随后交给 OpenPI adapter。
        """

        if self.config.is_dummy:
            frames = {
                name: np.zeros(
                    (self.config.image_height, self.config.image_width, 3),
                    dtype=np.uint8,
                )
                for name in _CAMERA_NAMES
            }
        else:
            frames = self._read_camera_frames()

        observation = {
            "state": {"joint_position": self._compose_absolute_joint_state()},
            "frames": frames,
        }
        return copy.deepcopy(observation)

    def _limit_absolute_action(self, action: np.ndarray) -> np.ndarray:
        """对 π0.5 的绝对动作执行空间裁剪和单步变化限制。

        Args:
            action: 已确认形状为 ``(14,)`` 且数值有限的 π0.5 原始输出。

        Returns:
            仍然是绝对关节位置语义的 14 维动作。关节目标先被裁剪到安全动作
            空间，再限制为当前关节位置正负 ``max_joint_delta_per_step``。
            夹爪只做合法宽度裁剪，不转换为 delta。
        """

        limited = np.clip(action, self.action_space.low, self.action_space.high)
        current = self._compose_absolute_joint_state().astype(np.float64)
        max_delta = self.config.max_joint_delta_per_step
        for joint_slice in (slice(0, 6), slice(7, 13)):
            limited[joint_slice] = np.clip(
                limited[joint_slice],
                current[joint_slice] - max_delta,
                current[joint_slice] + max_delta,
            )
        return limited.astype(np.float64)

    def _send_absolute_action(self, action: np.ndarray) -> None:
        """把 14 维绝对动作拆成左右臂命令并交给 ARX SDK。

        Args:
            action: 经过安全限制后的 14 维绝对动作，顺序固定为
                ``[左6关节, 左夹爪, 右6关节, 右夹爪]``。

        Effects:
            分别调用左右控制器的 ``set_joint_cmd``。如果任意一条机械臂发送
            失败，立即尝试让两条机械臂都进入阻尼模式，然后把异常抛给上层。
        """

        try:
            self._left_controller.send_absolute_joint_position(action[:6], action[6])
            self._right_controller.send_absolute_joint_position(
                action[7:13], action[13]
            )
        except Exception:
            self._enter_damping_after_error()
            raise

    def _enter_damping_after_error(self) -> None:
        """在部分发送失败时尽力把左右臂同时切换到阻尼模式。"""

        for label, controller in (
            ("left", getattr(self, "_left_controller", None)),
            ("right", getattr(self, "_right_controller", None)),
        ):
            if controller is None:
                continue
            try:
                controller.set_to_damping()
            except Exception as exc:
                self._logger.error("%s 臂切换阻尼模式失败：%s", label, exc)

    def reset(self, *, seed=None, options=None):
        """开始一条新 rollout，但当前版本不会自动移动机械臂。

        Args:
            seed: Gym 标准随机种子。真机状态不是随机生成，因此当前未使用。
            options: Gym 标准 reset 选项。当前版本未定义额外选项。

        Returns:
            ``(observation, info)``。observation 含三路图像和当前 14 维状态，
            info 当前为空字典。

        Effects:
            清零 episode step 计数，并重新读取左右臂反馈。为了避免未经确认的
            自动运动，函数不会调用 SDK ``reset_to_home()``；需要操作员先把
            机械臂放到安全初始位姿，再启动 rollout。
        """

        del seed, options
        self._num_steps = 0
        if not self.config.is_dummy:
            self._read_robot_state()
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        """执行一次“π0.5 observation -> action -> ARX”的闭环。

        Args:
            action: π0.5 输出的 14 维绝对动作，顺序为
                ``[左 q1..q6, 左夹爪, 右 q1..q6, 右夹爪]``。

        Returns:
            Gymnasium 标准五元组：

            - ``observation``：执行后重新读取的三路 RGB 图像和 14 维状态；
            - ``reward``：当前动作桥接版本固定为 0.0；
            - ``terminated``：当前没有任务成功判断，固定为 False；
            - ``truncated``：达到 ``max_num_steps`` 时为 True；
            - ``info``：同时记录 π0.5 请求动作和经过安全限制的实际执行动作。

        Effects:
            真实模式下向左右 ARX 控制器下发绝对目标，等待本 Env 控制周期结束，
            然后读取新的机器人状态和三路图像。dummy 模式只更新内存状态。
            每一步都会在终端日志中输出模型原始动作和安全限制后的实际动作。

        Raises:
            ValueError: action 不是 ``(14,)``，或包含 NaN/Inf。
            RuntimeError: ARX SDK 下发或状态读取失败。
        """

        start_time = time.monotonic()
        requested_action = np.asarray(action, dtype=np.float64)
        if requested_action.shape != (self.ACTION_DIM,):
            raise ValueError(
                f"ARX 双臂 action 必须是 ({self.ACTION_DIM},)，"
                f"实际收到 {requested_action.shape}。"
            )
        if not np.all(np.isfinite(requested_action)):
            raise ValueError("π0.5 action 不能包含 NaN 或 Inf。")

        executed_action = self._limit_absolute_action(requested_action.copy())
        self._logger.info(
            "π0.5 生成的 14 维绝对动作：requested_action=%s, executed_action=%s",
            np.array2string(
                requested_action, precision=6, separator=", ", max_line_width=1000
            ),
            np.array2string(
                executed_action, precision=6, separator=", ", max_line_width=1000
            ),
        )
        if self.config.is_dummy:
            self._left_state.joint_position = executed_action[:6].copy()
            self._left_state.gripper_position = float(executed_action[6])
            self._right_state.joint_position = executed_action[7:13].copy()
            self._right_state.gripper_position = float(executed_action[13])
        else:
            self._send_absolute_action(executed_action)

        elapsed = time.monotonic() - start_time
        time.sleep(max(0.0, 1.0 / self.config.step_frequency - elapsed))

        if not self.config.is_dummy:
            try:
                self._read_robot_state()
            except Exception as exc:
                self._enter_damping_after_error()
                raise RuntimeError("ARX 动作执行后读取双臂状态失败。") from exc

        self._num_steps += 1
        observation = self._get_observation()
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "requested_action": requested_action.astype(np.float32),
            "executed_action": executed_action.astype(np.float32),
        }
        return observation, 0.0, False, truncated, info

    def close(self) -> None:
        """关闭三路相机，并让左右机械臂进入阻尼/被动状态。

        函数可以重复调用。相机和控制器分别关闭；即使某一路硬件关闭失败，也会
        继续尝试释放其他硬件，最后记录错误而不是让资源释放流程提前中断。
        """

        if self._closed:
            return
        self._closed = True

        for camera in self._cameras:
            try:
                camera.close()
            except Exception as exc:
                self._logger.warning("关闭相机失败：%s", exc)
        self._cameras = []

        for label, controller in (
            ("left", getattr(self, "_left_controller", None)),
            ("right", getattr(self, "_right_controller", None)),
        ):
            if controller is None:
                continue
            try:
                controller.close()
            except Exception as exc:
                self._logger.warning("关闭 %s ARX 控制器失败：%s", label, exc)
