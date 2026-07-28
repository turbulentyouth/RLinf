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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from PIL import Image

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .arx_x5_dual_controller import ArxX5JointController
from .arx_x5_dual_robot_state import ArxX5ArmState

# Match xense-openpi/examples/bi_arx5_real/real_env.py exactly. The reference
# checks policy actions against fixed 0.9-scaled joint thresholds and a slightly
# wider gripper interval before forwarding the unchanged action to the SDK.
_X5_JOINT_POSITION_MIN_RAW = np.array(
    [-3.14, -0.05, -0.2, -1.6, -1.57, -2.0], dtype=np.float64
)
_X5_JOINT_POSITION_MAX_RAW = np.array(
    [2.618, 3.5, 3.2, 1.55, 1.57, 2.0], dtype=np.float64
)
_X5_SAFETY_FACTOR = 0.9
_X5_JOINT_POSITION_LOW = _X5_JOINT_POSITION_MIN_RAW * _X5_SAFETY_FACTOR
_X5_JOINT_POSITION_HIGH = _X5_JOINT_POSITION_MAX_RAW * _X5_SAFETY_FACTOR
_X5_GRIPPER_POSITION_LOW = -0.1
_X5_GRIPPER_POSITION_HIGH = 1.8

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
class ArxX5DualPositionConfig:
    """Synchronized dual-arm target used by the reference trajectory helpers."""

    left_joints: list[float] = field(default_factory=lambda: [0.0] * 6)
    right_joints: list[float] = field(default_factory=lambda: [0.0] * 6)
    left_gripper: Optional[float] = None
    right_gripper: Optional[float] = None
    duration: float | None = 2.0

    def __post_init__(self) -> None:
        """Validate target shapes using the same six-or-seven-value semantics."""

        for name in ("left_joints", "right_joints"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (6,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite six-dimensional array.")
            setattr(self, name, value.tolist())
        for name in ("left_gripper", "right_gripper"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite or null.")
        if self.duration is not None and not np.isfinite(self.duration):
            raise ValueError("duration must be finite or null.")


@dataclass
class ArxX5DualStartPositionConfig(ArxX5DualPositionConfig):
    """Reference start pose, executed during connect and every reset."""

    left_joints: list[float] = field(
        default_factory=lambda: [0.0, 0.948, 0.858, -0.573, 0.0, 0.0]
    )
    right_joints: list[float] = field(
        default_factory=lambda: [0.0, 0.948, 0.858, -0.573, 0.0, 0.0]
    )
    left_gripper: Optional[float] = 1.57
    right_gripper: Optional[float] = 1.57


@dataclass
class ArxX5DualHomePositionConfig(ArxX5DualPositionConfig):
    """Reference all-zero home pose used by disconnect."""

    left_gripper: Optional[float] = 0.0
    right_gripper: Optional[float] = 0.0
    duration: float | None = None


@dataclass
class ArxX5DualRobotConfig:
    """ARX X5 双臂绝对关节位置环境配置。

    这个配置描述两条机械臂、三个 RGB 相机、14 维绝对关节动作、每次 reset
    的平滑起始位，以及退出前的平滑回零。

    Attributes:
        robot_model: 左右臂共同使用的 ARX 型号，当前应为 ``"X5"``。
        left_interface: 左臂 Linux CAN 接口，参考默认为 ``"can1"``。
        right_interface: 右臂 Linux CAN 接口，参考默认为 ``"can3"``。
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
        command_preview_time: SDK 对新绝对关节目标做插值的预览时间。参考项目
            使用 0.03 秒。
        kp_scale: 对 SDK 默认关节 ``kp`` 的缩放比例。参考项目使用 0.5。
        kd_scale: 对 SDK 默认关节 ``kd`` 的缩放比例。参考项目使用 1.5。
        interpolation_controller_dt: 双臂平滑轨迹的高层插值周期。参考项目
            使用 0.02 秒（50 Hz）。
        step_frequency: RLinf Env 的目标执行频率。它低于 SDK 后台控制频率。
        max_num_steps: 一条 rollout 最多执行多少个 Env step。
        task_description: 发送给 π0.5 的语言任务指令。
        skip_interface_check: 为 True 时跳过 Linux CAN 接口存在性与 UP 状态检查。
            仅在特殊调试场景下使用，正常运行应保持 False。
        dry_run: 为 True 时仍按参考流程完成 SDK 回零、start 和普通位置控制，
            但模型动作只记录，不调用 ``set_joint_cmd``。
        is_dummy: 为 True 时不连接机械臂和相机，使用零图像和内存状态测试链路。
        start_position: 每次 reset 执行的参考双臂起始位。
        home_position: 环境关闭前必定执行的双臂回零目标。
    """

    robot_model: str = "X5"
    left_interface: str = "can1"
    right_interface: str = "can3"

    head_camera_serial: Optional[str] = None
    left_wrist_camera_serial: Optional[str] = None
    right_wrist_camera_serial: Optional[str] = None
    camera_type: str = "realsense"
    head_camera_type: Optional[str] = None
    left_wrist_camera_type: Optional[str] = None
    right_wrist_camera_type: Optional[str] = None
    image_height: int = 224
    image_width: int = 224
    camera_frame_timeout: float = 0.2

    controller_dt: float = 0.002
    controller_warmup_time: float = 0.5
    command_preview_time: float = 0.03
    interpolation_controller_dt: float = 0.02
    kp_scale: float = 0.5
    kd_scale: float = 1.5
    step_frequency: float = 30.0
    max_num_steps: int = 300
    task_description: str = ""
    skip_interface_check: bool = False
    dry_run: bool = False
    is_dummy: bool = False
    start_position: ArxX5DualStartPositionConfig | dict[str, Any] = field(
        default_factory=ArxX5DualStartPositionConfig
    )
    home_position: ArxX5DualHomePositionConfig | dict[str, Any] = field(
        default_factory=ArxX5DualHomePositionConfig
    )

    def __post_init__(self) -> None:
        """检查配置是否能形成明确且安全的双臂控制协议。"""

        if isinstance(self.start_position, dict):
            self.start_position = ArxX5DualStartPositionConfig(**self.start_position)
        elif not isinstance(self.start_position, ArxX5DualStartPositionConfig):
            raise TypeError("start_position 必须是配置映射。")
        if isinstance(self.home_position, dict):
            self.home_position = ArxX5DualHomePositionConfig(**self.home_position)
        elif not isinstance(self.home_position, ArxX5DualHomePositionConfig):
            raise TypeError("home_position 必须是配置映射。")

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
        if self.kp_scale <= 0:
            raise ValueError("kp_scale 必须大于 0。")
        if self.kd_scale <= 0:
            raise ValueError("kd_scale 必须大于 0。")
        if self.interpolation_controller_dt <= 0:
            raise ValueError("interpolation_controller_dt 必须大于 0。")
        if self.step_frequency <= 0:
            raise ValueError("step_frequency 必须大于 0。")
        if self.max_num_steps <= 0:
            raise ValueError("max_num_steps 必须大于 0。")
        if self.dry_run and self.is_dummy:
            raise ValueError(
                "dry_run 和 is_dummy 不能同时启用；请选择真实硬件或纯模拟模式。"
            )

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
            真实模式和 hardware dry-run 都严格执行参考 connect 流程：创建双臂、
            SDK 回零、相机连接、平滑到 start、恢复普通位置控制。dummy 模式只
            创建内存状态和零图像。
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
        self._cameras: list[BaseCamera] = []

        if self.config.is_dummy:
            self._set_reference_action_limits()
        else:
            self._setup_hardware()
            if self.config.dry_run:
                self._logger.warning(
                    "ARX hardware dry-run 已启用：仍执行真实 connect/reset/start/home，"
                    "但不会发送模型动作。"
                )

        self._init_action_observation_spaces()

        if not self.config.is_dummy:
            self._open_cameras()
            # BiARX5.connect(go_to_start=True) performs one start motion before the
            # runtime issues its first episode reset, which performs it again.
            self._smooth_go_start()
            self._enable_dual_position_control()

    @property
    def task_description(self) -> str:
        """返回发送给 π0.5 和写入 LeRobot 数据集的语言任务指令。"""

        return self._task_description

    def _set_reference_action_limits(self) -> None:
        """Install the fixed policy-action thresholds from xense-openpi."""

        self._joint_position_low = np.stack([_X5_JOINT_POSITION_LOW] * 2)
        self._joint_position_high = np.stack([_X5_JOINT_POSITION_HIGH] * 2)
        self._gripper_position_low = np.full(
            2, _X5_GRIPPER_POSITION_LOW, dtype=np.float64
        )
        self._gripper_position_high = np.full(
            2, _X5_GRIPPER_POSITION_HIGH, dtype=np.float64
        )

    @staticmethod
    def _check_can_interface(interface: str) -> None:
        """检查 Linux CAN 接口是否存在且处于 UP 状态。

        与 ``toolkits/realworld_check/test_arx_x5_dual.py`` 保持一致：在构造
        控制器前尽早发现接口配置错误，避免后续 SDK 调用返回难以定位的通信
        超时。/sys/class/net 下的 flags 是十六进制字符串，需要按 16 进制解析。
        """

        iff_up = 0x1
        interface_path = Path("/sys/class/net") / interface
        flags_path = interface_path / "flags"
        if not flags_path.exists():
            raise RuntimeError(
                f"CAN 接口 {interface!r} 不存在；请先配置并启动 ARX CAN 适配器。"
            )
        flags = int(flags_path.read_text(encoding="utf-8").strip(), 16)
        if not flags & iff_up:
            raise RuntimeError(f"CAN 接口 {interface!r} 存在但未 UP。")

    def _setup_hardware(self) -> None:
        """Create both controllers and run the reference SDK home sequence."""

        if not self.config.skip_interface_check:
            self._check_can_interface(self.config.left_interface)
            self._check_can_interface(self.config.right_interface)

        self._left_controller = ArxX5JointController(
            model=self.config.robot_model,
            interface_name=self.config.left_interface,
            controller_dt=self.config.controller_dt,
            preview_time=self.config.command_preview_time,
        )
        time.sleep(self.config.controller_warmup_time)
        try:
            self._right_controller = ArxX5JointController(
                model=self.config.robot_model,
                interface_name=self.config.right_interface,
                controller_dt=self.config.controller_dt,
                preview_time=self.config.command_preview_time,
            )
            time.sleep(self.config.controller_warmup_time)
            self._left_controller.reset_to_home()
            self._right_controller.reset_to_home()
            self._set_dual_gravity_compensation()
        except Exception:
            self._left_controller.close()
            controller = getattr(self, "_right_controller", None)
            if controller is not None:
                controller.close()
            raise

        self._set_reference_action_limits()

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
        """将三个物理相机序列号映射到固定的 π0.5 图像名称。

        分辨率与帧率与 ``toolkits/realworld_check/test_arx_x5_dual.py`` 保持
        一致：640x480 @ 30 fps。最终返回给策略的图像会保持宽高比 resize，
        再用黑色像素补齐到 ``image_height x image_width``。
        """

        default_type = self.config.camera_type
        return [
            CameraInfo(
                name=_HEAD_CAMERA_NAME,
                serial_number=str(self.config.head_camera_serial),
                camera_type=self.config.head_camera_type or default_type,
                resolution=(640, 480),
                fps=30,
                enable_depth=False,
            ),
            CameraInfo(
                name=_LEFT_WRIST_CAMERA_NAME,
                serial_number=str(self.config.left_wrist_camera_serial),
                camera_type=self.config.left_wrist_camera_type or default_type,
                resolution=(640, 480),
                fps=30,
                enable_depth=False,
            ),
            CameraInfo(
                name=_RIGHT_WRIST_CAMERA_NAME,
                serial_number=str(self.config.right_wrist_camera_serial),
                camera_type=self.config.right_wrist_camera_type or default_type,
                resolution=(640, 480),
                fps=30,
                enable_depth=False,
            ),
        ]

    def _open_cameras(self) -> None:
        """Create and open all configured cameras in parallel like BiARX5."""

        try:
            specs = self._camera_specs()
            with ThreadPoolExecutor(max_workers=min(len(specs), 8)) as executor:
                self._cameras = list(executor.map(create_camera, specs))
                futures = [executor.submit(camera.open) for camera in self._cameras]
                for future in futures:
                    future.result()
        except Exception:
            self.close()
            raise

    def _close_cameras_parallel(self) -> None:
        """Close all cameras in parallel before destroying arm controllers."""

        if not self._cameras:
            return
        with ThreadPoolExecutor(max_workers=min(len(self._cameras), 8)) as executor:
            futures = [executor.submit(camera.close) for camera in self._cameras]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    self._logger.warning("Failed to close camera: %s", exc)
        self._cameras = []

    @staticmethod
    def _validate_camera_frame(name: str, frame: Any) -> np.ndarray:
        """检查相机帧的形状、类型和数值有效性。

        与 ``toolkits/realworld_check/test_arx_x5_dual.py`` 保持一致：要求
        返回 HWC uint8 图像且不含 NaN/Inf。返回的数组供后续 RGB 转换与 resize。
        """

        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(
                f"相机 {name!r} 必须返回 HWC 彩色图像，实际为 {array.shape}。"
            )
        if array.dtype != np.uint8:
            raise ValueError(f"相机 {name!r} 返回 dtype {array.dtype!r}；期望 uint8。")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"相机 {name!r} 图像包含 NaN 或 Inf。")
        return array

    @staticmethod
    def _resize_with_pad(
        image: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        """保持宽高比缩放图像，并用零像素居中补齐目标尺寸。

        该处理与 ``xense-openpi/examples/bi_arx5_real/env.py`` 使用的
        ``xense_client.image_tools.resize_with_pad`` 保持一致。例如原始
        ``480x640`` 图像送入 ``224x224`` 目标时，会缩放为 ``168x224``，
        然后在上下各补 28 行黑色像素，不裁剪画面，也不拉伸物体。

        Args:
            image: ``HWC uint8`` 图像。
            target_height: 输出图像高度。
            target_width: 输出图像宽度。

        Returns:
            形状为 ``(target_height, target_width, C)`` 的 ``uint8`` 图像。
        """

        height, width = image.shape[:2]
        if (height, width) == (target_height, target_width):
            return image

        ratio = max(width / target_width, height / target_height)
        resized_height = int(height / ratio)
        resized_width = int(width / ratio)
        resized = Image.fromarray(image).resize(
            (resized_width, resized_height), resample=Image.Resampling.BILINEAR
        )
        padded = Image.new(resized.mode, (target_width, target_height), 0)
        pad_height = max(0, int((target_height - resized_height) / 2))
        pad_width = max(0, int((target_width - resized_width) / 2))
        padded.paste(resized, (pad_width, pad_height))
        return np.asarray(padded)

    def _read_camera_frames(self) -> dict[str, np.ndarray]:
        """同步读取三路相机，并转换成 π0.5 所需的 RGB 图像。

        Returns:
            包含 ``base_0_rgb``、``left_wrist_0_rgb`` 和
            ``right_wrist_0_rgb`` 的字典。每张图像形状为
            ``(image_height, image_width, 3)``，类型为 ``uint8``。

        Raises:
            RuntimeError: 某相机在产生第一张有效图像之前就超时。
            ValueError: 相机返回的帧形状、类型或数值异常。

        Notes:
            RLinf 相机后端输出 BGR。函数先转成 RGB，再按照 xense-openpi 的
            BiARX5 推理方式保持宽高比缩放并补黑边。
        """

        frames: dict[str, np.ndarray] = {}
        for camera in self._cameras:
            name = camera.name
            frame = camera.get_frame(timeout=self.config.camera_frame_timeout)

            frame = self._validate_camera_frame(name, frame)
            # 当前 π0.5 只使用 RGB；如果相机后端额外附带深度通道，仅取前三个
            # BGR 通道，避免 ``[..., ::-1]`` 把深度误当成颜色。
            frame = frame[..., :3]

            rgb = np.ascontiguousarray(frame[..., ::-1], dtype=np.uint8)
            frames[name] = self._resize_with_pad(
                rgb,
                self.config.image_height,
                self.config.image_width,
            )
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

    def _check_action_safety(self, action: np.ndarray) -> tuple[bool, str]:
        """Apply the fixed action checks from xense-openpi without clipping."""

        for side, joint_slice in (("Left", slice(0, 6)), ("Right", slice(7, 13))):
            for index, (value, minimum, maximum) in enumerate(
                zip(
                    action[joint_slice],
                    _X5_JOINT_POSITION_LOW,
                    _X5_JOINT_POSITION_HIGH,
                    strict=True,
                )
            ):
                if value < minimum or value > maximum:
                    return (
                        False,
                        f"{side} joint {index + 1} out of range: {value:.4f} "
                        f"not in [{minimum:.4f}, {maximum:.4f}]",
                    )

        for side, index in (("Left", 6), ("Right", 13)):
            value = action[index]
            if value < _X5_GRIPPER_POSITION_LOW or value > _X5_GRIPPER_POSITION_HIGH:
                return (
                    False,
                    f"{side} gripper out of range: {value:.4f} not in "
                    f"[{_X5_GRIPPER_POSITION_LOW}, {_X5_GRIPPER_POSITION_HIGH}]",
                )
        return True, ""

    def _send_absolute_action(self, action: np.ndarray) -> None:
        """Forward one unchanged reference-format action to both SDK controllers."""

        self._left_controller.send_absolute_joint_position(action[:6], action[6])
        self._right_controller.send_absolute_joint_position(action[7:13], action[13])

    def _enable_dual_position_control(self) -> None:
        """Restore the reference repository's ordinary joint-position gains."""

        self._left_controller.set_to_normal_position_control(
            kp_scale=self.config.kp_scale, kd_scale=self.config.kd_scale
        )
        self._right_controller.set_to_normal_position_control(
            kp_scale=self.config.kp_scale, kd_scale=self.config.kd_scale
        )

    def _set_dual_gravity_compensation(self) -> None:
        """Switch both reference-mode trackers to gravity compensation."""

        self._left_controller.set_to_gravity_compensation()
        self._right_controller.set_to_gravity_compensation()

    def _hold_dual_current_position(self) -> None:
        """Seed both SDK interpolators with their measured joint positions."""

        self._left_controller.hold_current_position()
        self._right_controller.hold_current_position()

    def _build_position_target(self, cfg: ArxX5DualPositionConfig) -> np.ndarray:
        """Build the 14D target while holding unspecified gripper positions."""

        current = self._compose_absolute_joint_state().astype(np.float64)
        return np.concatenate(
            [
                np.asarray(cfg.left_joints, dtype=np.float64),
                np.asarray(
                    [current[6] if cfg.left_gripper is None else cfg.left_gripper]
                ),
                np.asarray(cfg.right_joints, dtype=np.float64),
                np.asarray(
                    [current[13] if cfg.right_gripper is None else cfg.right_gripper]
                ),
            ]
        )

    def _move_to_position(
        self, cfg: ArxX5DualPositionConfig, *, position_name: str
    ) -> None:
        """Run the reference 50 Hz synchronized ease-in-out trajectory."""

        if cfg.duration is None:
            self._read_robot_state()
            initial = self._compose_absolute_joint_state().astype(np.float64)
            initial_target = self._build_position_target(cfg)
            joint_indices = np.r_[0:6, 7:13]
            max_position_error = float(
                np.max(np.abs(initial_target[joint_indices] - initial[joint_indices]))
            )
            duration = max(max_position_error, 1.0) * 2.0
        else:
            duration = float(cfg.duration)

        self._hold_dual_current_position()
        self._enable_dual_position_control()

        # move_joint_trajectory fetches feedback again after the mode transition.
        self._read_robot_state()
        current = self._compose_absolute_joint_state().astype(np.float64)
        target = self._build_position_target(cfg)
        controller_dt = self.config.interpolation_controller_dt
        try:
            if duration <= 0:
                self._send_absolute_action(target)
                return

            steps = max(1, int(np.ceil(duration / controller_dt)))
            for step in range(1, steps + 1):
                progress = step / steps
                doubled_progress = progress * 2.0
                if doubled_progress < 1.0:
                    ratio = doubled_progress * doubled_progress / 2.0
                else:
                    doubled_progress -= 1.0
                    ratio = -(doubled_progress * (doubled_progress - 2.0) - 1.0) / 2.0
                self._send_absolute_action(current + (target - current) * ratio)
                time.sleep(controller_dt)
        except KeyboardInterrupt:
            self._logger.warning(
                "%s trajectory interrupted by user. Holding current pose.",
                position_name,
            )

    def _smooth_go_start(self) -> None:
        """Match BiARX5.smooth_go_start(duration=2.0)."""

        if self.config.is_dummy:
            return
        self._move_to_position(
            self.config.start_position, position_name="Start position"
        )
        self._logger.info("Successfully reached the BiARX5 start position")

    def _smooth_go_home(self) -> None:
        """Match BiARX5.smooth_go_home() including the final gravity mode."""

        self._move_to_position(self.config.home_position, position_name="Home position")
        self._set_dual_gravity_compensation()
        self._logger.info(
            "Successfully returned home and switched to gravity compensation"
        )

    def _enter_damping_after_error(self) -> None:
        """Best-effort reference fallback used for interrupted disconnect."""

        for label, controller in (
            ("left", getattr(self, "_left_controller", None)),
            ("right", getattr(self, "_right_controller", None)),
        ):
            if controller is None:
                continue
            try:
                controller.set_to_damping()
            except Exception as exc:
                self._logger.error("%s arm failed to enter damping: %s", label, exc)

    def reset(self, *, seed=None, options=None):
        """Run smooth_go_start on every real-hardware episode reset."""

        del seed, options
        self._num_steps = 0
        if not self.config.is_dummy:
            self._smooth_go_start()
            self._read_robot_state()
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        """Validate and execute one unchanged 14D absolute BiARX5 action."""

        start_time = time.monotonic()
        requested_action = np.asarray(action, dtype=np.float64)
        if requested_action.shape != (self.ACTION_DIM,):
            raise ValueError(
                f"ARX dual-arm action must be ({self.ACTION_DIM},), "
                f"got {requested_action.shape}."
            )
        if not np.all(np.isfinite(requested_action)):
            raise ValueError("ARX dual-arm action cannot contain NaN or Inf.")

        if not self.config.dry_run:
            is_safe, error_message = self._check_action_safety(requested_action)
            if not is_safe:
                raise ValueError(f"Unsafe action detected: {error_message}")
        executed_action = requested_action.copy()

        action_sent = False
        if self.config.is_dummy:
            self._left_state.joint_position = executed_action[:6].copy()
            self._left_state.gripper_position = float(executed_action[6])
            self._right_state.joint_position = executed_action[7:13].copy()
            self._right_state.gripper_position = float(executed_action[13])
        elif self.config.dry_run:
            self._logger.info(
                "hardware dry-run: policy action intercepted and not sent: %s",
                np.array2string(executed_action, precision=6, separator=", "),
            )
        else:
            if (
                self._left_controller.control_mode == "gravity_compensation"
                or self._right_controller.control_mode == "gravity_compensation"
            ):
                self._enable_dual_position_control()
            self._send_absolute_action(executed_action)
            action_sent = True

        if not self.config.is_dummy:
            self._read_robot_state()

        self._num_steps += 1
        observation = self._get_observation()

        elapsed = time.monotonic() - start_time
        time.sleep(max(0.0, 1.0 / self.config.step_frequency - elapsed))
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "requested_action": requested_action.astype(np.float32),
            "executed_action": executed_action.astype(np.float32),
            "action_sent": action_sent,
            "dry_run_mode": (
                "distributed"
                if self.config.is_dummy
                else "hardware"
                if self.config.dry_run
                else "disabled"
            ),
        }
        return observation, 0.0, False, truncated, info

    def close(self) -> None:
        """Match BiARX5.disconnect while keeping this method idempotent."""

        if self._closed:
            return
        self._closed = True

        if not self.config.is_dummy:
            try:
                self._smooth_go_home()
                self._left_controller.set_to_damping()
                self._right_controller.set_to_damping()
            except KeyboardInterrupt:
                self._logger.warning(
                    "Disconnect interrupted. Forcing damping mode on both arms."
                )
                self._enter_damping_after_error()
            except Exception as exc:
                self._logger.warning("Failed to disconnect ARX dual arms: %s", exc)

        self._close_cameras_parallel()

        for label, controller in (
            ("left", getattr(self, "_left_controller", None)),
            ("right", getattr(self, "_right_controller", None)),
        ):
            if controller is None:
                continue
            try:
                controller.close()
            except Exception as exc:
                self._logger.warning(
                    "Failed to release %s ARX controller: %s", label, exc
                )

        if not self.config.is_dummy:
            time.sleep(1.0)
