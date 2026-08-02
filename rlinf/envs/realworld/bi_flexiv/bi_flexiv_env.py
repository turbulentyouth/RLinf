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

"""连接 Flexiv Rizon4 双臂（RT）、三路相机和 π0.5 的 Gym 真机环境。"""

from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from PIL import Image

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.scheduler import WorkerInfo
from rlinf.utils.eval_events import emit_event
from rlinf.utils.logging import get_logger

from .bi_flexiv_robot_state import BiFlexivArmState

# 三个名称同时承担两层职责：
# 1. 它们是 Env 原始 observation["frames"] 中的键；
# 2. bi_flexiv 的 OpenPI 适配器会把它们分别放进 base / left wrist /
#    right wrist 图像槽位（对应 openpi BiFlexivInputs.EXPECTED_CAMERAS）。
_HEAD_CAMERA_NAME = "head"
_LEFT_WRIST_CAMERA_NAME = "left_wrist"
_RIGHT_WRIST_CAMERA_NAME = "right_wrist"
_CAMERA_NAMES = (
    _HEAD_CAMERA_NAME,
    _LEFT_WRIST_CAMERA_NAME,
    _RIGHT_WRIST_CAMERA_NAME,
)

# 20D 状态/动作中各字段的下标（与 xense-openpi bi_flexiv 数据集格式一致）。
_LEFT_TCP_SLICE = slice(0, 9)
_RIGHT_TCP_SLICE = slice(9, 18)
_LEFT_GRIPPER_INDEX = 18
_RIGHT_GRIPPER_INDEX = 19


@dataclass
class BiFlexivDualRobotConfig:
    """Flexiv Rizon4 双臂 TCP 环境配置。

    这个配置描述两条 RT 模式的 Rizon4 机械臂、三个 RGB 相机、20 维 TCP
    动作（左 TCP 9D + 右 TCP 9D + 双夹爪），以及两种 dry-run。

    所有 robot 控制参数与 xense-openpi
    ``examples/bi_flexiv_rizon4_rt`` 的 ``BiFlexivRizon4RTConfig`` 一一对应；
    底层连接、起始位与回零由 lerobot 侧的 ``BiFlexivRizon4RT`` 机器人
    封装管理（go_to_start / reset_to_initial_position / disconnect）。

    Attributes:
        bi_mount_type: 台架预设名，决定左右臂 SN 和 start/home 位姿。必须是
            lerobot-xense 预设之一：``forward-04`` / ``forward-05`` /
            ``forward-06`` / ``forward-dewu`` / ``diagonal-02``。
        use_force: 是否在观测中包含力反馈。
        go_to_start: connect 时是否先回起始位。
        stiffness_ratio: 笛卡尔阻抗刚度比例（0~1）。参考项目默认 0.2。
        inner_control_hz: RT 内环消费 Python 命令的频率。参考项目为 1000。
        interpolate_cmds: 是否在命令间做线性插值。
        enable_tactile_sensors: 是否启用手爪触觉传感器（观测会被策略忽略）。
        robot_log_level: lerobot 机器人封装的日志级别。
        head_camera_serial: 头部相机序列号。该相机作为 π0.5 主视角。
        left_wrist_camera_serial: 左腕相机序列号。
        right_wrist_camera_serial: 右腕相机序列号。
        camera_type: 三路相机共同使用的 RLinf 相机后端，默认 ``"realsense"``。
        head_camera_type: 头部相机后端；为空时回退到 ``camera_type``。
        left_wrist_camera_type: 左腕相机后端；为空时回退到 ``camera_type``。
        right_wrist_camera_type: 右腕相机后端；为空时回退到 ``camera_type``。
        image_height: 返回给 π0.5 的图像高度。
        image_width: 返回给 π0.5 的图像宽度。
        camera_frame_timeout: 单次等待相机帧的最长时间，单位为秒。
        reset_timeout: reset 轨迹允许的最长执行时间，单位为秒。参考项目为 15。
        step_frequency: RLinf Env 的目标执行频率。它远低于 RT 内环频率。
        max_num_steps: 一条 rollout 最多执行多少个 Env step。
        task_description: 发送给 π0.5 的语言任务指令。
        dry_run: 为 True 时仍完成真实 connect/reset/disconnect 流程，
            但模型动作只记录，不调用 ``send_action``。
        is_dummy: 为 True 时不连接机械臂和相机，使用零图像和内存状态测试链路。
        manual_episode_control_only: 为 True 时，外层 ``RealWorldEnv`` 保留
            （不用超时步数覆盖）键盘 Wrapper 上报的 truncated 信号，使
            ``KeyboardEvalControlWrapper`` 的左/右键中断能够触发 auto_reset。
            本环境自身不读取该值，仅作为配置透传字段存在。
    """

    bi_mount_type: str = "diagonal-02"
    use_force: bool = False
    go_to_start: bool = True
    stiffness_ratio: float = 0.2
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True
    enable_tactile_sensors: bool = False
    robot_log_level: str = "INFO"

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

    reset_timeout: float = 15.0
    step_frequency: float = 30.0
    max_num_steps: int = 300
    task_description: str = ""
    dry_run: bool = False
    is_dummy: bool = False
    manual_episode_control_only: bool = False

    def __post_init__(self) -> None:
        """检查配置是否能形成明确且安全的双臂控制协议。"""

        if self.bi_mount_type not in (
            "forward-04",
            "forward-05",
            "forward-06",
            "forward-dewu",
            "diagonal-02",
        ):
            raise ValueError(
                "bi_mount_type 必须是 lerobot-xense 预设之一：forward-04、"
                f"forward-05、forward-06、forward-dewu、diagonal-02；实际为 "
                f"{self.bi_mount_type!r}。"
            )
        if not 0.0 < self.stiffness_ratio <= 1.0:
            raise ValueError("stiffness_ratio 必须在 (0, 1] 区间内。")
        if self.inner_control_hz <= 0:
            raise ValueError("inner_control_hz 必须大于 0。")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height 和 image_width 必须大于 0。")
        if self.camera_frame_timeout <= 0:
            raise ValueError("camera_frame_timeout 必须大于 0。")
        if self.reset_timeout <= 0:
            raise ValueError("reset_timeout 必须大于 0。")
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
                    "真实 Flexiv π0.5 环境要求配置头部、左腕和右腕三路相机。"
                    f"缺少字段：{missing}。"
                )


class BiFlexivDualEnv(gym.Env):
    """Flexiv Rizon4 双臂 TCP 真机环境。

    数据流如下：

    ``三路 RGB 图像 + 20 维当前状态 -> observation -> π0.5``

    ``π0.5 的 20 维 TCP 动作 -> step(action) -> BiFlexivRizon4RT.send_action -> 双臂``

    20 维状态和动作使用完全相同的顺序（与 xense-openpi bi_flexiv 数据集一致）：

    ``[左 TCP xyz+r1..r6 (9), 右 TCP xyz+r1..r6 (9), 左夹爪, 右夹爪]``。

    策略训练使用 DeltaActions(mask=[True]*18 + [False]*2)：前 18 维 TCP 是
    相对当前状态的增量，后 2 维夹爪是绝对值。模型输出经 OpenPI 输出端的
    ``AbsoluteActions`` 还原成绝对 TCP 位姿后才到达本 Env，因此 step() 只
    透传、不再做任何增量累加。
    """

    metadata = {"render_modes": []}
    ACTION_DIM = 20
    TCP_DIM_PER_ARM = 9

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo] = None,
        hardware_info: Any = None,
        env_idx: int = 0,
    ) -> None:
        """创建双臂机器人连接、三路相机和 Gym 空间。

        Args:
            override_cfg: YAML 中 ``env.*.override_cfg`` 解析后的字典。
            worker_info: RLinf Env Worker 信息。当前第一版不额外启动控制器
                Worker，但保留参数以符合 ``RealWorldEnv`` 的 Gym 工厂接口。
            hardware_info: 调度器硬件信息。当前相机序列号直接从 YAML 读取，
                因此该参数暂未使用。
            env_idx: 当前环境编号，用于日志定位。

        Effects:
            真实模式和 hardware dry-run 都严格执行参考 connect 流程：创建
            lerobot 机器人封装、connect（可选 go_to_start）、相机连接。
            dummy 模式只创建内存状态和零图像。
        """

        del hardware_info
        self.config = BiFlexivDualRobotConfig(**override_cfg)
        self.env_idx = env_idx
        self._logger = get_logger()
        self._task_description = self.config.task_description
        self._num_steps = 0
        self.node_rank = worker_info.cluster_node_rank if worker_info else 0
        self.worker_rank = worker_info.rank if worker_info else 0
        self._closed = False
        # Serializes hardware I/O between step() (called from the eval thread)
        # and close() (which may run on a separate Ray actor thread during a
        # Ctrl-C shutdown), matching ArxX5DualEnv's _control_lock semantics.
        self._control_lock = threading.RLock()

        self._robot = None
        self._left_state = BiFlexivArmState()
        self._right_state = BiFlexivArmState()
        self._cameras: list[BaseCamera] = []

        if not self.config.is_dummy:
            self._open_cameras()
            self._setup_hardware()
            if self.config.dry_run:
                self._logger.warning(
                    "Flexiv hardware dry-run 已启用：仍执行真实 connect/reset/"
                    "disconnect，但不会发送模型动作。"
                )

        self._init_action_observation_spaces()
        # BiFlexivRizon4RTConfig(go_to_start=True) 在 connect 阶段已经把
        # 双臂移动到起始位；RT 控制环由 robot.connect 内部拉起并持续
        # 保持当前 TCP 位姿，无需像 ARX 那样显式切换控制模式。

    @property
    def task_description(self) -> str:
        """返回发送给 π0.5 和写入 LeRobot 数据集的语言任务指令。"""

        return self._task_description

    def _setup_hardware(self) -> None:
        """Connect both RT arms through the lerobot BiFlexivRizon4RT wrapper.

        延迟导入非常重要：GPU 推理节点不安装 lerobot-xense / flexiv_rt。
        只有被 Ray 放置到机器人电脑上的 Env Worker 才会执行到这里。
        """

        emit_event("connect_start", "env")
        try:
            from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import (
                BiFlexivRizon4RTConfig,
            )
            from lerobot.robots.utils import make_robot_from_config
        except ImportError as exc:
            raise RuntimeError(
                "BiFlexivDualEnv 需要 lerobot-xense（含 "
                "lerobot.robots.bi_flexiv_rizon4_rt）与 flexiv_rt。"
                "参考仓库 xense-openpi(-tactile) 的 bi_flexiv 示例依赖该分叉，"
                "上游 huggingface/lerobot 不包含此机器人类。"
            ) from exc

        robot_config = BiFlexivRizon4RTConfig(
            bi_mount_type=self.config.bi_mount_type,
            use_force=self.config.use_force,
            go_to_start=self.config.go_to_start,
            stiffness_ratio=self.config.stiffness_ratio,
            inner_control_hz=self.config.inner_control_hz,
            interpolate_cmds=self.config.interpolate_cmds,
            enable_tactile_sensors=self.config.enable_tactile_sensors,
            log_level=self.config.robot_log_level,
        )
        try:
            self._robot = make_robot_from_config(robot_config)
            self._robot.connect(calibrate=False, go_to_start=self.config.go_to_start)
        except Exception:
            robot, self._robot = self._robot, None
            if robot is not None:
                try:
                    robot.disconnect()
                except Exception:
                    pass
            raise
        emit_event("connect_done", "env")
        self._logger.info("BiFlexiv Rizon4 RT connected and ready")

    def _init_action_observation_spaces(self) -> None:
        """定义 π0.5 动作与 Env observation 的固定形状。

        TCP 位置的物理安全包线由工作空间决定，不在 Gym 空间里臆造数值：
        ``action_space`` 使用前 18 维无界 Box，夹爪两维限定 ``[0, 1]``。

        Effects:
            ``observation_space`` 包含一个 20 维状态和三张
            ``HWC uint8`` RGB 图像。
        """

        action_low = np.full(self.ACTION_DIM, -np.inf, dtype=np.float32)
        action_high = np.full(self.ACTION_DIM, np.inf, dtype=np.float32)
        action_low[_LEFT_GRIPPER_INDEX] = 0.0
        action_low[_RIGHT_GRIPPER_INDEX] = 0.0
        action_high[_LEFT_GRIPPER_INDEX] = 1.0
        action_high[_RIGHT_GRIPPER_INDEX] = 1.0
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
                        # 名称沿用 joint_position 只是沿用 RealWorldEnv 的
                        # 观测键约定；内容是 20D TCP+夹爪状态。
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

        分辨率与帧率与 arx_x5_dual 保持一致：640x480 @ 30 fps。最终返回给
        策略的图像会保持宽高比 resize，再用黑色像素补齐到
        ``image_height x image_width``。
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
        """Close all cameras in parallel before disconnecting the arms."""

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

        与 ``ArxX5DualEnv`` 保持一致：要求返回 HWC uint8 图像且不含
        NaN/Inf。返回的数组供后续 RGB 转换与 resize。
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

        该处理与 ``xense-openpi/examples/bi_flexiv_rizon4_rt/env.py`` 使用的
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
            包含 ``head``、``left_wrist`` 和 ``right_wrist`` 的字典。每张
            图像形状为 ``(image_height, image_width, 3)``，类型为 ``uint8``。

        Raises:
            RuntimeError: 某相机在产生第一张有效图像之前就超时。
            ValueError: 相机返回的帧形状、类型或数值异常。

        Notes:
            RLinf 相机后端输出 BGR。函数先转成 RGB，再按照 xense-openpi 的
            bi_flexiv 推理方式保持宽高比缩放并补黑边。
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

    @staticmethod
    def _copy_finite_vector(name: str, value: Any, expected_dim: int) -> np.ndarray:
        """复制并检查机器人 SDK 返回的一维向量。"""

        vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
        if vector.shape != (expected_dim,):
            raise RuntimeError(
                f"Flexiv 观测字段 {name!r} 应为 ({expected_dim},)，实际为 {vector.shape}。"
            )
        if not np.all(np.isfinite(vector)):
            raise RuntimeError(f"Flexiv 观测字段 {name!r} 包含 NaN 或 Inf。")
        return vector

    def _read_robot_state(self) -> None:
        """从 lerobot 机器人封装读取最新观测并缓存左右臂 TCP 状态。"""

        obs = self._robot.get_observation()
        left_tcp = np.concatenate(
            [
                self._copy_finite_vector(
                    "left_tcp.xyz", [obs[f"left_tcp.{axis}"] for axis in "xyz"], 3
                ),
                self._copy_finite_vector(
                    "left_tcp.r1-r6",
                    [obs[f"left_tcp.r{i}"] for i in range(1, 7)],
                    6,
                ),
            ]
        )
        right_tcp = np.concatenate(
            [
                self._copy_finite_vector(
                    "right_tcp.xyz", [obs[f"right_tcp.{axis}"] for axis in "xyz"], 3
                ),
                self._copy_finite_vector(
                    "right_tcp.r1-r6",
                    [obs[f"right_tcp.r{i}"] for i in range(1, 7)],
                    6,
                ),
            ]
        )
        left_gripper = float(obs["left_gripper.pos"])
        right_gripper = float(obs["right_gripper.pos"])
        if not np.all(np.isfinite([left_gripper, right_gripper])):
            raise RuntimeError("Flexiv 夹爪观测包含 NaN 或 Inf。")

        self._left_state = BiFlexivArmState(
            tcp_pose_9d=left_tcp, gripper_position=left_gripper
        )
        self._right_state = BiFlexivArmState(
            tcp_pose_9d=right_tcp, gripper_position=right_gripper
        )

    def _compose_state(self) -> np.ndarray:
        """按照和 action 完全一致的顺序生成 20 维当前状态。

        Returns:
            ``float32`` 数组：
            ``[左 TCP xyz+r1..r6, 右 TCP xyz+r1..r6, 左夹爪, 右夹爪]``。
        """

        return np.concatenate(
            [
                self._left_state.tcp_pose_9d,
                self._right_state.tcp_pose_9d,
                np.array([self._left_state.gripper_position]),
                np.array([self._right_state.gripper_position]),
            ]
        ).astype(np.float32)

    def _get_observation(self) -> dict[str, Any]:
        """生成 RLinf RealWorldEnv 能包装并发送给 π0.5 的 observation。

        Returns:
            字典包含：

            - ``state/joint_position``：20 维 TCP 和夹爪状态；
            - ``frames/head``：头部 RGB 图像；
            - ``frames/left_wrist``：左腕 RGB 图像；
            - ``frames/right_wrist``：右腕 RGB 图像。

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
            "state": {"joint_position": self._compose_state()},
            "frames": frames,
        }
        return copy.deepcopy(observation)

    @staticmethod
    def _build_action_dict(action: np.ndarray) -> dict[str, float]:
        """Build the per-key action dict that BiFlexivRizon4RT.send_action expects."""

        action_dict: dict[str, float] = {}
        action_dict["left_tcp.x"] = float(action[0])
        action_dict["left_tcp.y"] = float(action[1])
        action_dict["left_tcp.z"] = float(action[2])
        for i in range(6):
            action_dict[f"left_tcp.r{i + 1}"] = float(action[3 + i])
        action_dict["right_tcp.x"] = float(action[9])
        action_dict["right_tcp.y"] = float(action[10])
        action_dict["right_tcp.z"] = float(action[11])
        for i in range(6):
            action_dict[f"right_tcp.r{i + 1}"] = float(action[12 + i])
        action_dict["left_gripper.pos"] = float(np.clip(action[18], 0.0, 1.0))
        action_dict["right_gripper.pos"] = float(np.clip(action[19], 0.0, 1.0))
        return action_dict

    def _send_action(self, action: np.ndarray) -> None:
        """Forward one reference-format 20D action to both RT arms."""

        self._robot.send_action(self._build_action_dict(action))

    def reset(self, *, seed=None, options=None):
        """Reset both arms to the start pose, blocking until the RT trajectory ends."""

        del seed, options
        self._num_steps = 0
        if not self.config.is_dummy:
            emit_event("reset_start", "env")
            self._robot.reset_to_initial_position()
            # Block until the non-blocking RT trajectory actually finishes.
            # Phase 1: wait for rt_moving to become True (RT thread picks up
            # the request). Phase 2: wait for it to become False (trajectory
            # complete). Mirrors the reference real_env.py reset().
            t0 = time.time()
            while not self._robot.rt_moving:
                if time.time() - t0 > 1.0:
                    self._logger.warning(
                        "RT trajectory never started, proceeding anyway"
                    )
                    break
                time.sleep(0.001)
            while self._robot.rt_moving:
                if time.time() - t0 > self.config.reset_timeout:
                    self._logger.warning("Reset trajectory timeout, proceeding anyway")
                    break
                time.sleep(0.05)
            self._read_robot_state()
            emit_event("reset_done", "env")
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        """Validate and execute one 20D TCP action on both arms."""

        start_time = time.monotonic()
        requested_action = np.asarray(action, dtype=np.float64)
        if requested_action.shape != (self.ACTION_DIM,):
            raise ValueError(
                f"Flexiv dual-arm action must be ({self.ACTION_DIM},), "
                f"got {requested_action.shape}."
            )
        if not np.all(np.isfinite(requested_action)):
            raise ValueError("Flexiv dual-arm action cannot contain NaN or Inf.")

        executed_action = requested_action.copy()

        action_sent = False
        with self._control_lock:
            if self.config.is_dummy:
                self._left_state.tcp_pose_9d = executed_action[_LEFT_TCP_SLICE].copy()
                self._left_state.gripper_position = float(
                    executed_action[_LEFT_GRIPPER_INDEX]
                )
                self._right_state.tcp_pose_9d = executed_action[_RIGHT_TCP_SLICE].copy()
                self._right_state.gripper_position = float(
                    executed_action[_RIGHT_GRIPPER_INDEX]
                )
            elif self.config.dry_run:
                self._logger.info(
                    "hardware dry-run: policy action intercepted and not sent: %s",
                    np.array2string(executed_action, precision=6, separator=", "),
                )
            elif self._closed:
                # A shutdown is running on another thread; hold instead of
                # sending so we never fight the homing trajectory on the bus.
                self._logger.warning(
                    "Env is closing; policy action held instead of sent."
                )
            else:
                self._send_action(executed_action)
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
        """Disconnect both arms gracefully; keep this method idempotent."""

        if self._closed:
            return
        self._closed = True

        if not self.config.is_dummy and self._robot is not None:
            # Serialize against any in-flight step() so the homing trajectory
            # owns the connection. _closed is already set above, so once we
            # hold the lock no further action is sent.
            with self._control_lock:
                try:
                    emit_event("homing_start", "env")
                    # BiFlexivRizon4RT.disconnect() homes the arms (MoveJ)
                    # before releasing the SDK.
                    self._robot.disconnect()
                    emit_event("homing_done", "env")
                except Exception as exc:
                    emit_event("close_error", "env", error=str(exc))
                    self._logger.warning(
                        "Failed to disconnect Flexiv dual arms cleanly: %s", exc
                    )
                    try:
                        from lerobot.utils.robot_utils import (
                            emergency_stop_flexiv_rt_robot,
                        )

                        emergency_stop_flexiv_rt_robot(self._robot, self._logger)
                    except Exception as stop_exc:
                        self._logger.error(
                            "Flexiv emergency stop fallback failed: %s", stop_exc
                        )
                finally:
                    self._robot = None

        self._close_cameras_parallel()

        if not self.config.is_dummy:
            time.sleep(1.0)
        emit_event("close_done", "env")
