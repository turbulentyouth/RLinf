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

"""ARX X5 单臂状态的数据结构。"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArxX5ArmState:
    """保存一条 ARX X5 机械臂在某一时刻的完整反馈状态。

    这个类只负责保存 SDK 已经读回来的数据，不执行任何硬件操作。环境层会分别
    保存左臂和右臂各一个实例，然后按固定顺序拼成 π0.5 使用的 14 维状态：

    ``[左臂 6 关节, 左夹爪, 右臂 6 关节, 右夹爪]``。

    Attributes:
        timestamp: SDK 控制器时间戳，单位由 ARX SDK 定义。
        joint_position: 6 个关节的绝对角度，形状为 ``(6,)``，单位为弧度。
        joint_velocity: 6 个关节速度，形状为 ``(6,)``。
        joint_torque: 6 个关节力矩，形状为 ``(6,)``。
        gripper_position: 夹爪绝对位置。写入命令时使用的合法范围是
            ``[0, robot_config.gripper_width]``。
        gripper_velocity: 夹爪速度反馈。
        gripper_torque: 夹爪力矩反馈。
        eef_pose_6d: SDK 计算的末端 6D 位姿，形状为 ``(6,)``。当前绝对关节
            控制不会使用它下发动作，但保留该反馈便于排查机械臂状态。
    """

    timestamp: float = 0.0
    joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_torque: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gripper_position: float = 0.0
    gripper_velocity: float = 0.0
    gripper_torque: float = 0.0
    eef_pose_6d: np.ndarray = field(default_factory=lambda: np.zeros(6))
