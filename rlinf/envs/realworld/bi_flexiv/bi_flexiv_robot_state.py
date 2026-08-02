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

"""Flexiv Rizon4 单臂状态的数据结构。"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BiFlexivArmState:
    """保存一条 Flexiv Rizon4 机械臂在某一时刻的 TCP 与夹爪反馈。

    与 ARX 的关节空间不同，Flexiv 双臂策略工作在笛卡尔空间：9 维 TCP
    位姿（xyz + 6D 旋转表示 r1..r6，即旋转矩阵的前两列拉直）加 1 维
    归一化夹爪。环境层分别保存左右臂各一个实例，然后按固定顺序拼成
    π0.5 使用的 20 维状态：

    ``[左 TCP 9D, 右 TCP 9D, 左夹爪, 右夹爪]``。

    Attributes:
        tcp_pose_9d: 末端 TCP 位姿，形状为 ``(9,)``：
            ``[x, y, z, r1, r2, r3, r4, r5, r6]``。位置单位为米；6D 旋转
            表示取自旋转矩阵前两列，连续且无万向锁问题。
        gripper_position: 归一化夹爪位置，范围 ``[0, 1]``。
    """

    tcp_pose_9d: np.ndarray = field(default_factory=lambda: np.zeros(9))
    gripper_position: float = 0.0
