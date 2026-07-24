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

"""注册 ARX X5 双臂 Gym 环境。"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.arx_x5_dual.arx_x5_dual_env import ArxX5DualEnv


def create_arx_x5_dual_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    """根据 RLinf RealWorldEnv 传入的参数创建 ARX 双臂环境。

    Args:
        override_cfg: 环境 YAML 中 ``override_cfg`` 的普通 Python 字典。
        worker_info: 当前 Ray Env Worker 的节点和 rank 信息。
        hardware_info: 调度器分配的硬件信息。第一版直接从 YAML 读取 CAN 和
            相机配置，因此只向下透传，不进行专用硬件类型解析。
        env_idx: 当前 Worker 内环境编号。真机环境固定只创建一个实例。
        env_cfg: 完整环境配置。当前工厂不叠加遥操作 Wrapper，但保留该参数以
            符合 RLinf Gym 环境工厂的统一签名。

    Returns:
        已创建的 :class:`ArxX5DualEnv`。外层 ``RealWorldEnv`` 会负责向量化、
        observation 包装以及把动作 chunk 逐步传入 ``env.step``。
    """

    del env_cfg
    return ArxX5DualEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )


register(
    id="ArxX5DualEnv-v1",
    entry_point=(
        "rlinf.envs.realworld.arx_x5_dual.tasks:create_arx_x5_dual_env"
    ),
)
