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
from rlinf.envs.realworld.arx_x5_dual.recorder import ArxX5DualLeRobotRecorder
from rlinf.envs.realworld.common.wrappers import KeyboardEvalControlWrapper


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

    env = ArxX5DualEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    control = env_cfg.get("episode_control", {})
    recording = env_cfg.get("recording", {})
    recorder = None
    try:
        if recording and bool(recording.get("enabled", False)):
            task = str(
                recording.get("task") or override_cfg.get("task_description", "")
            )
            recorder = ArxX5DualLeRobotRecorder(
                repo_id=str(recording["repo_id"]),
                root=recording.get("root"),
                task=task,
                fps=int(
                    recording.get(
                        "fps", round(float(override_cfg.get("step_frequency", 30.0)))
                    )
                ),
                image_height=int(override_cfg.get("image_height", 224)),
                image_width=int(override_cfg.get("image_width", 224)),
                use_videos=bool(recording.get("use_videos", True)),
                image_writer_threads=int(recording.get("image_writer_threads", 4)),
                image_writer_processes=int(recording.get("image_writer_processes", 0)),
                resume=bool(recording.get("resume", False)),
            )
    except Exception:
        # Recorder 构造失败时，ArxX5DualEnv 已把双臂停在 start 位；异常若直接
        # 逃出 EnvWorker.__init__，Ray 会回收 worker 进程而跳过 close()，
        # 电机失去力矩导致双臂下坠。先尽力回 home 再原样抛出。close() 内部
        # 会记录自身的失败，这里不再重复打日志。
        try:
            env.close()
        except Exception:
            pass
        raise
    if control and bool(control.get("enabled", False)):
        env = KeyboardEvalControlWrapper(
            env,
            start_keys=(),
            success_keys=(),
            failure_keys=(),
            interrupt_keys=("Key.left", "Key.right"),
            reset_wait_seconds=float(control.get("reset_wait_seconds", 0.0)),
            continue_key="Key.right",
            preserve_env_done=True,
            episode_recorder=recorder,
            save_keys=("Key.right",),
            discard_keys=("Key.left",),
            exit_keys=("Key.esc",),
        )
    if recorder is not None and not bool(control.get("enabled", False)):
        recorder.close()
        raise ValueError(
            "ARX recording requires env.eval.episode_control.enabled=true."
        )
    return env


register(
    id="ArxX5DualEnv-v1",
    entry_point=("rlinf.envs.realworld.arx_x5_dual.tasks:create_arx_x5_dual_env"),
)
