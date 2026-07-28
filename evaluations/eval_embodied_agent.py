# Copyright 2025 The RLinf Authors.
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

import json
import signal

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.ckpt_convertor.convert_openpi_jax_to_torch import (
    prepare_openpi_checkpoint,
)
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def _raise_keyboard_interrupt(signum, frame) -> None:
    """Route SIGTERM through the runner's orderly KeyboardInterrupt cleanup."""

    del signum, frame
    raise KeyboardInterrupt


@hydra.main(
    version_base="1.1",
    config_path="libero",
    config_name="libero_spatial_starvla_eval",
)
def main(cfg) -> None:
    cfg.runner.task_type = "embodied_eval"
    cfg = prepare_openpi_checkpoint(cfg)
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement,
        max_concurrency=2,
    )
    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=env_placement,
        max_concurrency=2,
    )

    runner = EmbodiedEvalRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        runner.run()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == "__main__":
    main()
