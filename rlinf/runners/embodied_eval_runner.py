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

import signal
import typing
from contextlib import contextmanager

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    @contextmanager
    def _shield_termination_signals(self):
        """Ignore SIGINT/SIGTERM for the duration of the block.

        Real-world arms only hold torque while the env worker is alive, so the
        homing RPC must finish before the process exits. Without shielding, an
        operator pressing Ctrl-C again during the ~seconds-long home trajectory
        would raise KeyboardInterrupt, abort the wait, let the driver exit, and
        Ray would SIGKILL the env actor mid-home — dropping the arms. Runs in
        the driver main thread, so signal.signal() is valid here.
        """

        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.signal(sig, signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        try:
            yield
        finally:
            for sig, handler in previous.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass

    def _close_envs_safely(self):
        """Home/close the envs, shielded so Ctrl-C cannot abort the homing."""

        with self._shield_termination_signals():
            try:
                close_handle = self.env.close_envs()
                close_handle.wait()
            except Exception as exc:
                self.logger.warning(
                    "Failed to close evaluation environments: %s", exc
                )

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        try:
            env_results = env_handle.wait()
            if not env_decoupled_mode:
                rollout_handle.wait()
        except KeyboardInterrupt:
            self.logger.info(
                "Manual stop requested; halting workers so run()'s shielded "
                "close_envs() can home the arms."
            )
            # Stop stepping and break any blocking manual-control wait. Do NOT
            # block on the eval handles here: if a worker is wedged in recv, a
            # further Ctrl-C should fall straight through to run()'s shielded
            # close_envs(), which is idempotent and runs on a separate actor
            # thread, so the arms are homed regardless of the eval loop state.
            try:
                self.env.request_eval_stop().wait()
                self.rollout.request_eval_stop().wait()
            except Exception as exc:
                self.logger.warning("Failed to request cooperative stop: %s", exc)
            raise
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        continuous_cfg = self.cfg.runner.get("continuous_eval", {})
        continuous_enabled = bool(continuous_cfg.get("enabled", False))
        max_cycles = continuous_cfg.get("max_cycles", None)
        if max_cycles is not None and int(max_cycles) <= 0:
            raise ValueError(
                "runner.continuous_eval.max_cycles must be positive or null."
            )

        cycle = 0
        try:
            while True:
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                self.logger.info("Evaluation cycle %d metrics: %s", cycle, eval_metrics)
                self.metric_logger.log(step=cycle, data=eval_metrics)
                cycle += 1

                if not continuous_enabled:
                    break
                if max_cycles is not None and cycle >= int(max_cycles):
                    break
        except KeyboardInterrupt:
            self.logger.info(
                "Continuous inference stopped manually after %d completed cycles.",
                cycle,
            )
        finally:
            self._close_envs_safely()
            self.metric_logger.finish()
