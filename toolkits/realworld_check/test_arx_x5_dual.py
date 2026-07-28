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

"""Verify the production ARX X5 dual-arm start and shutdown lifecycle.

This checker directly instantiates ArxX5DualEnv. Construction and every
explicit reset use production smooth_go_start(), while the finally block calls
production close() to exercise home-on-normal-exit, exception, and
Ctrl+C/SIGTERM behavior.

No policy action is sent and no action-boundary check is performed.

Examples:
    python toolkits/realworld_check/test_arx_x5_dual.py --execute
    python toolkits/realworld_check/test_arx_x5_dual.py --execute \
        --interrupt-wait 300
    python toolkits/realworld_check/test_arx_x5_dual.py --execute \
        --simulate-error
"""

from __future__ import annotations

import argparse
import copy
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.envs.realworld.arx_x5_dual.arx_x5_dual_env import (  # noqa: E402
    ArxX5DualEnv,
)

DEFAULT_CONFIG = (
    REPO_ROOT
    / "examples"
    / "embodiment"
    / "config"
    / "env"
    / "realworld_arx_x5_dual_joint.yaml"
)
START_JOINTS = np.array([0.0, 0.948, 0.858, -0.573, 0.0, 0.0], dtype=np.float64)
HOME_JOINTS = np.zeros(6, dtype=np.float64)
START_GRIPPER = 1.57
HOME_GRIPPER = 0.0


@dataclass
class VerificationReport:
    """Collect failures without preventing shutdown."""

    failures: list[str] = field(default_factory=list)

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        """Record and print one check."""

        if condition:
            print(f"  PASS  {label}")
            return
        message = f"{label}: {detail}" if detail else label
        self.failures.append(message)
        print(f"  FAIL  {message}", file=sys.stderr)

    def check_close(
        self,
        actual: Any,
        expected: Any,
        label: str,
        *,
        atol: float = 1e-9,
    ) -> None:
        """Record an allclose comparison."""

        actual_array = np.asarray(actual)
        expected_array = np.asarray(expected)
        self.check(
            bool(np.allclose(actual_array, expected_array, atol=atol, rtol=0.0)),
            label,
            f"actual={actual_array}, expected={expected_array}, atol={atol}",
        )


class ObservedArxX5DualEnv(ArxX5DualEnv):
    """Record feedback after the production start and home helpers."""

    def __init__(self, *args: Any, settle_seconds: float = 0.2, **kwargs: Any) -> None:
        self.start_states: list[np.ndarray] = []
        self.home_states: list[np.ndarray] = []
        self.start_modes: list[tuple[str, str]] = []
        self.home_modes: list[tuple[str, str]] = []
        self._verification_settle_seconds = settle_seconds
        super().__init__(*args, **kwargs)

    def _sample_state(self) -> np.ndarray:
        if self._verification_settle_seconds > 0:
            time.sleep(self._verification_settle_seconds)
        self._read_robot_state()
        return self._compose_absolute_joint_state().astype(np.float64)

    def _control_modes(self) -> tuple[str, str]:
        return (
            self._left_controller.control_mode,
            self._right_controller.control_mode,
        )

    def _smooth_go_start(self) -> None:
        super()._smooth_go_start()
        self.start_states.append(self._sample_state())
        self.start_modes.append(self._control_modes())

    def _smooth_go_home(self) -> None:
        super()._smooth_go_home()
        self.home_states.append(self._sample_state())
        self.home_modes.append(self._control_modes())


class ExpectedTestError(RuntimeError):
    """Exception injected by --simulate-error."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move both ARX X5 arms to start with the production environment and "
            "verify home-on-exit."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--left-interface")
    parser.add_argument("--right-interface")
    parser.add_argument("--head-camera-serial")
    parser.add_argument("--left-wrist-camera-serial")
    parser.add_argument("--right-wrist-camera-serial")
    parser.add_argument("--skip-interface-check", action="store_true")
    parser.add_argument("--reset-count", type=int, default=1)
    parser.add_argument("--settle-seconds", type=float, default=0.2)
    parser.add_argument("--joint-tolerance", type=float, default=0.08)
    parser.add_argument("--gripper-tolerance", type=float, default=0.2)
    parser.add_argument(
        "--interrupt-wait",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Wait at start for Ctrl+C/SIGTERM before cleanup.",
    )
    parser.add_argument(
        "--simulate-error",
        action="store_true",
        help="Inject an expected exception after reaching start.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required confirmation because this test moves both physical arms.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        parser.error("--execute is required because this test moves both arms")
    if args.reset_count < 0:
        parser.error("--reset-count must be non-negative")
    if (
        min(
            args.settle_seconds,
            args.joint_tolerance,
            args.gripper_tolerance,
            args.interrupt_wait,
        )
        < 0
    ):
        parser.error("time and tolerance values must be non-negative")
    return args


def _load_override_cfg(args: argparse.Namespace) -> dict[str, Any]:
    with args.config.open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)
    if not isinstance(document, dict):
        raise ValueError(f"{args.config} must contain a YAML mapping")
    override_cfg = document.get("override_cfg", document)
    if not isinstance(override_cfg, dict):
        raise ValueError(f"{args.config} override_cfg must be a mapping")

    override_cfg = copy.deepcopy(override_cfg)
    cli_overrides = {
        "left_interface": args.left_interface,
        "right_interface": args.right_interface,
        "head_camera_serial": args.head_camera_serial,
        "left_wrist_camera_serial": args.left_wrist_camera_serial,
        "right_wrist_camera_serial": args.right_wrist_camera_serial,
    }
    override_cfg.update(
        {name: value for name, value in cli_overrides.items() if value is not None}
    )
    if args.skip_interface_check:
        override_cfg["skip_interface_check"] = True
    override_cfg["is_dummy"] = False
    override_cfg["dry_run"] = False
    return override_cfg


def _validate_config(env: ArxX5DualEnv, report: VerificationReport) -> None:
    print("\n[configuration] Checking reference control parameters ...")
    cfg = env.config
    report.check(cfg.robot_model == "X5", "robot model is X5")
    for actual, expected, label in (
        (cfg.controller_dt, 0.002, "SDK background period is 0.002 s"),
        (cfg.command_preview_time, 0.03, "SDK preview time is 0.03 s"),
        (cfg.interpolation_controller_dt, 0.02, "trajectory period is 0.02 s"),
        (cfg.kp_scale, 0.5, "position kp scale is 0.5"),
        (cfg.kd_scale, 1.5, "position kd scale is 1.5"),
    ):
        report.check_close(actual, expected, label)

    for actual, expected, label in (
        (cfg.start_position.left_joints, START_JOINTS, "left start joints match"),
        (cfg.start_position.right_joints, START_JOINTS, "right start joints match"),
        (cfg.home_position.left_joints, HOME_JOINTS, "left home joints are zero"),
        (cfg.home_position.right_joints, HOME_JOINTS, "right home joints are zero"),
    ):
        report.check_close(actual, expected, label)
    report.check_close(
        [cfg.start_position.left_gripper, cfg.start_position.right_gripper],
        [START_GRIPPER, START_GRIPPER],
        "start grippers match",
    )
    report.check_close(
        [cfg.home_position.left_gripper, cfg.home_position.right_gripper],
        [HOME_GRIPPER, HOME_GRIPPER],
        "home grippers are zero",
    )
    report.check(cfg.start_position.duration == 2.0, "start duration is 2.0 s")
    report.check(
        cfg.home_position.duration is None,
        "home uses automatic reference duration",
        f"actual={cfg.home_position.duration}",
    )


def _validate_controllers(env: ArxX5DualEnv, report: VerificationReport) -> None:
    print("\n[controllers] Checking production SDK mode and gains ...")
    for side, wrapper in (
        ("left", env._left_controller),
        ("right", env._right_controller),
    ):
        config = wrapper._controller_config
        gain = wrapper._controller.get_gain()
        report.check(
            bool(config.background_send_recv),
            f"{side} SDK background send/receive is enabled",
        )
        report.check(
            not bool(config.gravity_compensation),
            f"{side} SDK uses ordinary joint position control",
        )
        report.check(
            wrapper.control_mode == "position",
            f"{side} wrapper is in position mode at start",
            f"actual={wrapper.control_mode}",
        )
        report.check_close(
            gain.kp(),
            np.asarray(config.default_kp) * env.config.kp_scale,
            f"{side} kp matches reference scale",
        )
        report.check_close(
            gain.kd(),
            np.asarray(config.default_kd) * env.config.kd_scale,
            f"{side} kd matches reference scale",
        )


def _validate_pose(
    state: np.ndarray,
    joints: np.ndarray,
    gripper: float,
    name: str,
    args: argparse.Namespace,
    report: VerificationReport,
) -> None:
    report.check_close(
        state[:6], joints, f"{name} left joints", atol=args.joint_tolerance
    )
    report.check_close(
        state[7:13], joints, f"{name} right joints", atol=args.joint_tolerance
    )
    report.check_close(
        state[[6, 13]],
        [gripper, gripper],
        f"{name} grippers",
        atol=args.gripper_tolerance,
    )


def _validate_start(
    env: ObservedArxX5DualEnv,
    args: argparse.Namespace,
    report: VerificationReport,
) -> None:
    print("\n[start] Checking connect/reset start motions ...")
    expected_count = 1 + args.reset_count
    report.check(
        len(env.start_states) == expected_count,
        "smooth_go_start ran during connect and every reset",
        f"actual={len(env.start_states)}, expected={expected_count}",
    )
    for index, state in enumerate(env.start_states, start=1):
        _validate_pose(
            state,
            START_JOINTS,
            START_GRIPPER,
            f"start sample {index}",
            args,
            report,
        )
    report.check(
        all(modes == ("position", "position") for modes in env.start_modes),
        "both arms use position mode after every start",
        f"actual={env.start_modes}",
    )


def _validate_observation(
    observation: dict[str, Any], report: VerificationReport
) -> None:
    state = observation.get("state", {}).get("joint_position")
    report.check(
        isinstance(state, np.ndarray) and state.shape == (14,),
        "reset returned a 14-dimensional state",
        f"actual shape={getattr(state, 'shape', None)}",
    )
    frames = observation.get("frames", {})
    report.check(
        set(frames) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"},
        "reset returned all three camera frames",
        f"actual keys={sorted(frames)}",
    )


def _wait_for_interrupt(seconds: float) -> None:
    if seconds <= 0:
        return
    print(
        f"\n[interrupt] Holding at start for up to {seconds:.1f} s. Press Ctrl+C now."
    )
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(min(0.2, deadline - time.monotonic()))
    raise RuntimeError("interrupt wait expired without Ctrl+C or SIGTERM")


def _sigterm_as_keyboard_interrupt(signum: int, frame: Any) -> None:
    del signum, frame
    raise KeyboardInterrupt


def main() -> int:
    args = _parse_args()
    np.set_printoptions(linewidth=160, precision=6, suppress=True)
    report = VerificationReport()
    env: ObservedArxX5DualEnv | None = None
    observation: dict[str, Any] | None = None
    interrupted = False
    expected_error = False
    unexpected_error: BaseException | None = None

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _sigterm_as_keyboard_interrupt)
    print("ARX X5 dual-arm production lifecycle verification")
    print(f"  Config:      {args.config}")
    print(f"  Resets:      {args.reset_count}")
    print("  Motion:      production smooth_go_start only")
    print("  Policy step: disabled (no action or boundary check)")
    print(
        "  WARNING: both arms will move. Clear the workspace and keep the "
        "emergency stop ready."
    )

    try:
        print("\n[connect] Constructing the production environment ...")
        env = ObservedArxX5DualEnv(
            override_cfg=_load_override_cfg(args),
            settle_seconds=args.settle_seconds,
        )
        print("  PASS  connected, opened cameras, and reached start")
        _validate_config(env, report)

        for reset_index in range(args.reset_count):
            print(f"\n[reset {reset_index + 1}/{args.reset_count}] Moving to start ...")
            observation, _ = env.reset()
            print("  PASS  production reset completed")

        _validate_controllers(env, report)
        _validate_start(env, args, report)
        if observation is not None:
            _validate_observation(observation, report)

        if args.simulate_error:
            raise ExpectedTestError("injected exception after reaching start")
        _wait_for_interrupt(args.interrupt_wait)
    except ExpectedTestError as exc:
        expected_error = True
        print(f"\n[exception] Expected test exception caught: {exc}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupt] Interrupt caught; running production home cleanup ...")
    except BaseException as exc:
        unexpected_error = exc
        if args.verbose:
            traceback.print_exc()
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if env is not None:
            print("\n[shutdown] Calling production env.close() ...")
            try:
                env.close()
            except BaseException as exc:
                report.failures.append(f"env.close raised {type(exc).__name__}: {exc}")
                print(
                    f"  FAIL  env.close raised {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            report.check(
                len(env.home_states) == 1,
                "production close executed one home trajectory",
                f"actual samples={len(env.home_states)}",
            )
            if env.home_states:
                _validate_pose(
                    env.home_states[-1],
                    HOME_JOINTS,
                    HOME_GRIPPER,
                    "shutdown home",
                    args,
                    report,
                )
            report.check(
                env.home_modes == [("gravity_compensation", "gravity_compensation")],
                "both arms entered gravity compensation after home",
                f"actual={env.home_modes}",
            )
        signal.signal(signal.SIGTERM, previous_sigterm)

    print("\n[result]")
    if unexpected_error is not None or report.failures:
        for failure in report.failures:
            print(f"  - {failure}", file=sys.stderr)
        print("FAIL: ARX X5 lifecycle verification did not pass.", file=sys.stderr)
        return 1

    if expected_error:
        print("PASS: exception cleanup returned both arms home.")
    elif interrupted:
        print("PASS: interrupt cleanup returned both arms home.")
    else:
        print("PASS: start/reset and normal-exit home lifecycle verified.")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
