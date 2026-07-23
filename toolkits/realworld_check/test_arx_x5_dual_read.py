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

# ip -br link show type can
# ip monitor link

"""Validate feedback from two ARX X5 arms without setting motion targets.

Run this smoke test after installing the ``arx_x5_dual`` environment and
bringing up both CAN interfaces. The script disables the SDK background
control loop and does not call reset, calibration, trajectory, or
command-setting methods. The vendor ``recv_once`` implementation can still
transmit motor-enable/query frames, and X5 shutdown transitions the controller
to damping. Clear the workspace and keep an emergency stop ready.

Examples::

    python toolkits/realworld_check/test_arx_x5_dual_read.py
    python toolkits/realworld_check/test_arx_x5_dual_read.py \
        --left-interface can0 --right-interface can1
    python toolkits/realworld_check/test_arx_x5_dual_read.py \
        --samples 20 --interval 0.1
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

IFF_UP = 0x1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Feedback test for an ARX X5 dual-arm setup without motion targets."
        )
    )
    parser.add_argument(
        "--left-interface",
        default="can0",
        help="CAN network interface for the left arm (default: can0).",
    )
    parser.add_argument(
        "--right-interface",
        default="can1",
        help="CAN network interface for the right arm (default: can1).",
    )
    parser.add_argument(
        "--model",
        default="X5",
        choices=("X5", "L5"),
        help="ARX arm model used for both arms (default: X5).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of feedback samples to read from each arm (default: 5).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between samples (default: 0.2).",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.5,
        help="Seconds to wait after constructing both controllers (default: 0.5).",
    )
    parser.add_argument(
        "--skip-interface-check",
        action="store_true",
        help="Skip checking that both Linux network interfaces exist and are UP.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a traceback if an SDK operation fails.",
    )
    args = parser.parse_args()

    if args.left_interface == args.right_interface:
        parser.error("the left and right interfaces must be different")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def _check_interface(interface: str) -> None:
    interface_path = Path("/sys/class/net") / interface
    flags_path = interface_path / "flags"
    if not flags_path.exists():
        raise RuntimeError(
            f"network interface {interface!r} does not exist; configure and "
            "bring up the ARX CAN adapter before running this test"
        )

    flags = int(flags_path.read_text(encoding="utf-8").strip(), 16)
    if not flags & IFF_UP:
        raise RuntimeError(
            f"network interface {interface!r} exists but is DOWN"
        )

    operstate_path = interface_path / "operstate"
    operstate = (
        operstate_path.read_text(encoding="utf-8").strip()
        if operstate_path.exists()
        else "unknown"
    )
    print(f"  PASS  {interface}: UP (operstate={operstate})")


def _finite_vector(name: str, value: Any, expected_size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if vector.size != expected_size:
        raise ValueError(
            f"{name} has shape {vector.shape}; expected {expected_size} values"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinity: {vector}")
    return vector


def _finite_scalar(name: str, value: Any) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} is not finite: {scalar}")
    return scalar


def _load_sdk(verbose: bool) -> Any:
    print("\n[1] Importing arx5-interface ...")
    try:
        import arx5_interface as arx5
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        raise RuntimeError(
            "could not import arx5_interface; activate the environment created "
            "by requirements/install.sh and reinstall --env arx_x5_dual"
        ) from exc

    required_names = (
        "Arx5JointController",
        "RobotConfigFactory",
        "ControllerConfigFactory",
    )
    missing_names = [name for name in required_names if not hasattr(arx5, name)]
    if missing_names:
        raise RuntimeError(
            "arx5_interface is missing expected API members: "
            + ", ".join(missing_names)
        )

    try:
        package_version = version("arx5-interface")
    except PackageNotFoundError:
        package_version = "unknown"
    spec = importlib.util.find_spec("arx5_interface")
    print(f"  PASS  version: {package_version}")
    print(f"  PASS  module:  {spec.origin if spec else 'unknown'}")
    return arx5


def _make_controller(arx5: Any, model: str, interface: str) -> Any:
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )

    # Disable the continuous control loop. The vendor recv_once() path still
    # sends motor-enable/query frames, and X5 requires damping on shutdown.
    controller_config.background_send_recv = False
    controller_config.shutdown_to_passive = True
    return arx5.Arx5JointController(
        robot_config,
        controller_config,
        interface,
    )


def _print_controller_info(side: str, controller: Any) -> int:
    robot = controller.get_robot_config()
    config = controller.get_controller_config()
    gain = controller.get_gain()
    dof = int(robot.joint_dof)

    print(f"\n[{side} configuration]")
    print(f"  robot_model:             {robot.robot_model}")
    print(f"  joint_dof:               {dof}")
    print(f"  joint_pos_min:           {np.asarray(robot.joint_pos_min)}")
    print(f"  joint_pos_max:           {np.asarray(robot.joint_pos_max)}")
    print(f"  joint_vel_max:           {np.asarray(robot.joint_vel_max)}")
    print(f"  joint_torque_max:        {np.asarray(robot.joint_torque_max)}")
    print(f"  ee_vel_max:              {np.asarray(robot.ee_vel_max)}")
    print(f"  gripper_width:           {robot.gripper_width}")
    print(f"  gripper_open_readout:    {robot.gripper_open_readout}")
    print(f"  gripper_vel_max:         {robot.gripper_vel_max}")
    print(f"  gripper_torque_max:      {robot.gripper_torque_max}")
    print(f"  controller_dt:           {config.controller_dt}")
    print(f"  background_send_recv:    {config.background_send_recv}")
    print(f"  shutdown_to_passive:     {config.shutdown_to_passive}")
    print(f"  home_pose:               {np.asarray(controller.get_home_pose())}")
    print(f"  gain.kp:                 {np.asarray(gain.kp())}")
    print(f"  gain.kd:                 {np.asarray(gain.kd())}")
    print(f"  gain.gripper_kp:         {gain.gripper_kp}")
    print(f"  gain.gripper_kd:         {gain.gripper_kd}")
    return dof


def _read_feedback(controller: Any, side: str, dof: int) -> dict[str, Any]:
    controller.recv_once()
    joint = controller.get_joint_state()
    eef = controller.get_eef_state()

    return {
        "controller_timestamp": _finite_scalar(
            f"{side}.controller_timestamp", controller.get_timestamp()
        ),
        "joint_timestamp": _finite_scalar(
            f"{side}.joint_timestamp", joint.timestamp
        ),
        "joint_position": _finite_vector(
            f"{side}.joint_position", joint.pos(), dof
        ),
        "joint_velocity": _finite_vector(
            f"{side}.joint_velocity", joint.vel(), dof
        ),
        "joint_torque": _finite_vector(
            f"{side}.joint_torque", joint.torque(), dof
        ),
        "joint_gripper_position": _finite_scalar(
            f"{side}.joint_gripper_position", joint.gripper_pos
        ),
        "joint_gripper_velocity": _finite_scalar(
            f"{side}.joint_gripper_velocity", joint.gripper_vel
        ),
        "joint_gripper_torque": _finite_scalar(
            f"{side}.joint_gripper_torque", joint.gripper_torque
        ),
        "eef_timestamp": _finite_scalar(
            f"{side}.eef_timestamp", eef.timestamp
        ),
        "eef_pose_6d": _finite_vector(
            f"{side}.eef_pose_6d", eef.pose_6d(), 6
        ),
        "eef_gripper_position": _finite_scalar(
            f"{side}.eef_gripper_position", eef.gripper_pos
        ),
        "eef_gripper_velocity": _finite_scalar(
            f"{side}.eef_gripper_velocity", eef.gripper_vel
        ),
        "eef_gripper_torque": _finite_scalar(
            f"{side}.eef_gripper_torque", eef.gripper_torque
        ),
    }


def _print_feedback(side: str, feedback: dict[str, Any]) -> None:
    print(f"  {side}:")
    for name, value in feedback.items():
        if isinstance(value, np.ndarray):
            formatted = np.array2string(value, precision=6, separator=", ")
        else:
            formatted = f"{value:.6f}"
        print(f"    {name:>26}: {formatted}")


def _check_timestamp_progress(
    side: str, samples: list[dict[str, Any]]
) -> None:
    if len(samples) < 2:
        return

    joint_times = np.asarray([sample["joint_timestamp"] for sample in samples])
    if not np.any(np.diff(joint_times) > 0):
        raise RuntimeError(
            f"{side} feedback timestamps did not advance across samples; "
            "check the interface and arm power"
        )


def main() -> int:
    args = _parse_args()
    np.set_printoptions(linewidth=160, precision=6, suppress=False)

    print("ARX X5 dual-arm feedback SDK smoke test")
    print(f"  Python:          {sys.executable}")
    print(f"  Model:           {args.model}")
    print(f"  Left interface:  {args.left_interface}")
    print(f"  Right interface: {args.right_interface}")
    print(
        "  WARNING: recv_once() may send motor-enable/query frames; clear "
        "the workspace and keep an emergency stop ready."
    )

    controllers: dict[str, Any] = {}
    try:
        arx5 = _load_sdk(args.verbose)

        if args.skip_interface_check:
            print("\n[2] Linux network interface checks skipped")
        else:
            print("\n[2] Checking Linux network interfaces ...")
            _check_interface(args.left_interface)
            _check_interface(args.right_interface)

        print("\n[3] Constructing feedback controllers ...")
        controllers = {
            "left": _make_controller(arx5, args.model, args.left_interface),
            "right": _make_controller(arx5, args.model, args.right_interface),
        }
        print("  PASS  both Arx5JointController instances constructed")

        if args.warmup:
            time.sleep(args.warmup)

        dofs = {
            side: _print_controller_info(side, controller)
            for side, controller in controllers.items()
        }

        print("\n[4] Reading feedback with recv_once() ...")
        all_samples: dict[str, list[dict[str, Any]]] = {
            "left": [],
            "right": [],
        }
        for sample_index in range(1, args.samples + 1):
            print(f"\n  Sample {sample_index}/{args.samples}")
            for side, controller in controllers.items():
                feedback = _read_feedback(controller, side, dofs[side])
                all_samples[side].append(feedback)
                _print_feedback(side, feedback)
            if sample_index < args.samples and args.interval:
                time.sleep(args.interval)

        for side, samples in all_samples.items():
            _check_timestamp_progress(side, samples)

        print(
            "\nPASS: arx5-interface loaded and both arms returned finite, "
            "joint, gripper, and end-effector feedback."
        )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        controllers.clear()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
