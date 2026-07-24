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

"""Validate ARX X5 dual-arm feedback, cameras, and guarded joint control.

Joint mode follows the control structure used by Vertax42/lerobot-ARX5:
500 Hz SDK background communication, 50 Hz command interpolation, 30 ms command
preview, and reduced-stiffness position gains. If no joint target is supplied,
both selected arms move to the reference start joint position. Gripper position
is held unless explicitly supplied because the reference project and the local
SDK use different gripper units.

The three RealSense serial numbers match the reference project. Cameras run in
both headed and headless modes; headed mode additionally displays a GUI. Press
``q`` or Escape in the camera window to interrupt and enter damping mode.

Examples::

    # 检查串口设备
    ls -l /dev/ttyACM*

    # 启动 CAN 接口

    sudo slcand -o -f -s8 /dev/ttyACM0 can1
    sudo slcand -o -f -s8 /dev/ttyACM1 can3

    sudo ip link set can1 up
    sudo ip link set can3 up

    # 确认
    ip link show can1
    ip link show can3

    export AMENT_PREFIX_PATH="$PWD/.venv/arx5-conda-env${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"

    source .venv/bin/activate

    # Feedback and cameras, without a GUI.
    python toolkits/realworld_check/test_arx_x5_dual.py --mode read --headless

    # Move both arms to the reference start position and show camera GUI.
    python toolkits/realworld_check/test_arx_x5_dual.py \
        --mode joint --headless --duration 30 --execute

    # Explicit small target for the left arm, holding the right arm.
    python toolkits/realworld_check/test_arx_x5_dual.py \
        --mode joint --arm left \
        --left-joints 0.0 0.95 0.86 -0.57 0.0 0.0 \
        --execute
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import queue
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IFF_UP = 0x1

REFERENCE_START_JOINTS = np.array(
    [0.0, 0.948, 0.858, -0.573, 0.0, 0.0], dtype=np.float64
)
REFERENCE_CAMERAS = {
    "head": "230322271365",
    "left_wrist": "230422271416",
    "right_wrist": "230322274234",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read ARX X5 feedback or send guarded joint commands while "
            "checking three RealSense cameras."
        )
    )
    parser.add_argument(
        "--left-interface",
        default="can1",
        help="CAN network interface for the left arm (default: can1).",
    )
    parser.add_argument(
        "--right-interface",
        default="can3",
        help="CAN network interface for the right arm (default: can3).",
    )
    parser.add_argument(
        "--model",
        default="X5",
        choices=("X5", "L5"),
        help="ARX arm model used for both arms (default: X5).",
    )
    parser.add_argument(
        "--mode",
        choices=("read", "joint"),
        default="read",
        help="read: receive feedback; joint: send guarded joint commands.",
    )
    parser.add_argument(
        "--arm",
        choices=("left", "right", "both"),
        default="both",
        help="Arm to command in joint mode (default: both).",
    )
    parser.add_argument(
        "--left-joints",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Left-arm joint target in radians.",
    )
    parser.add_argument(
        "--right-joints",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Right-arm joint target in radians.",
    )
    parser.add_argument(
        "--left-gripper",
        type=float,
        help="Left gripper target in local SDK units; defaults to current position.",
    )
    parser.add_argument(
        "--right-gripper",
        type=float,
        help="Right gripper target in local SDK units; defaults to current position.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of feedback samples in read mode (default: 5).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between feedback samples (default: 0.2).",
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
        help="Skip checking that both Linux CAN interfaces exist and are UP.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a traceback if an SDK operation fails.",
    )

    control = parser.add_argument_group("joint control")
    control.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Interpolation duration. By default use the reference rule "
            "max(max_joint_delta, 1.0) * 2.0 seconds."
        ),
    )
    control.add_argument(
        "--control-period",
        type=float,
        default=0.02,
        help="High-level command period (default: 0.02 s / 50 Hz).",
    )
    control.add_argument(
        "--controller-dt",
        type=float,
        default=0.002,
        help="SDK background control period (default: 0.002 s / 500 Hz).",
    )
    control.add_argument(
        "--preview-time",
        type=float,
        default=0.03,
        help="SDK command preview time (default: 0.03 s).",
    )
    control.add_argument(
        "--kp-scale",
        type=float,
        default=0.5,
        help="Scale applied to SDK default joint kp (default: 0.5).",
    )
    control.add_argument(
        "--kd-scale",
        type=float,
        default=1.5,
        help="Scale applied to SDK default joint kd (default: 1.5).",
    )
    control.add_argument(
        "--joint-safety-factor",
        type=float,
        default=0.9,
        help="Fraction of each SDK joint range accepted as safe (default: 0.9).",
    )
    control.add_argument(
        "--velocity-safety-factor",
        type=float,
        default=0.2,
        help="Fraction of SDK joint velocity limits accepted (default: 0.2).",
    )
    control.add_argument(
        "--max-joint-delta",
        type=float,
        default=1.2,
        help="Maximum movement of any joint per invocation (default: 1.2 rad).",
    )
    control.add_argument(
        "--max-final-error",
        type=float,
        default=0.15,
        help="Maximum final joint tracking error (default: 0.15 rad).",
    )
    control.add_argument(
        "--execute",
        action="store_true",
        help="Required confirmation before sending motion commands.",
    )

    cameras = parser.add_argument_group("RealSense cameras")
    display = cameras.add_mutually_exclusive_group()
    display.add_argument(
        "--headed",
        dest="headless",
        action="store_false",
        help="Capture cameras and open an OpenCV GUI.",
    )
    display.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Capture and validate cameras without opening a GUI (default).",
    )
    parser.set_defaults(headless=True)
    cameras.add_argument(
        "--skip-cameras",
        action="store_true",
        help="Run the arm check without opening RealSense cameras.",
    )
    cameras.add_argument(
        "--head-camera-serial",
        default=REFERENCE_CAMERAS["head"],
        help="Head RealSense serial number.",
    )
    cameras.add_argument(
        "--left-wrist-camera-serial",
        default=REFERENCE_CAMERAS["left_wrist"],
        help="Left wrist RealSense serial number.",
    )
    cameras.add_argument(
        "--right-wrist-camera-serial",
        default=REFERENCE_CAMERAS["right_wrist"],
        help="Right wrist RealSense serial number.",
    )
    cameras.add_argument("--camera-width", type=int, default=640)
    cameras.add_argument("--camera-height", type=int, default=480)
    cameras.add_argument("--camera-fps", type=int, default=30)
    cameras.add_argument(
        "--camera-warmup",
        type=float,
        default=1.0,
        help="Seconds to warm up all cameras (default: 1.0).",
    )

    args = parser.parse_args()

    if args.left_interface == args.right_interface:
        parser.error("the left and right interfaces must be different")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("camera dimensions must be positive")
    if args.camera_fps <= 0:
        parser.error("--camera-fps must be positive")
    if args.camera_warmup < 0:
        parser.error("--camera-warmup must be non-negative")
    if not args.headless and not args.skip_cameras:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            parser.error("--headed requires DISPLAY or WAYLAND_DISPLAY")

    if args.mode == "read":
        if args.samples <= 0:
            parser.error("--samples must be positive")
        if args.interval < 0:
            parser.error("--interval must be non-negative")
    else:
        if not args.execute:
            parser.error("joint mode requires --execute")
        if args.duration is not None and args.duration <= 0:
            parser.error("--duration must be positive")
        if args.control_period <= 0:
            parser.error("--control-period must be positive")
        if args.controller_dt <= 0:
            parser.error("--controller-dt must be positive")
        if args.preview_time < 0:
            parser.error("--preview-time must be non-negative")
        if args.kp_scale <= 0 or args.kd_scale <= 0:
            parser.error("--kp-scale and --kd-scale must be positive")
        if not 0 < args.joint_safety_factor <= 1:
            parser.error("--joint-safety-factor must be in (0, 1]")
        if not 0 < args.velocity_safety_factor <= 1:
            parser.error("--velocity-safety-factor must be in (0, 1]")
        if args.max_joint_delta <= 0:
            parser.error("--max-joint-delta must be positive")
        if args.max_final_error <= 0:
            parser.error("--max-final-error must be positive")

    return args


def _camera_serials(args: argparse.Namespace) -> dict[str, str]:
    return {
        "head": args.head_camera_serial,
        "left_wrist": args.left_wrist_camera_serial,
        "right_wrist": args.right_wrist_camera_serial,
    }


def _validate_camera_frame(
    name: str, frame: Any, args: argparse.Namespace
) -> np.ndarray:
    array = np.asarray(frame)
    expected_shape = (args.camera_height, args.camera_width, 3)
    if array.shape != expected_shape:
        raise ValueError(
            f"camera {name!r} returned shape {array.shape}; expected {expected_shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(
            f"camera {name!r} returned dtype {array.dtype}; expected uint8"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"camera {name!r} contains NaN or infinity")
    return array


def _open_cameras(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_cameras:
        print("\n[2] RealSense camera checks skipped")
        return {}

    print("\n[2] Checking RealSense cameras ...")
    try:
        from rlinf.envs.realworld.common.camera.base_camera import CameraInfo
        from rlinf.envs.realworld.common.camera.realsense_camera import (
            RealSenseCamera,
        )
    except Exception as exc:
        raise RuntimeError(
            "RealSense support could not be imported: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    serials = _camera_serials(args)
    if len(set(serials.values())) != len(serials):
        raise ValueError("head and wrist camera serial numbers must be different")

    available = RealSenseCamera.get_device_serial_numbers()
    missing = {
        name: serial for name, serial in serials.items() if serial not in available
    }
    print(f"  Detected serials: {sorted(available)}")
    if missing:
        formatted = ", ".join(f"{name}={serial}" for name, serial in missing.items())
        raise RuntimeError(f"required RealSense cameras are missing: {formatted}")

    opened: dict[str, Any] = {}
    try:
        for name, serial in serials.items():
            info = CameraInfo(
                name=name,
                serial_number=serial,
                resolution=(args.camera_width, args.camera_height),
                fps=args.camera_fps,
                enable_depth=False,
            )
            camera = RealSenseCamera(info)
            camera.open()
            opened[name] = camera
            print(f"  PASS  opened {name}: serial={serial}")

        if args.camera_warmup:
            time.sleep(args.camera_warmup)
        for name, camera in opened.items():
            frame = _validate_camera_frame(name, camera.get_frame(timeout=5), args)
            print(
                f"  PASS  {name}: shape={frame.shape}, dtype={frame.dtype}, "
                f"mean={float(frame.mean()):.1f}"
            )
        return opened
    except Exception:
        _close_cameras(opened, args)
        raise


def _poll_cameras(
    cameras: dict[str, Any],
    cached_frames: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    if not cameras:
        return

    for name, camera in cameras.items():
        try:
            frame = camera.get_frame(timeout=0.001)
        except queue.Empty:
            continue
        cached_frames[name] = _validate_camera_frame(name, frame, args)

    if args.headless or len(cached_frames) != len(cameras):
        return

    import cv2

    panels = []
    for name in REFERENCE_CAMERAS:
        if name not in cached_frames:
            continue
        panel = cv2.resize(cached_frames[name], (426, 320))
        cv2.putText(
            panel,
            name,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    if panels:
        cv2.imshow("ARX X5 cameras", np.concatenate(panels, axis=1))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt


def _wait_with_cameras(
    duration: float,
    cameras: dict[str, Any],
    cached_frames: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        _poll_cameras(cameras, cached_frames, args)
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def _close_cameras(cameras: dict[str, Any], args: argparse.Namespace) -> None:
    for name, camera in cameras.items():
        try:
            camera.close()
            print(f"  Closed camera {name}")
        except Exception as exc:
            print(f"WARNING: failed to close camera {name}: {exc}", file=sys.stderr)
    if not args.headless:
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


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
        raise RuntimeError(f"network interface {interface!r} exists but is DOWN")

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
    print("\n[3] Importing pyarx ...")

    REPO_ROOT = Path(__file__).resolve().parents[2]
    arx5_prefix = REPO_ROOT / ".venv" / "arx5-conda-env"

    entries = [
        path
        for path in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
        if path
    ]
    if str(arx5_prefix) not in entries:
        os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(
            [str(arx5_prefix), *entries]
        )

    try:
        import pyarx as arx5
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        raise RuntimeError(
            "could not import pyarx; activate the environment created "
            "by requirements/install.sh and reinstall --env arx_x5_dual"
        ) from exc

    required_names = (
        "Arx5JointController",
        "RobotConfigFactory",
        "ControllerConfigFactory",
        "Gain",
        "JointState",
    )
    missing_names = [name for name in required_names if not hasattr(arx5, name)]
    if missing_names:
        raise RuntimeError(
            "pyarx is missing expected API members: " + ", ".join(missing_names)
        )

    try:
        package_version = version("pyarx")
    except PackageNotFoundError:
        package_version = "unknown"
    spec = importlib.util.find_spec("pyarx")
    print(f"  PASS  version: {package_version}")
    print(f"  PASS  module:  {spec.origin if spec else 'unknown'}")
    return arx5


def _make_controller(arx5: Any, args: argparse.Namespace, interface: str) -> Any:
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(args.model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )

    controller_config.shutdown_to_passive = True
    if args.mode == "joint":
        controller_config.controller_dt = args.controller_dt
        controller_config.default_preview_time = args.preview_time
        controller_config.background_send_recv = True
    else:
        controller_config.background_send_recv = False

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
    print(f"  gripper_width:           {robot.gripper_width}")
    print(f"  gripper_open_readout:    {robot.gripper_open_readout}")
    print(f"  controller_dt:           {config.controller_dt}")
    print(f"  default_preview_time:    {config.default_preview_time}")
    print(f"  background_send_recv:    {config.background_send_recv}")
    print(f"  shutdown_to_passive:     {config.shutdown_to_passive}")
    print(f"  gain.kp:                 {np.asarray(gain.kp())}")
    print(f"  gain.kd:                 {np.asarray(gain.kd())}")
    return dof


def _collect_feedback(controller: Any, side: str, dof: int) -> dict[str, Any]:
    joint = controller.get_joint_state()
    eef = controller.get_eef_state()
    return {
        "controller_timestamp": _finite_scalar(
            f"{side}.controller_timestamp", controller.get_timestamp()
        ),
        "joint_timestamp": _finite_scalar(f"{side}.joint_timestamp", joint.timestamp),
        "joint_position": _finite_vector(f"{side}.joint_position", joint.pos(), dof),
        "joint_velocity": _finite_vector(f"{side}.joint_velocity", joint.vel(), dof),
        "joint_torque": _finite_vector(f"{side}.joint_torque", joint.torque(), dof),
        "joint_gripper_position": _finite_scalar(
            f"{side}.joint_gripper_position", joint.gripper_pos
        ),
        "eef_pose_6d": _finite_vector(f"{side}.eef_pose_6d", eef.pose_6d(), 6),
    }


def _read_feedback(controller: Any, side: str, dof: int) -> dict[str, Any]:
    controller.recv_once()
    return _collect_feedback(controller, side, dof)


def _print_feedback(side: str, feedback: dict[str, Any]) -> None:
    print(f"  {side}:")
    for name, value in feedback.items():
        formatted = (
            np.array2string(value, precision=6, separator=", ")
            if isinstance(value, np.ndarray)
            else f"{value:.6f}"
        )
        print(f"    {name:>26}: {formatted}")


def _check_timestamp_progress(side: str, samples: list[dict[str, Any]]) -> None:
    if len(samples) < 2:
        return
    joint_times = np.asarray([sample["joint_timestamp"] for sample in samples])
    if not np.any(np.diff(joint_times) > 0):
        raise RuntimeError(
            f"{side} feedback timestamps did not advance across samples; "
            "check the interface and arm power"
        )


def _run_read_mode(
    controllers: dict[str, Any],
    dofs: dict[str, int],
    cameras: dict[str, Any],
    cached_frames: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    print("\n[5] Reading feedback with recv_once() ...")
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
        _poll_cameras(cameras, cached_frames, args)
        if sample_index < args.samples and args.interval:
            _wait_with_cameras(args.interval, cameras, cached_frames, args)

    for side, samples in all_samples.items():
        _check_timestamp_progress(side, samples)

    print("\nPASS: both arms and configured cameras returned valid feedback.")


def _selected_sides(arm: str) -> tuple[str, ...]:
    return ("left", "right") if arm == "both" else (arm,)


def _safe_joint_limits(robot: Any, factor: float) -> tuple[np.ndarray, np.ndarray]:
    raw_min = np.asarray(robot.joint_pos_min, dtype=np.float64)
    raw_max = np.asarray(robot.joint_pos_max, dtype=np.float64)
    # Match the reference implementation: move negative/positive X5 bounds
    # toward zero by the configured factor.
    return raw_min * factor, raw_max * factor


def _prepare_motion(
    arx5: Any,
    controllers: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    tuple[str, ...],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, float],
    dict[str, Any],
    float,
]:
    sides = _selected_sides(args.arm)
    no_joint_targets = all(getattr(args, f"{side}_joints") is None for side in sides)
    starts: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    gripper_starts: dict[str, float] = {}
    gripper_targets: dict[str, float] = {}
    command_buffers: dict[str, Any] = {}
    max_delta = 0.0

    print("\n[6] Safe joint-control startup ...")
    print("  1. Reading finite states while controllers are in damping mode")
    for side in sides:
        controller = controllers[side]
        robot = controller.get_robot_config()
        dof = int(robot.joint_dof)
        state = controller.get_joint_state()
        start = _finite_vector(f"{side}.start", state.pos(), dof)
        gripper_start = _finite_scalar(f"{side}.gripper_start", state.gripper_pos)

        requested = getattr(args, f"{side}_joints")
        if no_joint_targets:
            target = REFERENCE_START_JOINTS.copy()
            target_source = "reference start_position"
        elif requested is None:
            target = start.copy()
            target_source = "current position"
        else:
            target = _finite_vector(f"{side}.target", requested, dof)
            target_source = "command line"

        safe_min, safe_max = _safe_joint_limits(robot, args.joint_safety_factor)
        if np.any(target < safe_min) or np.any(target > safe_max):
            raise ValueError(
                f"{side} target {target} exceeds safe limits [{safe_min}, {safe_max}]"
            )

        delta = np.abs(target - start)
        side_max_delta = float(delta.max())
        if side_max_delta > args.max_joint_delta:
            raise ValueError(
                f"{side} maximum joint delta {side_max_delta:.3f} rad exceeds "
                f"--max-joint-delta={args.max_joint_delta:.3f}"
            )
        max_delta = max(max_delta, side_max_delta)

        gripper_target = getattr(args, f"{side}_gripper")
        if gripper_target is None:
            gripper_target = gripper_start
        gripper_target = _finite_scalar(f"{side}.gripper_target", gripper_target)
        if not 0.0 <= gripper_target <= float(robot.gripper_width):
            raise ValueError(
                f"{side} gripper target must be between 0 and "
                f"{robot.gripper_width}; got {gripper_target}"
            )

        starts[side] = start
        targets[side] = target
        gripper_starts[side] = gripper_start
        gripper_targets[side] = gripper_target

        command = arx5.JointState(dof)
        command.pos()[:] = start
        command.gripper_pos = gripper_start
        command_buffers[side] = command
        print(f"     {side}: source={target_source}, start={start}, target={target}")

    duration = args.duration if args.duration is not None else max(max_delta, 1.0) * 2.0

    print("  2. Checking required velocity against conservative limits")
    for side in sides:
        robot = controllers[side].get_robot_config()
        required_velocity = np.abs(targets[side] - starts[side]) / duration
        safe_velocity = (
            np.asarray(robot.joint_vel_max, dtype=np.float64)
            * args.velocity_safety_factor
        )
        if np.any(required_velocity > safe_velocity):
            raise ValueError(
                f"{side} required velocity {required_velocity} exceeds "
                f"safe velocity {safe_velocity}"
            )
        required_gripper_velocity = (
            abs(gripper_targets[side] - gripper_starts[side]) / duration
        )
        safe_gripper_velocity = (
            float(robot.gripper_vel_max) * args.velocity_safety_factor
        )
        if required_gripper_velocity > safe_gripper_velocity:
            raise ValueError(
                f"{side} required gripper velocity "
                f"{required_gripper_velocity:.4f} exceeds safe velocity "
                f"{safe_gripper_velocity:.4f}"
            )

    print("  3. Writing current positions into command buffers")
    for side in sides:
        controllers[side].set_joint_cmd(command_buffers[side])
    time.sleep(max(args.preview_time, args.controller_dt * 2))

    print(
        "  4. Enabling reference position gains "
        f"(kp x {args.kp_scale}, kd x {args.kd_scale})"
    )
    for side in sides:
        controller = controllers[side]
        config = controller.get_controller_config()
        dof = int(controller.get_robot_config().joint_dof)
        gain = arx5.Gain(dof)
        gain.kp()[:] = np.asarray(config.default_kp) * args.kp_scale
        gain.kd()[:] = np.asarray(config.default_kd) * args.kd_scale
        gain.gripper_kp = config.default_gripper_kp
        gain.gripper_kd = config.default_gripper_kd
        controller.set_gain(gain)

    print(
        f"  5. Prepared {duration:.3f} s motion at {1.0 / args.control_period:.1f} Hz"
    )
    return (
        sides,
        starts,
        targets,
        gripper_starts,
        gripper_targets,
        command_buffers,
        duration,
    )


def _command_joint_targets(
    arx5: Any,
    controllers: dict[str, Any],
    cameras: dict[str, Any],
    cached_frames: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    (
        sides,
        starts,
        targets,
        gripper_starts,
        gripper_targets,
        command_buffers,
        duration,
    ) = _prepare_motion(arx5, controllers, args)

    steps = max(1, int(np.ceil(duration / args.control_period)))
    motion_start = time.monotonic()
    for step in range(1, steps + 1):
        ratio = step / steps
        alpha = 0.5 - 0.5 * np.cos(np.pi * ratio)

        for side in sides:
            command = command_buffers[side]
            command.pos()[:] = starts[side] + alpha * (targets[side] - starts[side])
            command.gripper_pos = gripper_starts[side] + alpha * (
                gripper_targets[side] - gripper_starts[side]
            )
            controllers[side].set_joint_cmd(command)

        _poll_cameras(cameras, cached_frames, args)
        deadline = motion_start + step * args.control_period
        lag = time.monotonic() - deadline
        if lag > 0.25:
            raise RuntimeError(
                f"control loop lagged by {lag:.3f} s; entering damping mode"
            )
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    _wait_with_cameras(
        args.preview_time + args.control_period,
        cameras,
        cached_frames,
        args,
    )

    print("\n[7] Verifying final tracking error ...")
    for side in sides:
        actual = _finite_vector(
            f"{side}.final",
            controllers[side].get_joint_state().pos(),
            targets[side].size,
        )
        error = np.abs(actual - targets[side])
        print(f"  {side}: target={targets[side]}")
        print(f"  {side}: actual={actual}")
        print(f"  {side}: abs_error={error}")
        if float(error.max()) > args.max_final_error:
            raise RuntimeError(
                f"{side} final tracking error {float(error.max()):.3f} rad "
                f"exceeds --max-final-error={args.max_final_error:.3f}"
            )


def _run_joint_mode(
    arx5: Any,
    controllers: dict[str, Any],
    cameras: dict[str, Any],
    cached_frames: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    print("\n[5] Joint position control selected")
    print(
        "  WARNING: motion commands will be transmitted; clear the workspace "
        "and keep an emergency stop ready."
    )
    _command_joint_targets(arx5, controllers, cameras, cached_frames, args)
    print("\nPASS: requested joint motion completed within safety checks.")


def main() -> int:
    args = _parse_args()
    np.set_printoptions(linewidth=160, precision=6, suppress=False)

    print("ARX X5 dual-arm SDK and RealSense smoke test")
    print(f"  Python:          {sys.executable}")
    print(f"  Mode:            {args.mode}")
    print(f"  Display:         {'headless' if args.headless else 'headed'}")
    print(f"  Model:           {args.model}")
    print(f"  Left interface:  {args.left_interface}")
    print(f"  Right interface: {args.right_interface}")
    print(
        "  WARNING: controller construction and recv_once() transmit CAN "
        "frames; clear the workspace and keep an emergency stop ready."
    )

    controllers: dict[str, Any] = {}
    cameras: dict[str, Any] = {}
    cached_frames: dict[str, np.ndarray] = {}
    try:
        print("\n[1] Checking Linux CAN interfaces ...")
        if args.skip_interface_check:
            print("  SKIP  interface checks disabled")
        else:
            _check_interface(args.left_interface)
            _check_interface(args.right_interface)

        cameras = _open_cameras(args)
        arx5 = _load_sdk(args.verbose)

        print("\n[4] Constructing controllers in damping mode ...")
        controllers = {
            "left": _make_controller(arx5, args, args.left_interface),
            "right": _make_controller(arx5, args, args.right_interface),
        }
        print("  PASS  both Arx5JointController instances constructed")

        if args.warmup:
            _wait_with_cameras(args.warmup, cameras, cached_frames, args)

        dofs = {
            side: _print_controller_info(side, controller)
            for side, controller in controllers.items()
        }
        for side, controller in controllers.items():
            feedback = _collect_feedback(controller, side, dofs[side])
            _print_feedback(side, feedback)

        if args.mode == "read":
            _run_read_mode(controllers, dofs, cameras, cached_frames, args)
        else:
            _run_joint_mode(arx5, controllers, cameras, cached_frames, args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted; entering damping mode.")
        return 130
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        if controllers:
            print("\n[shutdown] Setting both arms to damping ...")
        for side, controller in controllers.items():
            try:
                controller.set_to_damping()
                print(f"  PASS  {side} arm set to damping")
            except Exception as exc:
                print(
                    f"WARNING: failed to set {side} arm to damping: {exc}",
                    file=sys.stderr,
                )
        if controllers:
            time.sleep(0.1)
        controllers.clear()
        gc.collect()
        _close_cameras(cameras, args)


if __name__ == "__main__":
    raise SystemExit(main())
