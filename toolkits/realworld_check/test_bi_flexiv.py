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

"""Hardware smoke-check for the bi_flexiv (Flexiv Rizon4 dual-arm RT) rig.

Two modes:

* ``read`` (default): connect, then only *read* joint angles and TCP poses.
  No motion command is ever sent. Safe first check on a new rig.
* ``move``: after connect, translate both arms' TCP forward by 10 cm along a
  chosen world axis (default +X), hold, then drive them back to the start
  pose via ``reset_to_initial_position()``. Requires ``--execute``.

Both modes always run the production ``connect`` (with ``go_to_start``) and
``disconnect`` (which homes the arms) lifecycle, matching how the eval env
drives the hardware.

Run on the robot node, as root (RT 1 kHz needs SCHED_FIFO), venv preserved:

    # read-only, no motion
    sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        python toolkits/realworld_check/test_bi_flexiv.py --mode read

    # translate both TCPs +10 cm in world X, then back to start
    sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        python toolkits/realworld_check/test_bi_flexiv.py --mode move --execute
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Joint angle source: RobotStates.q (link-side encoder, rad), 7 DOF per Rizon4.
_JOINT_DOF = 7
_TCP_AXES = ("x", "y", "z")


@dataclass
class CheckReport:
    """Collect pass/fail lines without aborting the shutdown path."""

    failures: list[str] = field(default_factory=list)

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        """Record and print one check."""

        if condition:
            print(f"  PASS  {label}")
            return
        message = f"{label}: {detail}" if detail else label
        self.failures.append(message)
        print(f"  FAIL  {message}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read or move the bi_flexiv Flexiv Rizon4 dual arms."
    )
    parser.add_argument(
        "--mode",
        choices=("read", "move"),
        default="read",
        help="read = joint/TCP readout only; move = forward 10 cm then back to start.",
    )
    parser.add_argument(
        "--mount-type",
        default="diagonal-02",
        help="lerobot-xense rig preset (forward-04/05/06, forward-dewu, diagonal-02).",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="x",
        help="World axis for the move-mode translation.",
    )
    parser.add_argument(
        "--distance", type=float, default=0.10, help="Translation distance in metres."
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Settle time after each motion before reading back.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of read-mode samples to print.",
    )
    parser.add_argument(
        "--stiffness-ratio",
        type=float,
        default=0.2,
        help="Cartesian impedance stiffness ratio (matches eval env default).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required confirmation for --mode move (physically moves both arms).",
    )
    parser.add_argument(
        "--with-peripherals",
        action="store_true",
        help="Also connect cameras and grippers (full diagonal-02 preset). "
        "Default is arms-only: the check only needs joint/TCP readout and "
        "TCP motion, so it skips cameras and grippers to avoid unrelated "
        "hardware failures.",
    )
    args = parser.parse_args()

    if args.mode == "move" and not args.execute:
        parser.error("--mode move requires --execute because it moves both arms")
    if args.distance <= 0:
        parser.error("--distance must be positive")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be non-negative")
    return args


def _make_robot(mount_type: str, stiffness_ratio: float, with_peripherals: bool):
    """Build the lerobot-xense BiFlexivRizon4RT wrapper (lazy hardware import).

    By default (``with_peripherals=False``) the config is stripped to
    arms-only: no cameras and no grippers. The check only reads joint/TCP
    state and sends TCP targets, so connecting cameras/grippers would only
    add unrelated hardware-failure surface (``connect()`` requires every
    configured camera to come up, and ``is_connected``/``get_observation()``
    read all of them).
    """

    try:
        from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import (
            BiFlexivRizon4RTConfig,
        )
        from lerobot.robots.utils import make_robot_from_config
    except ImportError as exc:
        raise RuntimeError(
            "bi_flexiv check needs lerobot-xense (lerobot.robots.bi_flexiv_rizon4_rt) "
            "and flexiv_rt. Re-run requirements/install.sh for openpi + bi_flexiv."
        ) from exc

    config = BiFlexivRizon4RTConfig(
        bi_mount_type=mount_type,
        use_force=False,
        go_to_start=True,
        stiffness_ratio=stiffness_ratio,
        inner_control_hz=1000,
        interpolate_cmds=True,
        enable_tactile_sensors=False,
        log_level="INFO",
    )
    if not with_peripherals:
        # config.__post_init__ already built cameras and nested gripper configs
        # from the preset, so clear both the top-level switches AND the nested
        # objects: cameras are instantiated in the robot __init__ from
        # config.cameras, and grippers from config.{left,right}_gripper.
        config.cameras = {}
        config.left_use_gripper = False
        config.right_use_gripper = False
        config.left_gripper = None
        config.right_gripper = None
    return make_robot_from_config(config)


def _read_joint_angles(robot) -> tuple[np.ndarray, np.ndarray]:
    """Read 7-DOF joint angles (rad) from each arm's RobotStates.q."""

    left = np.asarray(robot._left_robot.states().q, dtype=np.float64)
    right = np.asarray(robot._right_robot.states().q, dtype=np.float64)
    return left, right


def _read_tcp_xyz(robot) -> tuple[np.ndarray, np.ndarray]:
    """Read current TCP xyz (m) from get_observation()."""

    obs = robot.get_observation()
    left = np.array([obs[f"left_tcp.{a}"] for a in _TCP_AXES], dtype=np.float64)
    right = np.array([obs[f"right_tcp.{a}"] for a in _TCP_AXES], dtype=np.float64)
    return left, right


def _print_readout(
    left_q: np.ndarray, right_q: np.ndarray, left_tcp: np.ndarray, right_tcp: np.ndarray
) -> None:
    np.set_printoptions(linewidth=160, precision=4, suppress=True)
    print(f"    left  q (rad): {left_q}")
    print(f"    right q (rad): {right_q}")
    print(f"    left  TCP xyz (m): {left_tcp}")
    print(f"    right TCP xyz (m): {right_tcp}")


def _run_read_mode(robot, args: argparse.Namespace, report: CheckReport) -> None:
    """Sample joint angles and TCP poses a few times; verify shapes and finiteness."""

    print(f"\n[read] Sampling joint angles and TCP poses ({args.samples}x) ...")
    left_q, right_q = _read_joint_angles(robot)
    left_tcp, right_tcp = _read_tcp_xyz(robot)
    print("  -- start pose (after go_to_start) --")
    _print_readout(left_q, right_q, left_tcp, right_tcp)
    for index in range(args.samples):
        left_q, right_q = _read_joint_angles(robot)
        left_tcp, right_tcp = _read_tcp_xyz(robot)
        report.check(
            left_q.shape == (_JOINT_DOF,) and right_q.shape == (_JOINT_DOF,),
            f"sample {index + 1}: both arms report {_JOINT_DOF} joint angles",
            f"left={left_q.shape}, right={right_q.shape}",
        )
        report.check(
            bool(np.all(np.isfinite(left_q)) and np.all(np.isfinite(right_q))),
            f"sample {index + 1}: joint angles are finite",
        )
        report.check(
            bool(np.all(np.isfinite(left_tcp)) and np.all(np.isfinite(right_tcp))),
            f"sample {index + 1}: TCP poses are finite",
        )
        _print_readout(left_q, right_q, left_tcp, right_tcp)
        if index + 1 < args.samples:
            time.sleep(0.2)


def _send_absolute_tcp(robot, left_tcp: np.ndarray, right_tcp: np.ndarray) -> None:
    """Send an absolute TCP target, holding current rotation and grippers.

    Builds the full action dict from the *current* observation, then overwrites
    only the xyz entries with the new targets. This keeps the 6D rotation and
    gripper commands unchanged so the motion is a pure translation.
    """

    obs = robot.get_observation()
    action = dict(obs)
    for axis, value in zip(_TCP_AXES, left_tcp, strict=True):
        action[f"left_tcp.{axis}"] = float(value)
    for axis, value in zip(_TCP_AXES, right_tcp, strict=True):
        action[f"right_tcp.{axis}"] = float(value)
    robot.send_action(action)


def _wait_rt_idle(robot, timeout: float) -> None:
    """Block until neither arm's RT thread is executing a trajectory."""

    deadline = time.monotonic() + timeout
    while robot.rt_moving:
        if time.monotonic() > deadline:
            print("  WARN  RT trajectory did not finish before timeout; continuing")
            return
        time.sleep(0.05)


def _run_move_mode(robot, args: argparse.Namespace, report: CheckReport) -> None:
    """Translate both TCPs forward, then return to the start pose."""

    axis_index = _TCP_AXES.index(args.axis)
    delta = np.zeros(3, dtype=np.float64)
    delta[axis_index] = args.distance

    left_start, right_start = _read_tcp_xyz(robot)
    left_target = left_start + delta
    right_target = right_start + delta
    print(
        f"\n[move] Translating both TCPs +{args.distance:.3f} m along world "
        f"'{args.axis}' ..."
    )
    print(f"    left  start {left_start} -> target {left_target}")
    print(f"    right start {right_start} -> target {right_target}")

    _send_absolute_tcp(robot, left_target, right_target)
    _wait_rt_idle(robot, timeout=args.settle_seconds + 5.0)
    time.sleep(args.settle_seconds)

    left_now, right_now = _read_tcp_xyz(robot)
    _print_readout(*_read_joint_angles(robot), left_now, right_now)
    report.check(
        bool(left_now[axis_index] > left_start[axis_index] + 0.5 * args.distance),
        f"left arm moved forward along '{args.axis}'",
        f"start={left_start[axis_index]:.4f}, now={left_now[axis_index]:.4f}",
    )
    report.check(
        bool(right_now[axis_index] > right_start[axis_index] + 0.5 * args.distance),
        f"right arm moved forward along '{args.axis}'",
        f"start={right_start[axis_index]:.4f}, now={right_now[axis_index]:.4f}",
    )

    print("\n[return] reset_to_initial_position() back to start ...")
    robot.reset_to_initial_position()
    _wait_rt_idle(robot, timeout=args.settle_seconds + 15.0)
    time.sleep(args.settle_seconds)

    left_back, right_back = _read_tcp_xyz(robot)
    _print_readout(*_read_joint_angles(robot), left_back, right_back)
    report.check(
        bool(np.allclose(left_back, left_start, atol=0.01)),
        "left arm returned to start TCP",
        f"start={left_start}, back={left_back}",
    )
    report.check(
        bool(np.allclose(right_back, right_start, atol=0.01)),
        "right arm returned to start TCP",
        f"start={right_start}, back={right_back}",
    )


def main() -> int:
    args = _parse_args()
    report = CheckReport()
    robot = None

    print("bi_flexiv (Flexiv Rizon4 dual-arm RT) hardware check")
    print(f"  Mode:        {args.mode}")
    print(f"  Mount type:  {args.mount_type}")
    print(
        f"  Peripherals: {'cameras + grippers' if args.with_peripherals else 'arms only'}"
    )
    if args.mode == "move":
        print(f"  Motion:      +{args.distance:.3f} m along '{args.axis}', then back")
        print(
            "  WARNING: both arms will move. Clear the workspace and keep the "
            "emergency stop ready."
        )

    try:
        print("\n[connect] Connecting and moving to start pose ...")
        robot = _make_robot(
            args.mount_type, args.stiffness_ratio, args.with_peripherals
        )
        robot.connect(calibrate=False, go_to_start=True)
        report.check(robot.is_connected, "both arms connected")
        report.check(
            robot.left_rt_running and robot.right_rt_running,
            "both RT control loops are running",
        )

        if args.mode == "read":
            _run_read_mode(robot, args, report)
        else:
            _run_move_mode(robot, args, report)
    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl+C caught; running shutdown cleanup ...")
    except BaseException as exc:  # noqa: BLE001 - report and still shut down
        report.failures.append(f"{type(exc).__name__}: {exc}")
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if robot is not None:
            print("\n[shutdown] disconnect() (homes both arms) ...")
            try:
                robot.disconnect()
                print("  PASS  disconnected and homed")
            except BaseException as exc:  # noqa: BLE001
                report.failures.append(f"disconnect raised {type(exc).__name__}: {exc}")
                print(f"  FAIL  disconnect raised: {exc}", file=sys.stderr)

    print("\n[result]")
    if report.failures:
        for failure in report.failures:
            print(f"  - {failure}", file=sys.stderr)
        print("FAIL: bi_flexiv hardware check did not pass.", file=sys.stderr)
        return 1
    print(f"PASS: bi_flexiv {args.mode}-mode check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
