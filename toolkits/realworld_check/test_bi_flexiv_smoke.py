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

"""Bi-Flexiv (dual Flexiv Rizon4 RT) hardware smoke test — no RLinf env involved.

Exercises, for BOTH arms at all times:

* flexiv_rt connection, fault clear, enable, operational wait
* joint positions / velocities (rad), TCP pose (wxyz quat), external wrench
* RT Cartesian streaming (1 kHz C++ thread) with tiny nudges on both arms
* RT non-blocking reset trajectory (``move_to_pose``) on both arms
* Xense serial grippers (auto-discovery by board-SN parity: odd -> left)

Config via environment variables (see STATION at the top of main()):

    LEFT_ROBOT_SN, RIGHT_ROBOT_SN        e.g. Rizon4s-062412
    HEAD_CAMERA_SERIAL                   realsense serial (empty to skip)
    LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA  /dev/v4l/by-id/* names (empty to skip)
    LEFT_GRIPPER_PORT, RIGHT_GRIPPER_PORT  explicit /dev/ttyUSB* (empty = auto)
    SKIP_MOVE=1                          connect/read only, no motion at all
    SMOKE_SAVE_FRAMES=<dir>              also save camera frames as PNG

Prereqs:
    bash requirements/install.sh embodied --env bi_flexiv
    getcap $(readlink -f .venv/bin/python)   # should show cap_sys_nice=ep

Run:
    python toolkits/realworld_check/test_bi_flexiv_smoke.py
"""

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

RESULTS: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))


@dataclass
class Station:
    """One physical bi-flexiv station. Fill in your own serials/poses."""

    left_robot_sn: str = ""
    right_robot_sn: str = ""
    # Start poses in DEGREES (MoveJ convention). Only used by the optional
    # `home` command; leave as None to disable homing.
    left_start_degree: list[float] | None = None
    right_start_degree: list[float] | None = None
    head_camera_serial: str = ""
    left_wrist_camera: str = ""  # /dev/v4l/by-id/ name
    right_wrist_camera: str = ""
    left_gripper_port: str = ""
    right_gripper_port: str = ""
    gripper_max_mm: float = 85.0
    stiffness_ratio: float = 0.2
    damping_ratio: list[float] = field(default_factory=lambda: [0.7] * 6)


class FlexivArmSmoke:
    """Minimal per-arm flexiv_rt driver for smoke testing (no Ray, no ROS)."""

    def __init__(self, side: str, sn: str, station: Station):
        import flexiv_rt as frt

        self._frt = frt
        self.side = side
        self.sn = sn
        self._station = station
        self.robot: "frt.Robot | None" = None
        self.cc: "frt.CartesianMotionForceControl | None" = None

    # ----------------------------------------------------------- lifecycle

    def connect(self) -> None:
        frt = self._frt
        print(f"[{self.side}] connecting to {self.sn} ...")
        self.robot = frt.Robot(self.sn, connect_retries=3, retry_interval_sec=1.0)
        if self.robot.fault():
            print(f"[{self.side}] fault detected, clearing ...")
            if not self.robot.ClearFault():
                raise RuntimeError(f"[{self.side}] failed to clear fault")
        self.robot.Enable()
        t0 = time.time()
        while not self.robot.operational():
            if time.time() - t0 > 30:
                raise TimeoutError(f"[{self.side}] not operational after 30 s")
            time.sleep(0.1)
        info = self.robot.info()
        print(
            f"[{self.side}] operational. model={info.model_name} dof={info.DoF} "
            f"K_x_nom={np.round(info.K_x_nom, 1).tolist()}"
        )

    def start_rt(self) -> None:
        """Switch to RT_CARTESIAN_MOTION_FORCE and start the 1 kHz thread."""
        frt = self._frt
        self._switch_mode(frt.Mode.RT_CARTESIAN_MOTION_FORCE)
        st = self._station
        kx = list(np.multiply(self.robot.info().K_x_nom, st.stiffness_ratio))
        self.robot.SetCartesianImpedance(kx, st.damping_ratio)
        self.robot.SetMaxContactWrench([30.0, 30.0, 30.0, 5.0, 5.0, 5.0])
        self.cc = self.robot.start_cartesian_control(
            task_name=f"SmokeRT_{self.side[0].upper()}",
            inner_control_hz=1000,
            interpolate_cmds=True,
        )
        # Seed with the live pose so the RT thread does not jump.
        self.cc.set_target_pose(list(self.robot.states().tcp_pose))
        time.sleep(0.2)
        print(
            f"[{self.side}] RT thread running: {self.cc.is_running()}, "
            f"stiffness={np.round(kx, 1).tolist()}"
        )

    def stop(self) -> None:
        # Destruction order matters (cc before robot), or the DDS transport
        # dies with 'terminate called without an active exception'.
        if self.cc is not None:
            try:
                self.cc.stop()
            except Exception as exc:
                print(f"[{self.side}] cc.stop() error: {exc}")
            self.cc = None
        if self.robot is not None:
            try:
                self.robot.Stop()
            except Exception as exc:
                print(f"[{self.side}] robot.Stop() error: {exc}")
            try:
                self.robot.close()
            except Exception:
                pass
            self.robot = None

    # --------------------------------------------------------------- state

    def _switch_mode(self, mode, timeout: float = 3.0) -> None:
        if self.robot.mode() == mode:
            return
        self.robot.SwitchMode(mode)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.robot.mode() == mode:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"[{self.side}] mode switch to {mode} failed (mode={self.robot.mode()})"
        )

    def tcp_pose(self) -> np.ndarray:
        """[x, y, z, qw, qx, qy, qz] in the robot base (world) frame."""
        if self.cc is not None and self.cc.is_running():
            return np.asarray(self.cc.get_state().tcp_pose, dtype=np.float64)
        return np.asarray(self.robot.states().tcp_pose, dtype=np.float64)

    def joints(self) -> tuple[np.ndarray, np.ndarray]:
        states = self.robot.states()
        return (
            np.asarray(states.q, dtype=np.float64),
            np.asarray(states.dq, dtype=np.float64),
        )

    def ext_wrench_tcp(self) -> np.ndarray:
        states = self.robot.states()
        return np.asarray(states.ext_wrench_in_tcp, dtype=np.float64)

    # -------------------------------------------------------------- motion

    def nudge_tcp(self, dx: float, dy: float, dz: float, seconds: float = 0.5) -> None:
        """Hold a small offset from the current pose for ``seconds`` then return."""
        assert self.cc is not None and self.cc.is_running()
        home = self.tcp_pose()
        target = home.copy()
        target[0] += dx
        target[1] += dy
        target[2] += dz
        self.cc.set_target_pose(target.tolist())
        time.sleep(seconds)
        self.cc.set_target_pose(home.tolist())
        time.sleep(seconds)

    def rt_move_to_pose(self, pose7: list[float], duration: float = 3.0) -> None:
        assert self.cc is not None and self.cc.is_running()
        self.cc.move_to_pose(pose7, duration_sec=duration)

    def is_moving(self) -> bool:
        return self.cc is not None and self.cc.is_moving()

    def move_joints_blocking(
        self, target_degree: list[float], vel_scale: int = 30
    ) -> None:
        """NRT MoveJ fallback (blocks until reachedTarget), then re-enter RT."""
        frt = self._frt
        was_rt = self.cc is not None and self.cc.is_running()
        if was_rt:
            self.cc.stop()
            self.cc = None
        self.robot.Stop()
        self._switch_mode(frt.Mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive(
            "MoveJ", {"target": target_degree, "jntVelScale": vel_scale}
        )
        t0 = time.time()
        while time.time() - t0 < 30:
            try:
                if self.robot.primitive_states().get("reachedTarget", 0) == 1:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise TimeoutError(f"[{self.side}] MoveJ did not complete in 30 s")
        if was_rt:
            self.start_rt()


class XenseGripperSmoke:
    """Minimal Xense serial gripper driver (normalized 0=closed, 1=open)."""

    def __init__(self, side: str, port: str, max_mm: float):
        self.side = side
        self.port = port
        self.max_mm = max_mm
        self._gripper = None

    def connect(self) -> None:
        from xensegripper import XenseSerialGripper

        port = self.port or self._auto_discover()
        print(f"[{self.side}] gripper on {port} ...")
        self._gripper = XenseSerialGripper(port=port, timeout=1.0)
        self.port = port

    def _auto_discover(self) -> str:
        """Board-SN parity convention from lerobot-xense: odd SN -> left."""
        found = self._scan_ports()
        for port, sn in found:
            want_odd = self.side == "left"
            if (sn % 2 == 1) == want_odd:
                return port
        raise RuntimeError(
            f"[{self.side}] no gripper matched SN parity "
            f"({'odd' if self.side == 'left' else 'even'}); scanned: {found}. "
            f"Set {self.side.upper()}_GRIPPER_PORT explicitly."
        )

    @staticmethod
    def _scan_ports() -> list[tuple[str, int]]:
        import glob

        from xensegripper import XenseSerialGripper

        results = []
        for port in sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")):
            try:
                g = XenseSerialGripper(port=port, timeout=0.3)
                info = g.get_device_info()
                sn = int(info.get("sn", info.get("board_sn", -1)))
                results.append((port, sn))
                g.release()
            except Exception:
                continue
        return results

    def set_normalized(self, value: float) -> None:
        assert 0.0 <= value <= 1.0
        self._gripper.set_position(value * self.max_mm, vmax=100.0, fmax=40.0)

    def position_normalized(self) -> float:
        status = self._gripper.get_gripper_status(timeout=0.5)
        if status is None or status.get("position") is None:
            raise RuntimeError(f"[{self.side}] gripper status unavailable")
        return float(status["position"]) / self.max_mm

    def close(self) -> None:
        if self._gripper is not None:
            try:
                self._gripper.set_position(self.max_mm, vmax=100.0, fmax=20.0)
                self._gripper.release()
            except Exception:
                pass
            self._gripper = None


# ---------------------------------------------------------------- cameras


def smoke_cameras(station: Station) -> None:
    from rlinf.envs.realworld.common.camera import CameraInfo, create_camera

    specs = []
    if station.head_camera_serial:
        specs.append(
            CameraInfo(
                name="head",
                serial_number=station.head_camera_serial,
                camera_type="realsense",
                resolution=(640, 480),
                fps=30,
            )
        )
    for name, serial in (
        ("left_wrist", station.left_wrist_camera),
        ("right_wrist", station.right_wrist_camera),
    ):
        if serial:
            specs.append(
                CameraInfo(
                    name=name,
                    serial_number=serial,
                    camera_type="lumos",
                    resolution=(640, 640),
                    fps=30,
                )
            )
    if not specs:
        report("cameras", True, "skipped (no serials configured)")
        return

    save_dir = os.environ.get("SMOKE_SAVE_FRAMES")
    for info in specs:
        try:
            cam = create_camera(info)
            cam.open()
            frame = cam.get_frame(timeout=5)
            cam.close()
            detail = f"{info.name}: frame shape={frame.shape} dtype={frame.dtype}"
            if save_dir:
                import cv2

                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f"{info.name}.png"), frame)
                detail += f" (saved to {save_dir}/{info.name}.png)"
            report(f"camera[{info.name}]", True, detail)
        except Exception as exc:
            report(f"camera[{info.name}]", False, str(exc))


# ------------------------------------------------------------------ main


def print_arm_state(arm: FlexivArmSmoke) -> None:
    q, dq = arm.joints()
    pose = arm.tcp_pose()
    wrench = arm.ext_wrench_tcp()
    print(f"[{arm.side}] q (rad)      = {np.round(q, 4).tolist()}")
    print(f"[{arm.side}] dq (rad/s)   = {np.round(dq, 4).tolist()}")
    print(
        f"[{arm.side}] tcp xyz (m)  = {np.round(pose[:3], 4).tolist()} "
        f"quat (wxyz) = {np.round(pose[3:], 4).tolist()}"
    )
    print(f"[{arm.side}] ext wrench  = {np.round(wrench, 2).tolist()}")


def main() -> int:
    station = Station(
        left_robot_sn=os.environ.get("LEFT_ROBOT_SN", "Rizon4s-062412"),
        right_robot_sn=os.environ.get("RIGHT_ROBOT_SN", "Rizon4s-062881"),
        head_camera_serial=os.environ.get("HEAD_CAMERA_SERIAL", ""),
        left_wrist_camera=os.environ.get("LEFT_WRIST_CAMERA", ""),
        right_wrist_camera=os.environ.get("RIGHT_WRIST_CAMERA", ""),
        left_gripper_port=os.environ.get("LEFT_GRIPPER_PORT", ""),
        right_gripper_port=os.environ.get("RIGHT_GRIPPER_PORT", ""),
    )
    skip_move = os.environ.get("SKIP_MOVE", "0") == "1"

    print("=" * 70)
    print("Bi-Flexiv smoke test (dual-arm at all times)")
    print(f"  left SN : {station.left_robot_sn}")
    print(f"  right SN: {station.right_robot_sn}")
    print(f"  SKIP_MOVE={skip_move}")
    print("=" * 70)

    left = FlexivArmSmoke("left", station.left_robot_sn, station)
    right = FlexivArmSmoke("right", station.right_robot_sn, station)
    left_grip = XenseGripperSmoke(
        "left", station.left_gripper_port, station.gripper_max_mm
    )
    right_grip = XenseGripperSmoke(
        "right", station.right_gripper_port, station.gripper_max_mm
    )

    try:
        # ---- connect both arms -------------------------------------------
        left.connect()
        right.connect()
        report("connect both arms", True)

        # ---- state reads --------------------------------------------------
        print_arm_state(left)
        print_arm_state(right)
        q_l, _ = left.joints()
        q_r, _ = right.joints()
        report(
            "read joint/tcp/wrench both arms",
            q_l.shape == (7,)
            and q_r.shape == (7,)
            and left.tcp_pose().shape == (7,)
            and right.tcp_pose().shape == (7,),
        )

        # ---- RT streaming both arms ---------------------------------------
        left.start_rt()
        right.start_rt()
        report(
            "start RT cartesian threads both arms",
            left.cc.is_running() and right.cc.is_running(),
        )

        if not skip_move:
            # Nudge both arms +z by 2 cm simultaneously, then back.
            left.nudge_tcp(0.0, 0.0, 0.02, seconds=0.8)
            right.nudge_tcp(0.0, 0.0, 0.02, seconds=0.8)
            report("nudge both arms +z 2 cm via RT stream", True)

            # RT non-blocking reset trajectory back to the current pose
            # (exercises move_to_pose without changing the arm position).
            left.rt_move_to_pose(left.tcp_pose().tolist(), duration=2.0)
            right.rt_move_to_pose(right.tcp_pose().tolist(), duration=2.0)
            t0 = time.time()
            while (left.is_moving() or right.is_moving()) and time.time() - t0 < 6:
                time.sleep(0.1)
            report(
                "RT move_to_pose trajectory both arms",
                not left.is_moving() and not right.is_moving(),
            )

        # ---- grippers ------------------------------------------------------
        for grip in (left_grip, right_grip):
            try:
                grip.connect()
                pos0 = grip.position_normalized()
                if skip_move:
                    report(f"gripper[{grip.side}] read", True, f"pos={pos0:.3f}")
                    continue
                grip.set_normalized(0.2)
                time.sleep(1.5)
                pos_closed = grip.position_normalized()
                grip.set_normalized(1.0)
                time.sleep(1.5)
                pos_open = grip.position_normalized()
                ok = pos_closed < 0.5 < pos_open
                report(
                    f"gripper[{grip.side}] close/open",
                    ok,
                    f"start={pos0:.3f} closed={pos_closed:.3f} open={pos_open:.3f}",
                )
            except Exception as exc:
                report(f"gripper[{grip.side}]", False, str(exc))

        # ---- cameras --------------------------------------------------------
        smoke_cameras(station)

    except Exception as exc:
        report("fatal", False, f"{type(exc).__name__}: {exc}")
    finally:
        for grip in (left_grip, right_grip):
            grip.close()
        left.stop()
        right.stop()

    print("=" * 70)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"Smoke test finished: {len(RESULTS) - n_fail} passed, {n_fail} failed.")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} — {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
