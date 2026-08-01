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
"""Robot-side live event log for real-world evaluation.

Tails the JSONL event file written by ``rlinf.utils.eval_events.emit_event``
(env worker, keyboard wrapper, robot env hooks), reads the operator's pedal
key presses locally via evdev, and prints every event as a timestamped line
as it arrives. Phase transitions (inference wait / reset wait / homing /
moving to the start pose / ...) and stall warnings are printed as extra
lines. While an episode is being saved (between ``recording_save_start``
and ``recording_saved``), an animated in-place progress line is shown on a
tty and erased when the save completes. Every received event (RLinf's
and local key presses) is also appended to a record file for post-hoc
analysis.

Usage (run on the robot node, node 1, inside the RLinf venv)::

    # follow /tmp/rlinf_eval_events_latest.jsonl, auto-detect keyboard
    python toolkits/realworld_check/eval_monitor.py

    # replay an existing file from the beginning
    python toolkits/realworld_check/eval_monitor.py --from-start

    # specify the pedal device explicitly
    RLINF_KEYBOARD_DEVICE=/dev/input/by-id/... python toolkits/realworld_check/eval_monitor.py

Each evaluation run writes its own timestamped event file under
``logs/eval_events/``; the ``_latest`` symlink always points at the active
one, and this monitor detects the symlink re-pointing across runs. The
monitor's own record file also lands in ``logs/eval_events/`` with a
timestamp.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EVENT_FILE = "/tmp/rlinf_eval_events_latest.jsonl"

KEYBOARD_POLL_S = 0.05
FILE_POLL_S = 0.1
WATCHDOG_POLL_S = 1.0
STALLED_S = 60.0
SAVE_BAR_REDRAW_S = 0.1
SAVE_BAR_WIDTH = 30
SAVE_BAR_BLOCK = 8

# Phase names (derived from the event stream).
PHASE_CLOSING = "CLOSING"
PHASE_HOMING = "HOMING"
PHASE_TO_START = "TO_START_POSE"
PHASE_RESET_WAIT = "RESET_WAIT"
PHASE_IDLE_WAIT = "IDLE_WAIT_PEDAL"
PHASE_WAITING_INFERENCE = "WAITING_INFERENCE"
PHASE_STEPPING = "STEPPING"
PHASE_UNKNOWN = "STARTING/UNKNOWN"


class _EventTail:
    """Incrementally read new JSONL events, surviving truncation/rotation."""

    def __init__(self, path: Path, from_start: bool):
        self.path = path
        self.from_start = from_start
        self._file = None
        self._inode: int | None = None

    def read_new(self) -> list[dict[str, Any]]:
        """Return events appended since the last call (parsed dicts only)."""
        try:
            stat = self.path.stat()
        except OSError:
            # Not created yet; drop any stale handle and wait.
            self._close()
            return []
        if (
            self._file is None
            or stat.st_ino != self._inode
            or stat.st_size < self._file.tell()
        ):
            # First open, rotation (inode changed), or truncation: reopen.
            self._close()
            try:
                self._file = self.path.open("r", encoding="utf-8", errors="replace")
            except OSError:
                return []
            self._inode = stat.st_ino
            if not self.from_start:
                self._file.seek(0, os.SEEK_END)
        events: list[dict[str, Any]] = []
        for line in self._file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                events.append(record)
        return events

    def _close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
            self._inode = None


class _MonitorState:
    """Derive the current eval phase from the event stream."""

    def __init__(self):
        self.last_ts_by_event: dict[str, float] = {}
        self.last_rlinf_ts: float | None = None
        self.evaluate_count = 0
        self.current_phase = PHASE_UNKNOWN

    def ingest(self, record: dict[str, Any], recv_ts: float) -> None:
        record.setdefault("ts", recv_ts)
        event = str(record.get("event", ""))
        self.last_ts_by_event[event] = record["ts"]
        if record.get("src") != "monitor":
            self.last_rlinf_ts = record["ts"]
        if event == "evaluate_start":
            self.evaluate_count += 1

    def _ts(self, event: str) -> float:
        return self.last_ts_by_event.get(event, 0.0)

    def update_phase(self) -> bool:
        """Recompute the phase; return True when it changed."""
        new_phase = self._derive_phase()
        changed = new_phase != self.current_phase
        self.current_phase = new_phase
        return changed

    def _derive_phase(self) -> str:
        if self._ts("homing_start") > self._ts("homing_done") and self._ts(
            "homing_start"
        ) >= self._ts("close_envs_start"):
            return PHASE_HOMING
        if self._ts("close_envs_start") > self._ts("close_envs_done"):
            return PHASE_CLOSING
        if self._ts("reset_wait_start") > self._ts("reset_wait_end"):
            return PHASE_RESET_WAIT
        if self._ts("idle_wait_start") > self._ts("idle_wait_end"):
            return PHASE_IDLE_WAIT
        if self._ts("go_start_start") > self._ts("go_start_done"):
            return PHASE_TO_START
        if self._ts("obs_sent") > self._ts("actions_received"):
            return PHASE_WAITING_INFERENCE
        if self._ts("actions_received") > 0.0:
            return PHASE_STEPPING
        return PHASE_UNKNOWN


def _format_ts(ts: float) -> str:
    """Wall-clock timestamp with milliseconds, e.g. ``12:34:56.789``."""
    return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int(ts % 1 * 1000):03d}"


def _format_event_line(record: dict[str, Any]) -> str:
    stamp = _format_ts(float(record.get("ts", 0.0)))
    extras = " ".join(
        f"{k}={v}"
        for k, v in record.items()
        if k not in ("ts", "src", "event", "recv_ts")
    )
    line = f"[{stamp}] {record.get('src', '?'):<8} {record.get('event', '?')}"
    if extras:
        line += f"  {extras}"
    return line


def _render_save_bar(now: float, elapsed: float, frames: Any) -> str:
    """Indeterminate bouncing-block progress line for an ongoing episode save.

    LeRobot's ``save_episode`` reports no percentage, so the bar animates a
    highlight sliding back and forth and shows wall-clock elapsed time.
    """
    span = SAVE_BAR_WIDTH - SAVE_BAR_BLOCK
    period = 2 * span
    pos = int(elapsed * 15) % period
    if pos > span:
        pos = period - pos
    cells = ["."] * SAVE_BAR_WIDTH
    for i in range(SAVE_BAR_BLOCK):
        cells[pos + i] = "#"
    line = (
        f"[{_format_ts(now)}] recorder  saving episode "
        f"|{''.join(cells)}| {elapsed:.1f}s"
    )
    if frames is not None:
        line += f" ({frames} frames)"
    return line


def _open_keyboard(no_keyboard: bool):
    """Create a KeyboardListener; degrade to None with a warning on failure."""
    if no_keyboard:
        return None
    try:
        from rlinf.envs.realworld.common.keyboard.keyboard_listener import (
            KeyboardListener,
        )

        return KeyboardListener()
    except Exception as exc:
        print(
            f"[monitor] WARNING: keyboard unavailable ({exc}); "
            "running without local key capture.",
            file=sys.stderr,
            flush=True,
        )
        return None


def _default_event_file() -> Path:
    """Follow an explicit RLINF_EVAL_EVENT_LOG, otherwise the latest symlink."""
    raw = os.environ.get("RLINF_EVAL_EVENT_LOG")
    if raw is not None and raw.strip().lower() not in {"", "off", "none", "0"}:
        return Path(raw)
    return Path(DEFAULT_EVENT_FILE)


def _default_record_path() -> Path:
    stamp = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    return REPO_ROOT / "logs" / "eval_events" / f"monitor_record-{stamp}.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Robot-side live event log for real-world evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--event-file",
        type=Path,
        default=_default_event_file(),
        help=(
            "JSONL event file to tail (default: RLINF_EVAL_EVENT_LOG if set, "
            "else the /tmp latest-run symlink)."
        ),
    )
    ap.add_argument(
        "--record",
        type=Path,
        default=_default_record_path(),
        help="Append every received event here with an added recv_ts field.",
    )
    ap.add_argument(
        "--keyboard-device",
        type=str,
        default=None,
        help=(
            "evdev device path for the pedal/keyboard "
            "(default: RLINF_KEYBOARD_DEVICE env var, then auto-detect)."
        ),
    )
    ap.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Do not read the local keyboard/pedal.",
    )
    ap.add_argument(
        "--from-start",
        action="store_true",
        help="Read the event file from the beginning (default: tail new only).",
    )
    args = ap.parse_args()

    if args.keyboard_device:
        os.environ["RLINF_KEYBOARD_DEVICE"] = args.keyboard_device
    listener = _open_keyboard(args.no_keyboard)

    tail = _EventTail(args.event_file, from_start=args.from_start)
    state = _MonitorState()

    try:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        record_file = args.record.open("a", encoding="utf-8")
    except OSError as exc:
        print(
            f"[monitor] WARNING: cannot open record file {args.record}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        record_file = None

    bar_enabled = sys.stdout.isatty()
    saving_since: float | None = None
    saving_frames: Any = None

    def emit(line: str) -> None:
        # The save bar lives on the current line without a newline; erase it
        # before printing so lines never interleave. It redraws on the next
        # bar tick (unless the save just ended).
        if saving_since is not None and bar_enabled:
            sys.stdout.write("\r\033[K")
        print(line, flush=True)

    def ingest(record: dict[str, Any]) -> None:
        nonlocal saving_since, saving_frames
        recv_ts = time.time()
        state.ingest(record, recv_ts)
        if record_file is not None:
            record_file.write(
                json.dumps(
                    {**record, "recv_ts": recv_ts}, ensure_ascii=False, default=str
                )
                + "\n"
            )
            record_file.flush()
        emit(_format_event_line(record))
        if state.update_phase():
            emit(
                f"[{_format_ts(recv_ts)}] --- phase -> {state.current_phase} "
                f"(eval #{state.evaluate_count}) ---"
            )
        event = record.get("event")
        if record.get("src") != "monitor":
            if event == "recording_save_start":
                saving_since = float(record.get("ts", recv_ts))
                saving_frames = record.get("frames")
            elif event in ("recording_saved", "recording_discarded"):
                # Save finished; the bar line was already erased by emit().
                saving_since = None
                saving_frames = None

    emit(
        f"[{_format_ts(time.time())}] monitor  start  "
        f"event_file={args.event_file} record={args.record}"
    )

    last_file_poll = 0.0
    last_watchdog = 0.0
    last_bar_draw = 0.0
    stall_warned = False
    try:
        while True:
            now = time.time()
            if listener is not None:
                for key in listener.pop_pressed_keys():
                    ingest(
                        {
                            "ts": now,
                            "src": "monitor",
                            "event": "key_pressed",
                            "key": key,
                        }
                    )
            if now - last_file_poll >= FILE_POLL_S:
                last_file_poll = now
                for record in tail.read_new():
                    ingest(record)
                    stall_warned = False
            if now - last_watchdog >= WATCHDOG_POLL_S:
                last_watchdog = now
                last_rlinf_ts = state.last_rlinf_ts
                if (
                    not stall_warned
                    and saving_since is None  # saving is a legit quiet period
                    and last_rlinf_ts is not None
                    and state.current_phase != PHASE_RESET_WAIT
                    and now - last_rlinf_ts > STALLED_S
                ):
                    stall_warned = True
                    emit(
                        f"[{_format_ts(now)}] *** STALLED: no RLinf event for "
                        f"{now - last_rlinf_ts:.1f}s "
                        f"(phase={state.current_phase}) ***"
                    )
            if (
                saving_since is not None
                and bar_enabled
                and now - last_bar_draw >= SAVE_BAR_REDRAW_S
            ):
                last_bar_draw = now
                sys.stdout.write(
                    "\r\033[K"
                    + _render_save_bar(now, now - saving_since, saving_frames)
                )
                sys.stdout.flush()
            time.sleep(KEYBOARD_POLL_S)
    except KeyboardInterrupt:
        pass
    finally:
        if saving_since is not None and bar_enabled:
            # Delete the in-place bar so the shell prompt starts clean.
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        if record_file is not None:
            record_file.close()
        print(
            f"[monitor] exited. Events recorded to {args.record}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
