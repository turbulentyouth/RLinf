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
"""Robot-side live status panel for real-world evaluation.

Tails the JSONL event file written by ``rlinf.utils.eval_events.emit_event``
(env worker, keyboard wrapper, robot env hooks), reads the operator's pedal
key presses locally via evdev, derives the current eval phase (inference
wait / reset wait / homing / ...), and renders a live terminal panel. Every
received event (RLinf's and local key presses) is also appended to a record
file for post-hoc analysis.

Usage (run on the robot node, node 1, inside the RLinf venv)::

    # live panel (follows /tmp/rlinf_eval_events_latest.jsonl, auto-detect keyboard)
    python toolkits/realworld_check/eval_monitor.py

    # ssh / no tty: line-by-line output instead of the panel
    python toolkits/realworld_check/eval_monitor.py --plain --from-start

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
REDRAW_S = 0.2
STALLED_S = 60.0
RECENT_EVENTS_SHOWN = 8
MAX_LINE_WIDTH = 80

# Phase names (panel first line).
PHASE_CLOSING = "CLOSING"
PHASE_HOMING = "HOMING"
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
        self.last_rlinf_event: dict[str, Any] | None = None
        self.last_key_event: dict[str, Any] | None = None
        self.last_actions: dict[str, Any] | None = None
        self.last_reset_wait: dict[str, Any] | None = None
        self.evaluate_count = 0
        self.recent: list[dict[str, Any]] = []

    def ingest(self, record: dict[str, Any], recv_ts: float) -> None:
        record.setdefault("ts", recv_ts)
        event = str(record.get("event", ""))
        self.last_ts_by_event[event] = record["ts"]
        if record.get("src") == "monitor":
            self.last_key_event = record
        else:
            self.last_rlinf_event = record
        if event == "actions_received":
            self.last_actions = record
        elif event == "reset_wait_start":
            self.last_reset_wait = record
        elif event == "evaluate_start":
            self.evaluate_count += 1
            self.last_actions = None  # new cycle: reset chunk progress
        self.recent.append(record)
        del self.recent[:-RECENT_EVENTS_SHOWN]

    def _ts(self, event: str) -> float:
        return self.last_ts_by_event.get(event, 0.0)

    def phase(self) -> tuple[str, float | None]:
        """Return (phase_name, reset_wait_remaining_s_or_None)."""
        if self._ts("homing_start") > self._ts("homing_done") and self._ts(
            "homing_start"
        ) >= self._ts("close_envs_start"):
            return PHASE_HOMING, None
        if self._ts("close_envs_start") > self._ts("close_envs_done"):
            return PHASE_CLOSING, None
        if self._ts("reset_wait_start") > self._ts("reset_wait_end"):
            remaining = None
            if self.last_reset_wait is not None:
                seconds = float(self.last_reset_wait.get("seconds", 0.0))
                deadline = self.last_reset_wait["ts"] + seconds
                # Event ts and local clock may differ across hosts; clamp.
                remaining = max(0.0, deadline - time.time())
            return PHASE_RESET_WAIT, remaining
        if self._ts("idle_wait_start") > self._ts("idle_wait_end"):
            return PHASE_IDLE_WAIT, None
        if self._ts("obs_sent") > self._ts("actions_received"):
            return PHASE_WAITING_INFERENCE, None
        if self._ts("actions_received") > 0.0:
            return PHASE_STEPPING, None
        return PHASE_UNKNOWN, None


def _format_event_line(record: dict[str, Any]) -> str:
    ts = float(record.get("ts", 0.0))
    stamp = time.strftime("%H:%M:%S", time.localtime(ts))
    extras = " ".join(
        f"{k}={v}"
        for k, v in record.items()
        if k not in ("ts", "src", "event", "recv_ts")
    )
    line = f"{stamp} {record.get('src', '?')} {record.get('event', '?')}"
    if extras:
        line += f" {extras}"
    return line[:MAX_LINE_WIDTH]


def _age(now: float, ts: float | None) -> str:
    if ts is None:
        return "-"
    return f"{max(0.0, now - ts):.1f}s ago"


def _render_panel(state: _MonitorState, now: float) -> str:
    phase, remaining = state.phase()
    phase_line = f"Phase: {phase}"
    if phase == PHASE_RESET_WAIT and remaining is not None:
        phase_line += f"  (remaining {remaining:.1f}s)"

    last = state.last_rlinf_event
    if last is not None:
        last_line = (
            f"Last RLinf event: {last.get('event', '?')}  ({_age(now, last['ts'])})"
        )
        if phase != PHASE_RESET_WAIT and now - last["ts"] > STALLED_S:
            last_line += "  *** STALLED ***"
    else:
        last_line = "Last RLinf event: -"

    key = state.last_key_event
    if key is not None:
        key_line = f"Last key: {key.get('key', '?')}  ({_age(now, key['ts'])})"
    else:
        key_line = "Last key: -"

    if state.last_actions is not None:
        chunk = state.last_actions.get("chunk", "?")
        n_chunks = state.last_actions.get("n_chunks", "?")
        progress = f"Chunks this episode: {chunk}/{n_chunks}"
    else:
        progress = "Chunks this episode: -"
    progress += f"    Cycle/evaluate count: {state.evaluate_count}"

    lines = [
        "=== RLinf Eval Monitor (robot side) ===",
        phase_line,
        last_line,
        key_line,
        progress,
        "--- recent events ---",
    ]
    lines.extend(_format_event_line(r) for r in reversed(state.recent))
    return "\n".join(line[:MAX_LINE_WIDTH] for line in lines)


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
        description="Robot-side live status monitor for real-world evaluation.",
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
        "--plain",
        action="store_true",
        help="Print events line-by-line instead of the ANSI panel (ssh/no tty).",
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

    def ingest(record: dict[str, Any]) -> None:
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
        if args.plain:
            print(_format_event_line(record), flush=True)

    if not args.plain:
        # Hide cursor; restored in the finally block.
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    last_file_poll = 0.0
    last_redraw = 0.0
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
            if not args.plain and now - last_redraw >= REDRAW_S:
                last_redraw = now
                sys.stdout.write("\033[2J\033[H" + _render_panel(state, now) + "\n")
                sys.stdout.flush()
            time.sleep(KEYBOARD_POLL_S)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.plain:
            sys.stdout.write("\033[?25h\n")
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
