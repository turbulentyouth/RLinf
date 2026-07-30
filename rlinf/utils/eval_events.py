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
"""Best-effort JSONL event emitter for real-world evaluation observability.

Key state-transition points in the eval loop (env worker, keyboard wrapper,
robot env) call :func:`emit_event` to append one JSON line per event to a
local file. A separate terminal-side monitor
(``toolkits/realworld_check/eval_monitor.py``) tails that file and renders a
live phase panel for the robot-side operator.

By default each run (i.e. each emitting process) writes its own timestamped
file ``logs/eval_events/<YYYYmmdd-HH:MM:SS>.jsonl`` under the repo root, so
records from different runs never mix. A stable symlink
``/tmp/rlinf_eval_events_latest.jsonl`` is updated to point at the active
file so the monitor can follow it without knowing the timestamp.

This is strictly observational: emission must never raise, and once a write
fails the emitter disables itself to avoid repeated errors on the control
path. Set ``RLINF_EVAL_EVENT_LOG`` to an explicit path to override the
default location, or to ``off``/``none``/``0``/empty to fully disable
emission (zero-overhead no-op).
"""

import json
import os
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LATEST_LINK = "/tmp/rlinf_eval_events_latest.jsonl"
_FALLBACK_DIR = Path("/tmp")
_ENV_VAR = "RLINF_EVAL_EVENT_LOG"
_DISABLED_VALUES = {"", "off", "none", "0"}

_write_lock = threading.Lock()
_write_failed = False
_resolved_default_path: str | None = None


def _logs_dir() -> Path:
    """Directory holding the per-run timestamped event files."""
    return _REPO_ROOT / "logs" / "eval_events"


def _resolve_default_path() -> str:
    """Pick a fresh timestamped path under logs/eval_events (once per process)."""
    stamp = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    try:
        base = _logs_dir()
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        base = _FALLBACK_DIR
    candidate = base / f"{stamp}.jsonl"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}-{suffix}.jsonl"
        suffix += 1
    return str(candidate)


def _update_latest_link(path: str) -> None:
    """Best-effort: point the stable /tmp symlink at the active event file."""
    try:
        tmp_link = _LATEST_LINK + ".tmp"
        try:
            os.unlink(tmp_link)
        except OSError:
            pass
        os.symlink(path, tmp_link)
        os.replace(tmp_link, _LATEST_LINK)
    except OSError:
        pass


def get_event_log_path() -> str | None:
    """Return the active event log path, or None if emission is disabled."""
    global _resolved_default_path
    raw = os.environ.get(_ENV_VAR)
    if raw is not None and raw.strip().lower() in _DISABLED_VALUES:
        return None
    if raw is not None:
        return raw
    with _write_lock:
        if _resolved_default_path is None:
            _resolved_default_path = _resolve_default_path()
            _update_latest_link(_resolved_default_path)
    return _resolved_default_path


def emit_event(event: str, src: str, **fields) -> None:
    """Append one JSONL event record; never raises.

    Args:
        event: Event name (e.g. ``"evaluate_start"``).
        src: Emitting component tag (e.g. ``"env_worker"``, ``"wrapper"``).
        **fields: Extra JSON-serializable payload merged into the record.
    """
    global _write_failed
    path = get_event_log_path()
    if path is None or _write_failed:
        return
    record = {"ts": time.time(), "src": src, "event": event, **fields}
    try:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with _write_lock:
            # Open/append/close per event so each line is flushed to the FS
            # immediately; a crash must not lose the last state transition.
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        _write_failed = True
    except Exception:
        _write_failed = True
