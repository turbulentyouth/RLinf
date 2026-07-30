# Copyright 2026 The RLinf Authors.

import json
import os

import pytest

from rlinf.utils import eval_events
from rlinf.utils.eval_events import emit_event, get_event_log_path


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    monkeypatch.setattr(eval_events, "_write_failed", False)
    monkeypatch.setattr(eval_events, "_resolved_default_path", None)


def test_emit_event_writes_jsonl(monkeypatch, tmp_path):
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("RLINF_EVAL_EVENT_LOG", str(log))

    emit_event("evaluate_start", "env_worker")
    emit_event("obs_sent", "env_worker", kind="reset", epoch=0)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "evaluate_start"
    assert first["src"] == "env_worker"
    assert isinstance(first["ts"], float)
    second = json.loads(lines[1])
    assert second["kind"] == "reset"
    assert second["epoch"] == 0


def test_emit_event_disabled_is_noop(monkeypatch, tmp_path):
    log = tmp_path / "events.jsonl"
    for value in ("", "off", "OFF", "none", "0"):
        monkeypatch.setenv("RLINF_EVAL_EVENT_LOG", value)
        assert get_event_log_path() is None
        emit_event("evaluate_start", "env_worker")
    assert not log.exists()


def test_emit_event_fields_passthrough_and_default_str(monkeypatch, tmp_path):
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("RLINF_EVAL_EVENT_LOG", str(log))

    emit_event("idle_wait_end", "wrapper", reason="start_key", key="a", obj=object())

    record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert record["reason"] == "start_key"
    assert record["key"] == "a"
    assert isinstance(record["obj"], str)


def test_default_path_is_timestamped_under_logs_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("RLINF_EVAL_EVENT_LOG", raising=False)
    monkeypatch.setattr(eval_events, "_logs_dir", lambda: tmp_path / "eval_events")
    link = tmp_path / "latest.jsonl"
    monkeypatch.setattr(eval_events, "_LATEST_LINK", str(link))

    path = get_event_log_path()
    assert path is not None
    assert path.startswith(str(tmp_path / "eval_events"))
    assert path.endswith(".jsonl")
    # Resolved once per process and stable across calls.
    assert get_event_log_path() == path
    # The stable latest symlink points at the active file.
    assert os.path.realpath(link) == os.path.realpath(path)

    # A second resolution in the same second must not clobber: simulate the
    # file already existing and reset the cache.
    (tmp_path / "eval_events" / os.path.basename(path)).touch()
    monkeypatch.setattr(eval_events, "_resolved_default_path", None)
    path2 = get_event_log_path()
    assert path2 is not None and path2 != path


def test_default_path_falls_back_to_tmp_when_logs_dir_uncreatable(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("RLINF_EVAL_EVENT_LOG", raising=False)

    def _raising_mkdir(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(eval_events, "_logs_dir", lambda: tmp_path / "no" / "way")
    monkeypatch.setattr("pathlib.Path.mkdir", _raising_mkdir)
    monkeypatch.setattr(eval_events, "_LATEST_LINK", str(tmp_path / "latest.jsonl"))

    path = get_event_log_path()
    assert path is not None
    assert os.path.dirname(path) == "/tmp"
    assert path.endswith(".jsonl")
    monkeypatch.setenv("RLINF_EVAL_EVENT_LOG", "/custom/path.jsonl")
    assert get_event_log_path() == "/custom/path.jsonl"


def test_emit_event_oserror_disables_silently(monkeypatch, tmp_path):
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("RLINF_EVAL_EVENT_LOG", str(log))

    def _raising_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _raising_open)
    # Must not raise, and disables itself after the first failure.
    emit_event("evaluate_start", "env_worker")
    assert eval_events._write_failed is True
    emit_event("evaluate_end", "env_worker")
    assert not log.exists()
