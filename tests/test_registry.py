import json

from app.pipeline.registry import list_runs, new_run_dir, promote, resolve_latest


def test_run_lifecycle(tmp_path):
    assert resolve_latest(tmp_path) is None
    run = new_run_dir(tmp_path, suffix="Test Run!")
    assert run.is_dir() and run.name.endswith("-testrun")
    assert resolve_latest(tmp_path) is None  # criado mas não promovido
    promote(tmp_path, run)
    assert resolve_latest(tmp_path) == run
    assert list_runs(tmp_path) == [run]


def test_bad_pointer_is_ignored(tmp_path):
    (tmp_path / "latest.json").write_text(json.dumps({"run": "../../etc"}))
    assert resolve_latest(tmp_path) is None
    (tmp_path / "latest.json").write_text("garbage")
    assert resolve_latest(tmp_path) is None
    (tmp_path / "latest.json").write_text(json.dumps({"run": "20260101T000000Z"}))
    assert resolve_latest(tmp_path) is None  # pasta inexistente
