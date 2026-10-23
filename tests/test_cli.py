"""Command routing must configure paths before importing a research workflow."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from civol import cli


@pytest.mark.parametrize("arguments", [[], ["--help"], *[
    [command, "--help"] for command in sorted(cli.NO_ARGUMENTS)
]])
def test_help_never_imports_or_runs_a_workflow(monkeypatch, capsys, arguments):
    def unexpected_import(name):
        pytest.fail(f"Help imported workflow {name}")

    monkeypatch.setattr(cli, "import_module", unexpected_import)
    if arguments == ["--help"]:
        with pytest.raises(SystemExit) as error:
            cli.main(arguments)
        assert error.value.code == 0
    else:
        cli.main(arguments)
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("command", [*sorted(cli.NO_ARGUMENTS), "doctor"])
def test_invalid_arguments_do_not_start_work(monkeypatch, command):
    def unexpected_work(*args, **kwargs):
        pytest.fail("Invalid arguments started work")

    monkeypatch.setattr(cli, "import_module", unexpected_work)
    monkeypatch.setattr(cli, "doctor", unexpected_work)
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--invalid-option"])
    assert error.value.code == 2


def test_paths_are_configured_before_import_and_workflow_arguments_are_forwarded(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    expected = {"CIVOL_DATA_DIR": str(tmp_path / "input"),
                "CIVOL_WORK_DIR": str(tmp_path / "generated"),
                "CIVOL_CACHE_DIR": str(tmp_path / "cached"),
                "CIVOL_GPU_PYTHON": str(tmp_path / "gpu-python")}
    for variable in expected:
        monkeypatch.delenv(variable, raising=False)
    previous_argv = sys.argv
    calls = []

    def workflow_main():
        calls.append(tuple(sys.argv))

    def import_workflow(name):
        assert name == "civol.application.run_application"
        assert {name: os.environ[name] for name in expected} == expected
        return SimpleNamespace(main=workflow_main)

    monkeypatch.setattr(cli, "import_module", import_workflow)
    cli.main(["--data-dir", "input", "--work-dir", "generated",
              "--cache-dir", "cached", "--gpu-python", "gpu-python",
              "application", "--limit", "2"])
    assert calls == [("civol application", "--limit", "2")]
    assert sys.argv is previous_argv
    assert not any(tmp_path.iterdir())


def test_workflow_failure_restores_process_arguments(monkeypatch):
    previous_argv = sys.argv

    def fail():
        raise RuntimeError("test workflow failed")

    monkeypatch.setattr(cli, "import_module", lambda name: SimpleNamespace(main=fail))
    with pytest.raises(RuntimeError, match="test workflow failed"):
        cli.main(["application", "--limit", "1"])
    assert sys.argv is previous_argv


def test_doctor_in_fresh_process_reports_overrides_without_creating_data(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    env = {name: value for name, value in os.environ.items()
           if not name.startswith("CIVOL_")}
    env["PYTHONPATH"] = str(source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "civol", "--data-dir", "input",
         "--work-dir", "generated", "--cache-dir", "cached",
         "--gpu-python", sys.executable, "doctor"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["paths"]["DATA"] == str(tmp_path / "input")
    assert report["paths"]["WORK"] == str(tmp_path / "generated")
    assert report["paths"]["CACHE"] == str(tmp_path / "cached")
    assert Path(report["paths"]["GPU_PYTHON"]) == Path(sys.executable).resolve()
    assert report["cache_manifest_exists"] is False
    assert report["versions"]["numpy"] != "not installed"
    assert not any(tmp_path.iterdir())
