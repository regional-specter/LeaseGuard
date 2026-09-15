"""Tests for Colab runtime bootstrap."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from leaseguard.colab import (
    _confirm_import,
    add_src_to_path,
    bootstrap,
    checkout_repo,
    has_project,
    in_colab,
    install_runtime,
    pip_extra_args,
    pip_install,
    try_editable_install,
)


def _write_checkout(root: Path) -> Path:
    package = root / "src" / "leaseguard"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'leaseguard-ai'\n")
    (package / "__init__.py").write_text('__version__ = "0.1.0"\n')
    return root


def test_in_colab_reads_release_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLAB_RELEASE_TAG", "2026.07")
    assert in_colab() is True


def test_in_colab_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    assert in_colab() is False


def test_pip_extra_args_colab_pep668(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "EXTERNALLY-MANAGED"
    marker.write_text("externally managed\n")
    monkeypatch.setenv("COLAB_RELEASE_TAG", "2026.07")
    monkeypatch.setattr("leaseguard.colab.sysconfig.get_path", lambda _name: str(tmp_path))
    assert pip_extra_args() == ["--break-system-packages"]


def test_pip_extra_args_empty_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    sys.modules.pop("google.colab", None)
    assert pip_extra_args() == []


def test_pip_extra_args_colab_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLAB_RELEASE_TAG", "2026.07")
    monkeypatch.setattr("leaseguard.colab.sysconfig.get_path", lambda _name: str(tmp_path))
    assert pip_extra_args() == []


def test_add_src_to_path_sets_pythonpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "LeaseGuard"
    (repo / "src").mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONPATH", "")
    src = add_src_to_path(repo)
    assert src == (repo / "src").resolve()
    assert str(src) in sys.path
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(src)


def test_has_project(tmp_path: Path) -> None:
    repo = _write_checkout(tmp_path / "LeaseGuard")
    assert has_project(repo) is True
    assert has_project(tmp_path) is False


def test_checkout_reuses_existing_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_checkout(tmp_path / "LeaseGuard")
    monkeypatch.chdir(tmp_path)
    assert checkout_repo("https://example.invalid/LeaseGuard.git", repo) == repo


def test_checkout_placeholder_without_drive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Set REPO_URL"):
        checkout_repo(
            "https://github.com/YOUR_USER/LeaseGuard.git",
            tmp_path / "LeaseGuard",
        )


def test_checkout_empty_dir_uses_drive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "LeaseGuard"
    repo.mkdir()
    drive = _write_checkout(tmp_path / "drive" / "repo")
    monkeypatch.chdir(tmp_path)
    result = checkout_repo(
        "https://github.com/YOUR_USER/LeaseGuard.git",
        repo,
        drive,
    )
    assert result == repo
    assert has_project(repo)


def test_checkout_clones_when_needed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "LeaseGuard"

    def fake_run(cmd: list[str], check: bool = False) -> object:
        dest = Path(cmd[3])
        _write_checkout(dest)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("leaseguard.colab.run", fake_run)
    monkeypatch.chdir(tmp_path)
    checkout_repo("https://github.com/regional-specter/LeaseGuard.git", repo)
    assert has_project(repo)


def test_checkout_clone_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "leaseguard.colab.run",
        lambda _cmd, check=False: types.SimpleNamespace(returncode=1),
    )
    with pytest.raises(SystemExit, match="git clone failed"):
        checkout_repo("https://github.com/regional-specter/LeaseGuard.git", tmp_path / "x")


def test_pip_install_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("leaseguard.colab._try_ipython_pip", lambda _spec: False)
    monkeypatch.setattr("leaseguard.colab.pip_extra_args", lambda: [])
    recorded: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], check: bool = False) -> object:
        recorded["cmd"] = cmd
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("leaseguard.colab.run", fake_run)
    pip_install("pydantic>=2.11,<3")
    assert recorded["cmd"][-1] == "pydantic>=2.11,<3"
    assert "-q" not in recorded["cmd"]


def test_pip_install_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("leaseguard.colab._try_ipython_pip", lambda _spec: False)
    monkeypatch.setattr("leaseguard.colab.pip_extra_args", lambda: ["--break-system-packages"])
    monkeypatch.setattr(
        "leaseguard.colab.run",
        lambda _cmd, check=False: types.SimpleNamespace(returncode=1),
    )
    with pytest.raises(RuntimeError, match="pip install failed"):
        pip_install("pydantic")


def test_pip_install_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        pip_install()


def test_pip_install_uses_ipython(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Shell:
        def __init__(self) -> None:
            self.spec = ""

        def run_line_magic(self, name: str, spec: str) -> None:
            assert name == "pip"
            self.spec = spec

    shell = _Shell()
    fake = types.ModuleType("IPython")
    fake.get_ipython = lambda: shell  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake)
    monkeypatch.setattr("leaseguard.colab.pip_extra_args", lambda: [])
    pip_install("pypdf")
    assert "pypdf" in shell.spec


def test_try_editable_install_false_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_packages: str) -> None:
        raise RuntimeError("editable failed")

    monkeypatch.setattr("leaseguard.colab.pip_install", fail)
    assert try_editable_install(tmp_path) is False


def test_install_runtime_falls_back_to_src(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_checkout(tmp_path / "LeaseGuard")
    calls: list[tuple[str, ...]] = []

    def fake_pip(*packages: str) -> None:
        calls.append(packages)
        if packages and packages[0] == "-e":
            raise RuntimeError("CalledProcessError")

    monkeypatch.setattr("leaseguard.colab.pip_install", fake_pip)
    monkeypatch.setattr("leaseguard.colab._confirm_import", lambda: None)
    monkeypatch.chdir(tmp_path)
    install_runtime(repo, inference=True)
    assert any(package[0] == "-e" for package in calls)
    assert any(any("pydantic" in part for part in package) for package in calls)
    assert any("transformers" in package for package in calls)


def test_install_runtime_skips_inference_deps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _write_checkout(tmp_path / "LeaseGuard")
    calls: list[tuple[str, ...]] = []

    def fake_pip(*packages: str) -> None:
        calls.append(packages)

    monkeypatch.setattr("leaseguard.colab.pip_install", fake_pip)
    monkeypatch.setattr("leaseguard.colab._confirm_import", lambda: None)
    monkeypatch.chdir(tmp_path)
    install_runtime(repo, inference=False)
    assert calls == [("-e", str(repo))]


def test_try_editable_install_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("leaseguard.colab.pip_install", lambda *_packages: None)
    assert try_editable_install(tmp_path) is True


def test_confirm_import(capsys: pytest.CaptureFixture[str]) -> None:
    _confirm_import()
    captured = capsys.readouterr()
    assert "leaseguard 0.1.0" in captured.out


def test_bootstrap_creates_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_checkout(tmp_path / "LeaseGuard")
    reports = tmp_path / "reports"
    monkeypatch.setattr("leaseguard.colab.try_editable_install", lambda _repo: True)
    monkeypatch.setattr("leaseguard.colab._confirm_import", lambda: None)
    monkeypatch.chdir(tmp_path)
    bootstrap("https://example.invalid/LeaseGuard.git", repo, extra_dirs=(reports,))
    assert reports.is_dir()


def test_pip_install_ipython_missing_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setattr("leaseguard.colab.pip_extra_args", lambda: [])
    monkeypatch.setattr(
        "leaseguard.colab.run",
        lambda _cmd, check=False: types.SimpleNamespace(returncode=0),
    )
    pip_install("pypdf")


def test_pip_install_ipython_none_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("IPython")
    fake.get_ipython = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake)
    monkeypatch.setattr("leaseguard.colab.pip_extra_args", lambda: [])
    monkeypatch.setattr(
        "leaseguard.colab.run",
        lambda _cmd, check=False: types.SimpleNamespace(returncode=0),
    )
    pip_install("pypdf")


def test_pip_install_ipython_magic_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Shell:
        def run_line_magic(self, _name: str, _spec: str) -> None:
            raise RuntimeError("magic failed")

    fake = types.ModuleType("IPython")
    fake.get_ipython = lambda: _Shell()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake)
    monkeypatch.setattr("leaseguard.colab.pip_extra_args", lambda: [])
    monkeypatch.setattr(
        "leaseguard.colab.run",
        lambda _cmd, check=False: types.SimpleNamespace(returncode=0),
    )
    pip_install("pypdf")
