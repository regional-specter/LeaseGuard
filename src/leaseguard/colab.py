"""Colab runtime bootstrap. Keep this module free of third-party imports."""

from __future__ import annotations

import os
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path
from subprocess import run

RUNTIME_DEPS: tuple[str, ...] = (
    "pydantic>=2.11,<3",
    "pypdf>=5.1,<6",
    "python-docx>=1.1,<2",
)
INFERENCE_DEPS: tuple[str, ...] = (
    "transformers",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
)


def in_colab() -> bool:
    """True when the process is a Google Colab kernel."""
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    return "google.colab" in sys.modules


def pip_extra_args() -> list[str]:
    """Return pip flags required on Colab when PEP 668 blocks system installs."""
    if not in_colab():
        return []
    marker = Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED"
    if marker.is_file():
        return ["--break-system-packages"]
    return []


def add_src_to_path(repo_dir: Path) -> Path:
    """Put ``src`` on ``sys.path`` and ``PYTHONPATH`` so imports work without hatchling."""
    src = (repo_dir / "src").resolve()
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if src_str not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([src_str, *parts]) if parts else src_str
    return src


def _try_ipython_pip(spec: str) -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    ipython = get_ipython()
    if ipython is None:
        return False
    extra = " ".join(pip_extra_args())
    magic = " ".join(part for part in ("install", extra, spec) if part)
    try:
        ipython.run_line_magic("pip", magic)
    except Exception as exc:
        print("%pip failed:", exc)
        return False
    return True


def pip_install(*packages: str) -> None:
    """Install packages, showing pip output. Never use ``-q``: Colab hides the real error."""
    if not packages:
        raise ValueError("pip_install requires at least one package spec")
    spec = " ".join(packages)
    if _try_ipython_pip(spec):
        return
    cmd = [sys.executable, "-m", "pip", "install", *pip_extra_args(), *packages]
    print("+", " ".join(cmd))
    completed = run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"pip install failed ({completed.returncode}): {spec}")


def try_editable_install(repo_dir: Path) -> bool:
    """Best-effort ``pip install -e .``. Hatchling/PEP 668 often fail on Colab."""
    try:
        pip_install("-e", str(repo_dir))
    except RuntimeError as exc:
        print(exc)
        return False
    return True


def _confirm_import() -> None:
    import leaseguard

    print(f"leaseguard {leaseguard.__version__} from {leaseguard.__file__}")


def install_runtime(repo_dir: Path | str, *, inference: bool = False) -> None:
    """Install LeaseGuard for a Colab notebook without requiring hatchling."""
    root = Path(repo_dir)
    os.chdir(root)
    add_src_to_path(root)
    if not try_editable_install(root):
        print("editable install unavailable; installing runtime deps and using src/")
        pip_install(*RUNTIME_DEPS)
    if inference:
        pip_install(*INFERENCE_DEPS)
    _confirm_import()


def has_project(path: Path) -> bool:
    """True when ``path`` is a LeaseGuard checkout with the src layout."""
    return (path / "pyproject.toml").is_file() and (path / "src" / "leaseguard").is_dir()


def checkout_repo(repo_url: str, repo_dir: Path, drive_fallback: Path | None = None) -> Path:
    """Clone or reuse a LeaseGuard checkout. Call this before importing other package modules."""
    if has_project(repo_dir):
        os.chdir(repo_dir)
        return repo_dir
    if repo_dir.exists() and repo_dir.is_dir() and not any(repo_dir.iterdir()):
        repo_dir.rmdir()
    if drive_fallback is not None and has_project(drive_fallback) and not repo_dir.exists():
        print(f"Using Drive checkout {drive_fallback}")
        os.symlink(drive_fallback, repo_dir, target_is_directory=True)
        os.chdir(repo_dir)
        return repo_dir
    if "YOUR_USER" in repo_url:
        hint = drive_fallback or (repo_dir.parent / "leaseguard" / "repo")
        raise SystemExit(
            "Set REPO_URL to your GitHub clone URL, for example "
            "https://github.com/regional-specter/LeaseGuard.git, "
            f"or copy the repo to {hint}."
        )
    completed = run(["git", "clone", repo_url, str(repo_dir)], check=False)
    if completed.returncode != 0 or not has_project(repo_dir):
        raise SystemExit(
            "git clone failed or the checkout is missing src/leaseguard. "
            "Check REPO_URL, repo visibility, and GitHub access."
        )
    os.chdir(repo_dir)
    return repo_dir


def bootstrap(
    repo_url: str,
    repo_dir: Path | str,
    *,
    inference: bool = False,
    drive_fallback: Path | str | None = None,
    extra_dirs: Sequence[Path] = (),
) -> Path:
    """Clone if needed, install runtime deps, and create output directories."""
    root = checkout_repo(
        repo_url,
        Path(repo_dir),
        None if drive_fallback is None else Path(drive_fallback),
    )
    add_src_to_path(root)
    install_runtime(root, inference=inference)
    for path in extra_dirs:
        path.mkdir(parents=True, exist_ok=True)
    return root
