"""
NanoControl vision setup script.

Creates a .venv, installs compatible versions of PaddlePaddle + PaddleOCR
+ pyautogui, verifies the install, and pre-downloads the OCR models.

Usage (from the repo root):
    python setup_vision.py

Safe to re-run: skips steps that are already done.

If doesnt work: [pip install "paddlepaddle==3.2.0" --force-reinstall --no-deps]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

# --- Config ---------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"

# Pinned for reproducibility. These versions were verified mutually compatible
# on Python 3.10-3.12 (CPU). Bump deliberately, not accidentally.
REQUIREMENTS = [
    "paddlepaddle==3.3.1",
    "paddleocr==3.5.0",
    "pyautogui>=0.9.54",
    "pillow>=10.0",
    "numpy>=1.26,<3",
]

MIN_PY = (3, 10)
MAX_PY = (3, 12)  # inclusive — 3.13 may work but isn't verified

# --- Helpers --------------------------------------------------------------

class Color:
    # Only colour if the terminal supports it
    _on = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    GREEN = "\033[92m" if _on else ""
    YELLOW = "\033[93m" if _on else ""
    RED = "\033[91m" if _on else ""
    BOLD = "\033[1m" if _on else ""
    END = "\033[0m" if _on else ""


def step(msg: str) -> None:
    print(f"\n{Color.BOLD}==> {msg}{Color.END}")


def ok(msg: str) -> None:
    print(f"{Color.GREEN}[OK]{Color.END} {msg}")


def warn(msg: str) -> None:
    print(f"{Color.YELLOW}[WARN]{Color.END} {msg}")


def fail(msg: str) -> int:
    print(f"{Color.RED}[FAIL]{Color.END} {msg}")
    return 1


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming output. Raises on non-zero exit."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


# --- Steps ----------------------------------------------------------------

def check_python_version() -> bool:
    v = sys.version_info
    step(f"Checking Python version ({v.major}.{v.minor}.{v.micro})")
    if (v.major, v.minor) < MIN_PY:
        fail(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ required, you have {v.major}.{v.minor}")
        return False
    if (v.major, v.minor) > MAX_PY:
        warn(
            f"Python {v.major}.{v.minor} is newer than the tested range "
            f"({MIN_PY[0]}.{MIN_PY[1]}-{MAX_PY[0]}.{MAX_PY[1]}). "
            "It may work but hasn't been verified."
        )
    else:
        ok("Python version is compatible")
    return True


def ensure_venv() -> None:
    step(f"Setting up virtual environment at {VENV_DIR}")
    if VENV_DIR.exists() and venv_python().exists():
        ok("Virtual environment already exists")
        return
    if VENV_DIR.exists():
        warn("Partial .venv found — removing and recreating")
        shutil.rmtree(VENV_DIR)
    venv.create(VENV_DIR, with_pip=True)
    ok(f"Created {VENV_DIR}")


def install_deps() -> None:
    step("Installing dependencies (this may take a few minutes on first run)")
    py = str(venv_python())
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", *REQUIREMENTS])
    ok("Dependencies installed")


def verify_install() -> bool:
    step("Verifying PaddlePaddle install")
    py = str(venv_python())
    try:
        run([py, "-c", "import paddle; paddle.utils.run_check()"])
        ok("PaddlePaddle health check passed")
        return True
    except subprocess.CalledProcessError:
        fail("PaddlePaddle health check failed — see output above")
        return False


def prefetch_models() -> None:
    step("Pre-downloading OCR models (~100 MB, one-time)")
    py = str(venv_python())
    prefetch_code = (
        "from paddleocr import PaddleOCR; "
        "import numpy as np; "
        "ocr = PaddleOCR(use_textline_orientation=True, lang='en'); "
        # Tiny dummy image just to force model initialisation
        "ocr.predict(np.zeros((64, 64, 3), dtype='uint8')); "
        "print('Models ready.')"
    )
    try:
        run([py, "-c", prefetch_code])
        ok("Models cached")
    except subprocess.CalledProcessError:
        warn(
            "Model prefetch failed (check your internet connection). "
            "Not fatal — models will download on first real OCR call."
        )


def print_next_steps() -> None:
    activate_cmd = (
        r".venv\Scripts\activate" if os.name == "nt"
        else "source .venv/bin/activate"
    )
    print(f"\n{Color.BOLD}{Color.GREEN}Setup complete.{Color.END}\n")
    print("To start using NanoControl vision:")
    print(f"  1. Activate the environment:  {Color.BOLD}{activate_cmd}{Color.END}")
    print(f"  2. Run your script:           {Color.BOLD}python -m nanocontrol.test_vision{Color.END}")
    print(f"     (or however you normally invoke it)\n")


# --- Main -----------------------------------------------------------------

def main() -> int:
    print(f"{Color.BOLD}NanoControl vision setup{Color.END}")
    print(f"Repo root: {REPO_ROOT}")

    if not check_python_version():
        return 1
    try:
        ensure_venv()
        install_deps()
    except subprocess.CalledProcessError as e:
        return fail(f"Command failed with exit code {e.returncode}")

    if not verify_install():
        return 1
    prefetch_models()
    print_next_steps()
    return 0


if __name__ == "__main__":
    sys.exit(main())