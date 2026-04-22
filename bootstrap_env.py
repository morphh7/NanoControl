"""
NanoControl environment bootstrap.

Run this once on any machine:

    python bootstrap_env.py

What it does (all automatic, safe to re-run):
  1. Verifies the host Python version (3.9 - 3.13).
  2. Creates (or reuses) a virtual environment at `.venv/`.
  3. Re-executes itself inside the venv so everything below installs
     into the venv and not your system Python.
  4. Detects whether a compatible NVIDIA GPU is present and, if so,
     picks the right PaddlePaddle-GPU wheel (CUDA 11.8 / 12.6 / 12.9).
     Falls back to the CPU wheel otherwise.
  5. Installs PaddlePaddle, PaddleOCR, pyautogui, Pillow, numpy.
  6. Installs this project in editable mode (`pip install -e .`) so
     `import nanocontrol` just works from anywhere inside the venv.
  7. Runs `paddle.utils.run_check()` to confirm the install.
  8. Pre-fetches the OCR models so the first real run is instant.

`setup.py` is a tiny setuptools shim for `pip install -e .` only; do not
run that file by hand. Use this script to bootstrap the environment.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --- Config ---------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"

MIN_PY = (3, 9)
MAX_PY = (3, 13)

# PaddlePaddle wheel selection. Versions per the official Paddle install
# docs (Windows/Linux, stable track).
PADDLE_CPU_SPEC = "paddlepaddle==3.3.0"
PADDLE_CPU_INDEX = "https://www.paddlepaddle.org.cn/packages/stable/cpu/"

PADDLE_GPU_SPEC = "paddlepaddle-gpu==3.2.2"
PADDLE_GPU_INDEXES = {
    "cu129": "https://www.paddlepaddle.org.cn/packages/stable/cu129/",
    "cu126": "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    "cu118": "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
}

# Paddle GPU wheels require compute capability >= 7.5 (Turing or newer).
MIN_GPU_COMPUTE_CAP = 7.5

# Other deps are declared in pyproject.toml; we install them via `-e .`.

# --- Pretty printing ------------------------------------------------------


class C:
    _on = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and os.name != "nt"
    # Colors on Windows are finicky across cmd/powershell/terminal; disabled
    # there by default to keep output readable in every shell.
    GREEN = "\033[92m" if _on else ""
    YELLOW = "\033[93m" if _on else ""
    RED = "\033[91m" if _on else ""
    BLUE = "\033[94m" if _on else ""
    BOLD = "\033[1m" if _on else ""
    END = "\033[0m" if _on else ""


def step(msg: str) -> None:
    print(f"\n{C.BOLD}==> {msg}{C.END}")


def ok(msg: str) -> None:
    print(f"{C.GREEN}[OK]{C.END} {msg}")


def info(msg: str) -> None:
    print(f"{C.BLUE}[..]{C.END} {msg}")


def warn(msg: str) -> None:
    print(f"{C.YELLOW}[WARN]{C.END} {msg}")


def fail(msg: str) -> int:
    print(f"{C.RED}[FAIL]{C.END} {msg}")
    return 1


# --- Helpers --------------------------------------------------------------


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def python_in_range(major: int, minor: int) -> bool:
    return MIN_PY <= (major, minor) <= MAX_PY


def find_compatible_base_python() -> Path | None:
    """
    Locate a Python interpreter that matches MIN_PY..MAX_PY.

    On Windows we query the `py` launcher for every installed version and
    pick the newest one in range. On POSIX we just probe `python3.13`,
    `python3.12`, ... on PATH. Returns None if nothing compatible is found.
    """
    # First, the interpreter that's running us right now.
    v = sys.version_info
    if python_in_range(v.major, v.minor):
        return Path(sys.executable)

    candidates: list[tuple[tuple[int, int], Path]] = []

    if os.name == "nt" and shutil.which("py"):
        # `py -0p` prints "  -3.12-64   C:\...\python.exe" per line.
        try:
            out = subprocess.run(
                ["py", "-0p"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            out = ""
        for line in out.splitlines():
            m = re.search(r"-?(\d+)\.(\d+)(?:-\d+)?\s+\*?\s*(.+)$", line)
            if not m:
                continue
            major, minor = int(m.group(1)), int(m.group(2))
            path_str = m.group(3).strip().strip('"')
            if not path_str:
                continue
            p = Path(path_str)
            if p.exists() and python_in_range(major, minor):
                candidates.append(((major, minor), p))
    else:
        for major, minor in [(3, m) for m in range(MAX_PY[1], MIN_PY[1] - 1, -1)]:
            exe = shutil.which(f"python{major}.{minor}")
            if exe:
                candidates.append(((major, minor), Path(exe)))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming output, raise on non-zero exit."""
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}")
    return subprocess.run(cmd, check=True, **kwargs)


def running_in_target_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == venv_python().resolve()
    except FileNotFoundError:
        return False


# --- Steps ----------------------------------------------------------------


def resolve_base_python() -> Path | None:
    """
    Pick the Python we'll use to create (or validate) the venv.

    If the user launched us with a compatible Python, we use that. Otherwise
    we search the system for a compatible one. Returns None on failure.
    """
    v = sys.version_info
    step(f"Checking Python ({v.major}.{v.minor}.{v.micro} on {platform.platform()})")

    if platform.machine().lower() not in {"amd64", "x86_64", "x64"}:
        warn(
            f"CPU architecture is {platform.machine()}. "
            "PaddlePaddle only ships x86_64 wheels; this may fail."
        )

    base = find_compatible_base_python()
    if base is None:
        fail(
            f"No Python in the supported range "
            f"({MIN_PY[0]}.{MIN_PY[1]}-{MAX_PY[0]}.{MAX_PY[1]}) was found. "
            "Install one (e.g. 3.12) and re-run."
        )
        return None

    if base.resolve() == Path(sys.executable).resolve():
        ok(f"Using current Python: {base}")
    else:
        info(
            f"Current Python ({v.major}.{v.minor}) is outside the tested "
            f"range; using {base} instead for the venv."
        )
    return base


def ensure_venv(base_python: Path, recreate: bool = False) -> None:
    step(f"Setting up virtual environment at {VENV_DIR}")
    if recreate and VENV_DIR.exists():
        info("--recreate requested, removing existing .venv")
        shutil.rmtree(VENV_DIR)

    if VENV_DIR.exists() and venv_python().exists():
        # If there's already a venv, sanity-check its Python version; if it's
        # out of range, rebuild it with our chosen base python.
        try:
            out = subprocess.run(
                [str(venv_python()), "-c",
                 "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip()
            major, minor = (int(x) for x in out.split("."))
            if python_in_range(major, minor):
                ok(f"Virtual environment already exists (Python {out})")
                return
            warn(
                f"Existing venv uses Python {out}, outside supported range. "
                "Rebuilding it with a compatible interpreter."
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            warn("Existing venv looks broken, rebuilding.")
        shutil.rmtree(VENV_DIR)

    elif VENV_DIR.exists():
        warn("Partial .venv found - removing and recreating")
        shutil.rmtree(VENV_DIR)

    # Build the venv using the *chosen base python* (not necessarily the one
    # that's currently running the script).
    run([str(base_python), "-m", "venv", str(VENV_DIR)])
    ok(f"Created {VENV_DIR} using {base_python}")


def reexec_in_venv(forwarded_args: list[str]) -> None:
    """If we're not already running inside the venv python, re-exec there."""
    if running_in_target_venv():
        return

    py = venv_python()
    if not py.exists():
        print(fail(f"Venv python not found at {py}"))
        sys.exit(1)

    step("Re-launching inside the virtual environment")
    info(f"using: {py}")

    cmd = [str(py), str(Path(__file__).resolve()), "--in-venv", *forwarded_args]
    # Use subprocess (not os.execv) so we behave identically on Windows,
    # where execv's semantics are weird.
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# --- GPU / CUDA detection ------------------------------------------------


def _nvidia_smi_cuda_version() -> str | None:
    """Return the driver-supported CUDA version (e.g. '12.6') or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
    return m.group(1) if m else None


def _nvidia_smi_compute_cap() -> float | None:
    """Return the highest compute capability reported by nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    caps: list[float] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            caps.append(float(line))
        except ValueError:
            pass
    return max(caps) if caps else None


def pick_paddle_install() -> tuple[str, str | None, str]:
    """
    Decide which Paddle wheel to install.

    Returns (package_spec, index_url_or_None, human_description).
    """
    step("Detecting GPU / CUDA")

    cuda = _nvidia_smi_cuda_version()
    cap = _nvidia_smi_compute_cap()

    if cuda is None:
        info("nvidia-smi not available (no NVIDIA driver found)")
        ok("Selecting CPU PaddlePaddle build")
        return (PADDLE_CPU_SPEC, PADDLE_CPU_INDEX, "CPU (no NVIDIA GPU detected)")

    info(f"Driver reports CUDA {cuda}")
    if cap is not None:
        info(f"GPU compute capability: {cap}")

    if cap is not None and cap < MIN_GPU_COMPUTE_CAP:
        warn(
            f"GPU compute capability {cap} < {MIN_GPU_COMPUTE_CAP} required by "
            "PaddlePaddle-GPU. Falling back to CPU build."
        )
        return (PADDLE_CPU_SPEC, PADDLE_CPU_INDEX, f"CPU (GPU cap {cap} too low)")

    try:
        major, minor = (int(x) for x in cuda.split("."))
    except ValueError:
        warn(f"Could not parse CUDA version '{cuda}', falling back to CPU.")
        return (PADDLE_CPU_SPEC, PADDLE_CPU_INDEX, f"CPU (unparseable CUDA '{cuda}')")

    if (major, minor) >= (12, 9):
        tag = "cu129"
    elif (major, minor) >= (12, 6):
        tag = "cu126"
    elif (major, minor) >= (11, 8):
        tag = "cu118"
    else:
        warn(
            f"CUDA {cuda} is older than the minimum Paddle-GPU wheel (11.8). "
            "Falling back to CPU."
        )
        return (PADDLE_CPU_SPEC, PADDLE_CPU_INDEX, f"CPU (CUDA {cuda} too old)")

    index = PADDLE_GPU_INDEXES[tag]
    ok(f"Selecting GPU PaddlePaddle build ({tag})")
    return (PADDLE_GPU_SPEC, index, f"GPU / {tag} (driver CUDA {cuda})")


# --- Install steps -------------------------------------------------------


def pip_install(*args: str) -> None:
    run([sys.executable, "-m", "pip", "install", *args])


def upgrade_pip() -> None:
    step("Upgrading pip / wheel / setuptools inside the venv")
    pip_install("--upgrade", "pip", "wheel", "setuptools")
    ok("pip toolchain up to date")


def warn_before_large_gpu_pip() -> None:
    """GPU wheels unpack many .dlls; OneDrive/AV/another process often causes WinError 32."""
    if os.name != "nt":
        return
    p = str(REPO_ROOT)
    p_lower = p.lower()
    if "onedrive" in p_lower or "dropbox" in p_lower:
        warn(
            "This repo is under a cloud-synced folder. OneDrive in particular can "
            "lock DLLs while pip overwrites `nvidia\\...\\cublasLt64_12.dll`, causing "
            "[WinError 32]. **Pause OneDrive** (system tray) for a few minutes, **close "
            "Cursor/VSCode** and any other terminals using this `.venv`, then re-run. "
            "If it keeps happening, copy the project to a non-synced path (e.g. C:\\\\dev). "
            "See README: Troubleshooting."
        )
    else:
        info(
            "Before this download: close other IDEs/terminals that activated this venv, "
            "or any stray python.exe, so DLLs in .venv are not in use."
        )


def install_paddle(spec: str, index_url: str | None) -> None:
    step(f"Installing {spec}")
    if "paddlepaddle-gpu" in spec:
        warn_before_large_gpu_pip()
    args = [spec]
    if index_url:
        args += ["-i", index_url]
    # `--upgrade` so re-runs pick up a newer pin if you bump it.
    pip_install("--upgrade", *args)
    ok(f"{spec} installed")


def install_project_and_deps() -> None:
    step("Installing NanoControl (editable) + runtime dependencies")
    pip_install("-e", str(REPO_ROOT))
    ok("`pip install -e .` complete")


def verify_install() -> bool:
    step("Verifying PaddlePaddle")
    try:
        run([sys.executable, "-c", "import paddle; paddle.utils.run_check()"])
        ok("PaddlePaddle health check passed")
        return True
    except subprocess.CalledProcessError:
        warn(
            "paddle.utils.run_check() reported a problem. "
            "Scroll up for details - often this is a harmless MKL/BLAS warning "
            "on CPU-only systems and OCR will still work."
        )
        # Do a softer fallback check: can we just import paddle + paddleocr?
        try:
            run([sys.executable, "-c", "import paddle, paddleocr"])
            ok("paddle + paddleocr import cleanly; continuing")
            return True
        except subprocess.CalledProcessError:
            fail("Import of paddle/paddleocr failed")
            return False


def prefetch_models() -> None:
    step("Pre-downloading OCR models (first run only, ~100 MB)")
    code = (
        "import numpy as np;"
        "from paddleocr import PaddleOCR;"
        "ocr = PaddleOCR("
        " lang='en',"
        " text_detection_model_name='PP-OCRv5_mobile_det',"
        " text_recognition_model_name='en_PP-OCRv5_mobile_rec',"
        " use_doc_orientation_classify=False,"
        " use_doc_unwarping=False,"
        " use_textline_orientation=False);"
        "ocr.predict(np.zeros((64,64,3), dtype='uint8'));"
        "print('Models ready.')"
    )
    try:
        run([sys.executable, "-c", code])
        ok("OCR models cached")
    except subprocess.CalledProcessError:
        warn(
            "Model prefetch failed (check your internet connection). "
            "Not fatal - models will download on first real OCR call."
        )


def print_next_steps(paddle_desc: str) -> None:
    activate_cmd = (
        r".venv\Scripts\activate"
        if os.name == "nt"
        else "source .venv/bin/activate"
    )
    print()
    print(f"{C.BOLD}{C.GREEN}Setup complete.{C.END}")
    print(f"  Paddle build: {paddle_desc}")
    print()
    print("Next steps:")
    print(f"  1. Activate the venv:  {C.BOLD}{activate_cmd}{C.END}")
    print(f"  2. Run a test:         {C.BOLD}python -m tests.test_vision{C.END}")
    print(f"     (or just `python tests/test_vision.py`)")
    print()
    print("Re-running this script is safe - it only fixes what's missing.")
    print()


# --- Main -----------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrap the NanoControl development environment."
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Delete .venv and build it from scratch.",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force the CPU PaddlePaddle build even if a GPU is detected.",
    )
    p.add_argument(
        "--skip-prefetch",
        action="store_true",
        help="Skip the OCR model prefetch step.",
    )
    p.add_argument(
        "--in-venv",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: set when we re-exec inside the venv
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    print(f"{C.BOLD}NanoControl environment bootstrap{C.END}")
    print(f"Repo root: {REPO_ROOT}")

    # Phase 1: host-python side of the script.
    if not args.in_venv:
        base = resolve_base_python()
        if base is None:
            return 1
        try:
            ensure_venv(base_python=base, recreate=args.recreate)
        except subprocess.CalledProcessError as e:
            return fail(f"venv creation failed with exit code {e.returncode}")
        except Exception as e:  # pragma: no cover - noisy setup errors
            return fail(f"Failed to create venv: {e}")

        forwarded = []
        if args.cpu:
            forwarded.append("--cpu")
        if args.skip_prefetch:
            forwarded.append("--skip-prefetch")
        reexec_in_venv(forwarded)  # re-enters main() in the venv
        return 0  # unreachable

    # Phase 2: we're now running inside .venv's python.
    try:
        upgrade_pip()

        if args.cpu:
            info("--cpu flag set; skipping GPU detection")
            spec, index, desc = (PADDLE_CPU_SPEC, PADDLE_CPU_INDEX, "CPU (forced)")
        else:
            spec, index, desc = pick_paddle_install()

        install_paddle(spec, index)
        install_project_and_deps()
    except subprocess.CalledProcessError as e:
        return fail(f"Command failed with exit code {e.returncode}")

    if not verify_install():
        return 1

    if not args.skip_prefetch:
        prefetch_models()

    print_next_steps(desc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
