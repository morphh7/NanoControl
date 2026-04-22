# NanoControl

Vision-driven desktop automation (OCR + input control) on top of PaddleOCR.

## Setup (any Windows / Linux / macOS machine)

Requirements: **Python 3.9 - 3.13** on your PATH. Nothing else.

```bash
python bootstrap_env.py
```

That single command:

1. Creates `.venv/` in the repo root.
2. Re-runs itself inside that venv.
3. Detects whether you have an NVIDIA GPU with a recent CUDA driver
   (11.8 / 12.6 / 12.9) and installs the matching PaddlePaddle-GPU wheel.
   Falls back to the CPU wheel automatically if not.
4. Installs PaddleOCR, pyautogui, Pillow, numpy.
5. Installs this project in editable mode (`pip install -e .`) so `import nanocontrol` works.
6. Verifies Paddle and pre-caches the OCR models.

Safe to re-run. It only fixes what's missing.

**Why not `setup.py`?** Pip calls `setup.py` internally with arguments like
`egg_info` when you run `pip install -e .`. The old project used one file
for both jobs, which broke editable installs. Now:
`bootstrap_env.py` = venv + Paddle; `setup.py` = tiny setuptools shim only.

### Useful flags

```bash
python bootstrap_env.py --cpu             # force CPU build (skip GPU detection)
python bootstrap_env.py --recreate        # nuke .venv and rebuild from scratch
python bootstrap_env.py --skip-prefetch   # don't pre-download OCR models
```

## Day-to-day use

After setup, activate the venv:

```bat
:: Windows (cmd / powershell)
.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then run things normally:

```bash
python tests/test_vision.py
```

## Project layout

```
src/nanocontrol/
    __init__.py
    utils.py
    actions/       # input control (mouse, keyboard, ...)
    vision/        # screen capture + OCR parsing
tests/
    test_vision.py
bootstrap_env.py   # run this to create venv + install Paddle + project
setup.py           # setuptools shim only (for pip install -e .)
pyproject.toml     # packaging metadata for editable install
```

## Troubleshooting

- **"paddlepaddle X.Y.Z not found"** — Paddle sometimes yanks a version
  from PyPI. Bump the pins at the top of `bootstrap_env.py` (`PADDLE_CPU_SPEC`,
  `PADDLE_GPU_SPEC`) and re-run.
- **GPU build installed but still slow / falling back to CPU** — open a
  Python shell inside the venv and run
  `python -c "import paddle; print(paddle.device.is_compiled_with_cuda(), paddle.device.cuda.device_count())"`.
  If that's `False / 0`, your driver is likely too old; run
  `python bootstrap_env.py --cpu` to force the CPU build.
- **`OSError: [WinError 32] ... cublasLt64_12.dll` (or another file in
  `site-packages\nvidia\`)** — something else has that DLL open. Typical
  causes, in order:
  1. **OneDrive** (you are on `...OneDrive\...`) — it indexes/syncs
     `.venv` and locks CUDA DLLs while pip rewrites them. In the
     system tray, **Pause syncing** for 2+ hours, then try again, or
     **move the whole project** to a folder that is not synced, e.g.
     `C:\dev\NanoControl`, and run `python bootstrap_env.py` from there.
  2. **Another process** — close this IDE’s terminals, any REPL, and
     **Task Manager → End** any `python.exe` that points at this project’s
     `.venv`. Retry in a new terminal.
  3. **Antivirus** — add an exclusion for the project’s `.venv` folder,
     or pause real-time protection during the install.
  4. If pip left things half-installed, run `python bootstrap_env.py --recreate`
     (after OneDrive is paused / project moved) for a clean venv.

- **OneDrive in general** — keeping `.venv` under a synced path is
  fragile. Prefer a local-only clone, or at least exclude `.venv` from
  OneDrive (Folder settings → "Always keep on this device" does not
  stop the sync service; pausing or moving the repo is more reliable).
