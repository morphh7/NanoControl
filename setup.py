"""
Setuptools shim for `pip install -e .` / wheel builds.

Project metadata lives in pyproject.toml. To create the venv and install
Paddle, run `python bootstrap_env.py` from the repo root — do not run this
file manually.
"""

from setuptools import setup

setup()
