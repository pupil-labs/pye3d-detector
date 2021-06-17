from pathlib import Path

import numpy as np


def abs_diff(x1, x2):
    return np.abs(x1 - x2)


def remove_file(path, missing_ok=True):
    path = Path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise


def input_dir() -> Path:
    path = _this_dir().joinpath("input")
    assert path.is_dir()  # sanity check
    return path


def output_dir() -> Path:
    path = _this_dir().joinpath("output")
    assert path.is_dir()  # sanity check
    return path


def _this_dir() -> Path:
    path = Path(__file__).resolve().parent
    assert path.is_dir()  # sanity check
    return path
