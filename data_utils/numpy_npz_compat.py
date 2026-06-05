import importlib
import sys

import numpy as np


def install_numpy_pickle_compat():
    """Allow NumPy 1.x to unpickle object arrays written by NumPy 2.x."""
    try:
        major_version = int(np.__version__.split(".", 1)[0])
    except ValueError:
        return
    if major_version >= 2:
        return

    numpy_core = importlib.import_module("numpy.core")
    sys.modules.setdefault("numpy._core", numpy_core)
    for module_name in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        try:
            module = importlib.import_module(f"numpy.core.{module_name}")
        except ImportError:
            continue
        sys.modules.setdefault(f"numpy._core.{module_name}", module)
