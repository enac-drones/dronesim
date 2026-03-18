"""Minimal setup.py for C extension configuration.

This file works with pyproject.toml (the primary build configuration).
It is kept for C extension support, which has limited TOML encoding support.
See: https://packaging.python.org/guides/writing-pyproject-toml/#c-extensions
"""

from setuptools import Extension, setup
import numpy

pywls_dir = "dronesim/control/pywls"

ext_modules = [
    Extension(
        name="dronesim.control.pywls",
        sources=[
            pywls_dir + "/pywls_module.c",
            pywls_dir + "/wls_alloc.c",
            pywls_dir + "/qr_solve.c",
            pywls_dir + "/r8lib_min.c",
        ],
        include_dirs=[
            numpy.get_include(),
            pywls_dir,
        ],
        extra_compile_args=["-O3"],
        define_macros=[
            ("WLS_N_U_MAX", "8"),
            ("WLS_N_V_MAX", "6"),
        ],
    ),
]


setup(
    ext_modules=ext_modules,
)
