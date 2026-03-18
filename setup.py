from setuptools import setup, Extension
import numpy

pywls_dir = "dronesim/control/pywls"

ext = Extension(
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

    # Optional: increase compile-time dimensions if needed.
    # These must be consistent for every translation unit that includes wls_alloc.h.
    define_macros=[
        ("WLS_N_U_MAX", "8"),
        ("WLS_N_V_MAX", "6"),
    ],
)

setup(
    name="dronesim",
    packages=["dronesim"],
    version="0.1.0",
    install_requires=[
        "numpy",
        "scipy",
        "Pillow",
        "matplotlib",
        "cycler",
        "gym",
        "pybullet",
    ],
    ext_modules=[ext]
)
