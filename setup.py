from setuptools import setup

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
)
