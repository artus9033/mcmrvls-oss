from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

deps = [line.strip() for line in open("requirements.txt").readlines()]

d = generate_distutils_setup(
    packages=["robot_driver"],
    scripts=[
        "scripts/cmd_vel_dt_mux.py",
        "scripts/local_planner.py",
        "scripts/DiffDriveLocalPlanner.py",
    ],
    package_dir={"": "scripts"},
    install_requires=deps,
)

setup(**d)
