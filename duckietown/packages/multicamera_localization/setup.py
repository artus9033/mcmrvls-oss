from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

deps = [line.strip() for line in open("requirements.txt").readlines()]

d = generate_distutils_setup(
    packages=["multicamera_localization"],
    scripts=[
        "scripts/robot_localization_node.py",
        "scripts/map_segmentation_node.py",
        "scripts/status_listener_node.py",
    ],
    package_dir={"": "scripts"},
    install_requires=deps,
)

setup(**d)
