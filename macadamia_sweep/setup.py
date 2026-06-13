from setuptools import setup
import os
from glob import glob

package_name = "macadamia_sweep"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Simple single-row macadamia/chestnut demo",
    license="MIT",
    entry_points={
        'console_scripts': [
    'simple_row_follower = macadamia_sweep.simple_row_follower:main',
    'sweep_logger = macadamia_sweep.sweep_logger:main',
    'nut_detector = macadamia_sweep.nut_detector:main',
    'nut_tracker = macadamia_sweep.nut_tracker:main',
],
    },
)
