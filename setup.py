from os import path
from setuptools import setup, find_packages

VERSION = "0.0.1"

HERE = path.abspath(path.dirname(__file__))

setup(
    name="gds_demo",
    version=VERSION,
    description="Demo for GDS application ",
    # long_description=LONG_DESCRIPTION,
    url="https://github.com/Yobota/yobota-simulation",
    author="Billy Liggins",
    email="billyliggins@gmail.com",
    license="UNLICENSED",
    packages=find_packages(
        exclude=[
            "analysis_scripts",
        ]
    ),
)
