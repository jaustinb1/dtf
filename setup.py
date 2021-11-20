from setuptools import setup, find_packages
import sys

setup(
    name="dtf",
    packages=[pkg for pkg in find_packages() if pkg.startswith("dtf")],
    description="distributed tensorflow",
    author="Jonathan Booher",
    author_email="jaustinb1@gmail.com",
    version="0.0",
)
