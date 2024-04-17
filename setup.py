from setuptools import setup, find_packages
from .__version__ import __version__

setup(
    name="cminor-mock",
    version=__version__,
    packages=find_packages(),
    install_requires=[],
    author="Cathoven A.I",
    author_email="contact@cathoven.com",
    description="Mocking library for Cminor",
    long_description=open("README.md").read(),
    url="https://github.com/yunusarli/cminor.git"
    )