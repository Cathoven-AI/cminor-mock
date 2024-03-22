# Copyright Cathoven A.I. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
from setuptools import setup, find_packages
from cminor.__version__ import __version__


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().strip().split('\n')

setup(
    name='solar',
    version=__version__,
    description='SOLAR: System of LAnguage Retention',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Cathoven A.I.',
    author_email='contact@cathoven.com',
    url = 'https://github.com/tacerdi/solar',
    packages=[],
    install_requires=requirements,
)
