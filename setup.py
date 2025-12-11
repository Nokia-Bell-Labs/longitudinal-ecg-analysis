# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause 

from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Remove comments and empty lines
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name='longitudinal_ecg_analysis',
    version='1.0.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
)
