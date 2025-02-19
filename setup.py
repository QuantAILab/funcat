#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from os.path import dirname, join
from setuptools import (
    find_packages,
    setup,
)

with open(join(dirname(__file__), 'funcat/VERSION.txt'), 'rb') as f:
    version = f.read().decode('ascii').strip()

with open('requirements.txt', 'r') as f:
    requirements = [l.strip('\n') for l in f.readlines()]

setup(
    name='funcat',
    version=version,
    description='funcat',
    packages=find_packages(exclude=[]),
    author='Hua Liang, Zirui Wei',
    url='https://github.com/cedricporter/funcat',
    author_email='et@everet.org, ziruiwei@gmail.com',
    license='Apache License v2',
    package_data={'': ['*.*']},
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
