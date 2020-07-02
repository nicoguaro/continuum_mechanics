#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['sympy>=1.3',
                'matplotlib>=3',
                'numpy',
                'scipy']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Nicolás Guarín-Zapata",
    author_email='nicoguarin@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Utilities for doing calculations in continuum mechanics.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='continuum_mechanics',
    name='continuum_mechanics',
    packages=find_packages(include=['continuum_mechanics']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nicoguaro/continuum_mechanics',
    version='0.2.1',
    zip_safe=False,
)
