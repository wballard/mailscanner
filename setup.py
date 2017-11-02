'''
Setup for mailscanner.
'''

import os
from sys import platform

from setuptools import find_packages, setup
from setuptools.extension import Extension


# Package details
setup(
    name='mailscanner',
    version='0.0.1',
    author='Will Ballard',
    author_email='wballard@mailframe.net',
    url='https://github.com/wballard/mailscanner',
    description='Tools for machine learning email',
    license='BSD 3-Clause License',
    packages=find_packages(),
    scripts=['bin/download-gmail', 'bin/prepare-replies-dataset', 'bin/prepare-replies-model'],
    install_requires=[
        'tqdm',
        'docopt',
        'vectoria',
        'smart_open',
        'connexion'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
