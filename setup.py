import os

from setuptools import setup, find_packages

with open('requirements.txt') as fd:
    install_requires = fd.read().splitlines()

setup(
    name='bhc',
    packages=find_packages(),
    install_requires=install_requires
)
