# -*- coding: utf-8 -*-
"""Custom local plugin to ad extra functionality to the SciPyLa site."""

# Third party imports
from setuptools import setup


setup(
    name='lektor-scipyla-plugin',
    author='Gonzalo Peña-Castellanos',
    author_email='goanpeca@gmail.com',
    url='https://github.com/scipy-latinamerica/scipylatam-webpage-2019/',
    version='0.1',
    license='MIT',
    py_modules=['lektor_scipyla_plugin'],
    install_requires=['Lektor'],
    entry_points={
        'lektor.plugins': [
            'scipyla-plugin = lektor_scipyla_plugin:SciPyLaPlugin',
        ]
    }
)