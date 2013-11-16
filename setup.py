#!/usr/bin/env python3
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

classifiers="""\
Development Status :: 1 - Planning
Environment :: Console
Intended Audience :: Developers
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Operating System :: OS Independent
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.2
Programming Language :: Python :: 3.3
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development :: Libraries :: Application Frameworks"""

setup(
    name='appblocks',
    version='0.1', 
    description="Flow based command line application framework for Python",
    long_description="",
    license='LGPLv3',
    classifiers=classifiers.splitlines(),
    platforms=['any'],
    packages=['appblocks'],
    url='http://zyga.github.io/appblocks/',
    author="Zygmunt Krynicki",
    author_email="<zkrynicki@gmail.com>")
