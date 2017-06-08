from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(
    name='agentq',
    packages=[
        package for package in find_packages()
        if package.startswith('agentq')
    ],
    install_requires=[
        'baselines',
        'tensorflow',
        'numpy',
        'rx',
    ],
    description="Agent Q, an exploratory Q Learning model.",
    author="AAorris",
    url='https://github.com/AAorris/agentq',
    author_email="aaorris@gmail.com",
    version="0.0.1"
)
