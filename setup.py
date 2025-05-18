# Used / Responsible for making the application into package

'''
🛠️ What is setup.py?
setup.py is the build script for Python projects — it tells Python how to install and package your application.

Think of it as the installation manual for your project. It contains metadata about your project (like name, version, author), and instructions on what code and dependencies are needed.

🧩 Why do we need it?
If you're building a Python project that:

Has modules or packages (your own .py files),

Uses external libraries (like numpy, pandas, etc.),

Or you want to install your project with pip install . (or make it pip-installable by others),

then you need a setup.py.
'''

from setuptools import find_packages,setup

'''
find_package:
Purpose: Tells setuptools which local Python packages/modules in your project should be included in the final distribution.

Think of it like: "Which code from my project should be part of this installable Python package?"
'''

'''
install_requires:
Purpose: Tells pip what external dependencies your project needs to run (e.g., numpy, pandas).

These are third-party libraries that need to be installed from PyPI.

Think of it like: "What does my project depend on that isn’t in the standard library or my own code?"
'''

from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(path: str) -> List[str]:
    with open(path) as f:
        return [
            line.strip() for line in f.readlines()
            if line.strip() and not line.startswith('-e')
        ]


setup(
    name='mlproject',
    version='0.0.1',
    author='Gunavazhagan',
    author_email='gunavazhagan8@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)