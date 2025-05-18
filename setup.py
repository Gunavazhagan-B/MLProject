# Used / Responsible for making the application into package

from setuptools import find_packages,setup

# Find_packages is used to find all the packages used in the application

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