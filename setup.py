import os
from setuptools import setup, find_packages


def read(f_name: str):
    return open(os.path.join(os.path.dirname(__file__), f_name)).read()


def read_requirements(f_name: str) -> list[str]:
    return list(
        open(os.path.join(os.path.dirname(__file__), f_name)).readlines()
    )


setup(
    name="custoch",
    version="0.0.1",
    author="Marcin Baranek",
    author_email="baranekmarcin47@gmail.com",
    description="custoch is a package for performing numerical experiments "
                "related to stochastic differential equations (SDE)",
    license="",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
    ],
    keywords="",
    url="https://github.com/MarcinBaranek/custoch",
    packages=find_packages(exclude=['custoch']),
    long_description=read('README.md'),
    install_requiers=read_requirements('requirements.txt'),
    include_package_data=True,
    zip_safe=False
)
