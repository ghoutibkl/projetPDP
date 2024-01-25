from distutils.core import setup

setup(
    description="package de test",
    author="magaye",
    author_email="magayendiaye56@gmail.com",
    name="projetPDP",
    packages=["projetPDP"],
    url="https://test.fr",
    version="0.1.0",
    entry_points={
    "console_scripts":
        ["projetPDP = projetPDP.main:main"],
}
)
