from setuptools import setup, find_packages

setup(
    name="pyhorn_cli",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "pyhorn_core",
        "typer>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "pyhorn=pyhorn_cli.main:run",
        ],
    },
)
