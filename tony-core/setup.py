from setuptools import setup, find_packages

setup(
    name="tony",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "tony=tony.cli:main"
        ]
    }
)
