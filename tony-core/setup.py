from setuptools import find_packages, setup


setup(
    name="tony",
    version="1.1.0",
    description="Grant and nonprofit ingestion, scoring, reporting, and dashboard tools",
    packages=find_packages(),
    include_package_data=True,
    package_data={"tony": ["default_config.json", "templates/*.html"]},
    install_requires=[
        "Flask>=3.0",
        "PyPDF2>=3.0",
        "numpy>=1.26",
        "openpyxl>=3.1",
        "pandas>=2.2",
        "plotly>=5.24",
        "pytest>=8.0",
        "requests>=2.32",
        "scikit-learn>=1.5",
    ],
    extras_require={
        "pdf": [
            "camelot-py>=0.11",
            "tabula-py>=2.10",
        ]
    },
    entry_points={"console_scripts": ["tony=tony.cli:main"]},
)
