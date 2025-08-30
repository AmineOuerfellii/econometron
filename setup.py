from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README.md
this_dir = Path(__file__).resolve().parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="econometron",
    version="0.0.3",
    author="Mohamed Amine Ouerfelli",
    author_email="mohamedamine.ouerfelli@outlook.com",
    description=(
      "Econometron is a Python library for advanced econometric analysis and time series forecasting,"
      "combining statistical rigor with modern computational methods. It offers a precise,"
      "extensible framework for exploring complex economic and financial dynamics."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://econometron.netlify.app",
    project_urls={
        "Source": "https://github.com/AmineOuerfellii/econometron",
        "Documentation": "https://econometron.netlify.app",
        "Tracker": "https://github.com/AmineOuerfellii/econometron/issues",
    },
    packages=find_packages(where=".", include=["econometron", "econometron.*"]),
    license="MIT",
    install_requires=[
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "scipy>=1.13.0",
        "matplotlib>=3.8.4",
        "statsmodels>=0.14.1",
        "sympy>=1.13.0",
        "torch>=1.13.1",
        "scikit-learn>=1.0.2",
        "torchinfo>=1.7.2",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
