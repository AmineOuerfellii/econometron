from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="econometron",
        version="0.1.0",
        author="Mohamed Amine Ouerfelli ",
        author_email="mohamedamine.ouerfelli@outlook.com",
        description="A Python package for solving ,simulating and estimating DSGE and VAR models.",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        url="https://econometron.netlify.app",
        packages=find_packages(include=["econometron", "econometron.*"]),
        license="MIT",
        install_requires=[
            "numpy>=1.23.5",
            "pandas>=1.5.3",
            "scipy>=1.13.0",
            "matplotlib>=3.8.4",
            "statsmodels>=0.14.1",
            "sympy>=1.13.0",
            "colorama"
        ],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Operating System :: OS Independent"
        ],
        python_requires='>=3.8',
    )
