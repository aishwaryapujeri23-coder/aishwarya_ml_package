from setuptools import setup, find_packages

setup(
    name             = "aishwarya_ml_package",
    version          = "1.0.0",
    author           = "Aishwarya",
    description      = "Complete ML library from scratch: preprocessing, regression, KNN, PCA, Neural Network, Pipeline",
    long_description = open("README.md", encoding="utf-8").read(),
    long_description_content_type = "text/markdown",
    packages         = find_packages(),
    python_requires  = ">=3.8",
    install_requires = ["numpy>=1.21", "scipy>=1.7"],
    extras_require   = {
        "plot" : ["matplotlib>=3.4"],
        "test" : ["pytest>=7.0"],
        "dev"  : ["matplotlib>=3.4", "pytest>=7.0"],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
