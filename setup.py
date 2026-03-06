from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    try:
        long_description = fh.read()
    except Exception:
        long_description = "A production-grade, educational machine learning and linear algebra foundation framework."

setup(
    name="mllense", 
    version="0.1.0",
    author="SANJAYRAM-DS",
    author_email="author@example.com",
    description="A foundational linear algebra and ML modeling framework built with educational traceability.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SANJAYRAM-DS/ml-lens",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "mypy"],
    },
)
