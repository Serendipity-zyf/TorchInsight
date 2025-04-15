from setuptools import setup, find_packages

# Define version (keep in sync with version.py)
VERSION = "0.1.0"

# Read README as long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="torchinsight",
    version=VERSION,
    author="PixelCookie",
    author_email="metazyf@gmail.com",
    description="Enhanced model analysis tool for PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Serendipity-zyf/torchinsight",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "colorama>=0.4.4",
    ],
    keywords=["pytorch", "deep-learning", "model-analysis", "visualization"],
)
