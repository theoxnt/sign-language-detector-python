#!/usr/bin/env python3
"""
Setup script for Sign Language Detector
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sign-language-detector",
    version="1.0.0",
    author="Sign Language Detector Team",
    description="ASL alphabet recognition system using computer vision and ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theoxnt/sign-language-detector-python",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.7.0",
        "mediapipe>=0.10.30",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "language-tool-python>=2.7.0",
    ],
    entry_points={
        "console_scripts": [
            "sign-lang=src.cli_enhanced:main",
        ],
    },
)
