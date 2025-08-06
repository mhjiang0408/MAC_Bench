#!/usr/bin/env python3
"""
Setup script for MAC_Bench CLI

Install the MAC_Bench command-line interface as a system command.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", "r", encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# CLI-specific requirements
cli_requirements = [
    'click>=8.0.0',
    'colorama>=0.4.4',
    'psutil>=5.8.0',
    'matplotlib>=3.5.0',
    'seaborn>=0.11.0'
]

setup(
    name="mac_bench_cli",
    version="1.0.0",
    author="Mohan Jiang",
    author_email="mhjiang0408@sjtu.edu.cn",
    description="Command-line interface for MAC_Bench multimodal benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mac-bench/MAC_Bench",
    project_urls={
        "Bug Reports": "https://github.com/mhjiang0408/MAC_Bench/issues",
        "Source": "https://github.com/mhjiang0408/MAC_Bench",
    },
    packages=find_packages(include=['mac_cli*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements + cli_requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'flake8>=4.0',
            'mypy>=0.950'
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
            'myst-parser>=0.17'
        ]
    },
    entry_points={
        'console_scripts': [
            'mac=mac_cli.cli:cli',
        ],
    },
    include_package_data=True,
    package_data={
        'mac_cli': [
            'templates/*',
            'configs/*'
        ]
    },
    zip_safe=False,
    keywords='a live multimodal benchmark cli for scientific understanding',
)