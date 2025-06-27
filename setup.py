"""Setup script for the step detection package."""

from setuptools import find_packages, setup

with open("README_Clean.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="step-detection",
    version="1.0.0",
    author="Step Detection Team",
    author_email="contact@stepdetection.com",
    description="A comprehensive solution for real-time step detection using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stepdetection/step-detection",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.10",
            "flake8>=5.0",
        ],
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "step-detection-demo=scripts.run_demo:main",
            "step-detection-api=scripts.run_api:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
