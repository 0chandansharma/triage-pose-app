from setuptools import setup, find_packages

setup(
    name="physiotrack",
    version="0.1.0",
    py_modules=[
        "angles", 
        "detector", 
        "models", 
        "utils"
    ],  # List individual modules
    install_requires=[
        "numpy>=1.20.0",
        "rtmlib>=0.0.13",
        "opencv-python>=4.5.0",
        "pydantic>=1.8.0",
        "anytree>=2.8.0"
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Minimal library for pose detection and angle calculation",
    keywords="pose detection, angle calculation, computer vision",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Include package data files
    include_package_data=True,
)