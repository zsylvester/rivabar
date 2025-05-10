from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rivabar",
    version="0.1.1",
    author="Zoltan Sylvester",
    author_email="zsylvester@gmail.com",  # Replace with your actual email
    description="A Python package to automatically extract channel centerlines and banklines from water index images of rivers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zsylvester/rivabar",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0.0",
        "matplotlib>=3.9.0",
        "pandas>=2.2.0",
        "scipy>=1.14.0",
        "scikit-image>=0.24.0",
        "opencv-python>=4.10.0",
        "sknw>=0.15",
        "networkx>=3.4.0",
        "scikit-learn>=1.5.0",
        "rasterio>=1.4.0",
        "shapely>=2.0.0",
        "libpysal>=4.12.0",
        "geopandas>=1.0.0",
        "momepy>=0.9.0",
        "tqdm>=4.67.0",
    ],
) 