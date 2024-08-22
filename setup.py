from setuptools import find_packages, setup

setup(
    name="torchclust",
    version="1.0.0",
    author="Daniel Ikechukwu",
    author_email="ttdanielik@gmail.com",
    description="Scalable Implementations of clustering algorithms written in Pytorch for running on the GPU.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/danny-1k/torchclust",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="machine learning deep learning pytorch clustering",
    python_requires=">=3.7",
)
