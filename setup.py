from setuptools import setup, find_packages

setup(
    name="waveclus-linlab",  # This is the name of your top-level package
    version="0.1.0",
    author="Zuohan Zhao",
    author_email="zzhmark@126.com",
    description="Lin Lab waveclus sorting pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DCLinLab/waveclus-sorting-pipeline",
    packages=find_packages(),  # This will find 'my_package' and its submodules
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        # List your dependencies here
    ],
    entry_points={
        "console": [
            "waveclus_linlab=waveclus_linlab.script:main",  # Entry point for CLI scripts
        ],
    },
)