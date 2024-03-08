from setuptools import setup
import setuptools
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(
    name='decile-spear',
    version='1.0.8',
    author='Ayush Maheshwari, Guttu Sai Abhishek',
    author_email='ayush.hakmn@gmail.com',
    url='https://github.com/decile-team/spear',
    license='LICENSE.txt',
    packages=setuptools.find_packages(),
    description='SPEAR is a library for data programming with semi-supervision that provides facility to programmatically label and build training data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "."},
    # packages=setuptools.find_packages(where="spear"),
    python_requires=">=3.6",
    install_requires=[
        "tqdm>=4.59.0",
        "torch>=1.8.0",
        "scikit_learn>=0.24.2",
        "matplotlib>=3.3.4",
        "pandas>=1.1.5",
        "numpy>=1.19.5",
        "scipy>=1.5.4",
        "tensorflow>=2.2.0",
        "TextBlob"
    ],
)
