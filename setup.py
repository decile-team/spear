from setuptools import setup
import setuptools

setup(
    name='spear',
    version='1.0.0',
    author='Ayush Maheshwari, Guttu Sai Abhishek',
    author_email='ayush.hakmn@gmail.com',
    #packages=['cords', 'cords/selectionstrategies', 'cords/utils'],
    url='https://github.com/decile-team/spear',
    license='LICENSE.txt',
    packages=setuptools.find_packages(),
    description='SPEAR is a library for data programming with semi-supervision that provides facility to programmatically label and build training data',
    install_requires=[
        "tqdm>=4.59.0",
        "torch>=1.8.0",
        "scikit_learn>=0.24.2",
        "matplotlib>=3.3.4",
        "pandas>=1.1.5",
        "numpy>=1.19.5",
        "scipy>=1.5.4"
    ],
)