from setuptools import setup, find_packages
from setuptools.command.install import install
from os import path
import subprocess

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    # $ pip install violet
    name='violet',
    version='0.0.1',
    description='A tool for H&E characterization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ding-lab/violet',
    author='Ding Lab',
    author_email='estorrs@wustl.edu',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='H&E slide image classification semi-supervised unsupervised clustering',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        ],
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'violet-dino=violet.models.main_dino:main',
        ],
    },
)
