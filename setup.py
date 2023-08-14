import os
import sys
import shutil
from setuptools import setup, find_packages, find_namespace_packages
from setuptools.command.install import install

setup(name='spyderwebb',
      version='1.0.5',
      description='JWST NIRSpec reduction software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/spyderwebb',
      scripts=['bin/spyderwebb'],
      requires=['numpy','astropy(>=4.0)','scipy','jwst'],
      zip_safe = False,
      include_package_data=True,
      packages=find_namespace_packages(where="python"),
      package_dir={"": "python"}      
)
