__author__ = 'lyuwenyu'

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

packages = find_packages()

setup(
    name='pdll',
    version='0.0.1',
    description='Python Deep Learning Library',
    author='lyuwenyu', 
    author_email='wenyu.lyu@qq.com',
    url='https://github.com/lyuwenyu/PDLL',
    packages=packages,
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    classifiers=[
        'Programming Language :: Python :: 3', 
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent'],
    python_requires='>=3.5',
    install_requires=['numpy>=1.15']
    )

