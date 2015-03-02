try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='PALplots',
    version='0.1.0',
    author='Jami L. Johnson and Kasper van Wijk,',
    author_email='jami.johnson@auckland.ac.nz',
    packages=['palplots','palplots.scripts'],
    license='GNU General Public License, Version 3 (LGPLv3)',
    url='https://github.com/PALab/PALplots',
    description= 'A software package for analysis of data acquired with PLACE automation',
    long_description=open('README.txt').read(),
    install_requires=['numpy>1.0.0', 'obspy','scipy', 'matplotlib', 'h5py', 'obspyh5'],
    entry_points={'console_scripts':['quickread = palplots.scripts.quickread:main',],},
    )



