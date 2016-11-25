from warnings import warn
try:
    from setuptools import setup
except ImportError:
    msg = 'distutils module can cause errors during installation.'
    msg += 'Installing setuptools is recommened prior to installation of PALplots.'
    warn(msg, ImportWarning)
    from distutils.core import setup

# we need to find out if numpy is installed
needed_for_setup = []
try:
    import numpy
except ImportError:
    needed_for_setup.append('numpy')

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

    setup_requires=needed_for_setup,
    install_requires=['numpy>1.0.0', 'obspy','scipy', 'matplotlib', 'h5py', 'obspyh5'],
    entry_points={'console_scripts':['quickread = palplots.scripts.quickread:main',],},
    )

