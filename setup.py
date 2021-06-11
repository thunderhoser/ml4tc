"""Setup file for ml4tc."""

from setuptools import setup

PACKAGE_NAMES = [
    'ml4tc', 'ml4tc.io', 'ml4tc.utils', 'ml4tc.machine_learning',
    'ml4tc.plotting', 'ml4tc.scripts'
]
KEYWORDS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'data science', 'weather', 'meteorology', 'hurricane', 'cyclone',
    'tropical cyclone', 'tropical', 'satellite'
]
SHORT_DESCRIPTION = (
    'End-to-end library for using machine learning to predict tropical-cyclone '
    'intensity.'
)
LONG_DESCRIPTION = SHORT_DESCRIPTION
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3'
]

PACKAGE_REQUIREMENTS = [
    'numpy',
    'scipy',
    'tensorflow',
    'keras',
    'scikit-learn',
    'scikit-image',
    'netCDF4',
    'pyproj',
    'opencv-python',
    'matplotlib',
    'pandas',
    'shapely',
    'geopy',
    'metpy'
]

if __name__ == '__main__':
    setup(
        name='ml4tc',
        version='0.1',
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author='Ryan Lagerquist',
        author_email='ryan.lagerquist@noaa.gov',
        url='https://github.com/thunderhoser/ml4tc',
        packages=PACKAGE_NAMES,
        scripts=[],
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        zip_safe=False,
        install_requires=PACKAGE_REQUIREMENTS
    )
