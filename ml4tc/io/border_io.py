"""Input/output methods for political borders."""

import os
import sys
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

VERTEX_DIMENSION_KEY = 'vertex'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'


def read_file(netcdf_file_name=None):
    """Reads borders from NetCDF file.

    :param netcdf_file_name: Path to input file.  If None, will look for file in
        repository.
    :return: latitudes_deg_n: See doc for `write_file`.
    :return: longitudes_deg_e: Same.
    """

    if netcdf_file_name is None:
        netcdf_file_name = '{0:s}/borders.nc'.format(THIS_DIRECTORY_NAME)

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    latitudes_deg_n = numpy.array(
        dataset_object.variables[LATITUDES_KEY][:]
    )
    longitudes_deg_e = numpy.array(
        dataset_object.variables[LONGITUDES_KEY][:]
    )
    dataset_object.close()

    return latitudes_deg_n, longitudes_deg_e
