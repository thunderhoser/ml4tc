"""Methods for reading and writing XBT (extended best track) data."""

import os
import sys
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils


def read_file(netcdf_file_name):
    """Reads XBT data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: xbt_table_xarray: xarray table.  Documentation in the xarray table
        should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(xbt_table_xarray, netcdf_file_name):
    """Writes XBT data to NetCDF file.

    :param xbt_table_xarray: xarray table in format returned by `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    xbt_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )
