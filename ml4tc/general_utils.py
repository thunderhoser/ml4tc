"""General utility methods."""

import os
import sys
import gzip
import shutil
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

GZIP_FILE_EXTENSION = '.gz'
DEGREES_TO_RADIANS = numpy.pi / 180


def compress_file(netcdf_file_name):
    """Compresses NetCDF file (turns it into a gzip file).

    :param netcdf_file_name: Path to NetCDF file.
    :raises: ValueError: if file is already gzipped.
    """

    error_checking.assert_is_string(netcdf_file_name)
    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        raise ValueError('File must not already be gzipped.')

    gzip_file_name = '{0:s}{1:s}'.format(netcdf_file_name, GZIP_FILE_EXTENSION)

    with open(netcdf_file_name, 'rb') as netcdf_handle:
        with gzip.open(gzip_file_name, 'wb') as gzip_handle:
            shutil.copyfileobj(netcdf_handle, gzip_handle)


def speed_and_heading_to_uv(storm_speeds_m_s01, storm_headings_deg):
    """Converts storm motion from speed-direction to u-v.

    :param storm_speeds_m_s01: numpy array of storm speeds (metres per
        second).
    :param storm_headings_deg: Same-shape numpy array of storm headings
        (degrees).
    :return: u_motions_m_s01: Same-shape numpy array of eastward motions (metres
        per second).
    :return: v_motions_m_s01: Same-shape numpy array of northward motions
        (metres per second).
    """

    error_checking.assert_is_geq_numpy_array(
        storm_speeds_m_s01, 0., allow_nan=True
    )

    error_checking.assert_is_geq_numpy_array(
        storm_headings_deg, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        storm_headings_deg, 360., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        storm_headings_deg,
        exact_dimensions=numpy.array(storm_speeds_m_s01.shape, dtype=int)
    )

    u_motions_m_s01 = storm_speeds_m_s01 * numpy.sin(
        storm_headings_deg * DEGREES_TO_RADIANS
    )
    v_motions_m_s01 = storm_speeds_m_s01 * numpy.cos(
        storm_headings_deg * DEGREES_TO_RADIANS
    )

    return u_motions_m_s01, v_motions_m_s01
