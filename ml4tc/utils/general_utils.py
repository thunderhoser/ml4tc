"""General utility methods."""

import gzip
import shutil
import numpy
from scipy.ndimage import distance_transform_edt
from gewittergefahr.gg_utils import error_checking

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


def fill_nans(data_matrix):
    """Fills NaN's with nearest neighbours.

    This method is adapted from the method `fill`, which you can find here:
    https://stackoverflow.com/posts/9262129/revisions

    :param data_matrix: numpy array of real-valued data.
    :return: data_matrix: Same but without NaN's.
    """

    error_checking.assert_is_real_numpy_array(data_matrix)

    indices = distance_transform_edt(
        numpy.isnan(data_matrix), return_distances=False, return_indices=True
    )
    return data_matrix[tuple(indices)]
