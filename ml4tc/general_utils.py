"""General utility methods."""

import os
import sys
import gzip
import shutil
import tempfile
import calendar
import numpy
from scipy.ndimage import distance_transform_edt

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import number_rounding
import longitude_conversion as lng_conversion
import time_conversion
import error_checking

GZIP_FILE_EXTENSION = '.gz'
TIME_FORMAT_FOR_LOG = '%Y-%m-%d-%H%M%S'

DEGREES_TO_RADIANS = numpy.pi / 180

MINUTES_TO_SECONDS = 60
DAYS_TO_SECONDS = 86400

TIME_FORMAT_FOR_ALTITUDE_ANGLE = '%Y %m %d %H %M %S'
BATCH_SIZE_FOR_ALTITUDE_ANGLE = 100
DEFAULT_EXE_NAME_FOR_ALTITUDE_ANGLE = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/solarpos/solarpos'
)
ERROR_STRING_FOR_ALTITUDE_ANGLE = (
    '\nUnix command failed (log messages shown above should explain why).'
)


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


def decompress_file(gzip_file_name):
    """Deompresses gzip file.

    :param gzip_file_name: Path to gzip file.
    :raises: ValueError: if file is not gzipped.
    """

    error_checking.assert_is_string(gzip_file_name)
    if not gzip_file_name.endswith(GZIP_FILE_EXTENSION):
        raise ValueError('File must be gzipped.')

    unzipped_file_name = gzip_file_name[:-len(GZIP_FILE_EXTENSION)]

    with gzip.open(gzip_file_name, 'rb') as gzip_handle:
        with open(unzipped_file_name, 'wb') as unzipped_handle:
            shutil.copyfileobj(gzip_handle, unzipped_handle)


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


def find_exact_times(
        actual_times_unix_sec, desired_times_unix_sec=None,
        first_desired_time_unix_sec=None, last_desired_time_unix_sec=None):
    """Finds desired times in array.

    D = number of desired times

    :param actual_times_unix_sec: 1-D numpy array of actual times.
    :param desired_times_unix_sec: length-D numpy array of desired times.
    :param first_desired_time_unix_sec: First desired time.
    :param last_desired_time_unix_sec: Last desired time.
    :return: desired_indices: length-D numpy array of indices into the array
        `actual_times_unix_sec`.
    :raises: ValueError: if cannot find actual time between
        `first_desired_time_unix_sec` and `last_desired_time_unix_sec`.
    """

    if desired_times_unix_sec is not None:
        error_checking.assert_is_integer_numpy_array(desired_times_unix_sec)
        error_checking.assert_is_numpy_array(
            desired_times_unix_sec, num_dimensions=1
        )

        return numpy.array([
            numpy.where(actual_times_unix_sec == t)[0][0]
            for t in desired_times_unix_sec
        ], dtype=int)

    error_checking.assert_is_integer(first_desired_time_unix_sec)
    error_checking.assert_is_integer(last_desired_time_unix_sec)
    error_checking.assert_is_geq(
        last_desired_time_unix_sec, first_desired_time_unix_sec
    )

    desired_indices = numpy.where(numpy.logical_and(
        actual_times_unix_sec >= first_desired_time_unix_sec,
        actual_times_unix_sec <= last_desired_time_unix_sec
    ))[0]

    if len(desired_indices) > 0:
        return desired_indices

    first_desired_time_string = time_conversion.unix_sec_to_string(
        first_desired_time_unix_sec, TIME_FORMAT_FOR_LOG
    )
    last_desired_time_string = time_conversion.unix_sec_to_string(
        last_desired_time_unix_sec, TIME_FORMAT_FOR_LOG
    )
    error_string = 'Cannot find any times between {0:s} and {1:s}.'.format(
        first_desired_time_string, last_desired_time_string
    )
    raise ValueError(error_string)


def create_latlng_grid(
        min_latitude_deg_n, max_latitude_deg_n, latitude_spacing_deg,
        min_longitude_deg_e, max_longitude_deg_e, longitude_spacing_deg):
    """Creates lat-long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param min_latitude_deg_n: Minimum latitude (deg N) in grid.
    :param max_latitude_deg_n: Max latitude (deg N) in grid.
    :param latitude_spacing_deg: Spacing (deg N) between grid points in adjacent
        rows.
    :param min_longitude_deg_e: Minimum longitude (deg E) in grid.
    :param max_longitude_deg_e: Max longitude (deg E) in grid.
    :param longitude_spacing_deg: Spacing (deg E) between grid points in
        adjacent columns.
    :return: grid_point_latitudes_deg: length-M numpy array with latitudes
        (deg N) of grid points.
    :return: grid_point_longitudes_deg: length-N numpy array with longitudes
        (deg E) of grid points.
    """

    # TODO(thunderhoser): Make this handle wrap-around issues.

    min_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg_e
    )
    max_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        max_longitude_deg_e
    )

    min_latitude_deg_n = number_rounding.floor_to_nearest(
        min_latitude_deg_n, latitude_spacing_deg
    )
    max_latitude_deg_n = number_rounding.ceiling_to_nearest(
        max_latitude_deg_n, latitude_spacing_deg
    )
    min_longitude_deg_e = number_rounding.floor_to_nearest(
        min_longitude_deg_e, longitude_spacing_deg
    )
    max_longitude_deg_e = number_rounding.ceiling_to_nearest(
        max_longitude_deg_e, longitude_spacing_deg
    )

    num_grid_rows = 1 + int(numpy.round(
        (max_latitude_deg_n - min_latitude_deg_n) / latitude_spacing_deg
    ))
    num_grid_columns = 1 + int(numpy.round(
        (max_longitude_deg_e - min_longitude_deg_e) / longitude_spacing_deg
    ))

    return grids.get_latlng_grid_points(
        min_latitude_deg=min_latitude_deg_n,
        min_longitude_deg=min_longitude_deg_e,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )


def simplify_scientific_notation(number_string):
    """Simplifies scientific notation in number string.

    :param number_string: Number represented as string.  This could be something
        like "9.6e-04", "9.6e+00", or "9.6e+04".
    :return: simplified_number_string: Simplified version.  This could be
        something like "9.6e-4", "9.6", or "9.6e4".
    """

    error_checking.assert_is_string(number_string)
    return number_string.replace('e+00', '').replace('e+0', 'e').replace(
        'e-0', 'e-'
    )


def get_solar_altitude_angles(
        valid_times_unix_sec, latitudes_deg_n, longitudes_deg_e,
        temporary_dir_name,
        fortran_exe_name=DEFAULT_EXE_NAME_FOR_ALTITUDE_ANGLE):
    """Computes solar altitude angle at every point on Earth's surface.

    :param valid_times_unix_sec: numpy array of valid times.
    :param latitudes_deg_n: numpy array of latitudes (deg north), with same
        shape as `valid_times_unix_sec`.
    :param longitudes_deg_e: numpy array of longitudes (deg east), with same
        shape as `valid_times_unix_sec`.
    :param temporary_dir_name: Name of directory for temporary text file.
    :param fortran_exe_name: Path to Fortran executable (pathless file name
        should probably be "solarpos").
    :return: altitude_angles_deg: length-P numpy array of solar altitude angles.
    """

    # Check input args.
    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    orig_dimensions = numpy.array(valid_times_unix_sec.shape, dtype=int)

    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        latitudes_deg_n, exact_dimensions=orig_dimensions
    )

    longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        longitudes_deg_e, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        longitudes_deg_e, exact_dimensions=orig_dimensions
    )

    error_checking.assert_is_string(temporary_dir_name)
    error_checking.assert_file_exists(fortran_exe_name)

    # Do actual stuff.
    valid_time_strings_1d = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_ALTITUDE_ANGLE)
        for t in numpy.ravel(valid_times_unix_sec)
    ]
    latitudes_deg_n_1d = numpy.ravel(latitudes_deg_n)
    longitudes_deg_e_1d = numpy.ravel(longitudes_deg_e)
    num_points = len(valid_time_strings_1d)

    temporary_file_name = tempfile.NamedTemporaryFile(
        dir=temporary_dir_name, delete=True
    ).name

    for i in range(0, num_points, BATCH_SIZE_FOR_ALTITUDE_ANGLE):
        if numpy.mod(i, 10 * BATCH_SIZE_FOR_ALTITUDE_ANGLE) == 0:
            print((
                'Have computed solar altitude angle for {0:d} of {1:d} '
                'points...'
            ).format(
                i, num_points
            ))

        first_index = i
        last_index = min([
            i + BATCH_SIZE_FOR_ALTITUDE_ANGLE, num_points
        ])

        command_string = '; '.join([
            '"{0:s}" {1:s} {2:.10f} {3:.10f} >> {4:s}'.format(
                fortran_exe_name, t, y, x, temporary_file_name
            )
            for t, y, x in zip(
                valid_time_strings_1d[first_index:last_index],
                latitudes_deg_n_1d[first_index:last_index],
                longitudes_deg_e_1d[first_index:last_index]
            )
        ])

        exit_code = os.system(command_string)
        if exit_code == 0:
            continue

        raise ValueError(ERROR_STRING_FOR_ALTITUDE_ANGLE)

    print('Have computed solar altitude angle for all {0:d} points!'.format(
        num_points
    ))

    found_header = False
    current_index = 0
    altitude_angles_deg_1d = numpy.full(num_points, numpy.nan)

    for this_line in open(temporary_file_name, 'r').readlines():
        if not found_header:
            found_header = this_line.split()[0] == 'Time'
            continue

        try:
            altitude_angles_deg_1d[current_index] = float(this_line.split()[3])
            found_header = False
            current_index += 1
        except ValueError:
            continue

    if os.path.isfile(temporary_file_name):
        os.remove(temporary_file_name)

    assert current_index == num_points
    return numpy.reshape(altitude_angles_deg_1d, orig_dimensions)


def get_solar_times(valid_times_unix_sec, longitudes_deg_e):
    """Returns local solar time (LST) for each point.

    P = number of points

    :param valid_times_unix_sec: length-P numpy array of valid times.
    :param longitudes_deg_e: length-P numpy array of longitudes (deg east).
    :return: solar_times_sec: length-P numpy array of solar times (seconds,
        ranging from 0...86399).
    """

    # Check input args.
    error_checking.assert_is_numpy_array(valid_times_unix_sec, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)

    num_examples = len(valid_times_unix_sec)
    expected_dim = numpy.array([num_examples], dtype=int)

    longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        longitudes_deg_e, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        longitudes_deg_e, exact_dimensions=expected_dim
    )

    # Do actual stuff.
    solar_times_sec = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        this_time_string = time_conversion.unix_sec_to_string(
            valid_times_unix_sec[i], '%Y-%j-%H-%M-%S'
        )
        this_julian_day = int(this_time_string.split('-')[1])

        this_time_string = time_conversion.unix_sec_to_string(
            valid_times_unix_sec[i], '%Y-%m-%d-%H-%M-%S'
        )
        this_year = int(this_time_string.split('-')[0])
        this_hour = int(this_time_string.split('-')[3])
        num_days_this_year = 365 + int(calendar.isleap(this_year))

        this_year_angle = this_julian_day - 1 + float(this_hour - 12) / 24
        this_gamma = 2 * (numpy.pi / num_days_this_year) * this_year_angle

        this_equation_of_time = 229.18 * (
            0.000075
            + 0.001868 * numpy.cos(this_gamma)
            - 0.032077 * numpy.sin(this_gamma)
            - 0.014615 * numpy.cos(2 * this_gamma)
            - 0.040849 * numpy.sin(2 * this_gamma)
        )

        # this_declination = (
        #     0.006918
        #     - 0.399912 * numpy.cos(this_gamma)
        #     + 0.070257 * numpy.sin(this_gamma)
        #     - 0.006758 * numpy.cos(2 * this_gamma)
        #     + 0.000907 * numpy.sin(2 * this_gamma)
        #     - 0.002697 * numpy.cos(3 * this_gamma)
        #     + 0.001480 * numpy.sin(3 * this_gamma)
        # )

        this_time_offset_sec = MINUTES_TO_SECONDS * (
            this_equation_of_time + 4 * longitudes_deg_e[i]
        )
        solar_times_sec[i] = numpy.mod(
            valid_times_unix_sec[i] + this_time_offset_sec,
            DAYS_TO_SECONDS
        )

    return solar_times_sec
