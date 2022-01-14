"""Helper methods for satellite data."""

import sys
import warnings
import numpy
import xarray
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

GRID_SPACING_METRES = 4000.
DEFAULT_NUM_GRID_ROWS = 480
DEFAULT_NUM_GRID_COLUMNS = 640

GRID_ROW_DIM = 'satellite_grid_row'
GRID_COLUMN_DIM = 'satellite_grid_column'
TIME_DIM = 'satellite_valid_time_unix_sec'

SATELLITE_NUMBER_KEY = 'satellite_number'
BAND_NUMBER_KEY = 'satellite_band_number'
BAND_WAVELENGTH_KEY = 'satellite_band_wavelength_micrometres'
SATELLITE_LONGITUDE_KEY = 'satellite_longitude_deg_e'
CYCLONE_ID_KEY = 'satellite_cyclone_id_string'
STORM_TYPE_KEY = 'satellite_storm_type_string'
STORM_NAME_KEY = 'satellite_storm_name'
STORM_LATITUDE_KEY = 'satellite_storm_latitude_deg_n'
STORM_LONGITUDE_KEY = 'satellite_storm_longitude_deg_e'
STORM_INTENSITY_KEY = 'satellite_storm_intensity_m_s01'
STORM_INTENSITY_NUM_KEY = 'satellite_storm_intensity_number'
STORM_MOTION_U_KEY = 'satellite_storm_u_motion_m_s01'
STORM_MOTION_V_KEY = 'satellite_storm_v_motion_m_s01'
STORM_DISTANCE_TO_LAND_KEY = 'satellite_storm_distance_to_land_metres'
STORM_RADIUS_VERSION1_KEY = 'satellite_storm_radius_version1_metres'
STORM_RADIUS_VERSION2_KEY = 'satellite_storm_radius_version2_metres'
STORM_RADIUS_FRACTIONAL_KEY = 'satellite_storm_radius_fractional'
SATELLITE_AZIMUTH_ANGLE_SIN_KEY = 'satellite_azimuth_angle_sin'
SATELLITE_AZIMUTH_ANGLE_COS_KEY = 'satellite_azimuth_angle_cos'
SATELLITE_ZENITH_ANGLE_KEY = 'satellite_zenith_angle_deg'
SOLAR_AZIMUTH_ANGLE_SIN_KEY = 'satellite_solar_azimuth_angle_sin'
SOLAR_AZIMUTH_ANGLE_COS_KEY = 'satellite_solar_azimuth_angle_cos'
SOLAR_ZENITH_ANGLE_KEY = 'satellite_solar_zenith_angle_deg'
SOLAR_ELEVATION_ANGLE_KEY = 'satellite_solar_elevation_angle_deg'
SOLAR_HOUR_ANGLE_SIN_KEY = 'satellite_solar_hour_angle_sin'
SOLAR_HOUR_ANGLE_COS_KEY = 'satellite_solar_hour_angle_cos'
BRIGHTNESS_TEMPERATURE_KEY = 'satellite_brightness_temp_kelvins'
GRID_LATITUDE_KEY = 'satellite_grid_latitude_deg_n'
GRID_LONGITUDE_KEY = 'satellite_grid_longitude_deg_e'

FIELD_NAMES = [
    SATELLITE_NUMBER_KEY, BAND_NUMBER_KEY, BAND_WAVELENGTH_KEY,
    SATELLITE_LONGITUDE_KEY, CYCLONE_ID_KEY, STORM_TYPE_KEY,
    STORM_NAME_KEY, STORM_LATITUDE_KEY, STORM_LONGITUDE_KEY,
    STORM_INTENSITY_KEY, STORM_INTENSITY_NUM_KEY, STORM_MOTION_U_KEY,
    STORM_MOTION_V_KEY, STORM_DISTANCE_TO_LAND_KEY, STORM_RADIUS_VERSION1_KEY,
    STORM_RADIUS_VERSION2_KEY, STORM_RADIUS_FRACTIONAL_KEY,
    SATELLITE_AZIMUTH_ANGLE_SIN_KEY, SATELLITE_AZIMUTH_ANGLE_COS_KEY,
    SATELLITE_ZENITH_ANGLE_KEY,
    SOLAR_AZIMUTH_ANGLE_SIN_KEY, SOLAR_AZIMUTH_ANGLE_COS_KEY,
    SOLAR_ZENITH_ANGLE_KEY, SOLAR_ELEVATION_ANGLE_KEY,
    SOLAR_HOUR_ANGLE_SIN_KEY, SOLAR_HOUR_ANGLE_COS_KEY,
    BRIGHTNESS_TEMPERATURE_KEY, GRID_LATITUDE_KEY, GRID_LONGITUDE_KEY
]

NORTH_ATLANTIC_ID_STRING = 'AL'
SOUTH_ATLANTIC_ID_STRING = 'SL'
NORTHEAST_PACIFIC_ID_STRING = 'EP'
NORTH_CENTRAL_PACIFIC_ID_STRING = 'CP'
NORTHWEST_PACIFIC_ID_STRING = 'WP'
NORTH_INDIAN_ID_STRING = 'IO'
SOUTHERN_HEMISPHERE_ID_STRING = 'SH'

VALID_BASIN_ID_STRINGS = [
    NORTH_ATLANTIC_ID_STRING, SOUTH_ATLANTIC_ID_STRING,
    NORTHEAST_PACIFIC_ID_STRING, NORTH_CENTRAL_PACIFIC_ID_STRING,
    NORTHWEST_PACIFIC_ID_STRING, NORTH_INDIAN_ID_STRING,
    SOUTHERN_HEMISPHERE_ID_STRING
]


def _find_storm_center_px_space(
        storm_latitude_deg_n, storm_longitude_deg_e, grid_latitudes_deg_n,
        grid_longitudes_deg_e):
    """Finds storm center in pixel space.

    M = number of rows in grid
    N = number of columns in grid

    :param storm_latitude_deg_n: Latitude (deg north) of storm center.
    :param storm_longitude_deg_e: Longitude (deg east) of storm center.
    :param grid_latitudes_deg_n: length-M numpy array of grid-point latitudes
        (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of grid-point longitudes
        (deg east).
    :return: storm_row: Row index (half-integer) of storm center.
    :return: storm_column: Column index (half-integer) of storm center.
    """

    num_rows = len(grid_latitudes_deg_n)
    row_indices = numpy.linspace(0, num_rows - 1, num=num_rows, dtype=float)

    interp_object = interp1d(
        x=grid_latitudes_deg_n, y=row_indices, kind='linear',
        bounds_error=True, assume_sorted=True
    )

    try:
        storm_row = number_rounding.round_to_half_integer(
            interp_object(storm_latitude_deg_n)
        )
    except ValueError:
        warning_string = (
            'POTENTIAL ERROR: Cannot find storm center in pixel space.  Storm '
            'latitude is {0:.4f} deg N, while min and max latitudes in grid '
            'are {1:.4f} and {2:.4f} deg N.'
        ).format(
            storm_latitude_deg_n, numpy.min(grid_latitudes_deg_n),
            numpy.max(grid_latitudes_deg_n)
        )

        warnings.warn(warning_string)
        return None, None

    num_columns = len(grid_longitudes_deg_e)
    column_indices = numpy.linspace(
        0, num_columns - 1, num=num_columns, dtype=float
    )

    grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitudes_deg_e, allow_nan=False
    )
    storm_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        storm_longitude_deg_e, allow_nan=False
    )

    if not numpy.all(numpy.diff(grid_longitudes_deg_e) > 0):
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e, allow_nan=False
        )
        storm_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            storm_longitude_deg_e, allow_nan=False
        )

    interp_object = interp1d(
        x=grid_longitudes_deg_e, y=column_indices, kind='linear',
        bounds_error=True, assume_sorted=True
    )

    try:
        storm_column = number_rounding.round_to_half_integer(
            interp_object(storm_longitude_deg_e)
        )
    except ValueError:
        warning_string = (
            'POTENTIAL ERROR: Cannot find storm center in pixel space.  Storm '
            'longitude is {0:.4f} deg E, while longitudes in grid are:'
        ).format(
            storm_longitude_deg_e
        )

        warnings.warn(warning_string)

        numpy.set_printoptions(threshold=sys.maxsize)
        print(grid_longitudes_deg_e)

        return None, None

    return storm_row, storm_column


def _crop_image_around_storm_center(
        data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        storm_row, storm_column, num_cropped_rows, num_cropped_columns):
    """Crops satellite image around storm center.

    M = number of grid rows before cropping
    N = number of grid columns before cropping
    m = number of grid rows after cropping
    n = number of grid columns after cropping

    :param data_matrix: 2-D numpy array of spatial data.
    :param grid_latitudes_deg_n: length-M numpy array of grid-point latitudes
        (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of grid-point longitudes
        (deg east).
    :param storm_row: Row index (half-integer) at storm center.
    :param storm_column: Column index (half-integer) at storm center.
    :param num_cropped_rows: m in the above discussion.  Must be even integer.
    :param num_cropped_columns: n in the above discussion.  Must be even
        integer.
    :return: data_matrix: m-by-n numpy array of spatial data, centered at storm
        center.
    :return: grid_latitudes_deg_n: length-m numpy array of grid-point latitudes
        (deg north).
    :return: grid_longitudes_deg_e: length-n numpy array of grid-point longitudes
        (deg east).
    """

    grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitudes_deg_e
    )

    using_negative_lng = not numpy.all(numpy.diff(grid_longitudes_deg_e) > 0)
    if using_negative_lng:
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e
        )

    num_total_rows = data_matrix.shape[0]
    num_total_columns = data_matrix.shape[1]

    error_checking.assert_is_integer(num_cropped_rows)
    error_checking.assert_is_greater(num_cropped_rows, 0)
    assert numpy.mod(num_cropped_rows, 2) == 0

    error_checking.assert_is_integer(num_cropped_columns)
    error_checking.assert_is_greater(num_cropped_columns, 0)
    assert numpy.mod(num_cropped_columns, 2) == 0

    half_num_cropped_rows = num_cropped_rows / 2 - 0.5
    first_row = int(numpy.round(
        storm_row - half_num_cropped_rows
    ))
    last_row = 1 + int(numpy.round(
        storm_row + half_num_cropped_rows
    ))

    half_num_cropped_columns = num_cropped_columns / 2 - 0.5
    first_column = int(numpy.round(
        storm_column - half_num_cropped_columns
    ))
    last_column = 1 + int(numpy.round(
        storm_column + half_num_cropped_columns
    ))

    if first_row < 0:
        num_padding_rows = -first_row
        padding_arg = ((num_padding_rows, 0), (0, 0))
        data_matrix = numpy.pad(data_matrix, pad_width=padding_arg, mode='edge')

        latitude_spacing_deg = numpy.diff(grid_latitudes_deg_n[:2])[0]
        end_value_arg = (
            grid_latitudes_deg_n[0] - num_padding_rows * latitude_spacing_deg,
            0
        )
        grid_latitudes_deg_n = numpy.pad(
            grid_latitudes_deg_n, pad_width=(num_padding_rows, 0),
            mode='linear_ramp', end_values=end_value_arg
        )

        first_row += num_padding_rows
        last_row += num_padding_rows
        num_total_rows += num_padding_rows

    if last_row > num_total_rows:
        num_padding_rows = last_row - num_total_rows
        padding_arg = ((0, num_padding_rows), (0, 0))
        data_matrix = numpy.pad(data_matrix, pad_width=padding_arg, mode='edge')

        latitude_spacing_deg = numpy.diff(grid_latitudes_deg_n[-2:])[0]
        end_value_arg = (
            0,
            grid_latitudes_deg_n[-1] + num_padding_rows * latitude_spacing_deg
        )
        grid_latitudes_deg_n = numpy.pad(
            grid_latitudes_deg_n, pad_width=(0, num_padding_rows),
            mode='linear_ramp', end_values=end_value_arg
        )

        del num_total_rows

    if first_column < 0:
        num_padding_columns = -first_column
        padding_arg = ((0, 0), (num_padding_columns, 0))
        data_matrix = numpy.pad(data_matrix, pad_width=padding_arg, mode='edge')

        longitude_spacing_deg = numpy.diff(grid_longitudes_deg_e[:2])[0]
        end_value_arg = (
            grid_longitudes_deg_e[0]
            - num_padding_columns * longitude_spacing_deg,
            0
        )
        grid_longitudes_deg_e = numpy.pad(
            grid_longitudes_deg_e, pad_width=(num_padding_columns, 0),
            mode='linear_ramp', end_values=end_value_arg
        )

        first_column += num_padding_columns
        last_column += num_padding_columns
        num_total_columns += num_padding_columns

    if last_column > num_total_columns:
        num_padding_columns = last_column - num_total_columns
        padding_arg = ((0, 0), (0, num_padding_columns))
        data_matrix = numpy.pad(data_matrix, pad_width=padding_arg, mode='edge')

        longitude_spacing_deg = numpy.diff(grid_longitudes_deg_e[-2:])[0]
        end_value_arg = (
            0,
            grid_longitudes_deg_e[-1]
            + num_padding_columns * longitude_spacing_deg
        )
        grid_longitudes_deg_e = numpy.pad(
            grid_longitudes_deg_e, pad_width=(0, num_padding_columns),
            mode='linear_ramp', end_values=end_value_arg
        )

        del num_total_columns

    if using_negative_lng:
        grid_longitudes_deg_e = (
            numpy.mod(grid_longitudes_deg_e + 180, 360) - 180
        )
    else:
        grid_longitudes_deg_e = numpy.mod(grid_longitudes_deg_e, 360)

    return (
        data_matrix[first_row:last_row, first_column:last_column],
        grid_latitudes_deg_n[first_row:last_row],
        lng_conversion.convert_lng_positive_in_west(
            grid_longitudes_deg_e[first_column:last_column]
        )
    )


def check_basin_id(basin_id_string):
    """Ensures that basin ID is valid.

    :param basin_id_string: Basin ID.
    :raises: ValueError: if `basin_id_strings not in VALID_BASIN_ID_STRINGS`.
    """

    error_checking.assert_is_string(basin_id_string)
    if basin_id_string in VALID_BASIN_ID_STRINGS:
        return

    error_string = (
        'Basin ID ("{0:s}") must be in the following list:\n{1:s}'
    ).format(basin_id_string, str(VALID_BASIN_ID_STRINGS))

    raise ValueError(error_string)


def get_cyclone_id(year, basin_id_string, cyclone_number):
    """Creates cyclone ID from metadata.

    :param year: Year (integer).
    :param basin_id_string: Basin ID (must be accepted by `check_basin_id`).
    :param cyclone_number: Cyclone number (integer).
    :return: cyclone_id_string: Cyclone ID.
    """

    error_checking.assert_is_integer(year)
    error_checking.assert_is_geq(year, 0)
    check_basin_id(basin_id_string)
    error_checking.assert_is_integer(cyclone_number)
    error_checking.assert_is_greater(cyclone_number, 0)

    return '{0:04d}{1:s}{2:02d}'.format(year, basin_id_string, cyclone_number)


def parse_cyclone_id(cyclone_id_string):
    """Parses metadata from cyclone ID.

    :param cyclone_id_string: Cyclone ID, formatted like "yyyybbcc", where yyyy
        is the year; bb is the basin ID; and cc is the cyclone number ([cc]th
        cyclone of the season in the given basin).
    :return: year: Year (integer).
    :return: basin_id_string: Basin ID.
    :return: cyclone_number: Cyclone number (integer).
    """

    error_checking.assert_is_string(cyclone_id_string)
    assert len(cyclone_id_string) == 8

    year = int(cyclone_id_string[:4])
    error_checking.assert_is_geq(year, 0)

    basin_id_string = cyclone_id_string[4:6]
    check_basin_id(basin_id_string)

    cyclone_number = int(cyclone_id_string[6:])
    error_checking.assert_is_greater(cyclone_number, 0)

    return year, basin_id_string, cyclone_number


def concat_tables_over_time(satellite_tables_xarray):
    """Concatenates tables with satellite data over the time dimension.

    :param satellite_tables_xarray: 1-D list of xarray tables in format returned
        by `satellite_io.read_file`.
    :return: satellite_table_xarray: One xarray table, in format returned by
        `satellite_io.read_file`, created by concatenating inputs.
    """

    num_rows_by_table = numpy.array([
        len(t.coords[GRID_ROW_DIM].values) for t in satellite_tables_xarray
    ], dtype=int)

    num_columns_by_table = numpy.array([
        len(t.coords[GRID_COLUMN_DIM].values) for t in satellite_tables_xarray
    ], dtype=int)

    if (
            len(numpy.unique(num_rows_by_table)) == 1 and
            len(numpy.unique(num_columns_by_table)) == 1
    ):
        return xarray.concat(objs=satellite_tables_xarray, dim=TIME_DIM)

    good_flags = numpy.logical_and(
        num_rows_by_table == DEFAULT_NUM_GRID_ROWS,
        num_columns_by_table == DEFAULT_NUM_GRID_COLUMNS
    )
    bad_indices = numpy.where(numpy.invert(good_flags))[0]

    for i in bad_indices:
        warning_string = (
            'POTENTIAL ERROR: table {0:d} of {1:d} has {2:d} grid rows and '
            '{3:d} grid columns (expected {4:d} rows and {5:d} columns).  This '
            'is weird.'
        ).format(
            i + 1, len(satellite_tables_xarray),
            num_rows_by_table[i], num_columns_by_table[i],
            DEFAULT_NUM_GRID_ROWS, DEFAULT_NUM_GRID_COLUMNS
        )

        warnings.warn(warning_string)

    good_indices = numpy.where(good_flags)[0]
    tables_to_concat = [satellite_tables_xarray[k] for k in good_indices]

    for this_table_xarray in tables_to_concat[1:]:
        assert numpy.array_equal(
            tables_to_concat[0].coords[GRID_ROW_DIM].values,
            this_table_xarray.coords[GRID_ROW_DIM].values
        )
        assert numpy.array_equal(
            tables_to_concat[0].coords[GRID_COLUMN_DIM].values,
            this_table_xarray.coords[GRID_COLUMN_DIM].values
        )

    return xarray.concat(objs=tables_to_concat, dim=TIME_DIM)


def crop_images_around_storm_centers(
        satellite_table_xarray, num_cropped_rows, num_cropped_columns):
    """Crops satellite image around storm center.

    M = number of grid rows after cropping
    N = number of grid columns after cropping

    :param satellite_table_xarray: xarray table in format returned
        by `satellite_io.read_file`.
    :param num_cropped_rows: M in the above discussion.  Must be even integer.
    :param num_cropped_columns: N in the above discussion.  Must be even
        integer.
    :return: satellite_table_xarray: Same as input but with only M-by-N images.
    """

    error_checking.assert_is_integer(num_cropped_rows)
    error_checking.assert_is_greater(num_cropped_rows, 0)
    error_checking.assert_is_integer(num_cropped_columns)
    error_checking.assert_is_greater(num_cropped_columns, 0)

    bad_object_flags = numpy.logical_or(
        numpy.isnan(satellite_table_xarray[STORM_LATITUDE_KEY].values),
        numpy.isnan(satellite_table_xarray[STORM_LONGITUDE_KEY].values)
    )
    good_object_indices = numpy.where(
        numpy.invert(bad_object_flags)
    )[0]
    satellite_table_xarray = satellite_table_xarray.isel(
        indexers={TIME_DIM: good_object_indices}
    )

    num_storm_objects = len(satellite_table_xarray.coords[TIME_DIM].values)
    brightness_temp_matrix_kelvins = numpy.full(
        (num_storm_objects, num_cropped_rows, num_cropped_columns), numpy.nan
    )
    grid_latitude_matrix_deg_n = numpy.full(
        (num_storm_objects, num_cropped_rows), numpy.nan
    )
    grid_longitude_matrix_deg_e = numpy.full(
        (num_storm_objects, num_cropped_columns), numpy.nan
    )

    t = satellite_table_xarray
    good_object_indices = []

    for i in range(num_storm_objects):
        storm_row, storm_column = _find_storm_center_px_space(
            storm_latitude_deg_n=t[STORM_LATITUDE_KEY].values[i],
            storm_longitude_deg_e=t[STORM_LONGITUDE_KEY].values[i],
            grid_latitudes_deg_n=t[GRID_LATITUDE_KEY].values[i, :],
            grid_longitudes_deg_e=t[GRID_LONGITUDE_KEY].values[i, :]
        )

        if storm_row is None:
            continue

        good_object_indices.append(i)

        (
            brightness_temp_matrix_kelvins[i, ...],
            grid_latitude_matrix_deg_n[i, :],
            grid_longitude_matrix_deg_e[i, :]
        ) = (
            _crop_image_around_storm_center(
                data_matrix=t[BRIGHTNESS_TEMPERATURE_KEY].values[i, ...],
                grid_latitudes_deg_n=t[GRID_LATITUDE_KEY].values[i, :],
                grid_longitudes_deg_e=t[GRID_LONGITUDE_KEY].values[i, :],
                storm_row=storm_row, storm_column=storm_column,
                num_cropped_rows=num_cropped_rows,
                num_cropped_columns=num_cropped_columns
            )
        )

    satellite_table_xarray = satellite_table_xarray.drop(
        [BRIGHTNESS_TEMPERATURE_KEY, GRID_LATITUDE_KEY, GRID_LONGITUDE_KEY]
    )

    this_dict = {
        GRID_ROW_DIM: numpy.linspace(
            0, num_cropped_rows - 1, num=num_cropped_rows, dtype=int
        ),
        GRID_COLUMN_DIM: numpy.linspace(
            0, num_cropped_columns - 1, num=num_cropped_columns, dtype=int
        )
    }
    satellite_table_xarray = satellite_table_xarray.assign_coords(this_dict)

    satellite_table_xarray[BRIGHTNESS_TEMPERATURE_KEY] = (
        (TIME_DIM, GRID_ROW_DIM, GRID_COLUMN_DIM),
        brightness_temp_matrix_kelvins
    )
    satellite_table_xarray[GRID_LATITUDE_KEY] = (
        (TIME_DIM, GRID_ROW_DIM),
        grid_latitude_matrix_deg_n
    )
    satellite_table_xarray[GRID_LONGITUDE_KEY] = (
        (TIME_DIM, GRID_COLUMN_DIM),
        grid_longitude_matrix_deg_e
    )

    good_object_indices = numpy.array(good_object_indices, dtype=int)
    satellite_table_xarray = satellite_table_xarray.isel(
        indexers={TIME_DIM: good_object_indices}
    )

    return satellite_table_xarray
