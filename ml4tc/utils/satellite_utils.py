"""Helper methods for satellite data."""

import warnings
import numpy
import xarray
from gewittergefahr.gg_utils import error_checking

NUM_GRID_ROWS = 480
NUM_GRID_COLUMNS = 640

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

    good_flags = numpy.logical_and(
        num_rows_by_table == NUM_GRID_ROWS,
        num_columns_by_table == NUM_GRID_COLUMNS
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
            NUM_GRID_ROWS, NUM_GRID_COLUMNS
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
