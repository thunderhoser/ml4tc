"""Helper methods for satellite data."""

import numpy
import xarray
from gewittergefahr.gg_utils import error_checking

GRID_ROW_DIM = 'grid_row'
GRID_COLUMN_DIM = 'grid_column'
TIME_DIM = 'valid_time_unix_sec'

SATELLITE_NUMBER_KEY = 'satellite_number'
BAND_NUMBER_KEY = 'band_number'
BAND_WAVELENGTH_KEY = 'band_wavelength_micrometres'
SATELLITE_LONGITUDE_KEY = 'satellite_longitude_deg_e'
STORM_ID_KEY = 'storm_id_string'
STORM_TYPE_KEY = 'storm_type_string'
STORM_NAME_KEY = 'storm_name'
STORM_LATITUDE_KEY = 'storm_latitude_deg_n'
STORM_LONGITUDE_KEY = 'storm_longitude_deg_e'
STORM_INTENSITY_KEY = 'storm_intensity_kt'
STORM_INTENSITY_NUM_KEY = 'storm_intensity_number'
STORM_MOTION_U_KEY = 'storm_u_motion_m_s01'
STORM_MOTION_V_KEY = 'storm_v_motion_m_s01'
STORM_DISTANCE_TO_LAND_KEY = 'storm_distance_to_land_metres'
STORM_RADIUS_VERSION1_KEY = 'storm_radius_version1_metres'
STORM_RADIUS_VERSION2_KEY = 'storm_radius_version2_metres'
STORM_RADIUS_FRACTIONAL_KEY = 'storm_radius_fractional'
SATELLITE_AZIMUTH_ANGLE_KEY = 'satellite_azimuth_angle_deg'
SATELLITE_ZENITH_ANGLE_KEY = 'satellite_zenith_angle_deg'
SOLAR_AZIMUTH_ANGLE_KEY = 'solar_azimuth_angle_deg'
SOLAR_ZENITH_ANGLE_KEY = 'solar_zenith_angle_deg'
SOLAR_ELEVATION_ANGLE_KEY = 'solar_elevation_angle_deg'
SOLAR_HOUR_ANGLE_KEY = 'solar_hour_angle_deg'
BRIGHTNESS_TEMPERATURE_KEY = 'brightness_temp_kelvins'
GRID_LATITUDE_KEY = 'grid_latitude_deg_n'
GRID_LONGITUDE_KEY = 'grid_longitude_deg_e'

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

    for this_table_xarray in satellite_tables_xarray[1:]:
        assert numpy.array_equal(
            satellite_tables_xarray[0].coords[GRID_ROW_DIM].values,
            this_table_xarray.coords[GRID_ROW_DIM].values
        )
        assert numpy.array_equal(
            satellite_tables_xarray[0].coords[GRID_COLUMN_DIM].values,
            this_table_xarray.coords[GRID_COLUMN_DIM].values
        )

    return xarray.concat(objs=satellite_tables_xarray, dim=TIME_DIM)
