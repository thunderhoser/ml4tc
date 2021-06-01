"""IO methods for processed ATCF files.

ATCF = Automated Tropical-cyclone-forecasting System
"""

import os
import glob
import numpy
import xarray
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tc.utils import satellite_utils

CYCLONE_ID_REGEX = '[0-9][0-9][0-9][0-9][A-Z][A-Z][0-9][0-9]'

STORM_OBJECT_DIM = 'storm_object_index'
WIND_THRESHOLD_DIM = 'wind_threshold_index'
WAVE_HEIGHT_THRESHOLD_DIM = 'wave_height_threshold_index'

CYCLONE_ID_KEY = 'cyclone_id_string'
VALID_TIME_KEY = 'valid_time_unix_sec'
TECHNIQUE_KEY = 'technique_string'
LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
INTENSITY_KEY = 'intensity_m_s01'
SEA_LEVEL_PRESSURE_KEY = 'sea_level_pressure_pa'
LAST_ISOBAR_PRESSURE_KEY = 'last_closed_isobar_pressure_pa'
LAST_ISOBAR_RADIUS_KEY = 'last_closed_isobar_radius_metres'
MAX_WIND_RADIUS_KEY = 'max_wind_radius_metres'
GUST_SPEED_KEY = 'gust_speed_m_s01'
EYE_DIAMETER_KEY = 'eye_diameter_metres'
MAX_SEA_HEIGHT_KEY = 'max_sea_height_metres'
MOTION_HEADING_KEY = 'motion_heading_deg'
MOTION_SPEED_KEY = 'motion_speed_m_s01'
SYSTEM_DEPTH_KEY = 'system_depth_string'
WIND_THRESHOLD_KEY = 'wind_threshold_m_s01'
WIND_RADIUS_CIRCULAR_KEY = 'wind_radius_circular_metres'
WIND_RADIUS_NE_QUADRANT_KEY = 'wind_radius_ne_quadrant_metres'
WIND_RADIUS_NW_QUADRANT_KEY = 'wind_radius_nw_quadrant_metres'
WIND_RADIUS_SW_QUADRANT_KEY = 'wind_radius_sw_quadrant_metres'
WIND_RADIUS_SE_QUADRANT_KEY = 'wind_radius_se_quadrant_metres'
WAVE_HEIGHT_THRESHOLD_KEY = 'wave_height_threshold_metres'
WAVE_HEIGHT_RADIUS_CIRCULAR_KEY = 'wave_height_radius_circular_metres'
WAVE_HEIGHT_RADIUS_NE_QUADRANT_KEY = 'wave_height_radius_ne_quadrant_metres'
WAVE_HEIGHT_RADIUS_NW_QUADRANT_KEY = 'wave_height_radius_nw_quadrant_metres'
WAVE_HEIGHT_RADIUS_SW_QUADRANT_KEY = 'wave_height_radius_sw_quadrant_metres'
WAVE_HEIGHT_RADIUS_SE_QUADRANT_KEY = 'wave_height_radius_se_quadrant_metres'


def find_file(directory_name, cyclone_id_string, raise_error_if_missing=True):
    """Finds NetCDF file with ATCF data.

    :param directory_name: Name of directory with ATCF data.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `satellite_utils.parse_cyclone_id`).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: atcf_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    satellite_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    atcf_file_name = '{0:s}/atcf_{1:s}.nc'.format(
        directory_name, cyclone_id_string
    )

    if os.path.isfile(atcf_file_name) or not raise_error_if_missing:
        return atcf_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        atcf_file_name
    )
    raise ValueError(error_string)


def find_cyclones(directory_name, raise_error_if_all_missing=True):
    """Finds all cyclones.

    :param directory_name: Name of directory with ATCF data.
    :param raise_error_if_all_missing: Boolean flag.  If no cyclones are found
        and `raise_error_if_all_missing == True`, will throw error.  If no
        cyclones are found and `raise_error_if_all_missing == False`, will
        return empty list.
    :return: cyclone_id_strings: List of cyclone IDs.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    file_pattern = '{0:s}/atcf_{1:s}.nc'.format(
        directory_name, CYCLONE_ID_REGEX
    )
    atcf_file_names = glob.glob(file_pattern)
    cyclone_id_strings = []

    for this_file_name in atcf_file_names:
        try:
            cyclone_id_strings.append(
                file_name_to_cyclone_id(this_file_name)
            )
        except:
            pass

    cyclone_id_strings.sort()

    if raise_error_if_all_missing and len(cyclone_id_strings) == 0:
        error_string = (
            'Could not find any cyclone IDs from files with pattern: "{0:s}"'
        ).format(file_pattern)

        raise ValueError(error_string)

    return cyclone_id_strings


def file_name_to_cyclone_id(atcf_file_name):
    """Parses cyclone ID from name of file with ATCF data.

    :param atcf_file_name: File path.
    :return: cyclone_id_string: Cyclone ID.
    """

    error_checking.assert_is_string(atcf_file_name)
    pathless_file_name = os.path.split(atcf_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    cyclone_id_string = extensionless_file_name.split('_')[-1]
    satellite_utils.parse_cyclone_id(cyclone_id_string)

    return cyclone_id_string


def read_file(netcdf_file_name):
    """Reads ATCF data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: atcf_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(atcf_table_xarray, netcdf_file_name):
    """Writes ATCF data to NetCDF file.

    :param atcf_table_xarray: xarray table in format returned by `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    atcf_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def concat_tables_over_storm_object(atcf_tables_xarray):
    """Concatenates tables with ATCF data over the storm-object dimension.

    :param atcf_tables_xarray: 1-D list of xarray tables in format returned by
        `read_file`.
    :return: atcf_table_xarray: One xarray table, in format returned by
        `read_file`, created by concatenating inputs.
    """

    num_storm_objects_found = 0

    for i in range(len(atcf_tables_xarray)):
        assert numpy.array_equal(
            atcf_tables_xarray[0].coords[WIND_THRESHOLD_DIM].values,
            atcf_tables_xarray[i].coords[WIND_THRESHOLD_DIM].values
        )
        assert numpy.array_equal(
            atcf_tables_xarray[0].coords[WAVE_HEIGHT_THRESHOLD_DIM].values,
            atcf_tables_xarray[i].coords[WAVE_HEIGHT_THRESHOLD_DIM].values
        )

        this_num_storm_objects = len(atcf_tables_xarray[i].index)
        these_indices = numpy.linspace(
            num_storm_objects_found,
            num_storm_objects_found + this_num_storm_objects - 1,
            num=this_num_storm_objects, dtype=int
        )
        num_storm_objects_found += this_num_storm_objects

        atcf_tables_xarray[i].assign_coords({
            STORM_OBJECT_DIM: these_indices
        })

    return xarray.concat(objs=atcf_tables_xarray, dim=STORM_OBJECT_DIM)
