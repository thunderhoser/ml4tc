"""IO methods for raw satellite data from CIRA."""

import os
import sys
import glob
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import longitude_conversion as lng_conversion
import error_checking
import general_utils
import satellite_utils

# TODO(thunderhoser): Maybe add equidistant coords as well?

TIME_FORMAT_IN_FILES = '%Y-%m-%dT%H:%M:%S'
TIME_FORMAT_IN_FILE_NAMES = '%Y%j%H%MM'

YEAR_REGEX = '[0-9][0-9][0-9][0-9]'
JULIAN_DAY_REGEX = '[0-3][0-9][0-9]'
HOUR_REGEX = '[0-2][0-9]'
MINUTE_REGEX = '[0-5][0-9]'

TOLERANCE = 1e-6
KM_TO_METRES = 1000.

LATITUDE_DIM = 'latitude'
LONGITUDE_DIM = 'longitude'
TIME_DIM = 'time'

SATELLITE_NUMBER_KEY = 'satellite_number'
BAND_NUMBER_KEY = 'band_id'
BAND_WAVELENGTH_KEY = 'band_wavelength'
SATELLITE_LONGITUDE_KEY = 'satellite_subpoint_longitude'
STORM_ID_KEY = 'storm_atcfid'
STORM_TYPE_KEY = 'storm_development_level'
STORM_NAME_KEY = 'storm_name'
STORM_LATITUDE_KEY = 'storm_latitude'
STORM_LONGITUDE_KEY = 'storm_longitude'
STORM_INTENSITY_KEY = 'storm_intensity'
STORM_INTENSITY_NUM_KEY = 'storm_current_intensity_number'
STORM_SPEED_KEY = 'storm_speed'
STORM_HEADING_KEY = 'storm_heading'
STORM_DISTANCE_TO_LAND_KEY = 'storm_distance_to_land'
STORM_RADIUS_KEY = 'R5'
STORM_RADIUS_FRACTIONAL_KEY = 'fR5'
SATELLITE_AZIMUTH_ANGLE_KEY = 'satellite_azimuth_angle'
SATELLITE_ZENITH_ANGLE_KEY = 'satellite_zenith_angle'
SOLAR_AZIMUTH_ANGLE_KEY = 'solar_azimuth_angle'
SOLAR_ZENITH_ANGLE_KEY = 'solar_zenith_angle'
SOLAR_ELEVATION_ANGLE_KEY = 'solar_elevation_angle'
SOLAR_HOUR_ANGLE_KEY = 'solar_hour_angle'
BRIGHTNESS_TEMPERATURE_KEY = 'brightness_temperature'


def _singleton_to_array(input_var):
    """Converts singleton (unsized numpy array) to sized numpy array.

    :param input_var: Input variable (unsized or sized numpy array).
    :return: output_var: Output variable (sized numpy array).
    """

    try:
        _ = len(input_var)
        return input_var
    except TypeError:
        return numpy.array([input_var])


def find_file(
        top_directory_name, cyclone_id_string, valid_time_unix_sec,
        raise_error_if_missing=True):
    """Finds NetCDF file with satellite data.

    :param top_directory_name: Name of top-level directory with satellite data.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `satellite_utils.parse_cyclone_id`).
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: satellite_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    satellite_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    satellite_file_name = '{0:s}/{1:s}/{2:s}/{3:s}{4:s}_{5:s}.nc'.format(
        top_directory_name, cyclone_id_string[:4], cyclone_id_string,
        cyclone_id_string[4:], cyclone_id_string[2:4],
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES
        )
    )

    if os.path.isfile(satellite_file_name) or not raise_error_if_missing:
        return satellite_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        satellite_file_name
    )
    raise ValueError(error_string)


def find_files_one_cyclone(top_directory_name, cyclone_id_string,
                           raise_error_if_all_missing=True):
    """Finds all NetCDF files with satellite data for one cyclone.

    :param top_directory_name: Name of top-level directory with satellite data.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `satellite_utils.parse_cyclone_id`).
    :param raise_error_if_all_missing: Boolean flag.  If no files are found and
        `raise_error_if_all_missing == True`, will throw error.  If no files are
        found and `raise_error_if_all_missing == False`, will return empty list.
    :return: satellite_file_names: List of file paths.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    satellite_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    file_pattern = (
        '{0:s}/{1:s}/{2:s}/{3:s}{4:s}_{5:s}{6:s}{7:s}{8:s}M.nc'
    ).format(
        top_directory_name, cyclone_id_string[:4], cyclone_id_string,
        cyclone_id_string[4:], cyclone_id_string[2:4],
        YEAR_REGEX, JULIAN_DAY_REGEX, HOUR_REGEX, MINUTE_REGEX
    )

    satellite_file_names = glob.glob(file_pattern)
    satellite_file_names.sort()

    if raise_error_if_all_missing and len(satellite_file_names) == 0:
        error_string = 'Could not find any files with pattern: "{0:s}"'.format(
            file_pattern
        )
        raise ValueError(error_string)

    return satellite_file_names


def find_cyclones_one_year(top_directory_name, year,
                           raise_error_if_all_missing=True):
    """Finds all cyclones in one year.

    :param top_directory_name: Name of top-level directory with satellite data.
    :param year: Year (integer).
    :param raise_error_if_all_missing: Boolean flag.  If no cyclones are found
        and `raise_error_if_all_missing == True`, will throw error.  If no
        cyclones are found and `raise_error_if_all_missing == False`, will
        return empty list.
    :return: cyclone_id_strings: List of cyclone IDs.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(year)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    directory_pattern = '{0:s}/{1:04d}/{1:04d}[A-Z][A-Z][0-9][0-9]'.format(
        top_directory_name, year
    )
    directory_names = glob.glob(directory_pattern)
    cyclone_id_strings = []

    for this_directory_name in directory_names:
        this_cyclone_id_string = this_directory_name.split('/')[-1]

        try:
            satellite_utils.parse_cyclone_id(this_cyclone_id_string)
            cyclone_id_strings.append(this_cyclone_id_string)
        except:
            pass

    cyclone_id_strings.sort()

    if raise_error_if_all_missing and len(cyclone_id_strings) == 0:
        error_string = (
            'Could not find any cyclone IDs from directories with pattern: '
            '"{0:s}"'
        ).format(directory_pattern)

        raise ValueError(error_string)

    return cyclone_id_strings


def file_name_to_cyclone_id(satellite_file_name):
    """Parses cyclone ID from name of file with satellite data.

    :param satellite_file_name: File path.
    :return: cyclone_id_string: Cyclone ID.
    """

    error_checking.assert_is_string(satellite_file_name)
    cyclone_id_string = satellite_file_name.split('/')[-2]
    satellite_utils.parse_cyclone_id(cyclone_id_string)

    return cyclone_id_string


def read_file(netcdf_file_name):
    """Reads satellite data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: satellite_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    orig_table_xarray = xarray.open_dataset(netcdf_file_name)

    valid_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(
            str(t).split('.')[0], TIME_FORMAT_IN_FILES
        )
        for t in orig_table_xarray.coords[TIME_DIM].values
    ], dtype=int)

    grid_latitude_matrix_deg_n = numpy.expand_dims(
        orig_table_xarray.coords[LATITUDE_DIM].values, axis=0
    )
    grid_longitude_matrix_deg_e = numpy.expand_dims(
        orig_table_xarray.coords[LONGITUDE_DIM].values, axis=0
    )
    grid_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitude_matrix_deg_e, allow_nan=False
    )

    num_times = len(valid_times_unix_sec)
    grid_latitude_matrix_deg_n = numpy.repeat(
        grid_latitude_matrix_deg_n, axis=0, repeats=num_times
    )
    grid_longitude_matrix_deg_e = numpy.repeat(
        grid_longitude_matrix_deg_e, axis=0, repeats=num_times
    )

    num_grid_rows = grid_latitude_matrix_deg_n.shape[1]
    num_grid_columns = grid_longitude_matrix_deg_e.shape[1]
    grid_row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=int
    )
    grid_column_indices = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=int
    )

    metadata_dict = {
        satellite_utils.GRID_ROW_DIM: grid_row_indices,
        satellite_utils.GRID_COLUMN_DIM: grid_column_indices,
        satellite_utils.TIME_DIM: valid_times_unix_sec
    }

    satellite_numbers_float = _singleton_to_array(
        orig_table_xarray[SATELLITE_NUMBER_KEY].values
    )
    satellite_numbers = numpy.round(satellite_numbers_float).astype(int)
    assert numpy.allclose(
        satellite_numbers, satellite_numbers_float, atol=TOLERANCE
    )

    band_numbers_float = _singleton_to_array(
        orig_table_xarray[BAND_NUMBER_KEY].values
    )
    band_numbers = numpy.round(band_numbers_float).astype(int)
    assert numpy.allclose(band_numbers, band_numbers_float, atol=TOLERANCE)

    satellite_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=_singleton_to_array(
            orig_table_xarray[SATELLITE_LONGITUDE_KEY].values
        ),
        allow_nan=False
    )

    cyclone_id_string = file_name_to_cyclone_id(netcdf_file_name)
    cyclone_id_strings = numpy.array(
        [cyclone_id_string] * num_times, dtype='S10'
    )

    try:
        storm_type_strings = numpy.array([
            s for s in orig_table_xarray[STORM_TYPE_KEY].values
        ], dtype='S10')

        storm_names = numpy.array([
            s for s in orig_table_xarray[STORM_NAME_KEY].values
        ], dtype='S10')
    except TypeError:
        storm_type_strings = numpy.array(
            [orig_table_xarray[STORM_TYPE_KEY].values], dtype='S10'
        )
        storm_names = numpy.array(
            [orig_table_xarray[STORM_NAME_KEY].values], dtype='S10'
        )

    storm_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=_singleton_to_array(
            orig_table_xarray[STORM_LONGITUDE_KEY].values
        ),
        allow_nan=False
    )
    storm_distances_to_land_metres = _singleton_to_array(
        orig_table_xarray[STORM_DISTANCE_TO_LAND_KEY].values * KM_TO_METRES
    )
    storm_radii_metres = (
        orig_table_xarray[STORM_RADIUS_KEY].values[0] * KM_TO_METRES
    )

    these_dim = (satellite_utils.TIME_DIM,)
    these_dim_3d = (
        satellite_utils.TIME_DIM, satellite_utils.GRID_ROW_DIM,
        satellite_utils.GRID_COLUMN_DIM
    )

    u_motions_m_s01, v_motions_m_s01 = general_utils.speed_and_heading_to_uv(
        storm_speeds_m_s01=
        _singleton_to_array(orig_table_xarray[STORM_SPEED_KEY].values),
        storm_headings_deg=
        _singleton_to_array(orig_table_xarray[STORM_HEADING_KEY].values)
    )

    main_data_dict = {
        satellite_utils.SATELLITE_NUMBER_KEY: (these_dim, satellite_numbers),
        satellite_utils.BAND_NUMBER_KEY: (these_dim, band_numbers),
        satellite_utils.BAND_WAVELENGTH_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[BAND_WAVELENGTH_KEY].values
            )
        ),
        satellite_utils.SATELLITE_LONGITUDE_KEY: (
            these_dim, satellite_longitudes_deg_e
        ),
        satellite_utils.CYCLONE_ID_KEY: (these_dim, cyclone_id_strings),
        satellite_utils.STORM_TYPE_KEY: (these_dim, storm_type_strings),
        satellite_utils.STORM_NAME_KEY: (these_dim, storm_names),
        satellite_utils.STORM_LATITUDE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[STORM_LATITUDE_KEY].values
            )
        ),
        satellite_utils.STORM_LONGITUDE_KEY: (
            these_dim, storm_longitudes_deg_e
        ),
        satellite_utils.STORM_INTENSITY_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[STORM_INTENSITY_KEY].values
            )
        ),
        satellite_utils.STORM_INTENSITY_NUM_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[STORM_INTENSITY_NUM_KEY].values
            )
        ),
        satellite_utils.STORM_MOTION_U_KEY: (these_dim, u_motions_m_s01),
        satellite_utils.STORM_MOTION_V_KEY: (these_dim, v_motions_m_s01),
        satellite_utils.STORM_DISTANCE_TO_LAND_KEY: (
            these_dim, storm_distances_to_land_metres
        ),
        satellite_utils.STORM_RADIUS_VERSION1_KEY: (
            these_dim, storm_radii_metres[[0]]
        ),
        satellite_utils.STORM_RADIUS_VERSION2_KEY: (
            these_dim, storm_radii_metres[[1]]
        ),
        satellite_utils.STORM_RADIUS_FRACTIONAL_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[STORM_RADIUS_FRACTIONAL_KEY].values
            )
        ),
        satellite_utils.SATELLITE_AZIMUTH_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SATELLITE_AZIMUTH_ANGLE_KEY].values
            )
        ),
        satellite_utils.SATELLITE_ZENITH_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SATELLITE_ZENITH_ANGLE_KEY].values
            )
        ),
        satellite_utils.SOLAR_AZIMUTH_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SOLAR_AZIMUTH_ANGLE_KEY].values
            )
        ),
        satellite_utils.SOLAR_ZENITH_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SOLAR_ZENITH_ANGLE_KEY].values
            )
        ),
        satellite_utils.SOLAR_ELEVATION_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SOLAR_ELEVATION_ANGLE_KEY].values
            )
        ),
        satellite_utils.SOLAR_HOUR_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SOLAR_HOUR_ANGLE_KEY].values
            )
        ),
        satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
            these_dim_3d,
            orig_table_xarray[BRIGHTNESS_TEMPERATURE_KEY].values
        ),
        satellite_utils.GRID_LATITUDE_KEY: (
            (satellite_utils.TIME_DIM, satellite_utils.GRID_ROW_DIM),
            grid_latitude_matrix_deg_n
        ),
        satellite_utils.GRID_LONGITUDE_KEY: (
            (satellite_utils.TIME_DIM, satellite_utils.GRID_COLUMN_DIM),
            grid_longitude_matrix_deg_e
        )
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=metadata_dict)
