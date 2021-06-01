"""IO methods for raw ATCF files.

ATCF = Automated Tropical-cyclone-forecasting System
"""

import os
import glob
import gzip
import shutil
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import atcf_io
from ml4tc.utils import satellite_utils
from atcf import ABRead

# TODO(thunderhoser): Don't know what to do with TECHNUM or TAU.
# TODO(thunderhoser): These files are garbage and need way more quality control.

TOLERANCE = 1e-6
GZIP_FILE_EXTENSION = '.gz'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

NO_RADIUS_STRINGS = ['NAN', '']
CIRCLE_STRING = 'AAA'
QUADRANTS_STRING = 'NEQ'
VALID_RADIUS_TYPE_STRINGS = (
    NO_RADIUS_STRINGS + [CIRCLE_STRING, QUADRANTS_STRING]
)

BEST_TRACK_NAME = 'BEST'
MAX_NUM_WIND_THRESHOLDS = 3
MAX_NUM_WAVE_HEIGHT_THRESHOLDS = 10

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
NAUTICAL_MILES_TO_METRES = 1852.
FEET_TO_METRES = 1. / 3.2808
MB_TO_PASCALS = 100.

VALID_TIME_KEY = 'DTG'
TECHNIQUE_KEY = 'TECH'
LATITUDE_KEY = 'LAT'
LONGITUDE_KEY = 'LON'
INTENSITY_KEY = 'VMAX'
SEA_LEVEL_PRESSURE_KEY = 'MSLP'
LAST_ISOBAR_PRESSURE_KEY = 'POUTER'
LAST_ISOBAR_RADIUS_KEY = 'ROUTER'
MAX_WIND_RADIUS_KEY = 'RWM'
GUST_SPEED_KEY = 'GUSTS'
EYE_DIAMETER_KEY = 'EYE'
MAX_SEA_HEIGHT_KEY = 'MAXSEAS'
MOTION_HEADING_KEY = 'DIR'
MOTION_SPEED_KEY = 'SPEED'
SYSTEM_DEPTH_KEY = 'DEPTH'
WIND_THRESHOLD_KEY = 'RAD'
WIND_RADIUS_TYPE_KEY = 'WINDCODE'
WIND_RADIUS_CIRCULAR_KEY = 'RAD1'
WIND_RADIUS_NE_QUADRANT_KEY = 'RAD1'
WIND_RADIUS_SE_QUADRANT_KEY = 'RAD2'
WIND_RADIUS_SW_QUADRANT_KEY = 'RAD3'
WIND_RADIUS_NW_QUADRANT_KEY = 'RAD4'
WAVE_HEIGHT_THRESHOLD_KEY = 'SEAS'
WAVE_HEIGHT_RADIUS_TYPE_KEY = 'SEACODE'
WAVE_HEIGHT_RADIUS_CIRCULAR_KEY = 'SEAS1'
WAVE_HEIGHT_RADIUS_NE_QUADRANT_KEY = 'SEAS1'
WAVE_HEIGHT_RADIUS_SE_QUADRANT_KEY = 'SEAS2'
WAVE_HEIGHT_RADIUS_SW_QUADRANT_KEY = 'SEAS3'
WAVE_HEIGHT_RADIUS_NW_QUADRANT_KEY = 'SEAS4'

FIELD_RENAMING_DICT = {
    LATITUDE_KEY: atcf_io.LATITUDE_KEY,
    LONGITUDE_KEY: atcf_io.LONGITUDE_KEY,
    INTENSITY_KEY: atcf_io.INTENSITY_KEY,
    SEA_LEVEL_PRESSURE_KEY: atcf_io.SEA_LEVEL_PRESSURE_KEY,
    LAST_ISOBAR_PRESSURE_KEY: atcf_io.LAST_ISOBAR_PRESSURE_KEY,
    LAST_ISOBAR_RADIUS_KEY: atcf_io.LAST_ISOBAR_RADIUS_KEY,
    MAX_WIND_RADIUS_KEY: atcf_io.MAX_WIND_RADIUS_KEY,
    GUST_SPEED_KEY: atcf_io.GUST_SPEED_KEY,
    EYE_DIAMETER_KEY: atcf_io.EYE_DIAMETER_KEY,
    MAX_SEA_HEIGHT_KEY: atcf_io.MAX_SEA_HEIGHT_KEY,
    MOTION_HEADING_KEY: atcf_io.MOTION_HEADING_KEY,
    MOTION_SPEED_KEY: atcf_io.MOTION_SPEED_KEY
}

RAW_FIELD_TO_CONV_FACTOR = {
    LATITUDE_KEY: 1.,
    LONGITUDE_KEY: 1.,
    INTENSITY_KEY: KT_TO_METRES_PER_SECOND,
    SEA_LEVEL_PRESSURE_KEY: MB_TO_PASCALS,
    LAST_ISOBAR_PRESSURE_KEY: MB_TO_PASCALS,
    LAST_ISOBAR_RADIUS_KEY: NAUTICAL_MILES_TO_METRES,
    MAX_WIND_RADIUS_KEY: NAUTICAL_MILES_TO_METRES,
    GUST_SPEED_KEY: KT_TO_METRES_PER_SECOND,
    EYE_DIAMETER_KEY: NAUTICAL_MILES_TO_METRES,
    MAX_SEA_HEIGHT_KEY: FEET_TO_METRES,
    MOTION_HEADING_KEY: 1.,
    MOTION_SPEED_KEY: KT_TO_METRES_PER_SECOND
}

WIND_THRESHOLDS_KEY = 'wind_thresholds_kt'
WIND_RADII_CIRCULAR_KEY = 'wind_radii_circular_nm'
WIND_RADII_NE_QUADRANT_KEY = 'wind_radii_ne_quadrant_nm'
WIND_RADII_SE_QUADRANT_KEY = 'wind_radii_se_quadrant_nm'
WIND_RADII_SW_QUADRANT_KEY = 'wind_radii_sw_quadrant_nm'
WIND_RADII_NW_QUADRANT_KEY = 'wind_radii_nw_quadrant_nm'

WAVE_HEIGHT_THRESHOLDS_KEY = 'height_thresholds_feet'
WAVE_HEIGHT_RADII_CIRCULAR_KEY = 'height_radii_circular_nm'
WAVE_HEIGHT_RADII_NE_QUADRANT_KEY = 'height_radii_ne_quadrant_nm'
WAVE_HEIGHT_RADII_SE_QUADRANT_KEY = 'height_radii_se_quadrant_nm'
WAVE_HEIGHT_RADII_SW_QUADRANT_KEY = 'height_radii_sw_quadrant_nm'
WAVE_HEIGHT_RADII_NW_QUADRANT_KEY = 'height_radii_nw_quadrant_nm'


def _convert_to_numpy_floats(input_array):
    """Converts input array to numpy array of floats.

    :param input_array: Input array.
    :return: output_array: numpy array of floats.
    """

    try:
        _ = 0.1 * input_array
        return input_array.astype(float)
    except:
        return numpy.array([
            numpy.nan if s is None or s.strip() == ''
            else float(s)
            for s in input_array
        ])


def _convert_to_float(input_value):
    """Converts input value to float.

    :param input_value: Input value.
    :return: output_value: Float value.
    """

    try:
        _ = 0.1 * input_value
        return float(input_value)
    except:
        return (
            numpy.nan if input_value is None or input_value.strip() == ''
            else float(input_value)
        )


def _convert_to_string(input_value):
    """Converts input value to string.

    :param input_value: Input value.
    :return: output_value: Float value.
    """

    try:
        return str(input_value).strip().upper()
    except:
        return ''


def _convert_to_numpy_strings(input_array):
    """Converts input array to numpy array of strings.

    :param input_array: Input array.
    :return: output_array: numpy array of strings.
    """

    return numpy.array([_convert_to_string(s) for s in input_array])


def _read_wind_radii(atcf_object, row_indices):
    """Reads wind radii for one valid time.

    T = max number of possible wind thresholds

    :param atcf_object: Object (kind of like a pandas table) returned by the
        method `ABRead`.
    :param row_indices: 1-D numpy array of row indices for a single valid time.
    :return: wind_radius_dict: Dictionary with the following keys.  Many values
        in the numpy arrays will likely be NaN.
    wind_radius_dict['wind_thresholds_kt']: length-T numpy array of wind
        thresholds (knots).
    wind_radius_dict['wind_radii_circular_nm']: length-T numpy array of circular
        wind radii (nautical miles).
    wind_radius_dict['wind_radii_ne_quadrant_nm']: length-T numpy array of wind
        radii for northeast quadrant (nautical miles).
    wind_radius_dict['wind_radii_se_quadrant_nm']: Same but for southeast
        quadrant.
    wind_radius_dict['wind_radii_sw_quadrant_nm']: Same but for southwest
        quadrant.
    wind_radius_dict['wind_radii_nw_quadrant_nm']: Same but for northwest
        quadrant.

    :raises: ValueError: if radius type is not recognized.
    """

    wind_thresholds_kt = _convert_to_numpy_floats(
        atcf_object[WIND_THRESHOLD_KEY].values[row_indices]
    )
    radius_type_strings = _convert_to_numpy_strings(
        atcf_object[WIND_RADIUS_TYPE_KEY].values[row_indices]
    )
    threshold_and_type_strings = numpy.array([
        '{0:s}_{1:.6f}'.format(a, b)
        for a, b in zip(radius_type_strings, wind_thresholds_kt)
    ])

    unique_indices = numpy.unique(
        threshold_and_type_strings, return_index=True
    )[1]

    unique_indices = numpy.array([
        k for k in unique_indices if wind_thresholds_kt[k] > TOLERANCE
    ], dtype=int)

    row_indices = row_indices[unique_indices]

    wind_thresholds_kt = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)
    wind_radii_circular_nm = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)
    wind_radii_ne_quadrant_nm = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)
    wind_radii_se_quadrant_nm = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)
    wind_radii_sw_quadrant_nm = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)
    wind_radii_nw_quadrant_nm = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)

    for i, r in enumerate(row_indices):
        wind_thresholds_kt[i] = _convert_to_float(
            atcf_object[WIND_THRESHOLD_KEY].values[r]
        )
        this_radius_type_string = _convert_to_string(
            atcf_object[WIND_RADIUS_TYPE_KEY].values[r]
        )

        if this_radius_type_string not in VALID_RADIUS_TYPE_STRINGS:
            error_string = (
                'Radius type ("{0:s}") is not in the following list:\n{1:s}'
            ).format(
                this_radius_type_string, str(VALID_RADIUS_TYPE_STRINGS)
            )

            raise ValueError(error_string)

        if this_radius_type_string in NO_RADIUS_STRINGS:
            continue

        if this_radius_type_string == CIRCLE_STRING:
            wind_radii_circular_nm[i] = _convert_to_float(
                atcf_object[WIND_RADIUS_CIRCULAR_KEY].values[r]
            )
            continue

        if this_radius_type_string == QUADRANTS_STRING:
            wind_radii_ne_quadrant_nm[i] = _convert_to_float(
                atcf_object[WIND_RADIUS_NE_QUADRANT_KEY].values[r]
            )
            wind_radii_se_quadrant_nm[i] = _convert_to_float(
                atcf_object[WIND_RADIUS_SE_QUADRANT_KEY].values[r]
            )
            wind_radii_sw_quadrant_nm[i] = _convert_to_float(
                atcf_object[WIND_RADIUS_SW_QUADRANT_KEY].values[r]
            )
            wind_radii_nw_quadrant_nm[i] = _convert_to_float(
                atcf_object[WIND_RADIUS_NW_QUADRANT_KEY].values[r]
            )

    return {
        WIND_THRESHOLDS_KEY: wind_thresholds_kt,
        WIND_RADII_CIRCULAR_KEY: wind_radii_circular_nm,
        WIND_RADII_NE_QUADRANT_KEY: wind_radii_ne_quadrant_nm,
        WIND_RADII_SE_QUADRANT_KEY: wind_radii_se_quadrant_nm,
        WIND_RADII_SW_QUADRANT_KEY: wind_radii_sw_quadrant_nm,
        WIND_RADII_NW_QUADRANT_KEY: wind_radii_nw_quadrant_nm
    }


def _read_wave_height_radii(atcf_object, row_indices):
    """Reads wave-height radii for one valid time.

    T = max number of possible wave-height thresholds

    :param atcf_object: Object (kind of like a pandas table) returned by the
        method `ABRead`.
    :param row_indices: 1-D numpy array of row indices for a single valid time.
    :return: wave_height_radius_dict: Dictionary with the following keys.  Many
        values in the numpy arrays will likely be NaN.
    wave_height_radius_dict['height_thresholds_feet']: length-T numpy array
        of height thresholds.
    wave_height_radius_dict['height_radii_circular_nm']: length-T numpy array of
        circular height radii (nautical miles).
    wave_height_radius_dict['height_radii_ne_quadrant_nm']: length-T numpy array
        of height radii for northeast quadrant (nautical miles).
    wave_height_radius_dict['height_radii_se_quadrant_nm']: Same but for
        southeast quadrant.
    wave_height_radius_dict['height_radii_sw_quadrant_nm']: Same but for
        southwest quadrant.
    wave_height_radius_dict['height_radii_nw_quadrant_nm']: Same but for
        northwest quadrant.

    :raises: ValueError: if radius type is not recognized.
    """

    height_thresholds_feet = _convert_to_numpy_floats(
        atcf_object[WAVE_HEIGHT_THRESHOLD_KEY].values[row_indices]
    )
    radius_type_strings = _convert_to_numpy_strings(
        atcf_object[WAVE_HEIGHT_RADIUS_TYPE_KEY].values[row_indices]
    )
    threshold_and_type_strings = numpy.array([
        '{0:s}_{1:.6f}'.format(a, b)
        for a, b in zip(radius_type_strings, height_thresholds_feet)
    ])

    unique_indices = numpy.unique(
        threshold_and_type_strings, return_index=True
    )[1]

    unique_indices = numpy.array([
        k for k in unique_indices if height_thresholds_feet[k] > TOLERANCE
    ], dtype=int)

    row_indices = row_indices[unique_indices]

    height_thresholds_feet = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )
    height_radii_circular_nm = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )
    height_radii_ne_quadrant_nm = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )
    height_radii_se_quadrant_nm = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )
    height_radii_sw_quadrant_nm = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )
    height_radii_nw_quadrant_nm = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )

    for i, r in enumerate(row_indices):
        height_thresholds_feet[i] = _convert_to_float(
            atcf_object[WAVE_HEIGHT_THRESHOLD_KEY].values[r]
        )
        this_radius_type_string = _convert_to_string(
            atcf_object[WAVE_HEIGHT_RADIUS_TYPE_KEY].values[r]
        )

        if this_radius_type_string not in VALID_RADIUS_TYPE_STRINGS:
            error_string = (
                'Radius type ("{0:s}") is not in the following list:\n{1:s}'
            ).format(
                this_radius_type_string, str(VALID_RADIUS_TYPE_STRINGS)
            )

            raise ValueError(error_string)

        if this_radius_type_string in NO_RADIUS_STRINGS:
            continue

        if this_radius_type_string == CIRCLE_STRING:
            height_radii_circular_nm[i] = _convert_to_float(
                atcf_object[WAVE_HEIGHT_RADIUS_CIRCULAR_KEY].values[r]
            )
            continue

        if this_radius_type_string == QUADRANTS_STRING:
            height_radii_ne_quadrant_nm[i] = _convert_to_float(
                atcf_object[WAVE_HEIGHT_RADIUS_NE_QUADRANT_KEY].values[r]
            )
            height_radii_se_quadrant_nm[i] = _convert_to_float(
                atcf_object[WAVE_HEIGHT_RADIUS_SE_QUADRANT_KEY].values[r]
            )
            height_radii_sw_quadrant_nm[i] = _convert_to_float(
                atcf_object[WAVE_HEIGHT_RADIUS_SW_QUADRANT_KEY].values[r]
            )
            height_radii_nw_quadrant_nm[i] = _convert_to_float(
                atcf_object[WAVE_HEIGHT_RADIUS_NW_QUADRANT_KEY].values[r]
            )

    return {
        WAVE_HEIGHT_THRESHOLDS_KEY: height_thresholds_feet,
        WAVE_HEIGHT_RADII_CIRCULAR_KEY: height_radii_circular_nm,
        WAVE_HEIGHT_RADII_NE_QUADRANT_KEY: height_radii_ne_quadrant_nm,
        WAVE_HEIGHT_RADII_SE_QUADRANT_KEY: height_radii_se_quadrant_nm,
        WAVE_HEIGHT_RADII_SW_QUADRANT_KEY: height_radii_sw_quadrant_nm,
        WAVE_HEIGHT_RADII_NW_QUADRANT_KEY: height_radii_nw_quadrant_nm
    }


def find_file(directory_name, cyclone_id_string, raise_error_if_missing=True):
    """Finds ASCII file with ATCF data.

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

    extensionless_file_name = '{0:s}/b{1:s}{2:s}'.format(
        directory_name, cyclone_id_string[4:].lower(), cyclone_id_string[:4]
    )
    atcf_file_name = '{0:s}.dat'.format(extensionless_file_name)
    if os.path.isfile(atcf_file_name):
        return atcf_file_name

    atcf_file_name = '{0:s}.dat.gz'.format(extensionless_file_name)
    if os.path.isfile(atcf_file_name):
        return atcf_file_name

    atcf_file_name = '{0:s}.txt'.format(extensionless_file_name)
    if os.path.isfile(atcf_file_name):
        return atcf_file_name

    atcf_file_name = '{0:s}.dat'.format(extensionless_file_name)
    if not raise_error_if_missing:
        return atcf_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        atcf_file_name
    )
    raise ValueError(error_string)


def file_name_to_cyclone_id(atcf_file_name):
    """Parses cyclone ID from file.

    :param atcf_file_name: Path to raw ATCF file.
    :return: cyclone_id_string: Cyclone ID.
    """

    pathless_file_name = os.path.split(atcf_file_name)[1]
    extensionless_file_name = pathless_file_name.split('.')[0]

    return satellite_utils.get_cyclone_id(
        year=int(extensionless_file_name[-4:]),
        basin_id_string=extensionless_file_name[-8:-6].upper(),
        cyclone_number=int(extensionless_file_name[-6:-4])
    )


def find_cyclones_one_year(directory_name, year,
                           raise_error_if_all_missing=True):
    """Finds all cyclones in one year.

    :param directory_name: Name of directory with ATCF data.
    :param year: Year (integer).
    :param raise_error_if_all_missing: Boolean flag.  If no cyclones are found
        and `raise_error_if_all_missing == True`, will throw error.  If no
        cyclones are found and `raise_error_if_all_missing == False`, will
        return empty list.
    :return: cyclone_id_strings: List of cyclone IDs.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(year)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    extensionless_file_pattern = '{0:s}/b[a-z][a-z][0-9][0-9]{1:04d}'.format(
        directory_name, year
    )
    atcf_file_names = glob.glob(
        '{0:s}.dat'.format(extensionless_file_pattern)
    )
    atcf_file_names += glob.glob(
        '{0:s}.dat.gz'.format(extensionless_file_pattern)
    )
    atcf_file_names += glob.glob(
        '{0:s}.txt'.format(extensionless_file_pattern)
    )

    cyclone_id_strings = [file_name_to_cyclone_id(f) for f in atcf_file_names]
    cyclone_id_strings.sort()

    if raise_error_if_all_missing and len(cyclone_id_strings) == 0:
        error_string = (
            'Could not find any cyclone IDs in directory: "{0:s}"'
        ).format(directory_name)

        raise ValueError(error_string)

    return cyclone_id_strings


def read_file(ascii_file_name):
    """Reads ATCF data from ASCII file.

    :param ascii_file_name: Path to input file.
    :return: atcf_table_xarray: xarray table.  Documentation in the xarray table
        should make values self-explanatory.
    """

    if ascii_file_name.endswith(GZIP_FILE_EXTENSION):
        temp_file_name = ascii_file_name[:-len(GZIP_FILE_EXTENSION)]

        with gzip.open(ascii_file_name, 'rb') as ascii_file_handle:
            with open(temp_file_name, 'wb') as temp_file_handle:
                shutil.copyfileobj(ascii_file_handle, temp_file_handle)

        atcf_object = ABRead(temp_file_name).deck
        os.remove(temp_file_name)
    else:
        atcf_object = ABRead(filename=ascii_file_name).deck

    # Find which rows use best-track.
    technique_strings = _convert_to_numpy_strings(
        atcf_object[TECHNIQUE_KEY].values
    )
    best_track_flags = numpy.array(
        [t == BEST_TRACK_NAME for t in technique_strings], dtype=bool
    )
    best_track_indices = numpy.where(best_track_flags)[0]

    # Find unique valid times.
    valid_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in atcf_object[VALID_TIME_KEY].values[best_track_indices]
    ], dtype=int)

    unique_times_unix_sec, these_indices = numpy.unique(
        valid_times_unix_sec, return_index=True
    )
    best_track_indices = best_track_indices[these_indices]
    technique_strings = technique_strings[best_track_indices]
    system_depth_strings = _convert_to_numpy_strings(
        atcf_object[SYSTEM_DEPTH_KEY].values[best_track_indices]
    )

    # Process metadata.
    cyclone_id_string = file_name_to_cyclone_id(ascii_file_name)

    num_entries = len(unique_times_unix_sec)
    cyclone_id_strings = [cyclone_id_string] * num_entries
    storm_object_indices = numpy.linspace(
        0, num_entries - 1, num=num_entries, dtype=int
    )
    wind_threshold_indices = numpy.linspace(
        0, MAX_NUM_WIND_THRESHOLDS - 1, num=MAX_NUM_WIND_THRESHOLDS, dtype=int
    )
    wave_height_threshold_indices = numpy.linspace(
        0, MAX_NUM_WAVE_HEIGHT_THRESHOLDS - 1,
        num=MAX_NUM_WAVE_HEIGHT_THRESHOLDS, dtype=int
    )

    metadata_dict = {
        atcf_io.STORM_OBJECT_DIM: storm_object_indices,
        atcf_io.WIND_THRESHOLD_DIM: wind_threshold_indices,
        atcf_io.WAVE_HEIGHT_THRESHOLD_DIM: wave_height_threshold_indices
    }

    # Process actual data.
    these_dim = (atcf_io.STORM_OBJECT_DIM,)
    main_data_dict = {
        atcf_io.CYCLONE_ID_KEY: (these_dim, cyclone_id_strings),
        atcf_io.VALID_TIME_KEY: (these_dim, unique_times_unix_sec),
        atcf_io.TECHNIQUE_KEY: (these_dim, technique_strings),
        atcf_io.SYSTEM_DEPTH_KEY: (these_dim, system_depth_strings)
    }

    for raw_field_name in FIELD_RENAMING_DICT:
        processed_field_name = FIELD_RENAMING_DICT[raw_field_name]
        these_values = _convert_to_numpy_floats(
            atcf_object[raw_field_name].values[best_track_indices]
        )

        if raw_field_name in RAW_FIELD_TO_CONV_FACTOR:
            these_values *= RAW_FIELD_TO_CONV_FACTOR[raw_field_name]

        main_data_dict[processed_field_name] = (these_dim, these_values)

    these_dim = (num_entries, MAX_NUM_WIND_THRESHOLDS)
    wind_threshold_matrix_m_s01 = numpy.full(these_dim, numpy.nan)
    wind_radius_matrix_circular_metres = numpy.full(these_dim, numpy.nan)
    wind_radius_matrix_ne_metres = numpy.full(these_dim, numpy.nan)
    wind_radius_matrix_se_metres = numpy.full(these_dim, numpy.nan)
    wind_radius_matrix_sw_metres = numpy.full(these_dim, numpy.nan)
    wind_radius_matrix_nw_metres = numpy.full(these_dim, numpy.nan)

    these_dim = (num_entries, MAX_NUM_WAVE_HEIGHT_THRESHOLDS)
    wave_height_threshold_matrix_metres = numpy.full(these_dim, numpy.nan)
    wave_height_radius_matrix_circular_metres = numpy.full(these_dim, numpy.nan)
    wave_height_radius_matrix_ne_metres = numpy.full(these_dim, numpy.nan)
    wave_height_radius_matrix_nw_metres = numpy.full(these_dim, numpy.nan)
    wave_height_radius_matrix_sw_metres = numpy.full(these_dim, numpy.nan)
    wave_height_radius_matrix_se_metres = numpy.full(these_dim, numpy.nan)

    for i in range(num_entries):
        row_indices = numpy.where(
            valid_times_unix_sec == unique_times_unix_sec[i]
        )[0]
        this_dict = _read_wind_radii(
            atcf_object=atcf_object, row_indices=row_indices
        )

        wind_threshold_matrix_m_s01[i, :] = this_dict[WIND_THRESHOLDS_KEY]
        wind_radius_matrix_circular_metres[i, :] = (
            this_dict[WIND_RADII_CIRCULAR_KEY]
        )
        wind_radius_matrix_ne_metres[i, :] = (
            this_dict[WIND_RADII_NE_QUADRANT_KEY]
        )
        wind_radius_matrix_se_metres[i, :] = (
            this_dict[WIND_RADII_SE_QUADRANT_KEY]
        )
        wind_radius_matrix_sw_metres[i, :] = (
            this_dict[WIND_RADII_SW_QUADRANT_KEY]
        )
        wind_radius_matrix_nw_metres[i, :] = (
            this_dict[WIND_RADII_NW_QUADRANT_KEY]
        )

        this_dict = _read_wave_height_radii(
            atcf_object=atcf_object, row_indices=row_indices
        )
        wave_height_threshold_matrix_metres[i, :] = (
            this_dict[WAVE_HEIGHT_THRESHOLDS_KEY]
        )
        wave_height_radius_matrix_circular_metres[i, :] = (
            this_dict[WAVE_HEIGHT_RADII_CIRCULAR_KEY]
        )
        wave_height_radius_matrix_ne_metres[i, :] = (
            this_dict[WAVE_HEIGHT_RADII_NE_QUADRANT_KEY]
        )
        wave_height_radius_matrix_se_metres[i, :] = (
            this_dict[WAVE_HEIGHT_RADII_SE_QUADRANT_KEY]
        )
        wave_height_radius_matrix_sw_metres[i, :] = (
            this_dict[WAVE_HEIGHT_RADII_SW_QUADRANT_KEY]
        )
        wave_height_radius_matrix_nw_metres[i, :] = (
            this_dict[WAVE_HEIGHT_RADII_NW_QUADRANT_KEY]
        )

    wind_threshold_matrix_m_s01 *= KT_TO_METRES_PER_SECOND
    wind_radius_matrix_circular_metres *= NAUTICAL_MILES_TO_METRES
    wind_radius_matrix_ne_metres *= NAUTICAL_MILES_TO_METRES
    wind_radius_matrix_nw_metres *= NAUTICAL_MILES_TO_METRES
    wind_radius_matrix_sw_metres *= NAUTICAL_MILES_TO_METRES
    wind_radius_matrix_se_metres *= NAUTICAL_MILES_TO_METRES

    wave_height_threshold_matrix_metres *= FEET_TO_METRES
    wave_height_radius_matrix_circular_metres *= NAUTICAL_MILES_TO_METRES
    wave_height_radius_matrix_ne_metres *= NAUTICAL_MILES_TO_METRES
    wave_height_radius_matrix_nw_metres *= NAUTICAL_MILES_TO_METRES
    wave_height_radius_matrix_sw_metres *= NAUTICAL_MILES_TO_METRES
    wave_height_radius_matrix_se_metres *= NAUTICAL_MILES_TO_METRES

    these_dim = (atcf_io.STORM_OBJECT_DIM, atcf_io.WIND_THRESHOLD_DIM)
    main_data_dict.update({
        atcf_io.WIND_THRESHOLD_KEY: (these_dim, wind_threshold_matrix_m_s01),
        atcf_io.WIND_RADIUS_CIRCULAR_KEY:
            (these_dim, wind_radius_matrix_circular_metres),
        atcf_io.WIND_RADIUS_NE_QUADRANT_KEY:
            (these_dim, wind_radius_matrix_ne_metres),
        atcf_io.WIND_RADIUS_SE_QUADRANT_KEY:
            (these_dim, wind_radius_matrix_se_metres),
        atcf_io.WIND_RADIUS_SW_QUADRANT_KEY:
            (these_dim, wind_radius_matrix_sw_metres),
        atcf_io.WIND_RADIUS_NW_QUADRANT_KEY:
            (these_dim, wind_radius_matrix_nw_metres)
    })

    these_dim = (atcf_io.STORM_OBJECT_DIM, atcf_io.WAVE_HEIGHT_THRESHOLD_DIM)
    main_data_dict.update({
        atcf_io.WAVE_HEIGHT_THRESHOLD_KEY:
            (these_dim, wave_height_threshold_matrix_metres),
        atcf_io.WAVE_HEIGHT_RADIUS_CIRCULAR_KEY:
            (these_dim, wave_height_radius_matrix_circular_metres),
        atcf_io.WAVE_HEIGHT_RADIUS_NE_QUADRANT_KEY:
            (these_dim, wave_height_radius_matrix_ne_metres),
        atcf_io.WAVE_HEIGHT_RADIUS_SE_QUADRANT_KEY:
            (these_dim, wave_height_radius_matrix_se_metres),
        atcf_io.WAVE_HEIGHT_RADIUS_SW_QUADRANT_KEY:
            (these_dim, wave_height_radius_matrix_sw_metres),
        atcf_io.WAVE_HEIGHT_RADIUS_NW_QUADRANT_KEY:
            (these_dim, wave_height_radius_matrix_nw_metres)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=metadata_dict)
