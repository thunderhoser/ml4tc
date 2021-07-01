"""Input/output methods for model predictions."""

import os
import sys
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import satellite_utils

EXAMPLE_DIMENSION_KEY = 'example'
CLASS_DIMENSION_KEY = 'class'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_character'

PROBABILITY_MATRIX_KEY = 'forecast_probability_matrix'
TARGET_CLASSES_KEY = 'target_classes'
CYCLONE_IDS_KEY = 'cyclone_id_strings'
INIT_TIMES_KEY = 'init_times_unix_sec'
STORM_LATITUDES_KEY = 'storm_latitudes_deg_n'
STORM_LONGITUDES_KEY = 'storm_longitudes_deg_e'
MODEL_FILE_KEY = 'model_file_name'

ONE_PER_EXAMPLE_KEYS = [
    PROBABILITY_MATRIX_KEY, TARGET_CLASSES_KEY, CYCLONE_IDS_KEY, INIT_TIMES_KEY,
    STORM_LATITUDES_KEY, STORM_LONGITUDES_KEY
]

MONTH_KEY = 'month'
BASIN_ID_KEY = 'basin_id_string'
GRID_ROW_KEY = 'grid_row'
GRID_COLUMN_KEY = 'grid_column'
METADATA_KEYS = [MONTH_KEY, BASIN_ID_KEY, GRID_ROW_KEY, GRID_COLUMN_KEY]


def find_file(directory_name, month=None, basin_id_string=None,
              grid_row=None, grid_column=None, raise_error_if_missing=True):
    """Finds NetCDF file with predictions.

    :param directory_name: Name of directory where file is expected.
    :param month: Month (integer from 1...12).  If file does not contain
        predictions for a specific month, leave this alone.
    :param basin_id_string: Basin ID (must be accepted by
        `satellite_utils.check_basin_id`).  If file does not contain predictions
        for a specific basin, leave this alone.
    :param grid_row: Grid row (non-negative integer).  If file does not contain
        predictions for a specific spatial region, leave this alone.
    :param grid_column: Same but for grid column.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: prediction_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)

    if month is not None:
        error_checking.assert_is_integer(month)
        error_checking.assert_is_geq(month, 1)
        error_checking.assert_is_leq(month, 12)

        prediction_file_name = '{0:s}/predictions_{1:s}={2:02d}.nc'.format(
            directory_name, MONTH_KEY.replace('_', '-'), month
        )

    elif basin_id_string is not None:
        satellite_utils.check_basin_id(basin_id_string)

        prediction_file_name = '{0:s}/predictions_{1:s}={2:s}.nc'.format(
            directory_name, BASIN_ID_KEY.replace('_', '-'), basin_id_string
        )

    elif grid_row is not None or grid_column is not None:
        error_checking.assert_is_integer(grid_row)
        error_checking.assert_is_geq(grid_row, 0)
        error_checking.assert_is_integer(grid_column)
        error_checking.assert_is_geq(grid_column, 0)

        prediction_file_name = (
            '{0:s}/{1:s}={2:03d}/predictions_{1:s}={2:03d}_{3:s}={4:03d}.nc'
        ).format(
            directory_name, GRID_ROW_KEY.replace('_', '-'), grid_row,
            GRID_COLUMN_KEY.replace('_', '-'), grid_column
        )

    else:
        prediction_file_name = '{0:s}/predictions.nc'.format(
            directory_name, grid_row, grid_column
        )

    if raise_error_if_missing and not os.path.isfile(prediction_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name
        )
        raise ValueError(error_string)

    return prediction_file_name


def file_name_to_metadata(prediction_file_name):
    """Parses metadata from file name.

    This method is the inverse of `find_file`.

    :param prediction_file_name: Path to NetCDF file with predictions.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['month']: See input doc for `find_file`.
    metadata_dict['basin_id_string']: Same.
    metadata_dict['grid_row']: Same.
    metadata_dict['grid_column']: Same.
    """

    error_checking.assert_is_string(prediction_file_name)

    metadata_dict = dict()
    for this_key in METADATA_KEYS:
        metadata_dict[this_key] = None

    pathless_file_name = os.path.split(prediction_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    words = extensionless_file_name.split('_')

    for this_key in METADATA_KEYS:
        this_key_with_dashes = this_key.replace('_', '-')
        if this_key_with_dashes not in words[-1]:
            continue

        metadata_dict[this_key] = words[-1].replace(
            this_key_with_dashes + '=', ''
        )

        if this_key != BASIN_ID_KEY:
            metadata_dict[this_key] = int(metadata_dict[this_key])

        break

    if metadata_dict[GRID_COLUMN_KEY] is not None:
        this_key_with_dashes = GRID_ROW_KEY.replace('_', '-')
        metadata_dict[GRID_ROW_KEY] = int(
            words[-2].replace(this_key_with_dashes + '=', '')
        )

    return metadata_dict


def write_file(
        netcdf_file_name, forecast_probability_matrix, target_classes,
        cyclone_id_strings, init_times_unix_sec, storm_latitudes_deg_n,
        storm_longitudes_deg_e, model_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples
    K = number of classes

    :param netcdf_file_name: Path to output file.
    :param forecast_probability_matrix: E-by-K numpy array of forecast
        probabilities.
    :param target_classes: length-E numpy array of target classes, all integers
        in range [0, K - 1].
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param init_times_unix_sec: length-E numpy array of forecast-initialization
        times.
    :param storm_latitudes_deg_n: length-E numpy array of latitudes (deg N).
    :param storm_longitudes_deg_e: length-E numpy array of longitudes (deg E).
    :param model_file_name: Path to file with trained model (readable by
        `neural_net.read_model`).
    """

    error_checking.assert_is_numpy_array(
        forecast_probability_matrix, num_dimensions=2
    )
    error_checking.assert_is_geq_numpy_array(forecast_probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(forecast_probability_matrix, 1.)

    num_examples = forecast_probability_matrix.shape[0]
    num_classes = forecast_probability_matrix.shape[1]
    expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_integer_numpy_array(target_classes)
    error_checking.assert_is_numpy_array(
        target_classes, exact_dimensions=expected_dim
    )
    error_checking.assert_is_geq_numpy_array(target_classes, 0)
    error_checking.assert_is_less_than_numpy_array(target_classes, num_classes)

    error_checking.assert_is_numpy_array(
        numpy.array(cyclone_id_strings), exact_dimensions=expected_dim
    )
    for this_id_string in cyclone_id_strings:
        _ = satellite_utils.parse_cyclone_id(this_id_string)

    error_checking.assert_is_integer_numpy_array(init_times_unix_sec)
    error_checking.assert_is_numpy_array(
        init_times_unix_sec, exact_dimensions=expected_dim
    )

    error_checking.assert_is_valid_lat_numpy_array(
        storm_latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        storm_latitudes_deg_n, exact_dimensions=expected_dim
    )

    print(len(storm_longitudes_deg_e))
    print(numpy.sum(storm_longitudes_deg_e > 360.))

    lng_conversion.convert_lng_positive_in_west(
        storm_longitudes_deg_e, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        storm_longitudes_deg_e, exact_dimensions=expected_dim
    )

    error_checking.assert_is_string(model_file_name)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(CLASS_DIMENSION_KEY, num_classes)

    dataset_object.createVariable(
        PROBABILITY_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, CLASS_DIMENSION_KEY)
    )
    dataset_object.variables[PROBABILITY_MATRIX_KEY][:] = (
        forecast_probability_matrix
    )

    dataset_object.createVariable(
        TARGET_CLASSES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[TARGET_CLASSES_KEY][:] = target_classes

    dataset_object.createVariable(
        INIT_TIMES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[INIT_TIMES_KEY][:] = init_times_unix_sec

    dataset_object.createVariable(
        STORM_LATITUDES_KEY, datatype=numpy.float32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[STORM_LATITUDES_KEY][:] = storm_latitudes_deg_n

    dataset_object.createVariable(
        STORM_LONGITUDES_KEY, datatype=numpy.float32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[STORM_LONGITUDES_KEY][:] = storm_longitudes_deg_e

    if num_examples == 0:
        num_id_characters = 1
    else:
        num_id_characters = numpy.max(numpy.array([
            len(id) for id in cyclone_id_strings
        ]))

    dataset_object.createDimension(CYCLONE_ID_CHAR_DIM_KEY, num_id_characters)

    string_format = 'S{0:d}'.format(num_id_characters)
    cyclone_ids_char_array = netCDF4.stringtochar(
        numpy.array(cyclone_id_strings, dtype=string_format)
    )

    dataset_object.createVariable(
        CYCLONE_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, CYCLONE_ID_CHAR_DIM_KEY)
    )
    dataset_object.variables[CYCLONE_IDS_KEY][:] = numpy.array(
        cyclone_ids_char_array
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['forecast_probability_matrix']: See doc for `write_file`.
    prediction_dict['target_classes']: Same.
    prediction_dict['cyclone_id_strings']: Same.
    prediction_dict['init_times_unix_sec']: Same.
    prediction_dict['storm_latitudes_deg_n']: Same.
    prediction_dict['storm_longitudes_deg_e']: Same.
    prediction_dict['model_file_name']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        PROBABILITY_MATRIX_KEY:
            dataset_object.variables[PROBABILITY_MATRIX_KEY][:],
        TARGET_CLASSES_KEY: dataset_object.variables[TARGET_CLASSES_KEY][:],
        CYCLONE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[CYCLONE_IDS_KEY][:])
        ],
        INIT_TIMES_KEY: dataset_object.variables[INIT_TIMES_KEY][:],
        STORM_LATITUDES_KEY: dataset_object.variables[STORM_LATITUDES_KEY][:],
        STORM_LONGITUDES_KEY: dataset_object.variables[STORM_LONGITUDES_KEY][:],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY))
    }

    dataset_object.close()
    return prediction_dict


def subset_by_index(prediction_dict, desired_indices):
    """Subsets examples by index.

    :param prediction_dict: See doc for `write_file`.
    :param desired_indices: 1-D numpy array of desired indices.
    :return: prediction_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_numpy_array(desired_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_indices)
    error_checking.assert_is_geq_numpy_array(desired_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_indices, len(prediction_dict[INIT_TIMES_KEY])
    )

    for this_key in ONE_PER_EXAMPLE_KEYS:
        if isinstance(prediction_dict[this_key], list):
            prediction_dict[this_key] = [
                prediction_dict[this_key][k] for k in desired_indices
            ]
        else:
            prediction_dict[this_key] = (
                prediction_dict[this_key][desired_indices, ...]
            )

    return prediction_dict


def subset_by_month(prediction_dict, desired_month):
    """Subsets examples by month.

    :param prediction_dict: See doc for `write_file`.
    :param desired_month: Desired month (integer from 1...12).
    :return: prediction_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_integer(desired_month)
    error_checking.assert_is_geq(desired_month, 1)
    error_checking.assert_is_leq(desired_month, 12)

    all_months = numpy.array([
        int(time_conversion.unix_sec_to_string(t, '%m'))
        for t in prediction_dict[INIT_TIMES_KEY]
    ], dtype=int)

    desired_indices = numpy.where(all_months == desired_month)[0]
    return subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices
    )


def subset_by_basin(prediction_dict, desired_basin_id_string):
    """Subsets examples by basin.

    :param prediction_dict: See doc for `write_file`.
    :param desired_basin_id_string: Desired basin ID (must be accepted by
        `satellite_utils.check_basin_id`).
    :return: prediction_dict: Same as input but with fewer examples.
    """

    satellite_utils.check_basin_id(desired_basin_id_string)

    all_basin_id_strings = numpy.array([
        satellite_utils.parse_cyclone_id(cid)[1]
        for cid in prediction_dict[CYCLONE_IDS_KEY]
    ])
    desired_indices = numpy.where(
        all_basin_id_strings == desired_basin_id_string
    )[0]

    return subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices
    )
