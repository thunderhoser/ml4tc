"""Input/output methods for model predictions."""

import os
import copy
import math
import numpy
import netCDF4
from scipy.stats import norm
from scipy.integrate import simps
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tc.utils import satellite_utils

TOLERANCE = 1e-6

EXAMPLE_DIMENSION_KEY = 'example'
CLASS_DIMENSION_KEY = 'class'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_character'
PREDICTION_SET_DIMENSION_KEY = 'prediction_set'
LEAD_TIME_DIMENSION_KEY = 'lead_time'
QUANTILE_DIMENSION_KEY = 'quantile'

PROBABILITY_MATRIX_KEY = 'forecast_probability_matrix'
TARGET_MATRIX_KEY = 'target_class_matrix'
CYCLONE_IDS_KEY = 'cyclone_id_strings'
INIT_TIMES_KEY = 'init_times_unix_sec'
STORM_LATITUDES_KEY = 'storm_latitudes_deg_n'
STORM_LONGITUDES_KEY = 'storm_longitudes_deg_e'
STORM_INTENSITY_CHANGES_KEY = 'storm_intensity_changes_m_s01'
MODEL_FILE_KEY = 'model_file_name'
ISOTONIC_MODEL_FILE_KEY = 'isotonic_model_file_name'
UNCERTAINTY_CALIB_MODEL_FILE_KEY = 'uncertainty_calib_model_file_name'
LEAD_TIMES_KEY = 'lead_times_hours'
QUANTILE_LEVELS_KEY = 'quantile_levels'

ONE_PER_EXAMPLE_KEYS = [
    PROBABILITY_MATRIX_KEY, TARGET_MATRIX_KEY, CYCLONE_IDS_KEY, INIT_TIMES_KEY,
    STORM_LATITUDES_KEY, STORM_LONGITUDES_KEY, STORM_INTENSITY_CHANGES_KEY
]

MONTH_KEY = 'month'
BASIN_ID_KEY = 'basin_id_string'
GRID_ROW_KEY = 'grid_row'
GRID_COLUMN_KEY = 'grid_column'
METADATA_KEYS = [MONTH_KEY, BASIN_ID_KEY, GRID_ROW_KEY, GRID_COLUMN_KEY]

GRID_ROW_DIMENSION_KEY = 'row'
GRID_COLUMN_DIMENSION_KEY = 'column'
GRID_LATITUDE_KEY = 'grid_latitude_deg_n'
GRID_LONGITUDE_KEY = 'grid_longitude_deg_e'


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
        netcdf_file_name, forecast_probability_matrix, target_class_matrix,
        cyclone_id_strings, init_times_unix_sec, storm_latitudes_deg_n,
        storm_longitudes_deg_e, storm_intensity_changes_m_s01, model_file_name,
        lead_times_hours, quantile_levels, isotonic_model_file_name,
        uncertainty_calib_model_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples
    K = number of classes
    L = number of lead times
    S = number of prediction sets

    :param netcdf_file_name: Path to output file.
    :param forecast_probability_matrix: E-by-K-by-L-by-S numpy array of forecast
        probabilities.
    :param target_class_matrix: E-by-L numpy array of target classes, all
        integers in range [0, K - 1].
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param init_times_unix_sec: length-E numpy array of forecast-initialization
        times.
    :param storm_latitudes_deg_n: length-E numpy array of latitudes (deg N).
    :param storm_longitudes_deg_e: length-E numpy array of longitudes (deg E).
    :param storm_intensity_changes_m_s01: length-E numpy of intensity changes
        corresponding to targets.  If prediction task is TD-to-TS (not rapid
        intensification), this should be None.
    :param model_file_name: Path to file with trained model (readable by
        `neural_net.read_model`).
    :param lead_times_hours: length-L numpy array of lead times.
    :param quantile_levels: If `forecast_probability_matrix` contains quantiles,
        this should be a length-(S - 1) numpy array of quantile levels, ranging
        from (0, 1).  Otherwise, this should be None.
    :param isotonic_model_file_name: Path to file with trained
        isotonic-regression model (readable by
        `isotonic_regression.read_model`).  If predictions do not have bias
        correction, make this None.
    :param uncertainty_calib_model_file_name: Path to file with trained
        uncertainty-calibration model (readable by
        `uncertainty_calibration.read_model`).  If predictions do not have
        calibrated uncertainty, make this None.
    """

    error_checking.assert_is_numpy_array(forecast_probability_matrix)

    while len(forecast_probability_matrix.shape) < 4:
        forecast_probability_matrix = numpy.expand_dims(
            forecast_probability_matrix, axis=-1
        )

    error_checking.assert_is_numpy_array(
        forecast_probability_matrix, num_dimensions=4
    )

    # TODO(thunderhoser): Allowing NaN is a hack for SHIPS predictions.
    error_checking.assert_is_geq_numpy_array(
        forecast_probability_matrix, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        forecast_probability_matrix, 1., allow_nan=True
    )

    num_examples = forecast_probability_matrix.shape[0]
    num_classes = forecast_probability_matrix.shape[1]
    num_lead_times = forecast_probability_matrix.shape[2]
    num_prediction_sets = forecast_probability_matrix.shape[3]

    expected_dim = numpy.array([num_examples, num_lead_times], dtype=int)

    error_checking.assert_is_integer_numpy_array(target_class_matrix)
    error_checking.assert_is_numpy_array(
        target_class_matrix, exact_dimensions=expected_dim
    )
    error_checking.assert_is_geq_numpy_array(target_class_matrix, 0)
    error_checking.assert_is_less_than_numpy_array(
        target_class_matrix, num_classes
    )

    expected_dim = numpy.array([num_examples], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(cyclone_id_strings), exact_dimensions=expected_dim
    )
    for this_id_string in cyclone_id_strings:
        _ = satellite_utils.parse_cyclone_id(this_id_string)

    error_checking.assert_is_integer_numpy_array(init_times_unix_sec)
    error_checking.assert_is_numpy_array(
        init_times_unix_sec, exact_dimensions=expected_dim
    )

    # TODO(thunderhoser): Allowing NaN is a HACK for real-time SHIPS data,
    # where I did not properly fill missing coordinates.
    error_checking.assert_is_valid_lat_numpy_array(
        storm_latitudes_deg_n, allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        storm_latitudes_deg_n, exact_dimensions=expected_dim
    )

    # TODO(thunderhoser): Allowing NaN is a HACK for real-time SHIPS data,
    # where I did not properly fill missing coordinates.
    lng_conversion.convert_lng_positive_in_west(
        storm_longitudes_deg_e, allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        storm_longitudes_deg_e, exact_dimensions=expected_dim
    )

    if storm_intensity_changes_m_s01 is not None:
        error_checking.assert_is_numpy_array_without_nan(
            storm_intensity_changes_m_s01
        )
        error_checking.assert_is_numpy_array(
            storm_intensity_changes_m_s01, exact_dimensions=expected_dim
        )

    error_checking.assert_is_string(model_file_name)

    if isotonic_model_file_name is None:
        isotonic_model_file_name = ''
    if uncertainty_calib_model_file_name is None:
        uncertainty_calib_model_file_name = ''

    error_checking.assert_is_string(isotonic_model_file_name)
    error_checking.assert_is_string(uncertainty_calib_model_file_name)

    expected_dim = numpy.array([num_lead_times], dtype=int)
    error_checking.assert_is_numpy_array(
        lead_times_hours, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(lead_times_hours)
    error_checking.assert_is_greater_numpy_array(lead_times_hours, 0)

    if quantile_levels is not None:
        expected_dim = numpy.array([num_prediction_sets - 1], dtype=int)
        error_checking.assert_is_numpy_array(
            quantile_levels, exact_dimensions=expected_dim
        )
        error_checking.assert_is_greater_numpy_array(quantile_levels, 0.)
        error_checking.assert_is_less_than_numpy_array(quantile_levels, 1.)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(
        ISOTONIC_MODEL_FILE_KEY, isotonic_model_file_name
    )
    dataset_object.setncattr(
        UNCERTAINTY_CALIB_MODEL_FILE_KEY, uncertainty_calib_model_file_name
    )
    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(CLASS_DIMENSION_KEY, num_classes)
    dataset_object.createDimension(LEAD_TIME_DIMENSION_KEY, num_lead_times)
    dataset_object.createDimension(
        PREDICTION_SET_DIMENSION_KEY, num_prediction_sets
    )

    these_dim = (
        EXAMPLE_DIMENSION_KEY, CLASS_DIMENSION_KEY, LEAD_TIME_DIMENSION_KEY,
        PREDICTION_SET_DIMENSION_KEY
    )
    dataset_object.createVariable(
        PROBABILITY_MATRIX_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[PROBABILITY_MATRIX_KEY][:] = (
        forecast_probability_matrix
    )

    dataset_object.createVariable(
        TARGET_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(EXAMPLE_DIMENSION_KEY, LEAD_TIME_DIMENSION_KEY)
    )
    dataset_object.variables[TARGET_MATRIX_KEY][:] = target_class_matrix

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

    if storm_intensity_changes_m_s01 is not None:
        dataset_object.createVariable(
            STORM_INTENSITY_CHANGES_KEY, datatype=numpy.float32,
            dimensions=EXAMPLE_DIMENSION_KEY
        )
        dataset_object.variables[STORM_INTENSITY_CHANGES_KEY][:] = (
            storm_intensity_changes_m_s01
        )

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

    dataset_object.createVariable(
        LEAD_TIMES_KEY, datatype=numpy.int32, dimensions=LEAD_TIME_DIMENSION_KEY
    )
    dataset_object.variables[LEAD_TIMES_KEY][:] = lead_times_hours

    if quantile_levels is not None:
        dataset_object.createDimension(
            QUANTILE_DIMENSION_KEY, num_prediction_sets - 1
        )

        dataset_object.createVariable(
            QUANTILE_LEVELS_KEY, datatype=numpy.float32,
            dimensions=QUANTILE_DIMENSION_KEY
        )
        dataset_object.variables[QUANTILE_LEVELS_KEY][:] = quantile_levels

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['forecast_probability_matrix']: See doc for `write_file`.
    prediction_dict['target_class_matrix']: Same.
    prediction_dict['cyclone_id_strings']: Same.
    prediction_dict['init_times_unix_sec']: Same.
    prediction_dict['storm_latitudes_deg_n']: Same.
    prediction_dict['storm_longitudes_deg_e']: Same.
    prediction_dict['storm_intensity_changes_m_s01']: Same.
    prediction_dict['model_file_name']: Same.
    prediction_dict['isotonic_model_file_name']: Same.
    prediction_dict['uncertainty_calib_model_file_name']: Same.
    prediction_dict['lead_times_hours']: Same.
    prediction_dict['quantile_levels']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        PROBABILITY_MATRIX_KEY:
            dataset_object.variables[PROBABILITY_MATRIX_KEY][:],
        CYCLONE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[CYCLONE_IDS_KEY][:])
        ],
        INIT_TIMES_KEY: dataset_object.variables[INIT_TIMES_KEY][:],
        STORM_LATITUDES_KEY: dataset_object.variables[STORM_LATITUDES_KEY][:],
        STORM_LONGITUDES_KEY: dataset_object.variables[STORM_LONGITUDES_KEY][:],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        QUANTILE_LEVELS_KEY: None
    }

    if PREDICTION_SET_DIMENSION_KEY not in dataset_object.dimensions:
        prediction_dict[PROBABILITY_MATRIX_KEY] = numpy.expand_dims(
            prediction_dict[PROBABILITY_MATRIX_KEY], axis=-1
        )

    if LEAD_TIME_DIMENSION_KEY not in dataset_object.dimensions:
        prediction_dict[PROBABILITY_MATRIX_KEY] = numpy.expand_dims(
            prediction_dict[PROBABILITY_MATRIX_KEY], axis=-2
        )

    try:
        prediction_dict[ISOTONIC_MODEL_FILE_KEY] = str(getattr(
            dataset_object, ISOTONIC_MODEL_FILE_KEY
        ))
    except:
        prediction_dict[ISOTONIC_MODEL_FILE_KEY] = ''

    if prediction_dict[ISOTONIC_MODEL_FILE_KEY] == '':
        prediction_dict[ISOTONIC_MODEL_FILE_KEY] = None

    try:
        prediction_dict[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = str(getattr(
            dataset_object, UNCERTAINTY_CALIB_MODEL_FILE_KEY
        ))
    except:
        prediction_dict[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = ''

    if prediction_dict[UNCERTAINTY_CALIB_MODEL_FILE_KEY] == '':
        prediction_dict[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = None

    if QUANTILE_LEVELS_KEY in dataset_object.variables:
        prediction_dict[QUANTILE_LEVELS_KEY] = (
            dataset_object.variables[QUANTILE_LEVELS_KEY][:]
        )
    else:
        prediction_dict[QUANTILE_LEVELS_KEY] = None

    if TARGET_MATRIX_KEY in dataset_object.variables:
        prediction_dict[TARGET_MATRIX_KEY] = (
            dataset_object.variables[TARGET_MATRIX_KEY][:]
        )
    else:
        prediction_dict[TARGET_MATRIX_KEY] = numpy.expand_dims(
            dataset_object.variables['target_classes'][:], axis=-1
        )

    if STORM_INTENSITY_CHANGES_KEY in dataset_object.variables:
        prediction_dict[STORM_INTENSITY_CHANGES_KEY] = (
            dataset_object.variables[STORM_INTENSITY_CHANGES_KEY][:]
        )
    else:
        num_examples = len(prediction_dict[INIT_TIMES_KEY])
        prediction_dict[STORM_INTENSITY_CHANGES_KEY] = numpy.full(
            num_examples, numpy.nan
        )

    if LEAD_TIMES_KEY in dataset_object.variables:
        prediction_dict[LEAD_TIMES_KEY] = (
            dataset_object.variables[LEAD_TIMES_KEY][:]
        )
    else:
        if numpy.mean(prediction_dict[TARGET_MATRIX_KEY] > 0) > 0.25:
            prediction_dict[LEAD_TIMES_KEY] = numpy.array([60000], dtype=int)
        else:
            prediction_dict[LEAD_TIMES_KEY] = numpy.array([24], dtype=int)

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


def subset_by_lead_time(prediction_dict, lead_times_hours):
    """Subsets data by lead time.

    :param prediction_dict: See doc for `write_file`.
    :param lead_times_hours: 1-D numpy array of desired lead times.
    :return: prediction_dict: Same as input but with fewer lead times.
    """

    error_checking.assert_is_numpy_array(lead_times_hours, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(lead_times_hours)
    error_checking.assert_is_greater_numpy_array(lead_times_hours, 0)
    lead_times_hours = numpy.unique(lead_times_hours)

    good_indices = numpy.array([
        numpy.where(prediction_dict[LEAD_TIMES_KEY] == t)[0][0]
        for t in lead_times_hours
    ], dtype=int)

    prediction_dict[PROBABILITY_MATRIX_KEY] = (
        prediction_dict[PROBABILITY_MATRIX_KEY][:, :, good_indices, :]
    )
    prediction_dict[TARGET_MATRIX_KEY] = (
        prediction_dict[TARGET_MATRIX_KEY][..., good_indices]
    )
    prediction_dict[LEAD_TIMES_KEY] = lead_times_hours

    return prediction_dict


def find_grid_metafile(prediction_dir_name, raise_error_if_missing=True):
    """Finds file with metadata for grid.

    This file is needed only if prediction files are split by space (one per
    grid cell).

    :param prediction_dir_name: Name of directory with prediction files.  The
        metafile is expected here.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: grid_metafile_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(prediction_dir_name)
    grid_metafile_name = '{0:s}/grid_metadata.nc'.format(prediction_dir_name)

    if raise_error_if_missing and not os.path.isfile(grid_metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            grid_metafile_name
        )
        raise ValueError(error_string)

    return grid_metafile_name


def write_grid_metafile(grid_latitudes_deg_n, grid_longitudes_deg_e,
                        netcdf_file_name):
    """Writes metadata for grid to NetCDF file.

    This file is needed only if prediction files are split by space (one per
    grid cell).

    M = number of rows in grid
    N = number of columns in grid

    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param netcdf_file_name: Path to output file.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(grid_latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(grid_latitudes_deg_n)

    error_checking.assert_is_numpy_array(
        grid_longitudes_deg_e, num_dimensions=1
    )
    grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitudes_deg_e
    )

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.createDimension(
        GRID_ROW_DIMENSION_KEY, len(grid_latitudes_deg_n)
    )
    dataset_object.createDimension(
        GRID_COLUMN_DIMENSION_KEY, len(grid_longitudes_deg_e)
    )

    dataset_object.createVariable(
        GRID_LATITUDE_KEY, datatype=numpy.float32,
        dimensions=GRID_ROW_DIMENSION_KEY
    )
    dataset_object.variables[GRID_LATITUDE_KEY][:] = grid_latitudes_deg_n

    dataset_object.createVariable(
        GRID_LONGITUDE_KEY, datatype=numpy.float32,
        dimensions=GRID_COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[GRID_LONGITUDE_KEY][:] = grid_longitudes_deg_e

    dataset_object.close()


def read_grid_metafile(netcdf_file_name):
    """Reads metadata for grid from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: grid_latitudes_deg_n: See doc for `write_grid_metafile`.
    :return: grid_longitudes_deg_e: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    grid_latitudes_deg_n = dataset_object.variables[GRID_LATITUDE_KEY][:]
    grid_longitudes_deg_e = dataset_object.variables[GRID_LONGITUDE_KEY][:]
    dataset_object.close()

    return grid_latitudes_deg_n, grid_longitudes_deg_e


def get_mean_predictions(prediction_dict):
    """Computes mean of predictive distribution for each example.

    E = number of examples
    L = number of lead times

    :param prediction_dict: Dictionary returned by `read_file`.
    :return: mean_probabilities: E-by-L numpy array of mean forecast
        probabilities.
    :raises: ValueError: if there are more than 2 classes.
    """

    num_classes = prediction_dict[PROBABILITY_MATRIX_KEY].shape[1]
    if num_classes > 2:
        raise ValueError('Cannot do this with more than 2 classes.')

    if prediction_dict[QUANTILE_LEVELS_KEY] is None:
        return numpy.mean(
            prediction_dict[PROBABILITY_MATRIX_KEY], axis=-1
        )[:, 1, :]

    return prediction_dict[PROBABILITY_MATRIX_KEY][:, 1, :, 0]


def get_median_predictions(prediction_dict):
    """Computes median of predictive distribution for each example.

    E = number of examples
    L = number of lead times

    :param prediction_dict: Dictionary returned by `read_file`.
    :return: median_probabilities: E-by-L numpy array of median forecast
        probabilities.
    :raises: ValueError: if there are more than 2 classes.
    """

    num_classes = prediction_dict[PROBABILITY_MATRIX_KEY].shape[1]
    if num_classes > 2:
        raise ValueError('Cannot do this with more than 2 classes.')

    if QUANTILE_LEVELS_KEY in prediction_dict:
        quantile_levels = prediction_dict[QUANTILE_LEVELS_KEY]
    else:
        quantile_levels = None

    if quantile_levels is None:
        return numpy.median(
            prediction_dict[PROBABILITY_MATRIX_KEY], axis=-1
        )[:, 1, :]

    median_index = 1 + numpy.where(
        numpy.absolute(quantile_levels - 0.5) <= TOLERANCE
    )[0][0]

    return prediction_dict[PROBABILITY_MATRIX_KEY][:, 1, :, median_index]


def get_predictive_stdevs(prediction_dict, use_fancy_quantile_method=True,
                          assume_large_sample_size=True):
    """Computes stdev of predictive distribution for each example.

    E = number of examples

    :param prediction_dict: Dictionary returned by `read_file`.
    :param use_fancy_quantile_method: Boolean flag.  If True, will use Equation
        15 from https://doi.org/10.1186/1471-2288-14-135.  If False, will treat
        each quantile-based estimate as a Monte Carlo estimate.
    :param assume_large_sample_size: [used only for fancy quantile method]
        Boolean flag.  If True, will assume large (essentially infinite) sample
        size.
    :return: prob_stdevs: length-E numpy array with standard deviations of
        forecast probabilities.
    :raises: ValueError: if there are more than 2 classes.
    :raises: ValueError: if there is only one prediction, rather than a
        distribution, per scalar example.
    """

    num_classes = prediction_dict[PROBABILITY_MATRIX_KEY].shape[1]
    if num_classes > 2:
        raise ValueError('Cannot do this with more than 2 classes.')

    num_prediction_sets = prediction_dict[PROBABILITY_MATRIX_KEY].shape[-1]
    if num_prediction_sets == 1:
        raise ValueError(
            'There is only one prediction, rather than a distribution, per '
            'scalar example.'
        )

    if QUANTILE_LEVELS_KEY in prediction_dict:
        quantile_levels = prediction_dict[QUANTILE_LEVELS_KEY]
    else:
        quantile_levels = None

    if quantile_levels is None:
        return numpy.std(
            prediction_dict[PROBABILITY_MATRIX_KEY], axis=-1, ddof=1
        )[:, 1, :]

    error_checking.assert_is_boolean(use_fancy_quantile_method)

    if not use_fancy_quantile_method:
        return numpy.std(
            prediction_dict[PROBABILITY_MATRIX_KEY][..., 1:], axis=-1, ddof=1
        )[:, 1, :]

    error_checking.assert_is_boolean(assume_large_sample_size)

    first_quartile_index = 1 + numpy.where(
        numpy.absolute(quantile_levels - 0.25) <= TOLERANCE
    )[0][0]
    third_quartile_index = 1 + numpy.where(
        numpy.absolute(quantile_levels - 0.75) <= TOLERANCE
    )[0][0]

    if assume_large_sample_size:
        psuedo_sample_size = 100
    else:
        psuedo_sample_size = int(numpy.round(
            0.25 * (len(quantile_levels) - 1)
        ))

    coeff_numerator = 2 * math.factorial(4 * psuedo_sample_size + 1)
    coeff_denominator = (
        math.factorial(psuedo_sample_size) *
        math.factorial(3 * psuedo_sample_size)
    )

    z_scores = numpy.linspace(-10, 10, num=1001, dtype=float)
    cumulative_densities = norm.cdf(z_scores, loc=0., scale=1.)
    prob_densities = norm.pdf(z_scores, loc=0., scale=1.)
    integrands = (
        z_scores *
        (cumulative_densities ** (3 * psuedo_sample_size)) *
        ((1. - cumulative_densities) ** psuedo_sample_size) *
        prob_densities
    )

    eta_value = (
        (coeff_numerator / coeff_denominator) *
        simps(y=integrands, x=z_scores, even='avg')
    )

    prob_iqr_values = (
        prediction_dict[PROBABILITY_MATRIX_KEY][..., 1, :, third_quartile_index]
        -
        prediction_dict[PROBABILITY_MATRIX_KEY][..., 1, :, first_quartile_index]
    )

    return prob_iqr_values / eta_value


def concat_over_ensemble_members(prediction_dicts):
    """Concatenates predictions over ensemble members.

    :param prediction_dicts: 1-D list of dictionaries, each in format returned
        by `read_file`, each containing a different set of ensemble members.
    :return: prediction_dict: A single dictionary, also in the format returned
        by `read_file`, containing all ensemble members.
    """

    for i in range(len(prediction_dicts)):
        these_cyclone_time_strings = [
            '{0:s}_{1:d}'.format(c, t) for c, t in zip(
                prediction_dicts[i][CYCLONE_IDS_KEY],
                prediction_dicts[i][INIT_TIMES_KEY]
            )
        ]

        sort_indices = numpy.argsort(numpy.array(these_cyclone_time_strings))
        prediction_dicts[i] = subset_by_index(
            prediction_dict=prediction_dicts[i], desired_indices=sort_indices
        )

    forecast_prob_matrix = prediction_dicts[0][PROBABILITY_MATRIX_KEY]

    for i in range(len(prediction_dicts)):
        assert (
            prediction_dicts[i][CYCLONE_IDS_KEY] ==
            prediction_dicts[0][CYCLONE_IDS_KEY]
        )
        assert numpy.array_equal(
            prediction_dicts[i][INIT_TIMES_KEY],
            prediction_dicts[0][INIT_TIMES_KEY]
        )
        assert numpy.array_equal(
            prediction_dicts[i][TARGET_MATRIX_KEY],
            prediction_dicts[0][TARGET_MATRIX_KEY]
        )
        assert numpy.allclose(
            prediction_dicts[i][STORM_LATITUDES_KEY],
            prediction_dicts[0][STORM_LATITUDES_KEY], atol=TOLERANCE
        )
        assert numpy.allclose(
            prediction_dicts[i][STORM_LONGITUDES_KEY],
            prediction_dicts[0][STORM_LONGITUDES_KEY], atol=TOLERANCE
        )
        assert numpy.allclose(
            prediction_dicts[i][STORM_INTENSITY_CHANGES_KEY],
            prediction_dicts[0][STORM_INTENSITY_CHANGES_KEY], atol=TOLERANCE
        )
        assert numpy.array_equal(
            prediction_dicts[i][LEAD_TIMES_KEY],
            prediction_dicts[0][LEAD_TIMES_KEY]
        )
        assert prediction_dicts[i][QUANTILE_LEVELS_KEY] is None

        if i == 0:
            continue

        forecast_prob_matrix = numpy.concatenate(
            (forecast_prob_matrix, prediction_dicts[i][PROBABILITY_MATRIX_KEY]),
            axis=-1
        )

    prediction_dict = copy.deepcopy(prediction_dicts[0])
    prediction_dict[PROBABILITY_MATRIX_KEY] = forecast_prob_matrix
    return prediction_dict
