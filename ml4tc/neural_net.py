"""Methods for training and applying neural nets."""

import os
import sys
import copy
import random
import pickle
import warnings
import numpy
import keras
import tensorflow.keras as tf_keras
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import error_checking
import custom_metrics
import example_io
import ships_io
import example_utils
import satellite_utils

TIME_FORMAT_FOR_LOG = '%Y-%m-%d-%H%M'

MINUTES_TO_SECONDS = 60
HOURS_TO_SECONDS = 3600
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

METRIC_DICT = {
    'accuracy': custom_metrics.accuracy,
    'binary_accuracy': custom_metrics.binary_accuracy,
    'binary_csi': custom_metrics.binary_csi,
    'binary_frequency_bias': custom_metrics.binary_frequency_bias,
    'binary_pod': custom_metrics.binary_pod,
    'binary_pofd': custom_metrics.binary_pofd,
    'binary_peirce_score': custom_metrics.binary_peirce_score,
    'binary_success_ratio': custom_metrics.binary_success_ratio,
    'binary_focn': custom_metrics.binary_focn
}

PLATEAU_PATIENCE_EPOCHS = 10
DEFAULT_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 30
LOSS_PATIENCE = 0.

EXAMPLE_FILE_KEY = 'example_file_name'
EXAMPLE_DIRECTORY_KEY = 'example_dir_name'
YEARS_KEY = 'years'
LEAD_TIME_KEY = 'lead_time_hours'
SATELLITE_LAG_TIMES_KEY = 'satellite_lag_times_minutes'
SHIPS_LAG_TIMES_KEY = 'ships_lag_times_hours'
SATELLITE_PREDICTORS_KEY = 'satellite_predictor_names'
SHIPS_PREDICTORS_LAGGED_KEY = 'ships_predictor_names_lagged'
SHIPS_PREDICTORS_FORECAST_KEY = 'ships_predictor_names_forecast'
NUM_POSITIVE_EXAMPLES_KEY = 'num_positive_examples_per_batch'
NUM_NEGATIVE_EXAMPLES_KEY = 'num_negative_examples_per_batch'
MAX_EXAMPLES_PER_CYCLONE_KEY = 'max_examples_per_cyclone_in_batch'
CLASS_CUTOFFS_KEY = 'class_cutoffs_m_s01'
SATELLITE_TIME_TOLERANCE_KEY = 'satellite_time_tolerance_sec'
SATELLITE_MAX_MISSING_TIMES_KEY = 'satellite_max_missing_times'
SHIPS_TIME_TOLERANCE_KEY = 'ships_time_tolerance_sec'
SHIPS_MAX_MISSING_TIMES_KEY = 'ships_max_missing_times'

DEFAULT_SATELLITE_PREDICTOR_NAMES = (
    set(satellite_utils.FIELD_NAMES) -
    set(example_utils.SATELLITE_METADATA_KEYS)
)
DEFAULT_SATELLITE_PREDICTOR_NAMES.remove(
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
)
DEFAULT_SATELLITE_PREDICTOR_NAMES = list(DEFAULT_SATELLITE_PREDICTOR_NAMES)

DEFAULT_SHIPS_PREDICTOR_NAMES_LAGGED = (
    set(ships_io.SATELLITE_FIELD_NAMES) -
    set(example_utils.SHIPS_METADATA_KEYS)
)
DEFAULT_SHIPS_PREDICTOR_NAMES_LAGGED.discard(ships_io.VALID_TIME_KEY)
DEFAULT_SHIPS_PREDICTOR_NAMES_LAGGED = list(
    DEFAULT_SHIPS_PREDICTOR_NAMES_LAGGED
)

DEFAULT_SHIPS_PREDICTOR_NAMES_FORECAST = (
    set(ships_io.FORECAST_FIELD_NAMES) -
    set(example_utils.SHIPS_METADATA_KEYS)
)
DEFAULT_SHIPS_PREDICTOR_NAMES_FORECAST.discard(ships_io.VALID_TIME_KEY)
DEFAULT_SHIPS_PREDICTOR_NAMES_FORECAST = list(
    DEFAULT_SHIPS_PREDICTOR_NAMES_FORECAST
)

DEFAULT_GENERATOR_OPTION_DICT = {
    LEAD_TIME_KEY: 24,
    SATELLITE_PREDICTORS_KEY: DEFAULT_SATELLITE_PREDICTOR_NAMES,
    SHIPS_PREDICTORS_LAGGED_KEY: DEFAULT_SHIPS_PREDICTOR_NAMES_LAGGED,
    SHIPS_PREDICTORS_FORECAST_KEY: DEFAULT_SHIPS_PREDICTOR_NAMES_FORECAST,
    NUM_POSITIVE_EXAMPLES_KEY: 8,
    NUM_NEGATIVE_EXAMPLES_KEY: 24,
    MAX_EXAMPLES_PER_CYCLONE_KEY: 6,
    CLASS_CUTOFFS_KEY: numpy.array([25 * KT_TO_METRES_PER_SECOND])
}

NUM_EPOCHS_KEY = 'num_epochs'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
TRAINING_OPTIONS_KEY = 'training_option_dict'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_OPTIONS_KEY = 'validation_option_dict'
EARLY_STOPPING_KEY = 'do_early_stopping'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_lr_multiplier'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY, EARLY_STOPPING_KEY,
    PLATEAU_LR_MUTIPLIER_KEY
]


def _find_desired_times(
        all_times_unix_sec, desired_times_unix_sec, tolerance_sec,
        max_num_missing_times):
    """Finds desired times.

    L = number of desired times

    :param all_times_unix_sec: 1-D numpy array with all times available.
    :param desired_times_unix_sec: length-L numpy array of desired times.
    :param tolerance_sec: Tolerance.
    :param max_num_missing_times: Max number of missing times allowed (M).  If
        number of missing times <= M, output array will contain -1 for each
        missing time.  If number of missing times > M, output will be None.
    :return: desired_indices: length-L numpy array of indices into
        `all_times_unix_sec`.  The output may also be None (see above).
    """

    desired_indices = []

    for t in desired_times_unix_sec:
        differences_sec = numpy.absolute(all_times_unix_sec - t)
        min_index = numpy.argmin(differences_sec)

        if differences_sec[min_index] > tolerance_sec:
            desired_time_string = time_conversion.unix_sec_to_string(
                t, TIME_FORMAT_FOR_LOG
            )
            found_time_string = time_conversion.unix_sec_to_string(
                all_times_unix_sec[min_index], TIME_FORMAT_FOR_LOG
            )

            warning_string = (
                'POTENTIAL ERROR: Could not find time within {0:d} seconds of '
                '{1:s}.  Nearest found time is {2:s}.'
            ).format(tolerance_sec, desired_time_string, found_time_string)

            warnings.warn(warning_string)

            desired_indices.append(-1)
            continue

        desired_indices.append(min_index)

    desired_indices = numpy.array(desired_indices, dtype=int)
    num_missing_times = numpy.sum(desired_indices == -1)
    num_found_times = numpy.sum(desired_indices != -1)

    if num_found_times == 0:
        desired_time_strings = [
            time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG)
            for t in desired_times_unix_sec
        ]

        warning_string = (
            'POTENTIAL ERROR: No times found.  Could not find time within {0:d}'
            ' seconds of any desired time:\n{1:s}'
        ).format(tolerance_sec, str(desired_time_strings))

        warnings.warn(warning_string)
        return None

    if num_missing_times > max_num_missing_times:
        bad_times_unix_sec = desired_times_unix_sec[desired_indices == -1]
        bad_time_strings = [
            time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG)
            for t in bad_times_unix_sec
        ]

        warning_string = (
            'POTENTIAL ERROR: Too many missing times.  Could not find time '
            'within {0:d} seconds of any of the following:\n{1:s}'
        ).format(tolerance_sec, str(bad_time_strings))

        warnings.warn(warning_string)
        return None

    return desired_indices


def _interp_missing_times_nonspatial(data_values, times_sec):
    """Interpolates to fill non-spatial data at missing times.

    This method handles data for only one example and one variable.

    T = number of time steps

    :param data_values: length-T numpy array of data values.
    :param times_sec: length-T numpy array of times (seconds).  Must be sorted
        in ascending order.
    :return: data_values: Same but without missing values.
    """

    missing_time_flags = numpy.isnan(data_values)
    if not numpy.any(missing_time_flags):
        return data_values

    assert not numpy.all(missing_time_flags)

    missing_time_indices = numpy.where(missing_time_flags)[0]
    found_time_indices = numpy.where(
        numpy.invert(missing_time_flags)
    )[0]

    if len(found_time_indices) == 1:
        data_values[missing_time_indices] = data_values[found_time_indices[0]]
        return data_values

    fill_values = (
        data_values[found_time_indices[0]],
        data_values[found_time_indices[-1]]
    )
    interp_object = interp1d(
        times_sec[found_time_indices], data_values[found_time_indices],
        kind='linear', bounds_error=False, assume_sorted=True,
        fill_value=fill_values
    )
    data_values[missing_time_indices] = interp_object(
        times_sec[missing_time_indices]
    )

    return data_values


def _interp_missing_times_spatial(data_matrix, times_sec):
    """Interpolates to fill spatial data at missing times.

    This method handles data for only one example and one variable.

    M = number of rows in grid
    N = number of columns in grid
    T = number of time steps

    :param data_matrix: M-by-N-by-T numpy array of data values.
    :param times_sec: length-T numpy array of times (seconds).  Must be sorted
        in ascending order.
    :return: data_matrix: Same but without missing values.
    """

    missing_time_flags = numpy.any(numpy.isnan(data_matrix), axis=(0, 1))
    if not numpy.any(missing_time_flags):
        return data_matrix

    assert not numpy.all(missing_time_flags)

    missing_time_indices = numpy.where(missing_time_flags)[0]
    found_time_indices = numpy.where(
        numpy.invert(missing_time_flags)
    )[0]

    for j in missing_time_indices:
        these_diffs = (j - found_time_indices).astype(float)
        these_diffs[these_diffs < 0] = numpy.inf

        if numpy.any(numpy.isfinite(these_diffs)):
            left_time_index = found_time_indices[
                numpy.argmin(numpy.absolute(these_diffs))
            ]
        else:
            left_time_index = -1

        these_diffs = (j - found_time_indices).astype(float)
        these_diffs[these_diffs > 0] = numpy.inf

        if numpy.any(numpy.isfinite(these_diffs)):
            right_time_index = found_time_indices[
                numpy.argmin(numpy.absolute(these_diffs))
            ]
        else:
            right_time_index = -1

        if left_time_index == -1:
            data_matrix[..., j] = data_matrix[..., right_time_index]
            continue

        if right_time_index == -1:
            data_matrix[..., j] = data_matrix[..., left_time_index]
            continue

        actual_time_diff_sec = times_sec[j] - times_sec[left_time_index]
        reference_time_diff_sec = (
            times_sec[right_time_index] - times_sec[left_time_index]
        )
        time_fraction = float(actual_time_diff_sec) / reference_time_diff_sec

        increment_matrix = time_fraction * (
            data_matrix[..., right_time_index] -
            data_matrix[..., left_time_index]
        )
        data_matrix[..., j] = (
            data_matrix[..., left_time_index] + increment_matrix
        )

    return data_matrix


def _interp_missing_times(data_matrix, times_sec):
    """Interpolates to fill data at missing times.

    E = number of examples
    T = number of time steps
    C = number of channels

    :param data_matrix: numpy array of data values.  Dimensions may be E x T x C
        or E x M x N x T x C, where M and N are spatial dimensions.  Either way,
        the first axis is the example axis; the last is the channel axis; and
        the second-last is the time axis.
    :param times_sec: length-T numpy array of times (seconds).  Must be sorted
        in ascending order.
    :return: data_matrix: Same but without missing values.
    """

    num_dimensions = len(data_matrix.shape)
    assert num_dimensions == 3 or num_dimensions == 5
    has_spatial_data = num_dimensions == 5

    num_examples = data_matrix.shape[0]
    num_channels = data_matrix.shape[-1]

    for i in range(num_examples):
        for k in range(num_channels):
            if has_spatial_data:
                data_matrix[i, ..., k] = _interp_missing_times_spatial(
                    data_matrix=data_matrix[i, ..., k], times_sec=times_sec
                )
            else:
                data_matrix[i, ..., k] = _interp_missing_times_nonspatial(
                    data_values=data_matrix[i, ..., k], times_sec=times_sec
                )

    assert not numpy.any(numpy.isnan(data_matrix))

    return data_matrix


def _discretize_intensity_change(intensity_change_m_s01, class_cutoffs_m_s01):
    """Discretizes intensity change into class.

    K = number of classes

    :param intensity_change_m_s01: Intensity change (metres per second).
    :param class_cutoffs_m_s01: numpy array (length K - 1) of class cutoffs.
    :return: class_flags: length-K numpy array of flags.  One value will be 1,
        and the others will be 0.
    """

    class_index = numpy.searchsorted(
        class_cutoffs_m_s01, intensity_change_m_s01, side='right'
    )
    class_flags = numpy.full(len(class_cutoffs_m_s01) + 1, 0, dtype=int)
    class_flags[class_index] = 1

    return class_flags


def _read_one_example_file(
        example_file_name, num_examples_desired, num_positive_examples_desired,
        num_negative_examples_desired, lead_time_hours,
        satellite_lag_times_minutes, ships_lag_times_hours,
        satellite_predictor_names, ships_predictor_names_lagged,
        ships_predictor_names_forecast, class_cutoffs_m_s01,
        satellite_time_tolerance_sec, satellite_max_missing_times,
        ships_time_tolerance_sec, ships_max_missing_times):
    """Reads one example file for generator.

    :param example_file_name: Path to input file.  Will be read by
        `example_io.read_file`.
    :param num_examples_desired: Number of total example desired.
    :param num_positive_examples_desired: Number of positive examples (in
        highest class) desired.
    :param num_negative_examples_desired: Number of negative examples (not in
        highest class) desired.
    :param lead_time_hours: See doc for `input_generator`.
    :param satellite_lag_times_minutes: Same.
    :param ships_lag_times_hours: Same.
    :param satellite_predictor_names: Same.
    :param ships_predictor_names_lagged: Same.
    :param ships_predictor_names_forecast: Same.
    :param class_cutoffs_m_s01: Same.
    :param satellite_time_tolerance_sec: Same.
    :param satellite_max_missing_times: Same.
    :param ships_time_tolerance_sec: Same.
    :param ships_max_missing_times: Same.
    :return: predictor_matrices: Same.
    :return: target_array: Same.
    :return: init_times_unix_sec: 1-D numpy array with forecast-initialization
        time for each example.
    """

    print('Reading data from: "{0:s}"...'.format(example_file_name))
    xt = example_io.read_file(example_file_name)

    all_init_times_unix_sec = (
        xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values
    )
    numpy.random.shuffle(all_init_times_unix_sec)

    satellite_lag_times_sec = satellite_lag_times_minutes * MINUTES_TO_SECONDS
    ships_lag_times_sec = ships_lag_times_hours * HOURS_TO_SECONDS
    lead_time_sec = lead_time_hours * HOURS_TO_SECONDS

    num_classes = len(class_cutoffs_m_s01) + 1
    num_positive_examples_found = 0
    num_negative_examples_found = 0

    satellite_time_indices_by_example = []
    ships_time_indices_by_example = []
    init_times_unix_sec = []

    if num_classes > 2:
        target_array = numpy.full((0, num_classes), -1, dtype=int)
    else:
        target_array = numpy.full(0, -1, dtype=int)

    for t in all_init_times_unix_sec:
        these_satellite_indices = _find_desired_times(
            all_times_unix_sec=
            xt.coords[example_utils.SATELLITE_TIME_DIM].values,
            desired_times_unix_sec=t - satellite_lag_times_sec,
            tolerance_sec=satellite_time_tolerance_sec,
            max_num_missing_times=satellite_max_missing_times
        )
        if these_satellite_indices is None:
            continue

        these_ships_indices = _find_desired_times(
            all_times_unix_sec=
            xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values,
            desired_times_unix_sec=t - ships_lag_times_sec,
            tolerance_sec=ships_time_tolerance_sec,
            max_num_missing_times=ships_max_missing_times
        )
        if these_ships_indices is None:
            continue

        # TODO(thunderhoser): Change list of desired times to all 6-hour steps
        # between t and t + lead_time_sec.
        if t + lead_time_sec > numpy.max(all_init_times_unix_sec):
            these_target_indices = _find_desired_times(
                all_times_unix_sec=
                xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values,
                desired_times_unix_sec=numpy.array([t], dtype=int),
                tolerance_sec=0, max_num_missing_times=0
            )

            intensity_change_m_s01 = -1 * xt[
                example_utils.STORM_INTENSITY_KEY
            ].values[these_target_indices[0]]
        else:
            these_target_indices = _find_desired_times(
                all_times_unix_sec=
                xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values,
                desired_times_unix_sec=
                numpy.array([t, t + lead_time_sec], dtype=int),
                tolerance_sec=0, max_num_missing_times=0
            )
            if these_target_indices is None:
                continue

            intensities_m_s01 = (
                xt[example_utils.STORM_INTENSITY_KEY].values[
                    these_target_indices[0]:(these_target_indices[-1] + 1)
                ]
            )
            intensity_change_m_s01 = numpy.max(
                intensities_m_s01 - intensities_m_s01[0]
            )

        these_flags = _discretize_intensity_change(
            intensity_change_m_s01=intensity_change_m_s01,
            class_cutoffs_m_s01=class_cutoffs_m_s01
        )

        if these_flags[-1] == 1:
            if num_positive_examples_found >= num_positive_examples_desired:
                continue

            num_positive_examples_found += 1

        if these_flags[-1] == 0:
            if num_negative_examples_found >= num_negative_examples_desired:
                continue

            num_negative_examples_found += 1

        if num_classes > 2:
            these_flags = numpy.expand_dims(these_flags, axis=0)
        else:
            these_flags = numpy.array([numpy.argmax(these_flags)], dtype=int)

        target_array = numpy.concatenate(
            (target_array, these_flags), axis=0
        )
        satellite_time_indices_by_example.append(
            these_satellite_indices
        )
        ships_time_indices_by_example.append(these_ships_indices)
        init_times_unix_sec.append(t)

        if (
                num_positive_examples_found >= num_positive_examples_desired and
                num_negative_examples_found >= num_negative_examples_desired
        ):
            break

        if (
                num_positive_examples_found + num_negative_examples_found >=
                num_examples_desired
        ):
            break

    init_times_unix_sec = numpy.array(init_times_unix_sec, dtype=int)

    num_examples = len(ships_time_indices_by_example)
    num_grid_rows = (
        xt[example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY].values.shape[1]
    )
    num_grid_columns = (
        xt[example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY].values.shape[2]
    )
    these_dim = (
        num_examples, num_grid_rows, num_grid_columns,
        len(satellite_lag_times_sec), 1
    )
    brightness_temp_matrix = numpy.full(these_dim, numpy.nan)

    these_dim = (
        num_examples, len(satellite_lag_times_sec),
        len(satellite_predictor_names)
    )
    satellite_predictor_matrix = numpy.full(these_dim, numpy.nan)

    num_ships_lag_times = len(
        xt.coords[example_utils.SHIPS_LAG_TIME_DIM].values
    )
    num_ships_forecast_hours = len(
        xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values
    )
    num_ships_channels_lagged = (
        num_ships_lag_times * len(ships_predictor_names_lagged)
    )
    num_ships_channels = (
        num_ships_channels_lagged +
        num_ships_forecast_hours * len(ships_predictor_names_forecast)
    )

    these_dim = (num_examples, len(ships_lag_times_sec), num_ships_channels)
    ships_predictor_matrix = numpy.full(these_dim, numpy.nan)

    all_satellite_predictor_names = (
        xt.coords[
            example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
        ].values.tolist()
    )
    satellite_predictor_indices = numpy.array([
        all_satellite_predictor_names.index(n)
        for n in satellite_predictor_names
    ], dtype=int)

    all_ships_predictor_names_lagged = (
        xt.coords[example_utils.SHIPS_PREDICTOR_LAGGED_DIM].values.tolist()
    )
    ships_predictor_indices_lagged = numpy.array([
        all_ships_predictor_names_lagged.index(n)
        for n in ships_predictor_names_lagged
    ], dtype=int)

    all_ships_predictor_names_forecast = (
        xt.coords[example_utils.SHIPS_PREDICTOR_FORECAST_DIM].values.tolist()
    )
    ships_predictor_indices_forecast = numpy.array([
        all_ships_predictor_names_forecast.index(n)
        for n in ships_predictor_names_forecast
    ], dtype=int)

    for i in range(num_examples):
        for j in range(len(satellite_lag_times_sec)):
            k = satellite_time_indices_by_example[i][j]

            brightness_temp_matrix[i, ..., j, 0] = xt[
                example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY
            ].values[k, ..., 0]

            satellite_predictor_matrix[i, j, :] = xt[
                example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY
            ].values[k, satellite_predictor_indices]

        for j in range(len(ships_lag_times_sec)):
            k = ships_time_indices_by_example[i][j]

            lagged_values = numpy.ravel(
                xt[
                    example_utils.SHIPS_PREDICTORS_LAGGED_KEY
                ].values[k, :, ships_predictor_indices_lagged]
            )
            ships_predictor_matrix[i, j, :len(lagged_values)] = lagged_values

            forecast_values = numpy.ravel(
                xt[
                    example_utils.SHIPS_PREDICTORS_FORECAST_KEY
                ].values[k, :, ships_predictor_indices_forecast]
            )
            ships_predictor_matrix[i, j, len(lagged_values):] = forecast_values

    brightness_temp_matrix = _interp_missing_times(
        data_matrix=brightness_temp_matrix,
        times_sec=-1 * satellite_lag_times_sec
    )
    satellite_predictor_matrix = _interp_missing_times(
        data_matrix=satellite_predictor_matrix,
        times_sec=-1 * satellite_lag_times_sec
    )
    ships_predictor_matrix = _interp_missing_times(
        data_matrix=ships_predictor_matrix,
        times_sec=-1 * ships_lag_times_sec
    )

    predictor_matrices = [
        brightness_temp_matrix, satellite_predictor_matrix,
        ships_predictor_matrix
    ]

    return predictor_matrices, target_array, init_times_unix_sec


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for `input_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_integer_numpy_array(option_dict[YEARS_KEY])
    error_checking.assert_is_numpy_array(
        option_dict[YEARS_KEY], num_dimensions=1
    )

    error_checking.assert_is_integer(option_dict[LEAD_TIME_KEY])
    assert numpy.mod(option_dict[LEAD_TIME_KEY], 6) == 0

    error_checking.assert_is_integer_numpy_array(
        option_dict[SATELLITE_LAG_TIMES_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[SATELLITE_LAG_TIMES_KEY], 0
    )
    error_checking.assert_is_numpy_array(
        option_dict[SATELLITE_LAG_TIMES_KEY], num_dimensions=1
    )
    option_dict[SATELLITE_LAG_TIMES_KEY] = numpy.sort(
        option_dict[SATELLITE_LAG_TIMES_KEY]
    )[::-1]

    error_checking.assert_is_integer_numpy_array(
        option_dict[SHIPS_LAG_TIMES_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[SHIPS_LAG_TIMES_KEY], 0
    )
    assert numpy.all(numpy.mod(option_dict[SHIPS_LAG_TIMES_KEY], 6) == 0)
    error_checking.assert_is_numpy_array(
        option_dict[SHIPS_LAG_TIMES_KEY], num_dimensions=1
    )
    option_dict[SHIPS_LAG_TIMES_KEY] = numpy.sort(
        option_dict[SHIPS_LAG_TIMES_KEY]
    )[::-1]

    # TODO(thunderhoser): Allow either list to be empty.
    error_checking.assert_is_string_list(option_dict[SATELLITE_PREDICTORS_KEY])
    error_checking.assert_is_string_list(
        option_dict[SHIPS_PREDICTORS_LAGGED_KEY]
    )
    error_checking.assert_is_string_list(
        option_dict[SHIPS_PREDICTORS_FORECAST_KEY]
    )

    error_checking.assert_is_integer(option_dict[NUM_POSITIVE_EXAMPLES_KEY])
    error_checking.assert_is_geq(option_dict[NUM_POSITIVE_EXAMPLES_KEY], 4)
    error_checking.assert_is_integer(option_dict[NUM_NEGATIVE_EXAMPLES_KEY])
    error_checking.assert_is_geq(option_dict[NUM_NEGATIVE_EXAMPLES_KEY], 4)
    error_checking.assert_is_integer(option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY])
    error_checking.assert_is_geq(option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY], 1)
    error_checking.assert_is_greater(
        option_dict[NUM_POSITIVE_EXAMPLES_KEY] +
        option_dict[NUM_NEGATIVE_EXAMPLES_KEY],
        option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY]
    )

    error_checking.assert_is_numpy_array(
        option_dict[CLASS_CUTOFFS_KEY], num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(option_dict[CLASS_CUTOFFS_KEY]), 0.
    )
    assert numpy.all(numpy.isfinite(option_dict[CLASS_CUTOFFS_KEY]))

    error_checking.assert_is_integer(option_dict[SATELLITE_TIME_TOLERANCE_KEY])
    error_checking.assert_is_geq(option_dict[SATELLITE_TIME_TOLERANCE_KEY], 0)
    error_checking.assert_is_integer(
        option_dict[SATELLITE_MAX_MISSING_TIMES_KEY]
    )
    error_checking.assert_is_geq(
        option_dict[SATELLITE_MAX_MISSING_TIMES_KEY], 0
    )
    error_checking.assert_is_integer(option_dict[SHIPS_TIME_TOLERANCE_KEY])
    error_checking.assert_is_geq(option_dict[SHIPS_TIME_TOLERANCE_KEY], 0)
    error_checking.assert_is_integer(option_dict[SHIPS_MAX_MISSING_TIMES_KEY])
    error_checking.assert_is_geq(option_dict[SHIPS_MAX_MISSING_TIMES_KEY], 0)

    return option_dict


def _write_metafile(
        pickle_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, do_early_stopping, plateau_lr_multiplier):
    """Writes metadata to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: See doc for `train_model`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param do_early_stopping: Same.
    :param plateau_lr_multiplier: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        EARLY_STOPPING_KEY: do_early_stopping,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_lr_multiplier
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def find_metafile(model_file_name, raise_error_if_missing=True):
    """Finds metafile for neural net.

    :param model_file_name: Path to trained model.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: metafile_name: Path to metafile.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name
        )
        raise ValueError(error_string)

    return metafile_name


def read_metafile(pickle_file_name):
    """Reads metadata for neural net from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['num_epochs']: See doc for `train_model`.
    metadata_dict['num_training_batches_per_epoch']: Same.
    metadata_dict['training_option_dict']: Same.
    metadata_dict['num_validation_batches_per_epoch']: Same.
    metadata_dict['validation_option_dict']: Same.
    metadata_dict['do_early_stopping']: Same.
    metadata_dict['plateau_lr_multiplier']: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    training_option_dict = metadata_dict[TRAINING_OPTIONS_KEY]
    validation_option_dict = metadata_dict[VALIDATION_OPTIONS_KEY]

    if SATELLITE_TIME_TOLERANCE_KEY not in training_option_dict:
        training_option_dict[SATELLITE_TIME_TOLERANCE_KEY] = 930
        training_option_dict[SATELLITE_MAX_MISSING_TIMES_KEY] = 1
        training_option_dict[SHIPS_TIME_TOLERANCE_KEY] = 0
        training_option_dict[SHIPS_MAX_MISSING_TIMES_KEY] = 1

        validation_option_dict[SATELLITE_TIME_TOLERANCE_KEY] = 3630
        validation_option_dict[SATELLITE_MAX_MISSING_TIMES_KEY] = int(1e10)
        validation_option_dict[SHIPS_TIME_TOLERANCE_KEY] = 21610
        validation_option_dict[SHIPS_MAX_MISSING_TIMES_KEY] = int(1e10)

    metadata_dict[TRAINING_OPTIONS_KEY] = training_option_dict
    metadata_dict[VALIDATION_OPTIONS_KEY] = validation_option_dict

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def create_inputs(option_dict):
    """Creates input data for neural net.

    This method is the same as `input_generator`, except that it returns all the
    data at once, rather than generating batches on the fly.

    :param option_dict: Dictionary with the following keys.
    option_dict['example_file_name']: Path to example file (will be read by
        `example_io.read_file`).
    option_dict['lead_time_hours']: See doc for `input_generator`.
    option_dict['satellite_lag_times_minutes']: Same.
    option_dict['ships_lag_times_hours']: Same.
    option_dict['satellite_predictor_names']: Same.
    option_dict['ships_predictor_names_lagged']: Same.
    option_dict['ships_predictor_names_forecast']: Same.
    option_dict['class_cutoffs_m_s01']: Same.
    option_dict['satellite_time_tolerance_sec']: Same.
    option_dict['satellite_max_missing_times']: Same.
    option_dict['ships_time_tolerance_sec']: Same.
    option_dict['ships_max_missing_times']: Same.

    :return: predictor_matrices: Same.
    :return: target_array: Same.
    :return: init_times_unix_sec: 1-D numpy array with forecast-initialization
        time for each example.
    """

    option_dict[EXAMPLE_DIRECTORY_KEY] = 'foo'
    option_dict[YEARS_KEY] = numpy.array([1900], dtype=int)
    option_dict[NUM_POSITIVE_EXAMPLES_KEY] = 8
    option_dict[NUM_NEGATIVE_EXAMPLES_KEY] = 8
    option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY] = 4

    option_dict = _check_generator_args(option_dict)

    example_file_name = option_dict[EXAMPLE_FILE_KEY]
    lead_time_hours = option_dict[LEAD_TIME_KEY]
    satellite_lag_times_minutes = option_dict[SATELLITE_LAG_TIMES_KEY]
    ships_lag_times_hours = option_dict[SHIPS_LAG_TIMES_KEY]
    satellite_predictor_names = option_dict[SATELLITE_PREDICTORS_KEY]
    ships_predictor_names_lagged = option_dict[SHIPS_PREDICTORS_LAGGED_KEY]
    ships_predictor_names_forecast = option_dict[SHIPS_PREDICTORS_FORECAST_KEY]
    class_cutoffs_m_s01 = option_dict[CLASS_CUTOFFS_KEY]
    satellite_time_tolerance_sec = option_dict[SATELLITE_TIME_TOLERANCE_KEY]
    satellite_max_missing_times = option_dict[SATELLITE_MAX_MISSING_TIMES_KEY]
    ships_time_tolerance_sec = option_dict[SHIPS_TIME_TOLERANCE_KEY]
    ships_max_missing_times = option_dict[SHIPS_MAX_MISSING_TIMES_KEY]

    predictor_matrices, target_array, init_times_unix_sec = (
        _read_one_example_file(
            example_file_name=example_file_name,
            num_examples_desired=int(1e10),
            num_positive_examples_desired=int(1e10),
            num_negative_examples_desired=int(1e10),
            lead_time_hours=lead_time_hours,
            satellite_lag_times_minutes=satellite_lag_times_minutes,
            ships_lag_times_hours=ships_lag_times_hours,
            satellite_predictor_names=satellite_predictor_names,
            ships_predictor_names_lagged=ships_predictor_names_lagged,
            ships_predictor_names_forecast=ships_predictor_names_forecast,
            class_cutoffs_m_s01=class_cutoffs_m_s01,
            satellite_time_tolerance_sec=satellite_time_tolerance_sec,
            satellite_max_missing_times=satellite_max_missing_times,
            ships_time_tolerance_sec=ships_time_tolerance_sec,
            ships_max_missing_times=ships_max_missing_times
        )
    )

    predictor_matrices = [p.astype('float16') for p in predictor_matrices]

    return predictor_matrices, target_array, init_times_unix_sec


def input_generator(option_dict):
    """Generates input data for neural net.

    E = number of examples per batch
    M = number of rows in satellite grid
    N = number of columns in satellite grid
    T_sat = number of lag times for satellite-based predictors
    T_ships = number of lag times for SHIPS predictors
    C_sat = number of channels for ungridded satellite-based predictors
    C_ships = number of channels for SHIPS predictors
    K = number of classes

    :param option_dict: Dictionary with the following keys.
    option_dict['example_dir_name']: Name of directory with example files.
        Files therein will be found by `example_io.find_file` and read by
        `example_io.read_file`.
    option_dict['years']: 1-D numpy array of training years.
    option_dict['lead_time_hours']: Lead time for predicting storm intensity.
    option_dict['satellite_lag_times_minutes']: 1-D numpy array of lag times for
        satellite-based predictors.
    option_dict['ships_lag_times_hours']: 1-D numpy array of lag times for SHIPS
        predictors.
    option_dict['satellite_predictor_names']: 1-D list with names of satellite-
        based predictors to use.
    option_dict['ships_predictor_names_lagged']: 1-D list with names of lagged
        SHIPS predictors to use.
    option_dict['ships_predictor_names_forecast']: 1-D list with names of
        forecast SHIPS predictors to use.
    option_dict['num_positive_examples_per_batch']: Number of positive examples
        (in highest class) per batch.
    option_dict['num_negative_examples_per_batch']: Number of negative examples
        (not in highest class) per batch.
    option_dict['max_examples_per_cyclone_in_batch']: Max number of examples
        (time steps) from one cyclone in a batch.
    option_dict['class_cutoffs_m_s01']: numpy array (length K - 1) of class
        cutoffs.
    option_dict['satellite_time_tolerance_sec']: Time tolerance (seconds) for
        satellite data.  For desired time t, if no data can be found within
        'satellite_time_tolerance_sec' of t, then missing data will be
        interpolated.
    option_dict['satellite_max_missing_times']: Max number of missing times for
        satellite data.  If more times are missing for example e, then e will
        not be used for training.
    option_dict['ships_time_tolerance_sec']: Same as
        'satellite_time_tolerance_sec' but for SHIPS data.
    option_dict['ships_max_missing_times']: Same as
        'satellite_max_missing_times' but for SHIPS data.

    :return: predictor_matrices: 1-D list with the following elements.

        brightness_temp_matrix: numpy array (E x M x N x T_sat x 1) of
        brightness temperatures.

        satellite_predictor_matrix: numpy array (E x T_sat x C_sat) of
        satellite-based predictors.

        ships_predictor_matrix: numpy array (E x T_ships x C_ships) of
        SHIPS predictors.

    :return: target_array: If K > 2, this is an E-by-K numpy array of integers
        (0 or 1), indicating true classes.  If K = 2, this is a length-E numpy
        array of integers (0 or 1).
    """

    option_dict = _check_generator_args(option_dict)

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lead_time_hours = option_dict[LEAD_TIME_KEY]
    satellite_lag_times_minutes = option_dict[SATELLITE_LAG_TIMES_KEY]
    ships_lag_times_hours = option_dict[SHIPS_LAG_TIMES_KEY]
    satellite_predictor_names = option_dict[SATELLITE_PREDICTORS_KEY]
    ships_predictor_names_lagged = option_dict[SHIPS_PREDICTORS_LAGGED_KEY]
    ships_predictor_names_forecast = option_dict[SHIPS_PREDICTORS_FORECAST_KEY]
    num_positive_examples_per_batch = option_dict[NUM_POSITIVE_EXAMPLES_KEY]
    num_negative_examples_per_batch = option_dict[NUM_NEGATIVE_EXAMPLES_KEY]
    max_examples_per_cyclone_in_batch = (
        option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY]
    )
    class_cutoffs_m_s01 = option_dict[CLASS_CUTOFFS_KEY]
    satellite_time_tolerance_sec = option_dict[SATELLITE_TIME_TOLERANCE_KEY]
    satellite_max_missing_times = option_dict[SATELLITE_MAX_MISSING_TIMES_KEY]
    ships_time_tolerance_sec = option_dict[SHIPS_TIME_TOLERANCE_KEY]
    ships_max_missing_times = option_dict[SHIPS_MAX_MISSING_TIMES_KEY]

    # TODO(thunderhoser): For SHIPS predictors, all lag times and forecast times
    # are currently flattened along the channel axis.  I might change this in
    # the future to make better use of time series.

    # Do actual stuff.
    cyclone_id_strings = example_io.find_cyclones(
        directory_name=example_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [satellite_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )
    good_flags = numpy.array([y in years for y in cyclone_years], dtype=bool)
    good_indices = numpy.where(good_flags)[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    random.shuffle(cyclone_id_strings)

    example_file_names = [
        example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=True
        )
        for c in cyclone_id_strings
    ]

    file_index = 0
    num_examples_per_batch = (
        num_positive_examples_per_batch + num_negative_examples_per_batch
    )

    while True:
        predictor_matrices = None
        target_array = None
        num_examples_in_memory = 0
        num_positive_examples_in_memory = 0
        num_negative_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if file_index == len(example_file_names):
                file_index = 0

            num_positive_examples_to_read = min([
                max_examples_per_cyclone_in_batch,
                num_positive_examples_per_batch -
                num_positive_examples_in_memory
            ])
            num_negative_examples_to_read = min([
                max_examples_per_cyclone_in_batch,
                num_negative_examples_per_batch -
                num_negative_examples_in_memory
            ])
            num_examples_to_read = min([
                max_examples_per_cyclone_in_batch,
                num_examples_per_batch - num_examples_in_memory
            ])

            these_predictor_matrices, this_target_array = (
                _read_one_example_file(
                    example_file_name=example_file_names[file_index],
                    num_examples_desired=num_examples_to_read,
                    num_positive_examples_desired=num_positive_examples_to_read,
                    num_negative_examples_desired=num_negative_examples_to_read,
                    lead_time_hours=lead_time_hours,
                    satellite_lag_times_minutes=satellite_lag_times_minutes,
                    ships_lag_times_hours=ships_lag_times_hours,
                    satellite_predictor_names=satellite_predictor_names,
                    ships_predictor_names_lagged=ships_predictor_names_lagged,
                    ships_predictor_names_forecast=
                    ships_predictor_names_forecast,
                    class_cutoffs_m_s01=class_cutoffs_m_s01,
                    satellite_time_tolerance_sec=satellite_time_tolerance_sec,
                    satellite_max_missing_times=satellite_max_missing_times,
                    ships_time_tolerance_sec=ships_time_tolerance_sec,
                    ships_max_missing_times=ships_max_missing_times
                )[:2]
            )

            file_index += 1

            if predictor_matrices is None:
                predictor_matrices = copy.deepcopy(these_predictor_matrices)
                target_array = this_target_array + 0
            else:
                for j in range(len(predictor_matrices)):
                    predictor_matrices[j] = numpy.concatenate(
                        (predictor_matrices[j], these_predictor_matrices[j]),
                        axis=0
                    )

                target_array = numpy.concatenate(
                    (target_array, this_target_array), axis=0
                )

            if len(target_array.shape) == 1:
                num_positive_examples_in_memory = numpy.sum(target_array == 1)
                num_negative_examples_in_memory = numpy.sum(target_array == 0)
            else:
                num_positive_examples_in_memory = numpy.sum(target_array[:, -1])
                num_negative_examples_in_memory = numpy.sum(
                    target_array[:, :-1]
                )

            num_examples_in_memory = (
                num_positive_examples_in_memory +
                num_negative_examples_in_memory
            )

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        target_array = target_array.astype('float16')

        if len(target_array.shape) == 1:
            print((
                'Yielding {0:d} examples with {1:d} positive examples!'
            ).format(
                len(target_array), int(numpy.sum(target_array))
            ))
        else:
            print((
                'Yielding {0:d} examples with the following class distribution:'
                '\n{1:s}'
            ).format(
                target_array.shape[0], str(numpy.sum(target_array, axis=0))
            ))

        yield predictor_matrices, target_array


def train_model(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        do_early_stopping=True,
        plateau_lr_multiplier=DEFAULT_LEARNING_RATE_MULTIPLIER):
    """Trains neural net.

    :param model_object: Untrained neural net (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param output_dir_name: Path to output directory (model and training history
        will be saved here).
    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param training_option_dict: See doc for `input_generator`.  This dictionary
        will be used to generate training data.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_option_dict: See doc for `input_generator`.  For
        validation only, the following values will replace corresponding values
        in `training_option_dict`:

    validation_option_dict['example_dir_name']
    validation_option_dict['years']

    :param do_early_stopping: Boolean flag.  If True, will stop training early
        if validation loss has not improved over last several epochs (see
        constants at top of file for what exactly this means).
    :param plateau_lr_multiplier: Multiplier for learning rate.  Learning
        rate will be multiplied by this factor upon plateau in validation
        performance.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 2)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 2)
    error_checking.assert_is_boolean(do_early_stopping)

    if do_early_stopping:
        error_checking.assert_is_greater(plateau_lr_multiplier, 0.)
        error_checking.assert_is_less_than(plateau_lr_multiplier, 1.)

    training_option_dict = _check_generator_args(training_option_dict)

    validation_keys_to_keep = [
        EXAMPLE_DIRECTORY_KEY, YEARS_KEY,
        SATELLITE_TIME_TOLERANCE_KEY, SHIPS_TIME_TOLERANCE_KEY
    ]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    validation_option_dict[SATELLITE_MAX_MISSING_TIMES_KEY] = int(1e10)
    validation_option_dict[SHIPS_MAX_MISSING_TIMES_KEY] = int(1e10)
    validation_option_dict = _check_generator_args(validation_option_dict)

    model_file_name = '{0:s}/model.h5'.format(output_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1
    )
    list_of_callback_objects = [history_object, checkpoint_object]

    if do_early_stopping:
        early_stopping_object = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=LOSS_PATIENCE,
            patience=EARLY_STOPPING_PATIENCE_EPOCHS, verbose=1, mode='min'
        )
        list_of_callback_objects.append(early_stopping_object)

        plateau_object = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=plateau_lr_multiplier,
            patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
            min_delta=LOSS_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS
        )
        list_of_callback_objects.append(plateau_object)

    training_generator = input_generator(training_option_dict)
    validation_generator = input_generator(validation_option_dict)

    metafile_name = find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    _write_metafile(
        pickle_file_name=metafile_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        do_early_stopping=do_early_stopping,
        plateau_lr_multiplier=plateau_lr_multiplier
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    """

    error_checking.assert_file_exists(hdf5_file_name)
    return tf_keras.models.load_model(
        hdf5_file_name, custom_objects=METRIC_DICT
    )


def apply_model(model_object, predictor_matrices, num_examples_per_batch,
                verbose=False):
    """Applies trained neural net.

    K = number of classes

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrices: See output doc for `input_generator`.
    :param num_examples_per_batch: Batch size.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: forecast_prob_array: If K > 2, this is an E-by-K numpy array of
        class probabilities.  If K = 2, this is a length-E numpy array of
        positive-class probabilities.
    """

    for this_matrix in predictor_matrices:
        error_checking.assert_is_numpy_array_without_nan(this_matrix)

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    num_examples = predictor_matrices[0].shape[0]
    num_examples_per_batch = min([num_examples_per_batch, num_examples])

    error_checking.assert_is_boolean(verbose)

    forecast_prob_array = None

    for i in range(0, num_examples, num_examples_per_batch):
        first_index = i
        last_index = min([
            i + num_examples_per_batch - 1, num_examples - 1
        ])
        these_indices = numpy.linspace(
            first_index, last_index,
            num=last_index - first_index + 1, dtype=int
        )

        if verbose:
            print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                first_index + 1, last_index + 1, num_examples
            ))

        this_prob_array = model_object.predict(
            [a[these_indices, ...] for a in predictor_matrices],
            batch_size=len(these_indices)
        )

        if forecast_prob_array is None:
            dimensions = (num_examples,) + this_prob_array.shape[1:]
            forecast_prob_array = numpy.full(dimensions, numpy.nan)

        forecast_prob_array[these_indices, ...] = this_prob_array

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    return forecast_prob_array
