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

import grids
import time_conversion
import file_system_utils
import error_checking
import data_augmentation
import custom_metrics
import example_io
import ships_io
import example_utils
import satellite_utils
import custom_losses

TOLERANCE = 1e-6
TIME_FORMAT_FOR_LOG = '%Y-%m-%d-%H%M'
MISSING_INDEX = int(1e12)

SHIPS_PREDICTORS_SANS_USABLE_FORECAST = [
    ships_io.THRESHOLD_EXCEEDANCE_KEY,
    ships_io.STORM_INTENSITY_KEY,
    ships_io.INTENSITY_KEY,
    ships_io.MINIMUM_SLP_KEY,
    ships_io.STORM_TYPE_KEY,
    ships_io.INTENSITY_CHANGE_M12HOURS_KEY,
    ships_io.INTENSITY_CHANGE_6HOURS_KEY,
    ships_io.VORTICITY_850MB_BIG_RING_KEY,
    ships_io.DIVERGENCE_200MB_BIG_RING_KEY,
    ships_io.SURFACE_PRESSURE_EDGE_KEY,
    ships_io.DIVERGENCE_200MB_CENTERED_BIG_RING_KEY,
    ships_io.SURFACE_PRESSURE_OUTER_RING_KEY,
    ships_io.SRH_1000TO700MB_OUTER_RING_KEY,
    ships_io.SRH_1000TO500MB_OUTER_RING_KEY
]

MINUTES_TO_SECONDS = 60
HOURS_TO_SECONDS = 3600
KT_TO_METRES_PER_SECOND = 1.852 / 3.6
MIN_TROP_STORM_INTENSITY_M_S01 = 34 * KT_TO_METRES_PER_SECOND

DUMMY_LATITUDES_DEG_N = numpy.linspace(50, 60, num=480, dtype=float)
DUMMY_LONGITUDES_DEG_E = numpy.linspace(-120, -110, num=640, dtype=float)

DUMMY_LATITUDE_MATRIX_DEG_N, DUMMY_LONGITUDE_MATRIX_DEG_E = (
    grids.latlng_vectors_to_matrices(
        unique_latitudes_deg=DUMMY_LATITUDES_DEG_N,
        unique_longitudes_deg=DUMMY_LONGITUDES_DEG_E
    )
)

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

SATELLITE_ROWS_KEY = 'satellite_rows_by_example'
SHIPS_GOES_ROWS_KEY = 'ships_goes_rows_by_example'
SHIPS_FORECAST_ROWS_KEY = 'ships_forecast_row_by_example'

PREDICTOR_MATRICES_KEY = 'predictor_matrices'
TARGET_MATRIX_KEY = 'target_class_matrix'
INIT_TIMES_KEY = 'init_times_unix_sec'
STORM_LATITUDES_KEY = 'storm_latitudes_deg_n'
STORM_LONGITUDES_KEY = 'storm_longitudes_deg_e'
STORM_INTENSITY_CHANGES_KEY = 'storm_intensity_changes_m_s01'
GRID_LATITUDE_MATRIX_KEY = 'grid_latitude_matrix_deg_n'
GRID_LONGITUDE_MATRIX_KEY = 'grid_longitude_matrix_deg_e'

PLATEAU_PATIENCE_EPOCHS = 10
DEFAULT_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 50
LOSS_PATIENCE = 0.

EXAMPLE_FILE_KEY = 'example_file_name'
EXAMPLE_DIRECTORY_KEY = 'example_dir_name'
YEARS_KEY = 'years'
LEAD_TIMES_KEY = 'lead_times_hours'
SATELLITE_PREDICTORS_KEY = 'satellite_predictor_names'
SATELLITE_LAG_TIMES_KEY = 'satellite_lag_times_minutes'
SHIPS_GOES_PREDICTORS_KEY = 'ships_goes_predictor_names'
SHIPS_GOES_LAG_TIMES_KEY = 'ships_goes_lag_times_hours'
SHIPS_FORECAST_PREDICTORS_KEY = 'ships_forecast_predictor_names'
SHIPS_MAX_FORECAST_HOUR_KEY = 'ships_max_forecast_hour'
NUM_POSITIVE_EXAMPLES_KEY = 'num_positive_examples_per_batch'
NUM_NEGATIVE_EXAMPLES_KEY = 'num_negative_examples_per_batch'
MAX_EXAMPLES_PER_CYCLONE_KEY = 'max_examples_per_cyclone_in_batch'
PREDICT_TD_TO_TS_KEY = 'predict_td_to_ts'
CLASS_CUTOFFS_KEY = 'class_cutoffs_m_s01'
NUM_GRID_ROWS_KEY = 'num_grid_rows'
NUM_GRID_COLUMNS_KEY = 'num_grid_columns'
USE_TIME_DIFFS_KEY = 'use_time_diffs_gridded_sat'
SATELLITE_TIME_TOLERANCE_KEY = 'satellite_time_tolerance_sec'
SATELLITE_MAX_MISSING_TIMES_KEY = 'satellite_max_missing_times'
SHIPS_TIME_TOLERANCE_KEY = 'ships_time_tolerance_sec'
SHIPS_MAX_MISSING_TIMES_KEY = 'ships_max_missing_times'
USE_CLIMO_KEY = 'use_climo_as_backup'
DATA_AUG_NUM_TRANS_KEY = 'data_aug_num_translations'
DATA_AUG_MAX_TRANS_KEY = 'data_aug_max_translation_px'
DATA_AUG_NUM_ROTATIONS_KEY = 'data_aug_num_rotations'
DATA_AUG_MAX_ROTATION_KEY = 'data_aug_max_rotation_deg'
DATA_AUG_NUM_NOISINGS_KEY = 'data_aug_num_noisings'
DATA_AUG_NOISE_STDEV_KEY = 'data_aug_noise_stdev'
WEST_PACIFIC_WEIGHT_KEY = 'west_pacific_weight'

ALL_SATELLITE_PREDICTOR_NAMES = (
    set(satellite_utils.FIELD_NAMES) -
    set(example_utils.SATELLITE_METADATA_KEYS)
)
# ALL_SATELLITE_PREDICTOR_NAMES.remove(
#     satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
# )
ALL_SATELLITE_PREDICTOR_NAMES = list(ALL_SATELLITE_PREDICTOR_NAMES)
DEFAULT_SATELLITE_PREDICTOR_NAMES = copy.deepcopy(ALL_SATELLITE_PREDICTOR_NAMES)

ALL_SHIPS_GOES_PREDICTOR_NAMES = (
    set(ships_io.SATELLITE_FIELD_NAMES) -
    set(example_utils.SHIPS_METADATA_KEYS)
)
ALL_SHIPS_GOES_PREDICTOR_NAMES.discard(ships_io.VALID_TIME_KEY)
ALL_SHIPS_GOES_PREDICTOR_NAMES.discard(ships_io.SATELLITE_LAG_TIME_KEY)
ALL_SHIPS_GOES_PREDICTOR_NAMES = list(ALL_SHIPS_GOES_PREDICTOR_NAMES)
DEFAULT_SHIPS_GOES_PREDICTOR_NAMES = copy.deepcopy(
    ALL_SHIPS_GOES_PREDICTOR_NAMES
)

ALL_SHIPS_FORECAST_PREDICTOR_NAMES = (
    set(ships_io.FORECAST_FIELD_NAMES) -
    set(example_utils.SHIPS_METADATA_KEYS)
)
ALL_SHIPS_FORECAST_PREDICTOR_NAMES = (
    ALL_SHIPS_FORECAST_PREDICTOR_NAMES |
    set(example_utils.SHIPS_METADATA_AND_FORECAST_KEYS)
)
ALL_SHIPS_FORECAST_PREDICTOR_NAMES.discard(ships_io.VALID_TIME_KEY)
ALL_SHIPS_FORECAST_PREDICTOR_NAMES.discard(ships_io.V_MOTION_KEY)
ALL_SHIPS_FORECAST_PREDICTOR_NAMES.discard(ships_io.V_MOTION_1000TO100MB_KEY)
ALL_SHIPS_FORECAST_PREDICTOR_NAMES.discard(ships_io.V_MOTION_OPTIMAL_KEY)

ALL_SHIPS_FORECAST_PREDICTOR_NAMES = list(ALL_SHIPS_FORECAST_PREDICTOR_NAMES)
DEFAULT_SHIPS_FORECAST_PREDICTOR_NAMES = copy.deepcopy(
    ALL_SHIPS_FORECAST_PREDICTOR_NAMES
)

DEFAULT_GENERATOR_OPTION_DICT = {
    LEAD_TIMES_KEY: numpy.array([24], dtype=int),
    SATELLITE_PREDICTORS_KEY: DEFAULT_SATELLITE_PREDICTOR_NAMES,
    SHIPS_GOES_PREDICTORS_KEY: DEFAULT_SHIPS_GOES_PREDICTOR_NAMES,
    SHIPS_FORECAST_PREDICTORS_KEY: DEFAULT_SHIPS_FORECAST_PREDICTOR_NAMES,
    SHIPS_MAX_FORECAST_HOUR_KEY: 0,
    NUM_POSITIVE_EXAMPLES_KEY: 8,
    NUM_NEGATIVE_EXAMPLES_KEY: 24,
    MAX_EXAMPLES_PER_CYCLONE_KEY: 6,
    CLASS_CUTOFFS_KEY: numpy.array([30 * KT_TO_METRES_PER_SECOND]),
    NUM_GRID_ROWS_KEY: None,
    NUM_GRID_COLUMNS_KEY: None,
    USE_TIME_DIFFS_KEY: False,
    PREDICT_TD_TO_TS_KEY: False,
    DATA_AUG_NUM_TRANS_KEY: 0,
    DATA_AUG_MAX_TRANS_KEY: None,
    DATA_AUG_NUM_ROTATIONS_KEY: 0,
    DATA_AUG_MAX_ROTATION_KEY: None,
    DATA_AUG_NUM_NOISINGS_KEY: 0,
    DATA_AUG_NOISE_STDEV_KEY: None,
    WEST_PACIFIC_WEIGHT_KEY: None
}

NUM_EPOCHS_KEY = 'num_epochs'
USE_CRPS_LOSS_KEY = 'use_crps_loss'
QUANTILE_LEVELS_KEY = 'quantile_levels'
CENTRAL_LOSS_WEIGHT_KEY = 'central_loss_function_weight'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
TRAINING_OPTIONS_KEY = 'training_option_dict'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_OPTIONS_KEY = 'validation_option_dict'
EARLY_STOPPING_KEY = 'do_early_stopping'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_lr_multiplier'
BNN_ARCHITECTURE_KEY = 'bnn_architecture_dict'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, USE_CRPS_LOSS_KEY, QUANTILE_LEVELS_KEY,
    CENTRAL_LOSS_WEIGHT_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY,
    EARLY_STOPPING_KEY, PLATEAU_LR_MUTIPLIER_KEY, BNN_ARCHITECTURE_KEY
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
        number of missing times <= M, output array will contain `MISSING_INDEX`
        for each missing time.  If number of missing times > M, output will be
        None.
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

            desired_indices.append(MISSING_INDEX)
            continue

        desired_indices.append(min_index)

    desired_indices = numpy.array(desired_indices, dtype=int)
    num_missing_times = numpy.sum(desired_indices == MISSING_INDEX)
    num_found_times = numpy.sum(desired_indices != MISSING_INDEX)

    if num_found_times == 0:
        # desired_time_strings = [
        #     time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG)
        #     for t in desired_times_unix_sec
        # ]
        #
        # warning_string = (
        #     'POTENTIAL ERROR: No times found.  Could not find time within {0:d}'
        #     ' seconds of any desired time:\n{1:s}'
        # ).format(tolerance_sec, str(desired_time_strings))
        #
        # warnings.warn(warning_string)

        return None

    if num_missing_times > max_num_missing_times:
        # bad_times_unix_sec = (
        #     desired_times_unix_sec[desired_indices == MISSING_INDEX]
        # )
        # bad_time_strings = [
        #     time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG)
        #     for t in bad_times_unix_sec
        # ]
        #
        # warning_string = (
        #     'POTENTIAL ERROR: Too many missing times.  Could not find time '
        #     'within {0:d} seconds of any of the following:\n{1:s}'
        # ).format(tolerance_sec, str(bad_time_strings))
        #
        # warnings.warn(warning_string)

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


def _find_all_desired_times(
        example_table_xarray, init_time_unix_sec, lead_times_sec,
        satellite_lag_times_sec, ships_goes_lag_times_sec, predict_td_to_ts,
        satellite_time_tolerance_sec, satellite_max_missing_times,
        ships_time_tolerance_sec, ships_max_missing_times,
        use_climo_as_backup, class_cutoffs_m_s01=None):
    """Finds all desired times (for predictors & target) at one fcst-init time.

    T = number of lag times for satellite data
    U = number of lag times for SHIPS GOES data
    L = number of lead times
    K = number of classes

    :param example_table_xarray: xarray table returned by
        `example_io.read_file`.
    :param init_time_unix_sec: Desired times will be found for this
        forecast-initialization time.
    :param lead_times_sec: See doc for `_read_one_example_file`.
    :param satellite_lag_times_sec: Same.
    :param ships_goes_lag_times_sec: Same.
    :param predict_td_to_ts: Same.
    :param satellite_time_tolerance_sec: Same.
    :param satellite_max_missing_times: Same.
    :param ships_time_tolerance_sec: Same.
    :param ships_max_missing_times: Same.
    :param use_climo_as_backup: Same.
    :param class_cutoffs_m_s01: Same.
    :return: satellite_indices: length-T numpy array of indices for satellite-
        based predictors.  These are indices into the satellite times in
        `example_table_xarray`.
    :return: ships_goes_indices: length-U numpy array of indices for SHIPS GOES
        predictors.  These are indices into the SHIPS times in
        `example_table_xarray`.
    :return: ships_forecast_index: Single index for SHIPS forecast predictors.
        This is an index into the SHIPS times in `example_table_xarray`.
    :return: target_class_matrix: L-by-K numpy array of integers in 0...1.
    :return: intensity_change_m_s01: Actual intensity change corresponding to
        target class.  If `predict_td_to_ts == True`, this is None.
    """

    xt = example_table_xarray

    if satellite_lag_times_sec is None:
        satellite_indices = None
    else:
        satellite_indices = _find_desired_times(
            all_times_unix_sec=
            xt.coords[example_utils.SATELLITE_TIME_DIM].values,
            desired_times_unix_sec=init_time_unix_sec - satellite_lag_times_sec,
            tolerance_sec=satellite_time_tolerance_sec,
            max_num_missing_times=satellite_max_missing_times
        )

    if (
            satellite_indices is None
            and satellite_lag_times_sec is not None
            and not use_climo_as_backup
    ):
        desired_time_strings = [
            time_conversion.unix_sec_to_string(
                init_time_unix_sec - t, TIME_FORMAT_FOR_LOG
            )
            for t in satellite_lag_times_sec
        ]

        warning_string = (
            'POTENTIAL ERROR: No satellite predictors found within {0:d} '
            'seconds of any desired time:\n{1:s}'
        ).format(satellite_time_tolerance_sec, str(desired_time_strings))

        warnings.warn(warning_string)
        return None, None, None, None, None

    if ships_goes_lag_times_sec is None:
        ships_goes_indices = None
    else:
        ships_goes_indices = _find_desired_times(
            all_times_unix_sec=
            xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values,
            desired_times_unix_sec=
            init_time_unix_sec - ships_goes_lag_times_sec,
            tolerance_sec=ships_time_tolerance_sec,
            max_num_missing_times=ships_max_missing_times
        )

    if (
            ships_goes_indices is None
            and ships_goes_lag_times_sec is not None
            and not use_climo_as_backup
    ):
        desired_time_strings = [
            time_conversion.unix_sec_to_string(
                init_time_unix_sec - t, TIME_FORMAT_FOR_LOG
            )
            for t in ships_goes_lag_times_sec
        ]

        warning_string = (
            'POTENTIAL ERROR: No SHIPS GOES predictors found within {0:d} '
            'seconds of any desired time:\n{1:s}'
        ).format(ships_time_tolerance_sec, str(desired_time_strings))

        warnings.warn(warning_string)

        return None, None, None, None, None

    if example_utils.SHIPS_METADATA_TIME_DIM in xt.coords:
        all_metadata_times_unix_sec = (
            xt.coords[example_utils.SHIPS_METADATA_TIME_DIM].values
        )
    else:
        all_metadata_times_unix_sec = (
            xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values
        )

    ships_forecast_index = numpy.where(
        all_metadata_times_unix_sec == init_time_unix_sec
    )[0][0]

    if predict_td_to_ts:
        init_time_index = numpy.where(
            all_metadata_times_unix_sec == init_time_unix_sec
        )[0][0]

        init_intensity_m_s01 = (
            xt[example_utils.STORM_INTENSITY_KEY].values[init_time_index]
        )
        if init_intensity_m_s01 >= MIN_TROP_STORM_INTENSITY_M_S01:
            return None, None, None, None, None

        zero_hour_index = numpy.where(
            xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values == 0
        )[0][0]
        init_storm_type_enum = (
            xt[ships_io.STORM_TYPE_KEY].values[init_time_index, zero_hour_index]
        )
        if init_storm_type_enum != 1:
            return None, None, None, None, None

        num_lead_times = len(lead_times_sec)
        target_class_matrix = numpy.full((num_lead_times, 2), 0, dtype=bool)

        for k in range(num_lead_times):
            num_desired_times = 1 + int(numpy.round(
                float(lead_times_sec[k]) / (6 * HOURS_TO_SECONDS)
            ))
            desired_times_unix_sec = numpy.linspace(
                init_time_unix_sec, init_time_unix_sec + lead_times_sec[k],
                num=num_desired_times, dtype=int
            )

            target_indices = _find_desired_times(
                all_times_unix_sec=all_metadata_times_unix_sec,
                desired_times_unix_sec=desired_times_unix_sec,
                tolerance_sec=0, max_num_missing_times=int(1e10)
            )
            target_indices = target_indices[target_indices != MISSING_INDEX]

            intensities_m_s01 = (
                xt[example_utils.STORM_INTENSITY_KEY].values[target_indices]
            )
            storm_type_enums = xt[ships_io.STORM_TYPE_KEY].values[
                target_indices, zero_hour_index
            ]

            target_class_matrix[k, 1] = numpy.any(numpy.logical_and(
                intensities_m_s01 >= MIN_TROP_STORM_INTENSITY_M_S01,
                storm_type_enums == 1
            ))
            target_class_matrix[k, 0] = numpy.invert(target_class_matrix[k, 1])

        return (
            satellite_indices, ships_goes_indices, ships_forecast_index,
            target_class_matrix.astype(int), None
        )

    lead_time_sec = lead_times_sec[0]

    num_desired_times = 1 + int(numpy.round(
        float(lead_time_sec) / (6 * HOURS_TO_SECONDS)
    ))
    desired_times_unix_sec = numpy.linspace(
        init_time_unix_sec, init_time_unix_sec + lead_time_sec,
        num=num_desired_times, dtype=int
    )
    target_indices = _find_desired_times(
        all_times_unix_sec=all_metadata_times_unix_sec,
        desired_times_unix_sec=desired_times_unix_sec,
        tolerance_sec=0,
        max_num_missing_times=numpy.sum(
            desired_times_unix_sec > numpy.max(all_metadata_times_unix_sec)
        )
    )

    if target_indices is None:
        valid_time_strings = [
            time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG)
            for t in desired_times_unix_sec
        ]

        warning_string = (
            'POTENTIAL ERROR: Cannot find intensity for at least one of '
            'the following times:\n{0:s}'
        ).format(str(valid_time_strings))

        warnings.warn(warning_string)
        return None, None, None, None, None

    target_indices = target_indices[target_indices != MISSING_INDEX]
    intensities_m_s01 = (
        xt[example_utils.STORM_INTENSITY_KEY].values[target_indices]
    )
    intensity_change_m_s01 = numpy.max(
        intensities_m_s01 - intensities_m_s01[0]
    )
    target_flags = _discretize_intensity_change(
        intensity_change_m_s01=intensity_change_m_s01,
        class_cutoffs_m_s01=class_cutoffs_m_s01
    )
    target_class_matrix = numpy.expand_dims(target_flags, axis=0)

    return (
        satellite_indices, ships_goes_indices, ships_forecast_index,
        target_class_matrix, intensity_change_m_s01
    )


def _read_non_predictors_one_file(
        example_table_xarray, num_examples_desired,
        num_positive_examples_desired, num_negative_examples_desired,
        lead_times_sec, satellite_lag_times_sec, ships_goes_lag_times_sec,
        predict_td_to_ts, satellite_time_tolerance_sec,
        satellite_max_missing_times, ships_time_tolerance_sec,
        ships_max_missing_times, use_climo_as_backup, all_init_times_unix_sec,
        class_cutoffs_m_s01=None):
    """Reads all but predictors from one example file.

    E = number of examples

    :param example_table_xarray: xarray table returned by
        `example_io.read_file`.
    :param num_examples_desired: See doc for `_read_one_example_file`.
    :param num_positive_examples_desired: Same.
    :param num_negative_examples_desired: Same.
    :param lead_times_sec: Same.
    :param satellite_lag_times_sec: Same.
    :param ships_goes_lag_times_sec: Same.
    :param predict_td_to_ts: Same.
    :param satellite_time_tolerance_sec: Same.
    :param satellite_max_missing_times: Same.
    :param ships_time_tolerance_sec: Same.
    :param ships_max_missing_times: Same.

    :param use_climo_as_backup: Same.
    :param all_init_times_unix_sec: Same.
    :param class_cutoffs_m_s01: Same.

    :return: data_dict: Dictionary with the following keys.
    data_dict['satellite_rows_by_example']: length-E list, where each element is
        either None or a 1-D numpy array of indices to satellite times needed
        for the given example.  These are row indices into
        `example_table_xarray`.
    data_dict['ships_goes_rows_by_example']: Same but for SHIPS GOES data.
    data_dict['ships_forecast_row_by_example']: length-E numpy array, where the
        [i]th element indexes the SHIPS init time needed for the [i]th example.
    data_dict['target_class_matrix']: See doc for `_read_one_example_file`.
    data_dict['init_times_unix_sec']: Same.
    data_dict['storm_latitudes_deg_n']: Same.
    data_dict['storm_longitudes_deg_e']: Same.
    data_dict['storm_intensity_changes_m_s01']: length-E numpy array of
        intensity changes corresponding to target classes.  If
        `predict_td_to_ts == True`, this is None.
    """

    xt = example_table_xarray
    if all_init_times_unix_sec is None:
        all_init_times_unix_sec = (
            xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values
        )

    numpy.random.shuffle(all_init_times_unix_sec)

    num_positive_examples_found = 0
    num_negative_examples_found = 0

    satellite_rows_by_example = []
    ships_goes_rows_by_example = []
    ships_forecast_row_by_example = []
    init_times_unix_sec = []
    storm_latitudes_deg_n = []
    storm_longitudes_deg_e = []
    storm_intensity_changes_m_s01 = []
    target_class_matrix = None

    for t in all_init_times_unix_sec:
        (
            these_satellite_indices,
            these_ships_goes_indices,
            this_ships_forecast_index,
            this_target_class_matrix,
            this_intensity_change_m_s01
        ) = _find_all_desired_times(
            example_table_xarray=xt, init_time_unix_sec=t,
            lead_times_sec=lead_times_sec,
            satellite_lag_times_sec=satellite_lag_times_sec,
            ships_goes_lag_times_sec=ships_goes_lag_times_sec,
            predict_td_to_ts=predict_td_to_ts,
            satellite_time_tolerance_sec=satellite_time_tolerance_sec,
            satellite_max_missing_times=satellite_max_missing_times,
            ships_time_tolerance_sec=ships_time_tolerance_sec,
            ships_max_missing_times=ships_max_missing_times,
            use_climo_as_backup=use_climo_as_backup,
            class_cutoffs_m_s01=class_cutoffs_m_s01
        )

        if this_target_class_matrix is None:
            continue

        if (
                these_satellite_indices is None
                and satellite_lag_times_sec is not None
                and not use_climo_as_backup
        ):
            continue

        if (
                these_ships_goes_indices is None
                and ships_goes_lag_times_sec is not None
                and not use_climo_as_backup
        ):
            continue

        if numpy.any(this_target_class_matrix[:, -1] == 1):
            if num_positive_examples_found >= num_positive_examples_desired:
                continue

            num_positive_examples_found += 1

        if not numpy.any(this_target_class_matrix[:, -1] == 1):
            if num_negative_examples_found >= num_negative_examples_desired:
                continue

            num_negative_examples_found += 1

        this_target_class_matrix = numpy.expand_dims(
            this_target_class_matrix, axis=0
        )

        if target_class_matrix is None:
            target_class_matrix = this_target_class_matrix + 0
        else:
            target_class_matrix = numpy.concatenate(
                (target_class_matrix, this_target_class_matrix), axis=0
            )

        satellite_rows_by_example.append(these_satellite_indices)
        ships_goes_rows_by_example.append(these_ships_goes_indices)
        ships_forecast_row_by_example.append(this_ships_forecast_index)
        init_times_unix_sec.append(t)

        if example_utils.SHIPS_METADATA_TIME_DIM in xt.coords:
            this_index = numpy.where(
                xt.coords[example_utils.SHIPS_METADATA_TIME_DIM].values == t
            )[0][0]
        else:
            this_index = numpy.where(
                xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values == t
            )[0][0]

        storm_latitudes_deg_n.append(
            xt[ships_io.STORM_LATITUDE_KEY].values[this_index]
        )
        storm_longitudes_deg_e.append(
            xt[ships_io.STORM_LONGITUDE_KEY].values[this_index]
        )
        storm_intensity_changes_m_s01.append(this_intensity_change_m_s01)

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
    storm_latitudes_deg_n = numpy.array(storm_latitudes_deg_n)
    storm_longitudes_deg_e = numpy.array(storm_longitudes_deg_e)

    if predict_td_to_ts:
        storm_intensity_changes_m_s01 = None
    else:
        storm_intensity_changes_m_s01 = numpy.array(
            storm_intensity_changes_m_s01
        )

    return {
        SATELLITE_ROWS_KEY: satellite_rows_by_example,
        SHIPS_GOES_ROWS_KEY: ships_goes_rows_by_example,
        SHIPS_FORECAST_ROWS_KEY:
            numpy.array(ships_forecast_row_by_example, dtype=int),
        TARGET_MATRIX_KEY: target_class_matrix,
        INIT_TIMES_KEY: init_times_unix_sec,
        STORM_LATITUDES_KEY: storm_latitudes_deg_n,
        STORM_LONGITUDES_KEY: storm_longitudes_deg_e,
        STORM_INTENSITY_CHANGES_KEY: storm_intensity_changes_m_s01
    }


def _read_brightness_temp_one_file(
        example_table_xarray, table_rows_by_example, lag_times_sec,
        num_grid_rows=None, num_grid_columns=None):
    """Reads brightness-temperature grids from one example file.

    E = number of examples

    :param example_table_xarray: xarray table returned by
        `example_io.read_file`.
    :param table_rows_by_example: length-E list, where each element is either
        None or a 1-D numpy array of indices to satellite times needed for the
        given example.  These are row indices into `example_table_xarray`.
    :param lag_times_sec: 1-D numpy array of lag times for model.
    :param num_grid_rows: Number of rows to keep in grid.  If None, will keep
        all rows.
    :param num_grid_columns: Same but for columns.
    :return: brightness_temp_matrix: See output doc for
        `_read_one_example_file`.
    :return: grid_latitude_matrix_deg_n: Same.
    :return: grid_longitude_matrix_deg_e: Same.
    """

    xt = example_table_xarray

    num_examples = len(table_rows_by_example)
    num_lag_times = len(lag_times_sec)
    num_grid_rows_orig = (
        xt[example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY].values.shape[1]
    )
    num_grid_columns_orig = (
        xt[example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY].values.shape[2]
    )

    these_dim = (
        num_examples, num_grid_rows_orig, num_grid_columns_orig,
        num_lag_times, 1
    )
    brightness_temp_matrix = numpy.full(these_dim, numpy.nan)

    regular_grids = len(xt[satellite_utils.GRID_LATITUDE_KEY].values.shape) == 2

    if regular_grids:
        grid_latitude_matrix_deg_n = numpy.full(
            (num_examples, num_grid_rows_orig, num_lag_times), numpy.nan
        )
        grid_longitude_matrix_deg_e = numpy.full(
            (num_examples, num_grid_columns_orig, num_lag_times), numpy.nan
        )
    else:
        dimensions = (
            num_examples, num_grid_rows_orig, num_grid_columns_orig,
            num_lag_times
        )
        grid_latitude_matrix_deg_n = numpy.full(dimensions, numpy.nan)
        grid_longitude_matrix_deg_e = numpy.full(dimensions, numpy.nan)

    for i in range(num_examples):
        for j in range(len(lag_times_sec)):
            if table_rows_by_example[i] is None:
                brightness_temp_matrix[i, ..., j, 0] = 0.

                if regular_grids:
                    grid_latitude_matrix_deg_n[i, :, j] = (
                        DUMMY_LATITUDES_DEG_N[:num_grid_rows_orig]
                    )
                    grid_longitude_matrix_deg_e[i, :, j] = (
                        DUMMY_LONGITUDES_DEG_E[:num_grid_columns_orig]
                    )
                else:
                    grid_latitude_matrix_deg_n[i, ..., j] = (
                        DUMMY_LATITUDE_MATRIX_DEG_N[
                            :num_grid_rows_orig, :num_grid_columns_orig
                        ]
                    )
                    grid_longitude_matrix_deg_e[i, ..., j] = (
                        DUMMY_LONGITUDE_MATRIX_DEG_E[
                            :num_grid_rows_orig, :num_grid_columns_orig
                        ]
                    )

                continue

            k = table_rows_by_example[i][j]

            if k == MISSING_INDEX:
                if regular_grids:
                    grid_latitude_matrix_deg_n[i, :, j] = (
                        DUMMY_LATITUDES_DEG_N[:num_grid_rows_orig]
                    )
                    grid_longitude_matrix_deg_e[i, :, j] = (
                        DUMMY_LONGITUDES_DEG_E[:num_grid_columns_orig]
                    )
                else:
                    grid_latitude_matrix_deg_n[i, ..., j] = (
                        DUMMY_LATITUDE_MATRIX_DEG_N[
                            :num_grid_rows_orig, :num_grid_columns_orig
                        ]
                    )
                    grid_longitude_matrix_deg_e[i, ..., j] = (
                        DUMMY_LONGITUDE_MATRIX_DEG_E[
                            :num_grid_rows_orig, :num_grid_columns_orig
                        ]
                    )

                continue

            try:
                these_latitudes_deg_n = (
                    xt[satellite_utils.GRID_LATITUDE_KEY].values[k, ...]
                )
                these_longitudes_deg_e = (
                    xt[satellite_utils.GRID_LONGITUDE_KEY].values[k, ...]
                )

                if regular_grids:
                    assert satellite_utils.is_regular_grid_valid(
                        latitudes_deg_n=these_latitudes_deg_n,
                        longitudes_deg_e=these_longitudes_deg_e
                    )[0]
            except:
                if regular_grids:
                    these_latitudes_deg_n = (
                        0. + DUMMY_LATITUDES_DEG_N[:num_grid_rows_orig]
                    )
                    these_longitudes_deg_e = (
                        0. + DUMMY_LONGITUDES_DEG_E[:num_grid_columns_orig]
                    )
                else:
                    these_latitudes_deg_n = 0. + DUMMY_LATITUDE_MATRIX_DEG_N[
                        :num_grid_rows_orig, :num_grid_columns_orig
                    ]
                    these_longitudes_deg_e = 0. + DUMMY_LONGITUDE_MATRIX_DEG_E[
                        :num_grid_rows_orig, num_grid_columns_orig
                    ]

            grid_latitude_matrix_deg_n[i, ..., j] = these_latitudes_deg_n
            grid_longitude_matrix_deg_e[i, ..., j] = these_longitudes_deg_e

            brightness_temp_matrix[i, ..., j, 0] = xt[
                example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY
            ].values[k, ..., 0]

    if num_grid_rows is not None:
        error_checking.assert_is_less_than(num_grid_rows, num_grid_rows_orig)

        first_index = int(numpy.round(
            num_grid_rows_orig / 2 - num_grid_rows / 2
        ))
        last_index = int(numpy.round(
            num_grid_rows_orig / 2 + num_grid_rows / 2
        ))

        brightness_temp_matrix = (
            brightness_temp_matrix[:, first_index:last_index, ...]
        )
        grid_latitude_matrix_deg_n = (
            grid_latitude_matrix_deg_n[:, first_index:last_index, ...]
        )
        grid_longitude_matrix_deg_e = (
            grid_longitude_matrix_deg_e[:, first_index:last_index, ...]
        )

    if num_grid_columns is not None:
        error_checking.assert_is_less_than(
            num_grid_columns, num_grid_columns_orig
        )

        first_index = int(numpy.round(
            num_grid_columns_orig / 2 - num_grid_columns / 2
        ))
        last_index = int(numpy.round(
            num_grid_columns_orig / 2 + num_grid_columns / 2
        ))

        brightness_temp_matrix = (
            brightness_temp_matrix[:, :, first_index:last_index, ...]
        )
        grid_latitude_matrix_deg_n = (
            grid_latitude_matrix_deg_n[:, :, first_index:last_index, ...]
        )
        grid_longitude_matrix_deg_e = (
            grid_longitude_matrix_deg_e[:, :, first_index:last_index, ...]
        )

    brightness_temp_matrix = _interp_missing_times(
        data_matrix=brightness_temp_matrix, times_sec=-1 * lag_times_sec
    )

    return (
        brightness_temp_matrix, grid_latitude_matrix_deg_n,
        grid_longitude_matrix_deg_e
    )


def _read_scalar_satellite_one_file(
        example_table_xarray, table_rows_by_example, lag_times_sec,
        predictor_names):
    """Reads scalar satellite predictors from one example file.

    :param example_table_xarray: See doc for `_read_brightness_temp_one_file`.
    :param table_rows_by_example: Same.
    :param lag_times_sec: Same.
    :param predictor_names: 1-D list of predictors to read.
    :return: satellite_predictor_matrix: See output doc for
        `_read_one_example_file`.
    """

    num_examples = len(table_rows_by_example)
    num_lag_times = len(lag_times_sec)
    num_channels = len(predictor_names)
    predictor_matrix = numpy.full(
        (num_examples, num_lag_times, num_channels), numpy.nan
    )

    xt = example_table_xarray

    all_predictor_names = (
        xt.coords[
            example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
        ].values.tolist()
    )
    predictor_indices = numpy.array([
        all_predictor_names.index(n) for n in predictor_names
    ], dtype=int)

    for i in range(num_examples):
        for j in range(num_lag_times):
            if table_rows_by_example[i] is None:
                predictor_matrix[i, j, :] = 0.
                continue

            k = table_rows_by_example[i][j]
            if k == MISSING_INDEX:
                continue

            predictor_matrix[i, j, :] = xt[
                example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY
            ].values[k, predictor_indices]

    return _interp_missing_times(
        data_matrix=predictor_matrix, times_sec=-1 * lag_times_sec
    )


def _read_ships_one_file(
        example_table_xarray, goes_predictor_names, goes_lag_times_sec,
        goes_rows_by_example, forecast_predictor_names, max_forecast_hour,
        forecast_row_by_example):
    """Reads SHIPS predictors from one example file.

    E = number of examples

    :param example_table_xarray: See doc for `_read_brightness_temp_one_file`.
    :param goes_predictor_names: 1-D list of GOES predictors to read.
    :param goes_lag_times_sec: 1-D numpy array of lag times for GOES predictors.
    :param goes_rows_by_example: length-E list, where the [i]th element is a
        1-D numpy array of table rows needed for the [i]th example.
    :param forecast_predictor_names: 1-D list of forecast predictors to read.
    :param max_forecast_hour: Maximum forecast hour to read.
    :param forecast_row_by_example: length-E numpy array of row indices.
    :return: ships_predictor_matrix: See output doc for
        `_read_one_example_file`.
    """

    if goes_predictor_names is None:
        num_goes_predictors = 0
    else:
        num_goes_predictors = len(goes_predictor_names)

    if forecast_predictor_names is None:
        num_forecast_predictors = 0
    else:
        num_forecast_predictors = len(forecast_predictor_names)

    if goes_lag_times_sec is None:
        num_goes_lag_times = 0
    else:
        num_goes_lag_times = len(goes_lag_times_sec)

    num_examples = len(forecast_row_by_example)
    num_forecast_hours = int(numpy.round(max_forecast_hour / 6)) + 1

    goes_predictor_matrix = numpy.full(
        (num_examples, num_goes_lag_times, num_goes_predictors), numpy.nan
    )
    forecast_predictor_matrix = numpy.full(
        (num_examples, num_forecast_hours, num_forecast_predictors), numpy.nan
    )
    xt = example_table_xarray

    if num_goes_predictors == 0:
        predictor_indices_goes = numpy.array([], dtype=int)
    else:
        this_coord_name = example_utils.SHIPS_PREDICTOR_LAGGED_DIM
        predictor_indices_goes = numpy.array([
            xt.coords[this_coord_name].values.tolist().index(n)
            for n in goes_predictor_names
        ], dtype=int)

    if num_forecast_predictors == 0:
        predictor_indices_forecast = numpy.array([], dtype=int)
    else:
        this_coord_name = example_utils.SHIPS_PREDICTOR_FORECAST_DIM
        predictor_indices_forecast = numpy.array([
            xt.coords[this_coord_name].values.tolist().index(n)
            for n in forecast_predictor_names
        ], dtype=int)

    for i in range(num_examples):
        for j in range(num_goes_lag_times):
            if goes_rows_by_example[i] is None:
                goes_predictor_matrix[i, j, :] = 0.
                continue

            k = goes_rows_by_example[i][j]
            if k == MISSING_INDEX:
                continue

            goes_predictor_matrix[i, j, :] = _ships_goes_xarray_to_keras(
                example_table_xarray=xt,
                init_time_index=k,
                predictor_indices=predictor_indices_goes
            )

        forecast_predictor_matrix[i, ...] = _ships_forecasts_xarray_to_keras(
            example_table_xarray=xt,
            init_time_index=forecast_row_by_example[i],
            predictor_indices=predictor_indices_forecast,
            max_forecast_hour=max_forecast_hour
        )

    if num_goes_predictors > 0:
        goes_predictor_matrix = _interp_missing_times(
            data_matrix=goes_predictor_matrix, times_sec=-1 * goes_lag_times_sec
        )

    if num_forecast_predictors > 0:
        dummy_times_unix_sec = 6 * HOURS_TO_SECONDS * numpy.linspace(
            0, num_forecast_hours - 1, num=num_forecast_hours, dtype=int
        )
        forecast_predictor_matrix = _interp_missing_times(
            data_matrix=forecast_predictor_matrix,
            times_sec=dummy_times_unix_sec
        )

    return combine_ships_predictors(
        ships_goes_predictor_matrix_3d=goes_predictor_matrix,
        ships_forecast_predictor_matrix_3d=forecast_predictor_matrix
    )


def _read_one_example_file(
        example_file_name, num_examples_desired, num_positive_examples_desired,
        num_negative_examples_desired, lead_times_hours,
        satellite_predictor_names, satellite_lag_times_minutes,
        ships_goes_predictor_names, ships_goes_lag_times_hours,
        ships_forecast_predictor_names, ships_max_forecast_hour,
        predict_td_to_ts, satellite_time_tolerance_sec,
        satellite_max_missing_times, ships_time_tolerance_sec,
        ships_max_missing_times, use_climo_as_backup, class_cutoffs_m_s01=None,
        num_grid_rows=None, num_grid_columns=None, init_times_unix_sec=None):
    """Reads one example file for generator.

    E = number of examples per batch
    M = number of rows in satellite grid
    N = number of columns in satellite grid
    T_sat = number of lag times for satellite-based predictors
    C_sat = number of channels for ungridded satellite-based predictors
    C_ships = number of scalar SHIPS predictors
    K = number of classes
    L = number of lead times

    :param example_file_name: Path to input file.  Will be read by
        `example_io.read_file`.
    :param num_examples_desired: Number of total example desired.
    :param num_positive_examples_desired: Number of positive examples (in
        highest class) desired.
    :param num_negative_examples_desired: Number of negative examples (not in
        highest class) desired.
    :param lead_times_hours: See doc for `input_generator`.
    :param satellite_predictor_names: Same.
    :param satellite_lag_times_minutes: Same.
    :param ships_goes_predictor_names: Same.
    :param ships_goes_lag_times_hours: Same.
    :param ships_forecast_predictor_names: Same.
    :param ships_max_forecast_hour: Same.
    :param predict_td_to_ts: Same.
    :param satellite_time_tolerance_sec: Same.
    :param satellite_max_missing_times: Same.
    :param ships_time_tolerance_sec: Same.
    :param ships_max_missing_times: Same.
    :param use_climo_as_backup: Same.
    :param class_cutoffs_m_s01: Same.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param init_times_unix_sec: 1-D numpy array of initial times for which to
        read examples.  If None, will read all initial times in file.

    :return: data_dict: Dictionary with the following keys.
    data_dict['predictor_matrices']: 1-D list with one or more of the following
        elements.

        brightness_temp_matrix: numpy array (E x M x N x T_sat x 1) of
        brightness temperatures.

        satellite_predictor_matrix: numpy array (E x T_sat x C_sat) of
        satellite-based predictors.

        ships_predictor_matrix: numpy array (E x C_ships) of SHIPS predictors.

    data_dict['target_class_matrix']: E-by-L-by-K numpy array of integers
        (0 or 1), indicating true classes.
    data_dict['init_times_unix_sec']: length-E numpy array of forecast-
        initialization times.
    data_dict['storm_latitudes_deg_n']: length-E numpy array of storm latitudes
        (deg N).
    data_dict['storm_longitudes_deg_e']: length-E numpy array of storm
        longitudes (deg E).
    data_dict['storm_intensity_changes_m_s01']: length-E numpy array of
        intensity changes corresponding to targets.  If
        `predict_td_to_ts == True`, this is None.
    data_dict['grid_latitude_matrix_deg_n']: numpy array of grid latitudes (deg
        north).  If regular grids, this array will have dimensions
        E x M x T_sat; if irregular, will have dimensions E x M x N x T_sat.
    data_dict['grid_longitude_matrix_deg_e']: numpy array of grid longitudes
        (deg east).  If regular grids, this array will have dimensions
        E x N x T_sat; if irregular, will have dimensions E x M x N x T_sat.
    """

    if satellite_lag_times_minutes is None:
        satellite_lag_times_sec = None
    else:
        satellite_lag_times_sec = (
            satellite_lag_times_minutes * MINUTES_TO_SECONDS
        )

    if ships_goes_lag_times_hours is None:
        ships_goes_lag_times_sec = None
    else:
        ships_goes_lag_times_sec = ships_goes_lag_times_hours * HOURS_TO_SECONDS

    lead_times_sec = lead_times_hours * HOURS_TO_SECONDS

    print('Reading data from: "{0:s}"...'.format(example_file_name))
    xt = example_io.read_file(example_file_name)

    data_dict = _read_non_predictors_one_file(
        example_table_xarray=xt,
        num_examples_desired=num_examples_desired,
        num_positive_examples_desired=num_positive_examples_desired,
        num_negative_examples_desired=num_negative_examples_desired,
        lead_times_sec=lead_times_sec,
        satellite_lag_times_sec=satellite_lag_times_sec,
        ships_goes_lag_times_sec=ships_goes_lag_times_sec,
        predict_td_to_ts=predict_td_to_ts,
        satellite_time_tolerance_sec=satellite_time_tolerance_sec,
        satellite_max_missing_times=satellite_max_missing_times,
        ships_time_tolerance_sec=ships_time_tolerance_sec,
        ships_max_missing_times=ships_max_missing_times,
        use_climo_as_backup=use_climo_as_backup,
        all_init_times_unix_sec=init_times_unix_sec,
        class_cutoffs_m_s01=class_cutoffs_m_s01
    )

    satellite_rows_by_example = data_dict.pop(SATELLITE_ROWS_KEY)
    ships_goes_rows_by_example = data_dict.pop(SHIPS_GOES_ROWS_KEY)
    ships_forecast_row_by_example = data_dict.pop(SHIPS_FORECAST_ROWS_KEY)

    if (
            satellite_lag_times_sec is None or
            satellite_predictor_names is None or
            satellite_utils.BRIGHTNESS_TEMPERATURE_KEY not in
            satellite_predictor_names
    ):
        brightness_temp_matrix = None
        grid_latitude_matrix_deg_n = None
        grid_longitude_matrix_deg_e = None
    else:
        (
            brightness_temp_matrix,
            grid_latitude_matrix_deg_n,
            grid_longitude_matrix_deg_e
        ) = _read_brightness_temp_one_file(
            example_table_xarray=xt,
            table_rows_by_example=satellite_rows_by_example,
            lag_times_sec=satellite_lag_times_sec,
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns
        )

    if satellite_predictor_names is None:
        scalar_predictor_names = None
    else:
        scalar_predictor_names = [
            n for n in satellite_predictor_names
            if n != satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
        ]

        if len(scalar_predictor_names) == 0:
            scalar_predictor_names = None

    if scalar_predictor_names is None:
        satellite_predictor_matrix = None
    else:
        satellite_predictor_matrix = _read_scalar_satellite_one_file(
            example_table_xarray=xt,
            table_rows_by_example=satellite_rows_by_example,
            lag_times_sec=satellite_lag_times_sec,
            predictor_names=scalar_predictor_names
        )

    if (
            ships_goes_predictor_names is None and
            ships_forecast_predictor_names is None
    ):
        ships_predictor_matrix = None
    else:
        ships_predictor_matrix = _read_ships_one_file(
            example_table_xarray=xt,
            goes_predictor_names=ships_goes_predictor_names,
            goes_lag_times_sec=ships_goes_lag_times_sec,
            goes_rows_by_example=ships_goes_rows_by_example,
            forecast_predictor_names=ships_forecast_predictor_names,
            max_forecast_hour=ships_max_forecast_hour,
            forecast_row_by_example=ships_forecast_row_by_example
        )

    predictor_matrices = [
        brightness_temp_matrix, satellite_predictor_matrix,
        ships_predictor_matrix
    ]

    data_dict.update({
        PREDICTOR_MATRICES_KEY: predictor_matrices,
        GRID_LATITUDE_MATRIX_KEY: grid_latitude_matrix_deg_n,
        GRID_LONGITUDE_MATRIX_KEY: grid_longitude_matrix_deg_e
    })

    return data_dict


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

    error_checking.assert_is_integer_numpy_array(option_dict[LEAD_TIMES_KEY])
    error_checking.assert_is_greater_numpy_array(option_dict[LEAD_TIMES_KEY], 0)
    assert numpy.all(numpy.mod(option_dict[LEAD_TIMES_KEY], 6) == 0)

    if option_dict[SATELLITE_LAG_TIMES_KEY] is None:
        option_dict[SATELLITE_PREDICTORS_KEY] = None
    else:
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

    if option_dict[SHIPS_GOES_PREDICTORS_KEY] is None:
        option_dict[SHIPS_GOES_LAG_TIMES_KEY] = None

    assert not (
        option_dict[SATELLITE_LAG_TIMES_KEY] is None
        and option_dict[SHIPS_GOES_PREDICTORS_KEY] is None
        and option_dict[SHIPS_FORECAST_PREDICTORS_KEY] is None
    )

    if option_dict[SHIPS_GOES_LAG_TIMES_KEY] is None:
        option_dict[SHIPS_GOES_PREDICTORS_KEY] = None
    else:
        error_checking.assert_is_integer_numpy_array(
            option_dict[SHIPS_GOES_LAG_TIMES_KEY]
        )
        error_checking.assert_is_geq_numpy_array(
            option_dict[SHIPS_GOES_LAG_TIMES_KEY], 0
        )
        assert numpy.all(
            numpy.mod(option_dict[SHIPS_GOES_LAG_TIMES_KEY], 6) == 0
        )

        error_checking.assert_is_numpy_array(
            option_dict[SHIPS_GOES_LAG_TIMES_KEY], num_dimensions=1
        )
        option_dict[SHIPS_GOES_LAG_TIMES_KEY] = numpy.sort(
            option_dict[SHIPS_GOES_LAG_TIMES_KEY]
        )[::-1]

    if option_dict[SATELLITE_PREDICTORS_KEY] is not None:
        error_checking.assert_is_string_list(
            option_dict[SATELLITE_PREDICTORS_KEY]
        )

    if option_dict[SHIPS_GOES_PREDICTORS_KEY] is not None:
        error_checking.assert_is_string_list(
            option_dict[SHIPS_GOES_PREDICTORS_KEY]
        )

    if option_dict[SHIPS_FORECAST_PREDICTORS_KEY] is None:
        option_dict[SHIPS_MAX_FORECAST_HOUR_KEY] = 0
    else:
        error_checking.assert_is_string_list(
            option_dict[SHIPS_FORECAST_PREDICTORS_KEY]
        )

        error_checking.assert_is_integer(
            option_dict[SHIPS_MAX_FORECAST_HOUR_KEY]
        )
        error_checking.assert_is_geq(
            option_dict[SHIPS_MAX_FORECAST_HOUR_KEY], 0
        )
        assert numpy.mod(option_dict[SHIPS_MAX_FORECAST_HOUR_KEY], 6) == 0

    error_checking.assert_is_integer(option_dict[NUM_POSITIVE_EXAMPLES_KEY])
    error_checking.assert_is_geq(option_dict[NUM_POSITIVE_EXAMPLES_KEY], 1)
    error_checking.assert_is_integer(option_dict[NUM_NEGATIVE_EXAMPLES_KEY])
    error_checking.assert_is_geq(option_dict[NUM_NEGATIVE_EXAMPLES_KEY], 1)
    error_checking.assert_is_integer(option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY])
    error_checking.assert_is_geq(option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY], 1)
    error_checking.assert_is_greater(
        option_dict[NUM_POSITIVE_EXAMPLES_KEY] +
        option_dict[NUM_NEGATIVE_EXAMPLES_KEY],
        option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY]
    )

    error_checking.assert_is_boolean(option_dict[PREDICT_TD_TO_TS_KEY])

    if option_dict[PREDICT_TD_TO_TS_KEY]:
        option_dict[CLASS_CUTOFFS_KEY] = None
    else:
        assert len(option_dict[LEAD_TIMES_KEY]) == 1

        error_checking.assert_is_numpy_array(
            option_dict[CLASS_CUTOFFS_KEY], num_dimensions=1
        )
        error_checking.assert_is_greater_numpy_array(
            numpy.diff(option_dict[CLASS_CUTOFFS_KEY]), 0.
        )
        assert numpy.all(numpy.isfinite(option_dict[CLASS_CUTOFFS_KEY]))

    if option_dict[NUM_GRID_ROWS_KEY] is not None:
        error_checking.assert_is_integer(option_dict[NUM_GRID_ROWS_KEY])
        error_checking.assert_is_greater(option_dict[NUM_GRID_ROWS_KEY], 0)
        assert numpy.mod(option_dict[NUM_GRID_ROWS_KEY], 2) == 0

    if option_dict[NUM_GRID_COLUMNS_KEY] is not None:
        error_checking.assert_is_integer(option_dict[NUM_GRID_COLUMNS_KEY])
        error_checking.assert_is_greater(option_dict[NUM_GRID_COLUMNS_KEY], 0)
        assert numpy.mod(option_dict[NUM_GRID_COLUMNS_KEY], 2) == 0

    error_checking.assert_is_boolean(option_dict[USE_TIME_DIFFS_KEY])
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

    if option_dict[DATA_AUG_NUM_TRANS_KEY] > 0:
        error_checking.assert_is_integer(option_dict[DATA_AUG_MAX_TRANS_KEY])
        error_checking.assert_is_greater(option_dict[DATA_AUG_MAX_TRANS_KEY], 0)
    else:
        option_dict[DATA_AUG_NUM_TRANS_KEY] = 0
        option_dict[DATA_AUG_MAX_TRANS_KEY] = None

    if option_dict[DATA_AUG_NUM_ROTATIONS_KEY] > 0:
        error_checking.assert_is_integer(
            option_dict[DATA_AUG_NUM_ROTATIONS_KEY]
        )
        error_checking.assert_is_greater(
            option_dict[DATA_AUG_MAX_ROTATION_KEY], 0.
        )
        error_checking.assert_is_leq(
            option_dict[DATA_AUG_MAX_ROTATION_KEY], 180.
        )
    else:
        option_dict[DATA_AUG_NUM_ROTATIONS_KEY] = 0
        option_dict[DATA_AUG_MAX_ROTATION_KEY] = None

    if option_dict[DATA_AUG_NUM_NOISINGS_KEY] > 0:
        error_checking.assert_is_integer(option_dict[DATA_AUG_NUM_NOISINGS_KEY])
        error_checking.assert_is_greater(
            option_dict[DATA_AUG_NOISE_STDEV_KEY], 0.
        )
    else:
        option_dict[DATA_AUG_NUM_NOISINGS_KEY] = 0
        option_dict[DATA_AUG_NOISE_STDEV_KEY] = None

    if option_dict[WEST_PACIFIC_WEIGHT_KEY] is not None:
        error_checking.assert_is_greater(
            option_dict[WEST_PACIFIC_WEIGHT_KEY], 1.
        )

    return option_dict


def _ships_goes_xarray_to_keras(
        example_table_xarray, init_time_index, predictor_indices):
    """Converts SHIPS GOES predictors from xarray format to Keras format.

    :param example_table_xarray: xarray table returned by
        `example_io.read_file`.
    :param init_time_index: Will extract predictors from the [i]th SHIPS
        initialization time, where i = `init_time_index`.
    :param predictor_indices: 1-D numpy array with indices of predictor
        variables to use.
    :return: predictor_values_1d: 1-D numpy array with desired SHIPS GOES
        predictors.
    """

    error_checking.assert_is_integer(init_time_index)
    error_checking.assert_is_geq(init_time_index, 0)
    error_checking.assert_is_integer_numpy_array(predictor_indices)
    error_checking.assert_is_geq_numpy_array(predictor_indices, 0)

    predictor_values = example_table_xarray[
        example_utils.SHIPS_PREDICTORS_LAGGED_KEY
    ].values[init_time_index, ...]

    predictor_values = predictor_values[:, predictor_indices]

    merged_lag_times_index = numpy.where(numpy.isnan(
        example_table_xarray.coords[example_utils.SHIPS_LAG_TIME_DIM].values
    ))[0][0]

    return predictor_values[merged_lag_times_index, :]


def _ships_forecasts_xarray_to_keras(
        example_table_xarray, init_time_index, predictor_indices,
        max_forecast_hour, test_mode=False):
    """Converts SHIPS forecast predictors from xarray format to Keras format.

    H = number of forecast hours
    P = number of predictor variables

    :param example_table_xarray: xarray table returned by
        `example_io.read_file`.
    :param init_time_index: Will extract predictors from the [i]th SHIPS
        initialization time, where i = `init_time_index`.
    :param predictor_indices: 1-D numpy array with indices of predictor
        variables to use.
    :param max_forecast_hour: Max forecast hour.
    :param test_mode: Leave this alone.
    :return: predictor_matrix: H-by-P numpy array of desired values.
    """

    error_checking.assert_is_integer(init_time_index)
    error_checking.assert_is_geq(init_time_index, 0)
    error_checking.assert_is_integer_numpy_array(predictor_indices)
    error_checking.assert_is_geq_numpy_array(predictor_indices, 0)
    error_checking.assert_is_integer(max_forecast_hour)
    error_checking.assert_is_geq(max_forecast_hour, 0)

    xt = example_table_xarray

    first_index = numpy.where(
        xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values == 0
    )[0][0]

    last_index = 1 + numpy.where(
        xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values ==
        max_forecast_hour
    )[0][0]

    predictor_matrix = xt[example_utils.SHIPS_PREDICTORS_FORECAST_KEY].values[
        init_time_index, ...
    ]
    predictor_matrix = (
        predictor_matrix[first_index:last_index, predictor_indices]
    )

    if test_mode:
        return predictor_matrix

    for i, j in enumerate(predictor_indices):
        this_predictor_name = xt.coords[
            example_utils.SHIPS_PREDICTOR_FORECAST_DIM
        ].values[j]

        if (
                this_predictor_name not in SHIPS_PREDICTORS_SANS_USABLE_FORECAST
        ):
            continue

        print((
            'Cannot use true forecast values for SHIPS predictor "{0:s}".'
        ).format(
            this_predictor_name
        ))

        predictor_matrix[:, i] = predictor_matrix[0, i]

    return predictor_matrix


def _write_metafile(
        pickle_file_name, num_epochs, use_crps_loss, quantile_levels,
        central_loss_function_weight, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, do_early_stopping, plateau_lr_multiplier,
        bnn_architecture_dict):
    """Writes metadata to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: See doc for `train_model`.
    :param use_crps_loss: Same.
    :param quantile_levels: Same.
    :param central_loss_function_weight: Same.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param do_early_stopping: Same.
    :param plateau_lr_multiplier: Same.
    :param bnn_architecture_dict: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        USE_CRPS_LOSS_KEY: use_crps_loss,
        QUANTILE_LEVELS_KEY: quantile_levels,
        CENTRAL_LOSS_WEIGHT_KEY: central_loss_function_weight,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        EARLY_STOPPING_KEY: do_early_stopping,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_lr_multiplier,
        BNN_ARCHITECTURE_KEY: bnn_architecture_dict
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def _create_multiply_function(real_number):
    """Creates function that multiplies input by real number.

    :param real_number: Multiplier.
    :return: multiply_function: Function handle.
    """

    def multiply_function(input_array):
        """Multiplies input array by real number.

        :param input_array: numpy array.
        :return: output_array: numpy array.
        """

        return input_array * real_number

    return multiply_function


def _multiply_a_function(orig_function_handle, real_number):
    """Multiplies function by a real number.

    :param orig_function_handle: Handle for function to be multiplied.
    :param real_number: Real number.
    :return: new_function_handle: Handle for new function, which is the original
        function multiplied by the given number.
    """

    this_function_handle = _create_multiply_function(real_number)
    return lambda x, y: this_function_handle(orig_function_handle(x, y))


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
        metafile_name = metafile_name.replace(
            '/home/ralager/condo/swatwork/ralager', ''
        )

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name
        )
        raise ValueError(error_string)

    return metafile_name


def _augment_data(
        predictor_matrices, target_array, num_translations, max_translation_px,
        num_rotations, max_rotation_deg, num_noisings, noise_stdev):
    """Performs data augmentation.

    This method applies each augmentation separately.  For example, a satellite
    image can be rotated *or* translated *or* noised, but not a combination of
    the three.

    :param predictor_matrices: See doc for `input_generator`.
    :param target_array: numpy array of target values, where the first axis is
        the example axis.
    :param num_translations: Number of translations per example.  This argument
        applies only to satellite images.
    :param max_translation_px: Max translation (pixels).  This argument applies
        only to satellite images.
    :param num_rotations: Number of rotations per example.  This argument
        applies only to satellite images.
    :param max_rotation_deg: Max absolute rotation angle (degrees).  This
        argument applies only to satellite images.
    :param num_noisings: Number of noisings per example.  This argument applies
        to all predictor types.
    :param noise_stdev: Standard deviation of Gaussian noise.  This argument
        applies to all predictor types.
    :return: predictor_matrices: Same as input but with more examples.  Also,
        the new examples will be slightly different than the original examples.
    :return: target_array: Same as input but with more examples.  The new
        examples will be just copies of the old examples (i.e., data
        augmentation changes only the predictors, not the targets).
    """

    if len(predictor_matrices[0].shape) < 4:
        return predictor_matrices, target_array

    orig_num_examples = predictor_matrices[0].shape[0]
    num_matrices = len(predictor_matrices)

    if num_translations > 0:
        x_offsets_px, y_offsets_px = data_augmentation.get_translations(
            num_translations=num_translations,
            max_translation_pixels=max_translation_px,
            num_grid_rows=predictor_matrices[0].shape[1],
            num_grid_columns=predictor_matrices[0].shape[2]
        )

        print('Applying {0:d} translations for DATA AUGMENTATION...'.format(
            num_translations
        ))
    else:
        x_offsets_px = numpy.array([])
        y_offsets_px = numpy.array([])

    for k in range(num_translations):
        this_matrix = data_augmentation.shift_radar_images(
            radar_image_matrix=predictor_matrices[0][:orig_num_examples, ...],
            x_offset_pixels=x_offsets_px[k],
            y_offset_pixels=y_offsets_px[k]
        )

        predictor_matrices[0] = numpy.concatenate(
            (predictor_matrices[0], this_matrix), axis=0
        )
        target_array = numpy.concatenate(
            (target_array, target_array[:orig_num_examples, ...]), axis=0
        )

        for j in range(1, num_matrices):
            predictor_matrices[j] = numpy.concatenate((
                predictor_matrices[j],
                predictor_matrices[j][:orig_num_examples, ...]
            ), axis=0)

    if num_rotations > 0:
        rotation_angles_deg = data_augmentation.get_rotations(
            num_rotations=num_rotations,
            max_absolute_rotation_angle_deg=max_rotation_deg
        )

        print('Applying {0:d} rotations for DATA AUGMENTATION...'.format(
            num_rotations
        ))
    else:
        rotation_angles_deg = numpy.array([])

    for k in range(num_rotations):
        this_matrix = data_augmentation.rotate_radar_images(
            radar_image_matrix=predictor_matrices[0][:orig_num_examples, ...],
            ccw_rotation_angle_deg=rotation_angles_deg[k]
        )

        predictor_matrices[0] = numpy.concatenate(
            (predictor_matrices[0], this_matrix), axis=0
        )
        target_array = numpy.concatenate(
            (target_array, target_array[:orig_num_examples, ...]), axis=0
        )

        for j in range(1, num_matrices):
            predictor_matrices[j] = numpy.concatenate((
                predictor_matrices[j],
                predictor_matrices[j][:orig_num_examples, ...]
            ), axis=0)

    if num_noisings > 0:
        print('Applying {0:d} noisings for DATA AUGMENTATION...'.format(
            num_noisings
        ))

    for k in range(num_noisings):
        for j in range(num_matrices):
            this_matrix = numpy.random.normal(
                loc=0., scale=noise_stdev,
                size=predictor_matrices[j][:orig_num_examples, ...].shape
            )
            this_matrix = (
                this_matrix + predictor_matrices[j][:orig_num_examples, ...]
            )
            predictor_matrices[j] = numpy.concatenate(
                (predictor_matrices[j], this_matrix), axis=0
            )

        target_array = numpy.concatenate(
            (target_array, target_array[:orig_num_examples, ...]), axis=0
        )

    return predictor_matrices, target_array


def _apply_model_td_to_ts(
        model_object, predictor_matrices, num_examples_per_batch,
        use_dropout, verbose):
    """Applies trained model (inference mode) for TD-to-TS prediction.

    :param model_object: See doc for `apply_model`.
    :param predictor_matrices: Same.
    :param num_examples_per_batch: Same.
    :param use_dropout: Same.
    :param verbose: Same.
    :return: forecast_prob_matrix: See doc for `apply_model`.
    """

    num_examples = predictor_matrices[0].shape[0]
    forecast_prob_matrix = None

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

        if use_dropout:
            this_prob_matrix = model_object(
                [a[these_indices, ...] for a in predictor_matrices],
                training=True
            ).numpy()
        else:
            this_prob_matrix = model_object.predict_on_batch(
                [a[these_indices, ...] for a in predictor_matrices]
            )

        this_prob_matrix = numpy.expand_dims(this_prob_matrix, axis=-3)
        this_prob_matrix = numpy.concatenate(
            (1. - this_prob_matrix, this_prob_matrix), axis=-3
        )

        if forecast_prob_matrix is None:
            dimensions = (num_examples,) + this_prob_matrix.shape[1:]
            forecast_prob_matrix = numpy.full(dimensions, numpy.nan)

        forecast_prob_matrix[these_indices, ...] = this_prob_matrix

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    return forecast_prob_matrix


def _apply_model_ri(
        model_object, predictor_matrices, num_examples_per_batch,
        use_dropout, verbose):
    """Applies trained model (inference mode) for rapid intensification.

    :param model_object: See doc for `apply_model`.
    :param predictor_matrices: Same.
    :param num_examples_per_batch: Same.
    :param use_dropout: Same.
    :param verbose: Same.
    :return: forecast_prob_matrix: See doc for `apply_model`.
    """

    num_examples = predictor_matrices[0].shape[0]
    forecast_prob_matrix = None

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

        if use_dropout:
            these_predictions = model_object(
                [a[these_indices, ...] for a in predictor_matrices],
                training=True
            ).numpy()
        else:
            these_predictions = model_object.predict_on_batch(
                [a[these_indices, ...] for a in predictor_matrices]
            )

        if isinstance(these_predictions, list):

            # Current shape is E x S or E x K x S.
            this_prob_matrix = numpy.stack(these_predictions, axis=-1)

            # If necessary, add class axis to get shape E x K x S.
            if len(this_prob_matrix.shape) == 2:
                this_prob_matrix = numpy.expand_dims(this_prob_matrix, axis=-2)
                this_prob_matrix = numpy.concatenate(
                    (1. - this_prob_matrix, this_prob_matrix), axis=-2
                )

            # Add lead-time axis to get shape E x K x L x S.
            this_prob_matrix = numpy.expand_dims(this_prob_matrix, axis=-2)
        else:

            # Current shape is E x L x S or E x K x L x S.
            this_prob_matrix = these_predictions + 0.

            # Make sure that shape is E x K x L x S.
            if len(this_prob_matrix.shape) == 3:
                this_prob_matrix = numpy.expand_dims(this_prob_matrix, axis=-3)
                this_prob_matrix = numpy.concatenate(
                    (1. - this_prob_matrix, this_prob_matrix), axis=-3
                )

        if forecast_prob_matrix is None:
            dimensions = (num_examples,) + this_prob_matrix.shape[1:]
            forecast_prob_matrix = numpy.full(dimensions, numpy.nan)

        forecast_prob_matrix[these_indices, ...] = this_prob_matrix

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    return forecast_prob_matrix


def read_metafile(pickle_file_name):
    """Reads metadata for neural net from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['num_epochs']: See doc for `train_model`.
    metadata_dict['quantile_levels']: Same.
    metadata_dict['use_crps_loss']: Same.
    metadata_dict['central_loss_function_weight']: Same.
    metadata_dict['num_training_batches_per_epoch']: Same.
    metadata_dict['training_option_dict']: Same.
    metadata_dict['num_validation_batches_per_epoch']: Same.
    metadata_dict['validation_option_dict']: Same.
    metadata_dict['do_early_stopping']: Same.
    metadata_dict['plateau_lr_multiplier']: Same.
    metadata_dict['bnn_architecture_dict']: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def combine_ships_predictors(ships_goes_predictor_matrix_3d,
                             ships_forecast_predictor_matrix_3d):
    """Combines SHIPS predictors into one matrix.

    E = number of examples
    T_lag = number of lag times
    T_fcst = number of forecast hours
    P_goes = number of GOES-based predictor variables
    P_fcst = number of forecast predictor variables
    P = total number of predictors = T_lag * P_goes + T_fcst * P_fcst

    :param ships_goes_predictor_matrix_3d: numpy array (E x T_lag x P_goes) of
        GOES-based predictors.
    :param ships_forecast_predictor_matrix_3d: numpy array (E x T_fcst x P_fcst)
        of forecast predictors.
    :return: ships_predictor_matrix_2d: E-by-P numpy array with all predictors.
    """

    error_checking.assert_is_numpy_array(
        ships_goes_predictor_matrix_3d, num_dimensions=3
    )
    error_checking.assert_is_numpy_array_without_nan(
        ships_goes_predictor_matrix_3d
    )

    error_checking.assert_is_numpy_array(
        ships_forecast_predictor_matrix_3d, num_dimensions=3
    )
    error_checking.assert_is_numpy_array_without_nan(
        ships_forecast_predictor_matrix_3d
    )

    error_checking.assert_equals(
        ships_goes_predictor_matrix_3d.shape[0],
        ships_forecast_predictor_matrix_3d.shape[0]
    )

    num_examples = ships_goes_predictor_matrix_3d.shape[0]
    this_second_dim = (
        ships_goes_predictor_matrix_3d.shape[1] *
        ships_goes_predictor_matrix_3d.shape[2]
    )
    ships_goes_predictor_matrix_2d = numpy.reshape(
        ships_goes_predictor_matrix_3d, (num_examples, this_second_dim)
    )

    this_second_dim = (
        ships_forecast_predictor_matrix_3d.shape[1] *
        ships_forecast_predictor_matrix_3d.shape[2]
    )
    ships_forecast_predictor_matrix_2d = numpy.reshape(
        ships_forecast_predictor_matrix_3d, (num_examples, this_second_dim)
    )

    return numpy.concatenate(
        (ships_goes_predictor_matrix_2d, ships_forecast_predictor_matrix_2d),
        axis=1
    )


def separate_ships_predictors(ships_predictor_matrix_2d, num_goes_predictors,
                              num_forecast_predictors, num_forecast_hours):
    """Separates SHIPS predictors into two matrices.

    This method is the inverse of `combine_ships_predictors`.

    :param ships_predictor_matrix_2d: See doc for `combine_ships_predictors`.
    :param num_goes_predictors: Number of GOES-based predictor variables.
    :param num_forecast_predictors: Number of forecast predictor variables.
    :param num_forecast_hours: Number of forecast hours.
    :return: ships_goes_predictor_matrix_3d: See doc for
        `combine_ships_predictors`.
    :return: ships_forecast_predictor_matrix_3d: Same.
    """

    error_checking.assert_is_numpy_array(
        ships_predictor_matrix_2d, num_dimensions=2
    )
    error_checking.assert_is_numpy_array_without_nan(ships_predictor_matrix_2d)

    error_checking.assert_is_integer(num_goes_predictors)
    error_checking.assert_is_geq(num_goes_predictors, 0)
    error_checking.assert_is_integer(num_forecast_predictors)
    error_checking.assert_is_geq(num_forecast_predictors, 0)
    error_checking.assert_is_integer(num_forecast_hours)
    error_checking.assert_is_geq(num_forecast_hours, 0)

    num_examples = ships_predictor_matrix_2d.shape[0]

    if num_goes_predictors == 0:
        ships_goes_predictor_matrix_3d = numpy.full((num_examples, 0, 0), 0.)
        num_goes_entries = 0
    else:
        num_entries = ships_predictor_matrix_2d.shape[1]
        num_forecast_entries = num_forecast_predictors * num_forecast_hours
        num_goes_entries = num_entries - num_forecast_entries
        num_goes_lag_times = int(numpy.round(
            float(num_goes_entries) / num_goes_predictors
        ))

        ships_goes_predictor_matrix_3d = numpy.reshape(
            ships_predictor_matrix_2d[:, :num_goes_entries],
            (num_examples, num_goes_lag_times, num_goes_predictors)
        )

    if num_forecast_predictors + num_forecast_hours == 0:
        ships_forecast_predictor_matrix_3d = numpy.full(
            (num_examples, 0, 0), 0.
        )
    else:
        ships_forecast_predictor_matrix_3d = numpy.reshape(
            ships_predictor_matrix_2d[:, num_goes_entries:],
            (num_examples, num_forecast_hours, num_forecast_predictors)
        )

    return ships_goes_predictor_matrix_3d, ships_forecast_predictor_matrix_3d


def create_inputs(option_dict):
    """Creates input data for neural net.

    This method is the same as `input_generator`, except that it returns all the
    data at once, rather than generating batches on the fly.

    :param option_dict: Dictionary with the following keys.
    option_dict['example_file_name']: Path to example file (will be read by
        `example_io.read_file`).
    option_dict['lead_times_hours']: See doc for `input_generator`.
    option_dict['satellite_predictor_names']: Same.
    option_dict['satellite_lag_times_minutes']: Same.
    option_dict['ships_goes_predictor_names']: Same.
    option_dict['ships_goes_lag_times_hours']: Same.
    option_dict['ships_forecast_predictor_names']: Same.
    option_dict['ships_max_forecast_hour']: Same.
    option_dict['predict_td_to_ts']: Same.
    option_dict['satellite_time_tolerance_sec']: Same.
    option_dict['satellite_max_missing_times']: Same.
    option_dict['ships_time_tolerance_sec']: Same.
    option_dict['ships_max_missing_times']: Same.
    option_dict['use_climo_as_backup']: Same.
    option_dict['class_cutoffs_m_s01']: Same.
    option_dict['num_grid_rows']: Same.
    option_dict['num_grid_columns']: Same.
    option_dict['use_time_diffs_gridded_sat']: Same.

    :return: data_dict: See doc for `_read_one_example_file`.
    """

    option_dict[EXAMPLE_DIRECTORY_KEY] = 'foo'
    option_dict[YEARS_KEY] = numpy.array([1900], dtype=int)
    option_dict[NUM_POSITIVE_EXAMPLES_KEY] = 8
    option_dict[NUM_NEGATIVE_EXAMPLES_KEY] = 8
    option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY] = 4

    option_dict = _check_generator_args(option_dict)

    example_file_name = option_dict[EXAMPLE_FILE_KEY]
    lead_times_hours = option_dict[LEAD_TIMES_KEY]
    satellite_predictor_names = option_dict[SATELLITE_PREDICTORS_KEY]
    satellite_lag_times_minutes = option_dict[SATELLITE_LAG_TIMES_KEY]
    ships_goes_predictor_names = option_dict[SHIPS_GOES_PREDICTORS_KEY]
    ships_goes_lag_times_hours = option_dict[SHIPS_GOES_LAG_TIMES_KEY]
    ships_forecast_predictor_names = option_dict[SHIPS_FORECAST_PREDICTORS_KEY]
    ships_max_forecast_hour = option_dict[SHIPS_MAX_FORECAST_HOUR_KEY]
    predict_td_to_ts = option_dict[PREDICT_TD_TO_TS_KEY]
    satellite_time_tolerance_sec = option_dict[SATELLITE_TIME_TOLERANCE_KEY]
    satellite_max_missing_times = option_dict[SATELLITE_MAX_MISSING_TIMES_KEY]
    ships_time_tolerance_sec = option_dict[SHIPS_TIME_TOLERANCE_KEY]
    ships_max_missing_times = option_dict[SHIPS_MAX_MISSING_TIMES_KEY]
    use_climo_as_backup = option_dict[USE_CLIMO_KEY]
    class_cutoffs_m_s01 = option_dict[CLASS_CUTOFFS_KEY]
    num_grid_rows = option_dict[NUM_GRID_ROWS_KEY]
    num_grid_columns = option_dict[NUM_GRID_COLUMNS_KEY]
    use_time_diffs_gridded_sat = option_dict[USE_TIME_DIFFS_KEY]

    data_dict = _read_one_example_file(
        example_file_name=example_file_name,
        num_examples_desired=int(1e10),
        num_positive_examples_desired=int(1e10),
        num_negative_examples_desired=int(1e10),
        lead_times_hours=lead_times_hours,
        satellite_predictor_names=satellite_predictor_names,
        satellite_lag_times_minutes=satellite_lag_times_minutes,
        ships_goes_lag_times_hours=ships_goes_lag_times_hours,
        ships_goes_predictor_names=ships_goes_predictor_names,
        ships_forecast_predictor_names=ships_forecast_predictor_names,
        ships_max_forecast_hour=ships_max_forecast_hour,
        predict_td_to_ts=predict_td_to_ts,
        satellite_time_tolerance_sec=satellite_time_tolerance_sec,
        satellite_max_missing_times=satellite_max_missing_times,
        ships_time_tolerance_sec=ships_time_tolerance_sec,
        ships_max_missing_times=ships_max_missing_times,
        use_climo_as_backup=use_climo_as_backup,
        class_cutoffs_m_s01=class_cutoffs_m_s01,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns
    )

    data_dict[PREDICTOR_MATRICES_KEY] = [
        None if m is None else m.astype('float16')
        for m in data_dict[PREDICTOR_MATRICES_KEY]
    ]

    if (
            use_time_diffs_gridded_sat and
            satellite_predictor_names is not None and
            satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
            in satellite_predictor_names
    ):
        num_lag_times = len(satellite_lag_times_minutes)

        for j in range(num_lag_times - 1):
            data_dict[PREDICTOR_MATRICES_KEY][0][..., j, 0] = (
                data_dict[PREDICTOR_MATRICES_KEY][0][..., -1, 0] -
                data_dict[PREDICTOR_MATRICES_KEY][0][..., j, 0]
            )

    return data_dict


def input_generator(option_dict):
    """Generates input data for neural net.

    E = number of examples
    K = number of classes
    L = number of lead times

    :param option_dict: Dictionary with the following keys.
    option_dict['example_dir_name']: Name of directory with example files.
        Files therein will be found by `example_io.find_file` and read by
        `example_io.read_file`.
    option_dict['years']: 1-D numpy array of training years.
    option_dict['lead_times_hours']: Lead times for predicting storm intensity.
    option_dict['satellite_predictor_names']: 1-D list with names of scalar
        satellite predictors to use.  If you do not want scalar satellite
        predictors, make this None.
    option_dict['satellite_lag_times_minutes']: 1-D numpy array of lag times for
        satellite-based predictors.  If you do not want any satellite predictors
        (brightness-temperature grids or scalars), make this None.
    option_dict['ships_goes_predictor_names']: 1-D list with names of SHIPS GOES
        predictors to use.  If you do not want GOES SHIPS predictors, make this
        None.
    option_dict['ships_goes_lag_times_hours']: 1-D numpy array of model lag
        times for SHIPS GOES predictors.  If you do not want SHIPS GOES
        predictors, make this None.
    option_dict['ships_forecast_predictor_names']: 1-D list with names of
        forecast SHIPS predictors to use.  If you do not want forecast SHIPS
        predictors, make this None.
    option_dict['ships_max_forecast_hour']: Max forecast hour to include in
        SHIPS predictors.
    option_dict['num_positive_examples_per_batch']: Number of positive examples
        (in highest class) per batch.
    option_dict['num_negative_examples_per_batch']: Number of negative examples
        (not in highest class) per batch.
    option_dict['max_examples_per_cyclone_in_batch']: Max number of examples
        (time steps) from one cyclone in a batch.
    option_dict['predict_td_to_ts']: Boolean flag.  If True, will predict
        transition from tropical depression to tropical storm.  If False, will
        predict rapid intensification.
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
    option_dict['use_climo_as_backup']: Boolean flag.  If True, for examples
        where a certain predictor type (either satellite or SHIPS) is not found,
        will assume climatological values.  If False, for examples where a
        certain predictor type is not found, will not use this example.
    option_dict['class_cutoffs_m_s01']:
        [used only if `predict_td_to_ts == False`]
        numpy array (length K - 1) of class cutoffs.
    option_dict['num_grid_rows']: Number of rows to keep in brightness-
        temperature grid.  If you want to keep all rows, make this None.
    option_dict['num_grid_columns']: Same but for columns.
    option_dict['use_time_diffs_gridded_sat']: Boolean flag.  If True, will turn
        gridded satellite data at non-zero lag times into temporal differences.
    option_dict['data_aug_num_translations']: Number of translations per example
        for data augmentation.  You can make this 0.
    option_dict['data_aug_max_translation_px']: Max translation (pixels) for
        data augmentation.  Used only if data_aug_num_translations > 0.
    option_dict['data_aug_num_rotations']: Number of rotations per example for
        data augmentation.  You can make this 0.
    option_dict['data_aug_max_rotation_deg']: Max absolute rotation (degrees)
        for data augmentation.  Used only if data_aug_num_rotations > 0.
    option_dict['data_aug_num_noisings']: Number of noisings per example for
        data augmentation.  You can make this 0.
    option_dict['data_aug_noise_stdev']: Standard deviation of noise for data
        augmentation.  Used only if data_aug_num_noisings > 0.

    :return: predictor_matrices: See output doc for `_read_one_example_file`.
        However, for this generator, any undesired predictor type will be
        omitted from the list.  For example, if scalar satellite predictors are
        undesired, the list will contain only
        [brightness_temperature_matrix, ships_predictor_matrix].

    :return: target_array: If prediction task is TD-to-TS, this is an E-by-L
        numpy array of integers in 0...1.  Else, if task is RI with > 2 classes,
        this is an E-by-K numpy array of integers in 0...1.  Else, if task is RI
        with 2 classes, this is a length-E numpy array of integers in 0...1.
    """

    option_dict = _check_generator_args(option_dict)

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lead_times_hours = option_dict[LEAD_TIMES_KEY]
    satellite_predictor_names = option_dict[SATELLITE_PREDICTORS_KEY]
    satellite_lag_times_minutes = option_dict[SATELLITE_LAG_TIMES_KEY]
    ships_goes_predictor_names = option_dict[SHIPS_GOES_PREDICTORS_KEY]
    ships_goes_lag_times_hours = option_dict[SHIPS_GOES_LAG_TIMES_KEY]
    ships_forecast_predictor_names = option_dict[SHIPS_FORECAST_PREDICTORS_KEY]
    ships_max_forecast_hour = option_dict[SHIPS_MAX_FORECAST_HOUR_KEY]
    num_positive_examples_per_batch = option_dict[NUM_POSITIVE_EXAMPLES_KEY]
    num_negative_examples_per_batch = option_dict[NUM_NEGATIVE_EXAMPLES_KEY]
    max_examples_per_cyclone_in_batch = (
        option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY]
    )
    predict_td_to_ts = option_dict[PREDICT_TD_TO_TS_KEY]
    satellite_time_tolerance_sec = option_dict[SATELLITE_TIME_TOLERANCE_KEY]
    satellite_max_missing_times = option_dict[SATELLITE_MAX_MISSING_TIMES_KEY]
    ships_time_tolerance_sec = option_dict[SHIPS_TIME_TOLERANCE_KEY]
    ships_max_missing_times = option_dict[SHIPS_MAX_MISSING_TIMES_KEY]
    use_climo_as_backup = option_dict[USE_CLIMO_KEY]
    class_cutoffs_m_s01 = option_dict[CLASS_CUTOFFS_KEY]
    num_grid_rows = option_dict[NUM_GRID_ROWS_KEY]
    num_grid_columns = option_dict[NUM_GRID_COLUMNS_KEY]
    use_time_diffs_gridded_sat = option_dict[USE_TIME_DIFFS_KEY]
    data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    data_aug_max_translation_px = option_dict[DATA_AUG_MAX_TRANS_KEY]
    data_aug_num_rotations = option_dict[DATA_AUG_NUM_ROTATIONS_KEY]
    data_aug_max_rotation_deg = option_dict[DATA_AUG_MAX_ROTATION_KEY]
    data_aug_num_noisings = option_dict[DATA_AUG_NUM_NOISINGS_KEY]
    data_aug_noise_stdev = option_dict[DATA_AUG_NOISE_STDEV_KEY]
    west_pacific_weight = option_dict[WEST_PACIFIC_WEIGHT_KEY]

    use_data_augmentation = (
        data_aug_num_translations + data_aug_num_rotations +
        data_aug_num_noisings
        > 0
    )

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
    init_times_by_file_unix_sec = [None] * len(example_file_names)
    num_examples_per_batch = (
        num_positive_examples_per_batch + num_negative_examples_per_batch
    )

    # TODO(thunderhoser): This is a HACK to allow no-oversampling.
    num_examples_per_batch = min([num_examples_per_batch, 16])

    while True:
        predictor_matrices = None
        target_array = None
        sample_weights = None
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

            this_data_dict = _read_one_example_file(
                example_file_name=example_file_names[file_index],
                num_examples_desired=num_examples_to_read,
                num_positive_examples_desired=num_positive_examples_to_read,
                num_negative_examples_desired=num_negative_examples_to_read,
                lead_times_hours=lead_times_hours,
                satellite_predictor_names=satellite_predictor_names,
                satellite_lag_times_minutes=satellite_lag_times_minutes,
                ships_goes_predictor_names=ships_goes_predictor_names,
                ships_goes_lag_times_hours=ships_goes_lag_times_hours,
                ships_forecast_predictor_names=ships_forecast_predictor_names,
                ships_max_forecast_hour=ships_max_forecast_hour,
                predict_td_to_ts=predict_td_to_ts,
                satellite_time_tolerance_sec=satellite_time_tolerance_sec,
                satellite_max_missing_times=satellite_max_missing_times,
                ships_time_tolerance_sec=ships_time_tolerance_sec,
                ships_max_missing_times=ships_max_missing_times,
                use_climo_as_backup=use_climo_as_backup,
                class_cutoffs_m_s01=class_cutoffs_m_s01,
                num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
                init_times_unix_sec=init_times_by_file_unix_sec[file_index]
            )

            init_times_by_file_unix_sec[file_index] = (
                this_data_dict[INIT_TIMES_KEY]
            )
            this_cyclone_id_string = example_io.file_name_to_cyclone_id(
                example_file_names[file_index]
            )
            file_index += 1

            this_target_class_matrix = this_data_dict[TARGET_MATRIX_KEY]
            if this_target_class_matrix is None:
                continue

            these_predictor_matrices = [
                m for m in this_data_dict[PREDICTOR_MATRICES_KEY]
                if m is not None
            ]

            if predict_td_to_ts:
                this_target_array = this_target_class_matrix[..., -1]
            elif len(class_cutoffs_m_s01) > 1:
                this_target_array = this_target_class_matrix[:, 0, :]
            else:
                this_target_array = this_target_class_matrix[:, 0, -1]

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

            if predict_td_to_ts:
                num_positive_examples_in_memory = numpy.sum(
                    numpy.any(target_array == 1, axis=1)
                )
                num_negative_examples_in_memory = numpy.sum(
                    numpy.all(target_array == 0, axis=1)
                )
            elif len(class_cutoffs_m_s01) > 1:
                num_positive_examples_in_memory = numpy.sum(target_array[:, -1])
                num_negative_examples_in_memory = numpy.sum(
                    target_array[:, :-1]
                )
            else:
                num_positive_examples_in_memory = numpy.sum(target_array == 1)
                num_negative_examples_in_memory = numpy.sum(target_array == 0)

            num_examples_in_memory = (
                num_positive_examples_in_memory +
                num_negative_examples_in_memory
            )

            if west_pacific_weight is None:
                continue

            this_flag = (
                satellite_utils.parse_cyclone_id(this_cyclone_id_string)[1] ==
                satellite_utils.NORTHWEST_PACIFIC_ID_STRING
            )
            these_sample_weights = numpy.full(
                this_target_array.shape[0],
                west_pacific_weight if this_flag else 1.
            )

            if sample_weights is None:
                sample_weights = these_sample_weights + 0.
            else:
                sample_weights = numpy.concatenate(
                    (sample_weights, these_sample_weights), axis=0
                )

        if use_data_augmentation:
            num_examples_before_da = target_array.shape[0]

            predictor_matrices, target_array = _augment_data(
                predictor_matrices=predictor_matrices,
                target_array=target_array,
                num_translations=data_aug_num_translations,
                max_translation_px=data_aug_max_translation_px,
                num_rotations=data_aug_num_rotations,
                max_rotation_deg=data_aug_max_rotation_deg,
                num_noisings=data_aug_num_noisings,
                noise_stdev=data_aug_noise_stdev
            )

            if west_pacific_weight is not None:
                num_repeats = int(numpy.round(
                    float(target_array.shape[0]) / num_examples_before_da
                ))
                sample_weights = numpy.tile(sample_weights, reps=num_repeats)

        if (
                use_time_diffs_gridded_sat and
                satellite_predictor_names is not None and
                satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
                in satellite_predictor_names
        ):
            num_lag_times = len(satellite_lag_times_minutes)

            for j in range(num_lag_times - 1):
                predictor_matrices[0][..., j, 0] = (
                    predictor_matrices[0][..., -1, 0] -
                    predictor_matrices[0][..., j, 0]
                )

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        target_array = target_array.astype('float16')

        if predict_td_to_ts:
            print((
                'Yielding {0:d} examples with {1:d} positive examples!'
            ).format(
                target_array.shape[0],
                numpy.sum(numpy.any(target_array == 1, axis=1))
            ))
        elif len(class_cutoffs_m_s01) > 1:
            print((
                'Yielding {0:d} examples with the following class distribution:'
                '\n{1:s}'
            ).format(
                target_array.shape[0], str(numpy.sum(target_array, axis=0))
            ))
        else:
            print((
                'Yielding {0:d} examples with {1:d} positive examples!'
            ).format(
                target_array.shape[0], numpy.sum(target_array == 1)
            ))

        if west_pacific_weight is None:
            yield predictor_matrices, target_array

        yield predictor_matrices, target_array, sample_weights


def train_model(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        use_crps_loss, bnn_architecture_dict,
        do_early_stopping=True, quantile_levels=None,
        central_loss_function_weight=None,
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

    :param use_crps_loss: Boolean flag.  If True, using CRPS as a loss function.
    :param bnn_architecture_dict: Dictionary with architecture options for
        Bayesian neural network (BNN).  If the model being trained is not
        Bayesian, make this None.
    :param do_early_stopping: Boolean flag.  If True, will stop training early
        if validation loss has not improved over last several epochs (see
        constants at top of file for what exactly this means).
    :param quantile_levels: 1-D numpy array of quantile levels for quantile
        regression.  Levels must range from (0, 1).  If the model is not doing
        quantile regression, make this None.
    :param central_loss_function_weight: Weight for loss function used to
        penalize central output.  If the model is not doing quantile regression,
        make this None.
    :param plateau_lr_multiplier: Multiplier for learning rate.  Learning
        rate will be multiplied by this factor upon plateau in validation
        performance.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_boolean(use_crps_loss)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 2)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 2)
    error_checking.assert_is_boolean(do_early_stopping)

    if do_early_stopping:
        error_checking.assert_is_greater(plateau_lr_multiplier, 0.)
        error_checking.assert_is_less_than(plateau_lr_multiplier, 1.)

    if quantile_levels is not None:
        error_checking.assert_is_numpy_array(quantile_levels, num_dimensions=1)
        error_checking.assert_is_greater_numpy_array(quantile_levels, 0.)
        error_checking.assert_is_less_than_numpy_array(quantile_levels, 1.)
        error_checking.assert_is_greater_numpy_array(
            numpy.diff(quantile_levels), 0.
        )

        error_checking.assert_is_geq(central_loss_function_weight, 1.)

    training_option_dict = _check_generator_args(training_option_dict)

    validation_keys_to_keep = [
        EXAMPLE_DIRECTORY_KEY, YEARS_KEY,
        SATELLITE_TIME_TOLERANCE_KEY, SHIPS_TIME_TOLERANCE_KEY
    ]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    num_data_augmentations = (
        training_option_dict[DATA_AUG_NUM_TRANS_KEY] +
        training_option_dict[DATA_AUG_NUM_ROTATIONS_KEY] +
        training_option_dict[DATA_AUG_NUM_NOISINGS_KEY]
    )
    validation_option_dict[NUM_POSITIVE_EXAMPLES_KEY] *= (
        1 + num_data_augmentations
    )
    validation_option_dict[NUM_NEGATIVE_EXAMPLES_KEY] *= (
        1 + num_data_augmentations
    )
    validation_option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY] *= (
        1 + num_data_augmentations
    )

    validation_option_dict[SATELLITE_MAX_MISSING_TIMES_KEY] = int(1e10)
    validation_option_dict[SHIPS_MAX_MISSING_TIMES_KEY] = int(1e10)
    validation_option_dict[DATA_AUG_NUM_TRANS_KEY] = 0
    validation_option_dict[DATA_AUG_NUM_ROTATIONS_KEY] = 0
    validation_option_dict[DATA_AUG_NUM_NOISINGS_KEY] = 0

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
        use_crps_loss=use_crps_loss, quantile_levels=quantile_levels,
        central_loss_function_weight=central_loss_function_weight,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        do_early_stopping=do_early_stopping,
        plateau_lr_multiplier=plateau_lr_multiplier,
        bnn_architecture_dict=bnn_architecture_dict
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

    metafile_name = find_metafile(
        model_file_name=hdf5_file_name, raise_error_if_missing=True
    )
    metadata_dict = read_metafile(metafile_name)

    quantile_levels = metadata_dict[QUANTILE_LEVELS_KEY]
    use_crps_loss = metadata_dict[USE_CRPS_LOSS_KEY]
    bnn_architecture_dict = metadata_dict[BNN_ARCHITECTURE_KEY]

    if bnn_architecture_dict is not None:
        from ml4tc.machine_learning import cnn_architecture_bayesian

        # TODO(thunderhoser): Calling `create_crps_model_ri` will not work if
        # I introduce more methods to cnn_architecture_bayesian.py.
        model_object = cnn_architecture_bayesian.create_crps_model_ri(
            option_dict_gridded_sat=
            bnn_architecture_dict['option_dict_gridded_sat'],
            option_dict_ungridded_sat=
            bnn_architecture_dict['option_dict_ungridded_sat'],
            option_dict_ships=bnn_architecture_dict['option_dict_ships'],
            option_dict_dense=bnn_architecture_dict['option_dict_dense']
        )

        model_object.load_weights(hdf5_file_name)
        return model_object

    if quantile_levels is None and not use_crps_loss:
        return tf_keras.models.load_model(
            hdf5_file_name, custom_objects=METRIC_DICT
        )

    if use_crps_loss:
        custom_object_dict = {
            'loss': custom_losses.crps_loss()
        }
        model_object = tf_keras.models.load_model(
            hdf5_file_name, custom_objects=custom_object_dict, compile=False
        )
        model_object.compile(
            loss=custom_object_dict['loss'], optimizer=keras.optimizers.Adam(),
            metrics=[]
        )

        return model_object

    option_dict = metadata_dict[TRAINING_OPTIONS_KEY]
    predict_td_to_ts = option_dict[PREDICT_TD_TO_TS_KEY]

    if predict_td_to_ts:
        loss_function = custom_losses.quantile_loss_plus_xentropy_3d_output(
            quantile_levels=quantile_levels,
            central_loss_weight=metadata_dict[CENTRAL_LOSS_WEIGHT_KEY]
        )
        custom_object_dict = {'loss': loss_function}

        model_object = tf_keras.models.load_model(
            hdf5_file_name, custom_objects=custom_object_dict, compile=False
        )
        model_object.compile(
            loss=custom_object_dict['loss'], optimizer=keras.optimizers.Adam(),
            metrics=[]
        )

        return model_object

    central_loss_function = _multiply_a_function(
        orig_function_handle=keras.losses.binary_crossentropy,
        real_number=metadata_dict[CENTRAL_LOSS_WEIGHT_KEY]
    )

    custom_object_dict = {
        'central_output_loss': central_loss_function
    }
    loss_dict = {'central_output': central_loss_function}
    metric_list = []

    for k in range(len(quantile_levels)):
        this_loss_function = custom_losses.quantile_loss(
            quantile_level=quantile_levels[k]
        )
        loss_dict['quantile_output{0:03d}'.format(k + 1)] = (
            this_loss_function
        )
        custom_object_dict['quantile_output{0:03d}_loss'.format(k + 1)] = (
            this_loss_function
        )

    custom_object_dict['loss'] = loss_dict

    model_object = tf_keras.models.load_model(
        hdf5_file_name, custom_objects=custom_object_dict, compile=False
    )
    model_object.compile(
        loss=custom_object_dict['loss'], optimizer=keras.optimizers.Adam(),
        metrics=metric_list
    )

    return model_object


def apply_model(
        model_object, model_metadata_dict, predictor_matrices,
        num_examples_per_batch, use_dropout=False, verbose=False):
    """Applies trained neural net (inference mode).

    E = number of examples
    K = number of classes
    L = number of lead times
    S = number of prediction sets

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param model_metadata_dict: Dictionary returned by `read_metafile`.
    :param predictor_matrices: See output doc for `input_generator`.
    :param num_examples_per_batch: Batch size.
    :param use_dropout: Boolean flag.  If True, will keep dropout in all layers
        turned on.  Using dropout at inference time is called "Monte Carlo
        dropout".
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: forecast_prob_matrix: E-by-K-by-L-by-S numpy array of class
        probabilities.
    """

    for this_matrix in predictor_matrices:
        error_checking.assert_is_numpy_array_without_nan(this_matrix)

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    num_examples = predictor_matrices[0].shape[0]
    num_examples_per_batch = min([num_examples_per_batch, num_examples])

    error_checking.assert_is_boolean(use_dropout)
    error_checking.assert_is_boolean(verbose)

    if use_dropout:
        for layer_object in model_object.layers:
            if 'batch' in layer_object.name.lower():
                print('Layer "{0:s}" set to NON-TRAINABLE!'.format(
                    layer_object.name
                ))
                layer_object.trainable = False

    option_dict = model_metadata_dict[VALIDATION_OPTIONS_KEY]

    if option_dict[PREDICT_TD_TO_TS_KEY]:
        forecast_prob_matrix = _apply_model_td_to_ts(
            model_object=model_object, predictor_matrices=predictor_matrices,
            num_examples_per_batch=num_examples_per_batch,
            use_dropout=use_dropout, verbose=verbose
        )
    else:
        forecast_prob_matrix = _apply_model_ri(
            model_object=model_object, predictor_matrices=predictor_matrices,
            num_examples_per_batch=num_examples_per_batch,
            use_dropout=use_dropout, verbose=verbose
        )

    forecast_prob_matrix = numpy.maximum(forecast_prob_matrix, 0.)
    forecast_prob_matrix = numpy.minimum(forecast_prob_matrix, 1.)
    return forecast_prob_matrix
