"""Plots predictions for many storms, one map per time step."""

import copy
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import example_io
from ml4tc.io import prediction_io
from ml4tc.io import border_io
from ml4tc.utils import satellite_utils
from ml4tc.utils import normalization
from ml4tc.machine_learning import neural_net
from ml4tc.plotting import plotting_utils
from ml4tc.plotting import satellite_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H'
TIME_INTERVAL_SEC = 21600

CYCLONE_IDS_KEY = prediction_io.CYCLONE_IDS_KEY
PROBABILITY_MATRIX_KEY = prediction_io.PROBABILITY_MATRIX_KEY
TARGET_MATRIX_KEY = prediction_io.TARGET_MATRIX_KEY
LEAD_TIMES_KEY = prediction_io.LEAD_TIMES_KEY
QUANTILE_LEVELS_KEY = prediction_io.QUANTILE_LEVELS_KEY

LABEL_COLOUR = numpy.full(3, 0.)
LABEL_FONT_SIZE = 16

CYCLONE_ID_BOUNDING_BOX_DICT = {
    'alpha': 0.5,
    'edgecolor': numpy.full(3, 0.),
    'linewidth': 1,
    'facecolor': numpy.full(3, 1.)
}

PREDICTION_BOUNDING_BOX_DICT = {
    'alpha': 0.25,
    'edgecolor': numpy.full(3, 0.),
    'linewidth': 1,
    'facecolor': numpy.full(3, 1.)
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

MODEL_METAFILE_ARG_NAME = 'input_model_metafile_name'
EXAMPLE_DIR_ARG_NAME = 'input_norm_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
MIN_LATITUDE_ARG_NAME = 'min_latitude_deg_n'
MAX_LATITUDE_ARG_NAME = 'max_latitude_deg_n'
MIN_LONGITUDE_ARG_NAME = 'min_longitude_deg_e'
MAX_LONGITUDE_ARG_NAME = 'max_longitude_deg_e'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
FIRST_TIME_ARG_NAME = 'first_init_time_string'
LAST_TIME_ARG_NAME = 'last_init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_METAFILE_HELP_STRING = (
    'Path to metafile for model.  Will be read by `neural_net.read_metafile`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with normalized learning examples.  Files therein will '
    'be found by `example_io.find_file` and read by `example_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be used to denormalize '
    'brightness-temperature maps before plotting).  Will be read by '
    '`normalization.read_file`.'
)
PREDICTION_FILE_HELP_STRING = (
    'Path to file with predictions and targets.  Will be read by '
    '`prediction_io.read_file`.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level to plot.  For example, if this is 0.95, will plot 95% '
    'confidence interval of probabilities for each cyclone.  If you want to '
    'plot only the mean and not a confidence interval, leave this argument '
    'alone.'
)
MIN_LATITUDE_HELP_STRING = 'Minimum latitude (deg north) in map.'
MAX_LATITUDE_HELP_STRING = 'Max latitude (deg north) in map.'
MIN_LONGITUDE_HELP_STRING = 'Minimum longitude (deg east) in map.'
MAX_LONGITUDE_HELP_STRING = 'Max longitude (deg east) in map.'
CYCLONE_IDS_HELP_STRING = (
    'List of IDs for cyclones to plot.  If you would rather specify a start/end'
    ' time, leave this argument alone.'
)
FIRST_TIME_HELP_STRING = (
    'First initialization time to plot (format "yyyy-mm-dd-HH").'
)
LAST_TIME_HELP_STRING = (
    'Last initialization time to plot (format "yyyy-mm-dd-HH").'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_METAFILE_ARG_NAME, type=str, required=True,
    help=MODEL_METAFILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=False, default='',
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=-1,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LATITUDE_ARG_NAME, type=float, required=True,
    help=MIN_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LATITUDE_ARG_NAME, type=float, required=True,
    help=MAX_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LONGITUDE_ARG_NAME, type=float, required=True,
    help=MIN_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LONGITUDE_ARG_NAME, type=float, required=True,
    help=MAX_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_IDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=CYCLONE_IDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=False, default='',
    help=FIRST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=False, default='',
    help=LAST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _subset_data(
        data_dict, min_latitude_deg_n, max_latitude_deg_n, min_longitude_deg_e,
        max_longitude_deg_e, longitude_positive_in_west, cyclone_id_string,
        first_init_time_unix_sec, last_init_time_unix_sec):
    """Subsets data by time and location.

    E = number of examples

    :param data_dict: Dictionary returned by `neural_net.create_inputs`.
    :param min_latitude_deg_n: See documentation at top of file.
    :param max_latitude_deg_n: Same.
    :param min_longitude_deg_e: Same.
    :param max_longitude_deg_e: Same.
    :param longitude_positive_in_west: Boolean flag, indicating longitude
        format.
    :param cyclone_id_string: Cyclone ID.
    :param first_init_time_unix_sec: See documentation at top of file.
    :param last_init_time_unix_sec: Same.
    :return: data_dict: Subset version of input, containing fewer examples and
        an extra key.
    data_dict['cyclone_id_strings']: length-E list of cyclone IDs.
    """

    good_latitude_flags = numpy.logical_and(
        data_dict[neural_net.STORM_LATITUDES_KEY] >= min_latitude_deg_n,
        data_dict[neural_net.STORM_LATITUDES_KEY] <= max_latitude_deg_n
    )

    if longitude_positive_in_west:
        storm_longitudes_deg_e = (
            lng_conversion.convert_lng_positive_in_west(
                data_dict[neural_net.STORM_LONGITUDES_KEY]
            )
        )
    else:
        storm_longitudes_deg_e = (
            lng_conversion.convert_lng_negative_in_west(
                data_dict[neural_net.STORM_LONGITUDES_KEY]
            )
        )

    good_longitude_flags = numpy.logical_and(
        storm_longitudes_deg_e >= min_longitude_deg_e,
        storm_longitudes_deg_e <= max_longitude_deg_e
    )
    good_location_flags = numpy.logical_and(
        good_latitude_flags, good_longitude_flags
    )
    good_time_flags = numpy.logical_and(
        data_dict[neural_net.INIT_TIMES_KEY] >= first_init_time_unix_sec,
        data_dict[neural_net.INIT_TIMES_KEY] <= last_init_time_unix_sec
    )
    good_indices = numpy.where(
        numpy.logical_and(good_location_flags, good_time_flags)
    )[0]

    if len(good_indices) == 0:
        return None

    data_dict[neural_net.PREDICTOR_MATRICES_KEY] = [
        None if a is None else a[good_indices, ...]
        for a in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]

    for this_key in [
            neural_net.TARGET_MATRIX_KEY, neural_net.INIT_TIMES_KEY,
            neural_net.STORM_LATITUDES_KEY, neural_net.STORM_LONGITUDES_KEY,
            neural_net.GRID_LATITUDE_MATRIX_KEY,
            neural_net.GRID_LONGITUDE_MATRIX_KEY
    ]:
        data_dict[this_key] = data_dict[this_key][good_indices, ...]

    data_dict[CYCLONE_IDS_KEY] = [cyclone_id_string] * len(good_indices)

    return data_dict


def _concat_data(data_dicts):
    """Concatenates many examples into the same dictionary.

    :param data_dicts: 1-D list of dictionaries returned by
        `_subset_data`.
    :return: data_dict: Single dictionary, created by concatenating inputs.
    """

    num_matrices = len(data_dicts[0][neural_net.PREDICTOR_MATRICES_KEY])
    data_dict = {
        neural_net.PREDICTOR_MATRICES_KEY: []
    }

    for k in range(num_matrices):
        if data_dicts[0][neural_net.PREDICTOR_MATRICES_KEY][k] is None:
            data_dict[neural_net.PREDICTOR_MATRICES_KEY][k] = None
            continue

        data_dict[neural_net.PREDICTOR_MATRICES_KEY][k] = numpy.concatenate(
            [d[neural_net.PREDICTOR_MATRICES_KEY][k] for d in data_dicts],
            axis=0
        )

    for this_key in [
            neural_net.TARGET_MATRIX_KEY, neural_net.INIT_TIMES_KEY,
            neural_net.STORM_LATITUDES_KEY, neural_net.STORM_LONGITUDES_KEY,
            neural_net.GRID_LATITUDE_MATRIX_KEY,
            neural_net.GRID_LONGITUDE_MATRIX_KEY
    ]:
        data_dict[this_key] = numpy.concatenate(
            [d[this_key] for d in data_dicts], axis=0
        )

    data_dict[CYCLONE_IDS_KEY] = numpy.concatenate(
        [numpy.array(d[CYCLONE_IDS_KEY]) for d in data_dicts], axis=0
    )
    data_dict[CYCLONE_IDS_KEY] = data_dict[CYCLONE_IDS_KEY].tolist()

    return data_dict


def _match_predictors_to_predictions(data_dict, prediction_file_name):
    """Matches predictors to predictions.

    E = number of examples
    K = number of classes
    L = number of lead times
    S = number of prediction sets

    :param data_dict: Dictionary returned by `_concat_data`.
    :param prediction_file_name: See documentation at top of file.
    :return: data_dict: Same as input but with extra keys.
    data_dict['forecast_prob_matrix']: E-by-K-by-L-by-S numpy array of forecast
        event probabilities.
    data_dict['target_class_matrix']: E-by-L numpy array of target classes, all
        integers in range [0, K - 1].
    data_dict['lead_times_hours']: length-L numpy array of lead times.
    data_dict['quantile_levels']: length-(S + 1) numpy array of quantile levels.
        This may also be None.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    prediction_dict[prediction_io.CYCLONE_IDS_KEY] = numpy.array(
        prediction_dict[prediction_io.CYCLONE_IDS_KEY]
    )

    good_indices = []
    num_examples = len(data_dict[neural_net.INIT_TIMES_KEY])

    for i in range(num_examples):
        this_index = numpy.where(numpy.logical_and(
            prediction_dict[prediction_io.INIT_TIMES_KEY] ==
            data_dict[neural_net.INIT_TIMES_KEY][i],
            prediction_dict[prediction_io.CYCLONE_IDS_KEY] ==
            data_dict[CYCLONE_IDS_KEY][i]
        ))[0][0]

        good_indices.append(this_index)

    idxs = numpy.array(good_indices, dtype=int)
    data_dict.pop(neural_net.TARGET_MATRIX_KEY)

    data_dict.update({
        PROBABILITY_MATRIX_KEY:
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][idxs, ...],
        TARGET_MATRIX_KEY:
            prediction_dict[prediction_io.TARGET_MATRIX_KEY][idxs, ...],
        LEAD_TIMES_KEY: prediction_dict[prediction_io.LEAD_TIMES_KEY],
        QUANTILE_LEVELS_KEY: prediction_dict[prediction_io.QUANTILE_LEVELS_KEY]
    })

    return data_dict


def _plot_brightness_temp_one_example(
        predictor_matrices_one_example, normalization_table_xarray,
        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e, axes_object):
    """Plots brightness-temperature map at 0-minute lag time for one example.

    M = number of rows in grid
    N = number of columns in grid
    L = number of lag times
    P = number of points in border set

    :param predictor_matrices_one_example: length-3 list, where each element is
        either None or a numpy array formatted in the same way as the training
        data.  The first axis (i.e., the example axis) of each numpy array
        should have length 1.
    :param normalization_table_xarray: xarray table returned by
        `normalization.read_file`.
    :param grid_latitude_matrix_deg_n: numpy array of latitudes (deg north).  If
        regular grids, this should have dimensions M x L.  If irregular grids,
        should have dimensions M x N x L.
    :param grid_longitude_matrix_deg_e: numpy array of longitudes (deg east).
        If regular grids, this should have dimensions N x L.  If irregular
        grids, should have dimensions M x N x L.
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Denormalize brightness temperatures.
    nt = normalization_table_xarray
    predictor_names_norm = list(
        nt.coords[normalization.SATELLITE_PREDICTOR_GRIDDED_DIM].values
    )

    k = predictor_names_norm.index(satellite_utils.BRIGHTNESS_TEMPERATURE_KEY)
    training_values = (
        nt[normalization.SATELLITE_PREDICTORS_GRIDDED_KEY].values[:, k]
    )
    training_values = training_values[numpy.isfinite(training_values)]

    brightness_temp_matrix_kelvins = normalization._denorm_one_variable(
        normalized_values_new=predictor_matrices_one_example[0][0, ..., -1, 0],
        actual_values_training=training_values
    )

    # Plot brightness temperatures.
    satellite_plotting.plot_2d_grid(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        latitude_array_deg_n=grid_latitude_matrix_deg_n[..., -1],
        longitude_array_deg_e=grid_longitude_matrix_deg_e[..., -1],
        cbar_orientation_string=None
    )


def _get_prediction_string(data_dict, example_index, predict_td_to_ts,
                           confidence_level):
    """Returns string with predictions and targets.

    :param data_dict: Dictionary returned by `_match_predictors_to_predictions`.
    :param example_index: Will create string for the [i]th example, where
        i = `example_index`.
    :param predict_td_to_ts: Boolean flag.  If True (False), prediction task is
        TD-to-TS (rapid intensification).
    :param confidence_level: See documentation at top of file.
    :return: prediction_string: String with predictions and targets.
    """

    i = example_index

    dummy_prediction_dict = {
        prediction_io.PROBABILITY_MATRIX_KEY: data_dict[PROBABILITY_MATRIX_KEY],
        prediction_io.QUANTILE_LEVELS_KEY: data_dict[QUANTILE_LEVELS_KEY]
    }
    mean_forecast_probs = (
        prediction_io.get_mean_predictions(dummy_prediction_dict)[i, :]
    )

    forecast_prob_matrix = data_dict[PROBABILITY_MATRIX_KEY][i, 1, ...]
    target_classes = data_dict[TARGET_MATRIX_KEY][i, :]
    lead_times_hours = data_dict[LEAD_TIMES_KEY]
    quantile_levels = data_dict[QUANTILE_LEVELS_KEY]

    label_string = 'Storm {0:s}'.format(
        data_dict[CYCLONE_IDS_KEY][i][-2:]
    )

    for k in range(len(lead_times_hours)):
        label_string += '\n{0:s} in next {1:d} h? {2:s}; '.format(
            'TS' if predict_td_to_ts else 'RI',
            lead_times_hours[k],
            'Yes' if target_classes[k] else 'No'
        )

        label_string += r'$p$ = '
        label_string += '{0:.2f}'.format(mean_forecast_probs[k]).lstrip('0')

        if confidence_level is not None:
            if quantile_levels is None:
                min_probability = numpy.percentile(
                    forecast_prob_matrix[k, :], 50 * (1. - confidence_level)
                )
                max_probability = numpy.percentile(
                    forecast_prob_matrix[k, :], 50 * (1. + confidence_level)
                )
            else:
                interp_object = interp1d(
                    x=quantile_levels, y=forecast_prob_matrix[k, 1:],
                    kind='linear', bounds_error=False, assume_sorted=True,
                    fill_value='extrapolate'
                )

                min_probability = interp_object(0.5 * (1. - confidence_level))
                max_probability = interp_object(0.5 * (1. + confidence_level))

            min_probability = numpy.maximum(min_probability, 0.)
            min_probability = numpy.minimum(min_probability, 1.)
            max_probability = numpy.maximum(max_probability, 0.)
            max_probability = numpy.minimum(max_probability, 1.)

            min_prob_string = '{0:.2f}'.format(min_probability).lstrip('0')
            max_prob_string = '{0:.2f}'.format(max_probability).lstrip('0')

            label_string += (
                ' ({0:.1f}'.format(100 * confidence_level).rstrip('.0')
            )
            label_string += '% CI: {0:s}-{1:s})'.format(
                min_prob_string, max_prob_string
            )

    return label_string


def _get_swmost_index(data_dict):
    """Returns index of southwesternmost tropical cyclone.

    :param data_dict: Dictionary returned by `_match_predictors_to_predictions`.
    :return: swmost_index: Index of southwesternmost tropical cyclone.
    """

    return numpy.argmin(
        data_dict[neural_net.STORM_LATITUDES_KEY] +
        data_dict[neural_net.STORM_LONGITUDES_KEY]
    )


def _get_nwmost_index(data_dict):
    """Returns index of northwesternmost tropical cyclone.

    :param data_dict: Dictionary returned by `_match_predictors_to_predictions`.
    :return: nwmost_index: Index of northwesternmost tropical cyclone.
    """

    return numpy.argmax(
        data_dict[neural_net.STORM_LATITUDES_KEY] -
        data_dict[neural_net.STORM_LONGITUDES_KEY]
    )


def _get_nemost_index(data_dict):
    """Returns index of northeasternmost tropical cyclone.

    :param data_dict: Dictionary returned by `_match_predictors_to_predictions`.
    :return: nemost_index: Index of northeasternmost tropical cyclone.
    """

    return numpy.argmax(
        data_dict[neural_net.STORM_LATITUDES_KEY] +
        data_dict[neural_net.STORM_LONGITUDES_KEY]
    )


def _run(model_metafile_name, norm_example_dir_name, normalization_file_name,
         prediction_file_name, confidence_level,
         min_latitude_deg_n, max_latitude_deg_n,
         min_longitude_deg_e, max_longitude_deg_e, cyclone_id_strings,
         first_init_time_string, last_init_time_string, output_dir_name):
    """Plots predictions for many storms, one map per time step.

    This is effectively the main method.

    :param model_metafile_name: See documentation at top of file.
    :param norm_example_dir_name: Same.
    :param normalization_file_name: Same.
    :param prediction_file_name: Same.
    :param confidence_level: Same.
    :param min_latitude_deg_n: Same.
    :param max_latitude_deg_n: Same.
    :param min_longitude_deg_e: Same.
    :param max_longitude_deg_e: Same.
    :param cyclone_id_strings: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    if confidence_level <= 0:
        confidence_level = None

    if confidence_level is not None:
        error_checking.assert_is_geq(confidence_level, 0.8)
        error_checking.assert_is_less_than(confidence_level, 1.)

    error_checking.assert_is_valid_latitude(min_latitude_deg_n)
    error_checking.assert_is_valid_latitude(max_latitude_deg_n)
    error_checking.assert_is_greater(max_latitude_deg_n, min_latitude_deg_n)

    min_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg_e, allow_nan=False
    )
    max_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        max_longitude_deg_e, allow_nan=False
    )
    longitude_positive_in_west = True

    if max_longitude_deg_e <= min_longitude_deg_e:
        min_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            min_longitude_deg_e, allow_nan=False
        )
        max_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            max_longitude_deg_e, allow_nan=False
        )
        error_checking.assert_is_greater(
            max_longitude_deg_e, min_longitude_deg_e
        )
        longitude_positive_in_west = False

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Find example files.
    if len(cyclone_id_strings) == 1 and cyclone_id_strings[0] == '':
        first_init_time_unix_sec = time_conversion.string_to_unix_sec(
            first_init_time_string, TIME_FORMAT
        )
        last_init_time_unix_sec = time_conversion.string_to_unix_sec(
            last_init_time_string, TIME_FORMAT
        )
        init_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=first_init_time_unix_sec,
            end_time_unix_sec=last_init_time_unix_sec,
            time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True
        )

        first_year = int(time_conversion.unix_sec_to_string(
            init_times_unix_sec[0], '%Y'
        ))
        last_year = int(time_conversion.unix_sec_to_string(
            init_times_unix_sec[-1], '%Y'
        ))
        years = numpy.linspace(
            first_year, last_year, num=last_year - first_year + 1, dtype=int
        )

        cyclone_id_strings = example_io.find_cyclones(
            directory_name=norm_example_dir_name,
            raise_error_if_all_missing=True
        )
        cyclone_years = numpy.array([
            satellite_utils.parse_cyclone_id(c)[0]
            for c in cyclone_id_strings
        ], dtype=int)

        good_flags = numpy.array(
            [c in years for c in cyclone_years], dtype=float
        )
        good_indices = numpy.where(good_flags)[0]
        cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    else:
        first_init_time_string = '1000-01-01-00'
        last_init_time_string = '3000-01-01-00'

        first_init_time_unix_sec = time_conversion.string_to_unix_sec(
            first_init_time_string, TIME_FORMAT
        )
        last_init_time_unix_sec = time_conversion.string_to_unix_sec(
            last_init_time_string, TIME_FORMAT
        )

    cyclone_id_strings.sort()

    example_file_names = [
        example_io.find_file(
            directory_name=norm_example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        for c in cyclone_id_strings
    ]

    # Read model metadata and determine target variable.
    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY][
        neural_net.USE_TIME_DIFFS_KEY
    ] = False

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )

    predict_td_to_ts = validation_option_dict[neural_net.PREDICT_TD_TO_TS_KEY]
    target_variable_string = 'TD-to-TS' if predict_td_to_ts else 'RI'

    # Read borders and determine spacing of grid lines (parallels/meridians).
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    parallel_spacing_deg = number_rounding.round_to_half_integer(
        0.1 * (max_latitude_deg_n - min_latitude_deg_n)
    )
    meridian_spacing_deg = number_rounding.round_to_half_integer(
        0.1 * (max_longitude_deg_e - min_longitude_deg_e)
    )

    # Read normalization params (will be used to denormalize brightness temps).
    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    # Read examples.
    data_dicts = []

    for i in range(len(example_file_names)):
        print(SEPARATOR_STRING)

        this_option_dict = copy.deepcopy(validation_option_dict)
        this_option_dict[neural_net.EXAMPLE_FILE_KEY] = example_file_names[i]
        this_data_dict = neural_net.create_inputs(this_option_dict)

        this_data_dict = _subset_data(
            data_dict=this_data_dict, min_latitude_deg_n=min_latitude_deg_n,
            max_latitude_deg_n=max_latitude_deg_n,
            min_longitude_deg_e=min_longitude_deg_e,
            max_longitude_deg_e=max_longitude_deg_e,
            longitude_positive_in_west=longitude_positive_in_west,
            cyclone_id_string=
            example_io.file_name_to_cyclone_id(example_file_names[i]),
            first_init_time_unix_sec=first_init_time_unix_sec,
            last_init_time_unix_sec=last_init_time_unix_sec
        )

        if this_data_dict is None:
            continue

        data_dicts.append(this_data_dict)

    print(SEPARATOR_STRING)
    data_dict = _concat_data(data_dicts)
    del data_dicts

    data_dict = _match_predictors_to_predictions(
        data_dict=data_dict, prediction_file_name=prediction_file_name
    )
    print(SEPARATOR_STRING)

    # Do actual plotting.
    unique_init_times_unix_sec = numpy.unique(
        data_dict[neural_net.INIT_TIMES_KEY]
    )

    for this_time_unix_sec in unique_init_times_unix_sec:
        example_indices_this_time = numpy.where(
            data_dict[neural_net.INIT_TIMES_KEY] == this_time_unix_sec
        )[0]

        if len(example_indices_this_time) == 0:
            continue

        data_dict_this_time = {}

        for this_key in [
                neural_net.STORM_LATITUDES_KEY, neural_net.STORM_LONGITUDES_KEY
        ]:
            data_dict_this_time[this_key] = data_dict[this_key][
                example_indices_this_time, ...
            ]

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        plotting_utils.plot_borders(
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            axes_object=axes_object
        )

        for j in example_indices_this_time:
            these_predictor_matrices = [
                None if a is None else a[[j], ...]
                for a in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
            ]

            _plot_brightness_temp_one_example(
                predictor_matrices_one_example=these_predictor_matrices,
                normalization_table_xarray=normalization_table_xarray,
                grid_latitude_matrix_deg_n=
                data_dict[neural_net.GRID_LATITUDE_MATRIX_KEY][j, ..., -1],
                grid_longitude_matrix_deg_e=
                data_dict[neural_net.GRID_LONGITUDE_MATRIX_KEY][j, ..., -1],
                axes_object=axes_object
            )

        for i in range(len(example_indices_this_time)):
            j = example_indices_this_time[i]

            # Plot cyclone ID near center.
            label_string = '{0:s}'.format(
                data_dict[CYCLONE_IDS_KEY][j][-2:]
            )

            this_latitude_deg_n = (
                data_dict[neural_net.STORM_LATITUDES_KEY][j] + 1.
            )
            vertical_alignment_string = 'bottom'

            if this_latitude_deg_n > max_latitude_deg_n:
                this_latitude_deg_n = (
                    data_dict[neural_net.STORM_LATITUDES_KEY][j] - 1.
                )
                vertical_alignment_string = 'top'

            axes_object.text(
                data_dict[neural_net.STORM_LONGITUDES_KEY][j],
                this_latitude_deg_n, label_string,
                color=LABEL_COLOUR, fontsize=LABEL_FONT_SIZE,
                bbox=CYCLONE_ID_BOUNDING_BOX_DICT,
                horizontalalignment='center',
                verticalalignment=vertical_alignment_string,
                zorder=1e10
            )

            # Print predictions and targets off side of map.
            label_string = _get_prediction_string(
                data_dict=data_dict, example_index=j,
                predict_td_to_ts=predict_td_to_ts,
                confidence_level=confidence_level
            )

            if i == _get_swmost_index(data_dict_this_time):
                x_coord = -0.1
                y_coord = 0.5
                horiz_alignment_string = 'right'
                vertical_alignment_string = 'center'
            elif i == _get_nwmost_index(data_dict_this_time):
                x_coord = 1.1
                y_coord = 0.5
                horiz_alignment_string = 'left'
                vertical_alignment_string = 'center'
            elif i == _get_nemost_index(data_dict_this_time):
                x_coord = 0.5
                y_coord = 1.1
                horiz_alignment_string = 'center'
                vertical_alignment_string = 'bottom'
            else:
                x_coord = 0.5
                y_coord = -0.1
                horiz_alignment_string = 'center'
                vertical_alignment_string = 'top'

            axes_object.text(
                x_coord, y_coord, label_string,
                color=LABEL_COLOUR, fontsize=LABEL_FONT_SIZE,
                bbox=PREDICTION_BOUNDING_BOX_DICT,
                horizontalalignment=horiz_alignment_string,
                verticalalignment=vertical_alignment_string,
                zorder=1e10, transform=axes_object.transAxes
            )

        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=
            numpy.linspace(min_latitude_deg_n, max_latitude_deg_n, num=100),
            plot_longitudes_deg_e=
            numpy.linspace(min_longitude_deg_e, max_longitude_deg_e, num=100),
            axes_object=axes_object,
            parallel_spacing_deg=parallel_spacing_deg,
            meridian_spacing_deg=meridian_spacing_deg,
            font_size=DEFAULT_FONT_SIZE
        )

        colour_map_object, colour_norm_object = (
            satellite_plotting.get_colour_scheme()
        )
        satellite_plotting.add_colour_bar(
            brightness_temp_matrix_kelvins=numpy.full((2, 2), 273.15),
            axes_object=axes_object,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical', font_size=DEFAULT_FONT_SIZE
        )

        init_time_string = time_conversion.unix_sec_to_string(
            this_time_unix_sec, TIME_FORMAT
        )
        title_string = (
            'Forecast probs and labels for {0:s}, init {1:s}'
        ).format(target_variable_string, init_time_string)

        axes_object.set_title(title_string)

        output_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, init_time_string
        )
        print('Saving figure to file: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_metafile_name=getattr(INPUT_ARG_OBJECT, MODEL_METAFILE_ARG_NAME),
        norm_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        min_latitude_deg_n=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_latitude_deg_n=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_longitude_deg_e=getattr(INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_longitude_deg_e=getattr(INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        cyclone_id_strings=getattr(INPUT_ARG_OBJECT, CYCLONE_IDS_ARG_NAME),
        first_init_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_init_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
