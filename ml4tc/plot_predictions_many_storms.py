"""Plots predictions for many storms, one map per time step."""

import os
import sys
import copy
import argparse
import warnings
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import longitude_conversion as lng_conversion
import number_rounding
import file_system_utils
import error_checking
import example_io
import prediction_io
import border_io
import satellite_utils
import normalization
import neural_net
import plotting_utils
import satellite_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H'
TIME_INTERVAL_SEC = 21600

CYCLONE_IDS_KEY = 'cyclone_id_strings'
FORECAST_PROBS_KEY = 'forecast_probabilities'

LABEL_COLOUR = numpy.full(3, 0.)
LABEL_BOUNDING_BOX_DICT = {
    'alpha': 0.5,
    'edgecolor': numpy.full(3, 0.),
    'linewidth': 1,
    'facecolor': numpy.full(3, 1.)
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

MODEL_METAFILE_ARG_NAME = 'input_model_metafile_name'
EXAMPLE_DIR_ARG_NAME = 'input_norm_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
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
            neural_net.TARGET_ARRAY_KEY, neural_net.INIT_TIMES_KEY,
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

        print(data_dicts[0][neural_net.PREDICTOR_MATRICES_KEY][k])

        data_dict[neural_net.PREDICTOR_MATRICES_KEY][k] = numpy.concatenate(
            [d[neural_net.PREDICTOR_MATRICES_KEY][k] for d in data_dicts],
            axis=0
        )

    for this_key in [
            neural_net.TARGET_ARRAY_KEY, neural_net.INIT_TIMES_KEY,
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

    :param data_dict: Dictionary returned by `_concat_data`.
    :param prediction_file_name: See documentation at top of file.
    :return: data_dict: Same as input but with an extra key.
    data_dict['forecast_probabilities']: length-E numpy array of forecast event
        probabilities.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    prediction_dict[prediction_io.CYCLONE_IDS_KEY] = numpy.array(
        prediction_dict[prediction_io.CYCLONE_IDS_KEY]
    )
    all_forecast_probs = prediction_io.get_mean_predictions(prediction_dict)

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

    good_indices = numpy.array(good_indices, dtype=int)
    data_dict[FORECAST_PROBS_KEY] = all_forecast_probs[good_indices]
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


def _run(model_metafile_name, norm_example_dir_name, normalization_file_name,
         prediction_file_name, min_latitude_deg_n, max_latitude_deg_n,
         min_longitude_deg_e, max_longitude_deg_e, cyclone_id_strings,
         first_init_time_string, last_init_time_string, output_dir_name):
    """Plots predictions for many storms, one map per time step.

    This is effectively the main method.

    :param model_metafile_name: See documentation at top of file.
    :param norm_example_dir_name: Same.
    :param normalization_file_name: Same.
    :param prediction_file_name: Same.
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

    # Read necessary files other than example files.
    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

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

    # Read model metadata.
    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY][
        neural_net.USE_TIME_DIFFS_KEY
    ] = False

    # Read examples.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
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
            print('\n\n\n\n\n\nNONE NONE NONE\n\n\n\n\n\n\n')
            continue

        data_dicts.append(this_data_dict)

    if len(data_dicts) == 0:
        warning_string = 'Cannot find any data to plot.  RETURNING.'
        warnings.warn(warning_string)
        return

    print(SEPARATOR_STRING)
    data_dict = _concat_data(data_dicts)
    del data_dicts

    data_dict = _match_predictors_to_predictions(
        data_dict=data_dict, prediction_file_name=prediction_file_name
    )
    print(SEPARATOR_STRING)

    unique_init_times_unix_sec = numpy.unique(
        data_dict[neural_net.INIT_TIMES_KEY]
    )
    parallel_spacing_deg = number_rounding.round_to_half_integer(
        0.1 * (max_latitude_deg_n - min_latitude_deg_n)
    )
    meridian_spacing_deg = number_rounding.round_to_half_integer(
        0.1 * (max_longitude_deg_e - min_longitude_deg_e)
    )
    target_variable_string = (
        'TD-to-TS' if validation_option_dict[neural_net.PREDICT_TD_TO_TS_KEY]
        else 'RI'
    )

    for this_time_unix_sec in unique_init_times_unix_sec:
        these_indices = numpy.where(
            data_dict[neural_net.INIT_TIMES_KEY] == this_time_unix_sec
        )[0]

        if len(these_indices) == 0:
            continue

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        plotting_utils.plot_borders(
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            axes_object=axes_object
        )

        for i in these_indices:
            these_predictor_matrices = [
                None if a is None else a[[i], ...]
                for a in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
            ]

            _plot_brightness_temp_one_example(
                predictor_matrices_one_example=these_predictor_matrices,
                normalization_table_xarray=normalization_table_xarray,
                grid_latitude_matrix_deg_n=
                data_dict[neural_net.GRID_LATITUDE_MATRIX_KEY][i, ..., -1],
                grid_longitude_matrix_deg_e=
                data_dict[neural_net.GRID_LONGITUDE_MATRIX_KEY][i, ..., -1],
                axes_object=axes_object
            )

        for i in these_indices:
            label_string = r'$p$ = '
            label_string += '{0:.2f}\n'.format(data_dict[FORECAST_PROBS_KEY][i])
            label_string += r'$y$ = '
            label_string += (
                'yes' if data_dict[neural_net.TARGET_ARRAY_KEY][i] == 1
                else 'no'
            )

            axes_object.text(
                data_dict[neural_net.STORM_LONGITUDES_KEY][i],
                data_dict[neural_net.STORM_LATITUDES_KEY][i],
                label_string,
                fontsize=FONT_SIZE, color=LABEL_COLOUR,
                bbox=LABEL_BOUNDING_BOX_DICT,
                horizontalalignment='center', verticalalignment='center',
                zorder=1e10
            )

        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=
            numpy.linspace(min_latitude_deg_n, max_latitude_deg_n, num=100),
            plot_longitudes_deg_e=
            numpy.linspace(min_longitude_deg_e, max_longitude_deg_e, num=100),
            axes_object=axes_object,
            parallel_spacing_deg=parallel_spacing_deg,
            meridian_spacing_deg=meridian_spacing_deg,
            font_size=FONT_SIZE
        )

        colour_map_object, colour_norm_object = (
            satellite_plotting.get_colour_scheme()
        )
        satellite_plotting.add_colour_bar(
            brightness_temp_matrix_kelvins=numpy.full((2, 2), 273.15),
            axes_object=axes_object,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical', font_size=FONT_SIZE
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
        min_latitude_deg_n=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_latitude_deg_n=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_longitude_deg_e=getattr(INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_longitude_deg_e=getattr(INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        cyclone_id_strings=getattr(INPUT_ARG_OBJECT, CYCLONE_IDS_ARG_NAME),
        first_init_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_init_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
