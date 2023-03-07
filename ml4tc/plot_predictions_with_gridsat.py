"""Plots predictions with GridSat in background, one map per time step."""

import os
import sys
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
import matplotlib.patches
from scipy.interpolate import interp1d

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
import imagemagick_utils
import prediction_io
import border_io
import general_utils
import neural_net
import plotting_utils
import satellite_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H'
TIME_INTERVAL_SEC = 21600
GRIDSAT_TIME_FORMAT = '%Y.%m.%d.%H'

CYCLONE_IDS_KEY = 'cyclone_id_strings'
FORECAST_PROBS_KEY = 'forecast_probabilities'

TC_CENTER_MARKER_TYPE = '*'
TC_CENTER_MARKER_SIZE = 32
TC_CENTER_MARKER_COLOUR = numpy.full(3, 1.)

LINE_COLOURS = 30 * [
    numpy.array([27, 158, 119], dtype=float) / 255,
    numpy.array([217, 95, 2], dtype=float) / 255,
    numpy.array([117, 112, 179], dtype=float) / 255
]
LINE_WIDTH = 4

MEAN_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
VIOLIN_LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
VIOLIN_LINE_WIDTH = 2.
VIOLIN_FACE_COLOUR = matplotlib.colors.to_rgba(c=VIOLIN_LINE_COLOUR, alpha=0.4)
VIOLIN_EDGE_COLOUR = matplotlib.colors.to_rgba(c=VIOLIN_LINE_COLOUR, alpha=0.4)
VIOLIN_EDGE_WIDTH = 0.

POSITIVE_CLASS_MARKER_TYPE = 'D'
POSITIVE_CLASS_MARKER_SIZE = 12
POLYGON_OPACITY = 0.5

LABEL_COLOUR = numpy.full(3, 0.)
LABEL_FONT_SIZE = 32

CYCLONE_ID_BOUNDING_BOX_DICT = {
    'alpha': 0.5,
    'edgecolor': numpy.full(3, 0.),
    'linewidth': 1,
    'facecolor': numpy.full(3, 1.)
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

MODEL_METAFILE_ARG_NAME = 'input_model_metafile_name'
PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
GRIDSAT_DIR_ARG_NAME = 'input_gridsat_dir_name'
SHOW_VIOLIN_PLOTS_ARG_NAME = 'show_violin_plots'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
MIN_LATITUDE_ARG_NAME = 'min_latitude_deg_n'
MAX_LATITUDE_ARG_NAME = 'max_latitude_deg_n'
MIN_LONGITUDE_ARG_NAME = 'min_longitude_deg_e'
MAX_LONGITUDE_ARG_NAME = 'max_longitude_deg_e'
FIRST_TIME_ARG_NAME = 'first_init_time_string'
LAST_TIME_ARG_NAME = 'last_init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_METAFILE_HELP_STRING = (
    'Path to metafile for model.  Will be read by `neural_net.read_metafile`.'
)
PREDICTION_FILE_HELP_STRING = (
    'Path to file with predictions and targets.  Will be read by '
    '`prediction_io.read_file`.'
)
GRIDSAT_DIR_HELP_STRING = (
    'Name of directory with GridSat data.  Files therein will be found '
    '`_find_gridsat_file` and read by `_read_gridsat_file`.'
)
SHOW_VIOLIN_PLOTS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will show violin plots (confidence intervals) to '
    'represent uncertainty.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    '[used only if {0:s} == 0] Confidence level to plot.  For example, if this '
    'is 0.95, will plot 95% confidence interval of probabilities for each '
    'cyclone.  If you want to plot only the mean and not a confidence interval, '
    'leave this argument alone.'
).format(SHOW_VIOLIN_PLOTS_ARG_NAME)

MIN_LATITUDE_HELP_STRING = 'Minimum latitude (deg north) in map.'
MAX_LATITUDE_HELP_STRING = 'Max latitude (deg north) in map.'
MIN_LONGITUDE_HELP_STRING = 'Minimum longitude (deg east) in map.'
MAX_LONGITUDE_HELP_STRING = 'Max longitude (deg east) in map.'
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
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDSAT_DIR_ARG_NAME, type=str, required=False, default='',
    help=GRIDSAT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHOW_VIOLIN_PLOTS_ARG_NAME, type=int, required=False, default=0,
    help=SHOW_VIOLIN_PLOTS_HELP_STRING
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


def _find_gridsat_file(directory_name, valid_time_unix_sec):
    """Finds GridSat file.

    :param directory_name: Name of directory where file is expected.
    :param valid_time_unix_sec: Valid time.
    :return: gridsat_file_name: Path to file.
    :raises: ValueError: if file is not found.
    """

    gridsat_file_name = '{0:s}/GRIDSAT-B1.{1:s}.v02r01.nc'.format(
        directory_name,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, GRIDSAT_TIME_FORMAT
        )
    )

    if os.path.isfile(gridsat_file_name):
        return gridsat_file_name

    error_string = 'Cannot find GridSat file.  Expected at: "{0:s}"'.format(
        gridsat_file_name
    )
    raise ValueError(error_string)


def _read_gridsat_file(
        netcdf_file_name, min_latitude_deg_n, max_latitude_deg_n,
        min_longitude_deg_e, max_longitude_deg_e, longitude_positive_in_west):
    """Reads GridSat data from NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to input file.
    :param min_latitude_deg_n: See documentation at top of this script.
    :param max_latitude_deg_n: Same.
    :param min_longitude_deg_e: Same.
    :param max_longitude_deg_e: Same.
    :param longitude_positive_in_west: Boolean flag.
    :return: brightness_temp_matrix_kelvins: M-by-N numpy array of brightness
        temperatures.
    :return: grid_latitudes_deg_n: length-M numpy array of grid latitudes.
    :return: grid_longitudes_deg_e: length-N numpy array of grid longitudes.
    """

    gridsat_table_xarray = xarray.open_dataset(netcdf_file_name)
    grid_latitudes_deg_n = gridsat_table_xarray.coords['lat'].values
    grid_longitudes_deg_e = gridsat_table_xarray.coords['lon'].values
    brightness_temp_matrix_kelvins = gridsat_table_xarray['irwin_cdr'].values[
        0, ...
    ]
    brightness_temp_matrix_kelvins = general_utils.fill_nans(
        brightness_temp_matrix_kelvins
    )

    if longitude_positive_in_west:
        grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            grid_longitudes_deg_e, allow_nan=False
        )
    else:
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e, allow_nan=False
        )

    latitude_spacing_deg = numpy.absolute(
        numpy.diff(grid_latitudes_deg_n[:2])
    )[0]
    good_indices = numpy.where(numpy.logical_and(
        grid_latitudes_deg_n >= min_latitude_deg_n - latitude_spacing_deg,
        grid_latitudes_deg_n <= max_latitude_deg_n + latitude_spacing_deg
    ))[0]

    grid_latitudes_deg_n = grid_latitudes_deg_n[good_indices]
    brightness_temp_matrix_kelvins = (
        brightness_temp_matrix_kelvins[good_indices, :]
    )

    longitude_spacing_deg = numpy.absolute(
        numpy.diff(grid_longitudes_deg_e[10:12])
    )[0]

    print(grid_longitudes_deg_e)
    print(min_longitude_deg_e)
    print(min_longitude_deg_e - longitude_spacing_deg)
    print(max_longitude_deg_e)
    print(max_longitude_deg_e + longitude_spacing_deg)

    good_indices = numpy.where(numpy.logical_and(
        grid_longitudes_deg_e >= min_longitude_deg_e - longitude_spacing_deg,
        grid_longitudes_deg_e <= max_longitude_deg_e + longitude_spacing_deg
    ))[0]

    grid_longitudes_deg_e = grid_longitudes_deg_e[good_indices]
    brightness_temp_matrix_kelvins = (
        brightness_temp_matrix_kelvins[:, good_indices]
    )

    return (
        brightness_temp_matrix_kelvins,
        grid_latitudes_deg_n,
        grid_longitudes_deg_e
    )


def _plot_predictions_with_violin(
        prediction_dict, example_index, predict_td_to_ts, output_file_name):
    """Plots predictions and targets for one init time, with violin plot.

    :param prediction_dict: See doc for `_plot_predictions_no_violin`.
    :param example_index: Index corresponding to the given init time.
    :param predict_td_to_ts: Boolean flag.  Should be True (False) if the
        prediction task is TD-to-TS (rapid intensification).
    :param output_file_name: See doc for `_plot_predictions_no_violin`.
    """

    target_classes = (
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][example_index, :]
    )
    all_forecast_prob_matrix = prediction_dict[
        prediction_io.PROBABILITY_MATRIX_KEY
    ][example_index, 1, ...]

    lead_times_hours = prediction_dict[prediction_io.LEAD_TIMES_KEY]
    quantile_levels = prediction_dict[prediction_io.QUANTILE_LEVELS_KEY]

    if quantile_levels is None:
        mean_forecast_probs = prediction_io.get_mean_predictions(
            prediction_dict
        )[example_index, :]
    else:
        new_quantile_levels = numpy.linspace(0, 1, num=1001, dtype=float)

        interp_object = interp1d(
            x=quantile_levels, y=all_forecast_prob_matrix[..., 1:],
            axis=-1, kind='linear', bounds_error=False, assume_sorted=True,
            fill_value='extrapolate'
        )

        new_forecast_prob_matrix = interp_object(new_quantile_levels)

        for j in range(len(lead_times_hours)):
            new_forecast_prob_matrix[j, :] = numpy.maximum(
                new_forecast_prob_matrix[j, :],
                numpy.min(all_forecast_prob_matrix[j, 1:])
            )
            new_forecast_prob_matrix[j, :] = numpy.minimum(
                new_forecast_prob_matrix[j, :],
                numpy.max(all_forecast_prob_matrix[j, 1:])
            )

        new_forecast_prob_matrix = numpy.maximum(new_forecast_prob_matrix, 0.)
        new_forecast_prob_matrix = numpy.minimum(new_forecast_prob_matrix, 1.)
        all_forecast_prob_matrix = new_forecast_prob_matrix + 0.

        # TODO(thunderhoser): This is actually the median.
        mean_forecast_probs = interp_object(0.5)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    legend_handles = []
    legend_strings = []

    if example_index > -1:
        violin_handles = axes_object.violinplot(
            numpy.transpose(all_forecast_prob_matrix),
            positions=lead_times_hours,
            vert=True, widths=4.8, showmeans=False, showmedians=False,
            showextrema=True
        )

        for part_name in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            try:
                this_handle = violin_handles[part_name]
            except:
                continue

            this_handle.set_edgecolor(VIOLIN_LINE_COLOUR)
            this_handle.set_linewidth(VIOLIN_LINE_WIDTH)

        for this_handle in violin_handles['bodies']:
            this_handle.set_facecolor(VIOLIN_FACE_COLOUR)
            this_handle.set_edgecolor(VIOLIN_EDGE_COLOUR)
            this_handle.set_linewidth(VIOLIN_EDGE_WIDTH)
            this_handle.set_alpha(1.)

        legend_handles.append(violin_handles['bodies'][0])
        legend_strings.append('Forecast distribution')

        this_handle = axes_object.plot(
            lead_times_hours, mean_forecast_probs,
            color=MEAN_COLOUR, linestyle='solid', linewidth=LINE_WIDTH
        )[0]

        legend_handles.append(this_handle)
        legend_strings.append('Mean forecast')

        title_string = 'Forecast for storm {0:s}'.format(
            prediction_dict[prediction_io.CYCLONE_IDS_KEY][example_index][-2:]
        )
        axes_object.set_title(title_string)

        positive_indices = numpy.where(target_classes == 1)[0]
        this_handle = axes_object.plot(
            lead_times_hours[positive_indices],
            mean_forecast_probs[positive_indices],
            linestyle='None', marker=POSITIVE_CLASS_MARKER_TYPE,
            markersize=POSITIVE_CLASS_MARKER_SIZE, markeredgewidth=0,
            markerfacecolor=numpy.full(3, 0.),
            markeredgecolor=numpy.full(3, 0.)
        )[0]

        legend_handles.append(this_handle)
        legend_strings.append(
            'TD-to-TS observed' if predict_td_to_ts else 'RI observed'
        )

        axes_object.legend(
            legend_handles, legend_strings,
            loc='center right', bbox_to_anchor=(0.95, 0.5),
            fancybox=True, shadow=True, ncol=1
        )
    else:
        axes_object.text(
            0.5, 0.5,
            'No {0:s}s at this time'.format('TD' if predict_td_to_ts else 'TC'),
            color=numpy.full(3, 0.),
            horizontalalignment='center', verticalalignment='center',
            transform=axes_object.transAxes
        )

    axes_object.set_ylabel(
        'TD-to-TS probability' if predict_td_to_ts else 'RI probability'
    )
    axes_object.set_xlabel('Lead time (hours)')
    axes_object.set_ylim(0, 1)

    if predict_td_to_ts:
        axes_object.set_xlim(0, numpy.max(lead_times_hours))
    else:
        axes_object.set_xlim(lead_times_hours[0] - 3, lead_times_hours[0] + 3)

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_predictions_no_violin(
        prediction_dict, example_indices, predict_td_to_ts, confidence_level,
        output_file_name):
    """Plots predictions and targets for one init time, with no violin plot.

    :param prediction_dict: Dictionary returned by `prediction_io.read_file`.
    :param example_indices: 1-D numpy array of indices corresponding to the
        given init time.
    :param predict_td_to_ts: Boolean flag.  Should be True (False) if the
        prediction task is TD-to-TS (rapid intensification).
    :param confidence_level: See documentation at top of file.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    lead_times_hours = prediction_dict[prediction_io.LEAD_TIMES_KEY]
    target_class_matrix = (
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][example_indices, :]
    )
    all_forecast_prob_matrix = prediction_dict[
        prediction_io.PROBABILITY_MATRIX_KEY
    ][example_indices, 1, ...]

    mean_forecast_prob_matrix = (
        prediction_io.get_mean_predictions(prediction_dict)[example_indices, :]
    )

    if predict_td_to_ts:
        lead_times_hours = lead_times_hours[0] + numpy.array([-2.5, 2.5])
        target_class_matrix = numpy.repeat(
            target_class_matrix, axis=2, repeats=2
        )
        all_forecast_prob_matrix = numpy.repeat(
            all_forecast_prob_matrix, axis=2, repeats=2
        )
        mean_forecast_prob_matrix = numpy.repeat(
            mean_forecast_prob_matrix, axis=2, repeats=2
        )

    quantile_levels = prediction_dict[prediction_io.QUANTILE_LEVELS_KEY]

    if confidence_level is None:
        min_forecast_prob_matrix = None
        max_forecast_prob_matrix = None
    else:
        if quantile_levels is None:
            min_forecast_prob_matrix = numpy.percentile(
                all_forecast_prob_matrix, 50 * (1. - confidence_level), axis=-1
            )
            max_forecast_prob_matrix = numpy.percentile(
                all_forecast_prob_matrix, 50 * (1. + confidence_level), axis=-1
            )
        else:
            interp_object = interp1d(
                x=quantile_levels, y=all_forecast_prob_matrix[..., 1:],
                axis=-1, kind='linear', bounds_error=False, assume_sorted=True,
                fill_value='extrapolate'
            )

            min_forecast_prob_matrix = interp_object(
                0.5 * (1. - confidence_level)
            )
            max_forecast_prob_matrix = interp_object(
                0.5 * (1. + confidence_level)
            )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_examples = len(example_indices)
    legend_handles = [None] * (num_examples + 1)
    legend_strings = [''] * (num_examples + 1)

    for i in range(len(example_indices)):
        legend_handles[i] = axes_object.plot(
            lead_times_hours, mean_forecast_prob_matrix[i, :],
            color=LINE_COLOURS[i], linestyle='solid', linewidth=LINE_WIDTH
        )[0]

        j = example_indices[i]
        legend_strings[i] = 'Fcst for storm {0:s}'.format(
            prediction_dict[prediction_io.CYCLONE_IDS_KEY][j][-2:]
        )

        positive_indices = numpy.where(target_class_matrix[i, :] == 1)[0]
        this_handle = axes_object.plot(
            lead_times_hours[positive_indices],
            mean_forecast_prob_matrix[i, :][positive_indices],
            linestyle='None', marker=POSITIVE_CLASS_MARKER_TYPE,
            markersize=POSITIVE_CLASS_MARKER_SIZE, markeredgewidth=0,
            markerfacecolor=numpy.full(3, 0.),
            markeredgecolor=numpy.full(3, 0.)
        )[0]

        if i == len(example_indices) - 1:
            legend_handles[-1] = this_handle
            legend_strings[-1] = (
                'TD-to-TS observed' if predict_td_to_ts else 'RI observed'
            )

        x_vertices = numpy.concatenate((
            lead_times_hours, lead_times_hours[::-1], lead_times_hours[[0]]
        ))
        y_vertices = numpy.concatenate((
            max_forecast_prob_matrix[i, :],
            min_forecast_prob_matrix[i, :][::-1],
            max_forecast_prob_matrix[i, :][[0]]
        ))
        polygon_coord_matrix = numpy.transpose(numpy.vstack((
            x_vertices, y_vertices
        )))

        polygon_colour = matplotlib.colors.to_rgba(
            LINE_COLOURS[i], POLYGON_OPACITY
        )
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    if len(example_indices) > 0:
        axes_object.legend(
            legend_handles, legend_strings,
            loc='center right', bbox_to_anchor=(0.95, 0.5),
            fancybox=True, shadow=True, ncol=1
        )
    else:
        axes_object.text(
            0.5, 0.5,
            'No {0:s}s at this time'.format('TD' if predict_td_to_ts else 'TC'),
            color=numpy.full(3, 0.),
            horizontalalignment='center', verticalalignment='center',
            transform=axes_object.transAxes
        )

    y_label_string = 'Forecast probability with {0:.0f}% CI'.format(
        100 * confidence_level
    )

    axes_object.set_ylabel(y_label_string)
    axes_object.set_xlabel('Lead time (hours)')
    axes_object.set_ylim(0, 1)

    if predict_td_to_ts:
        axes_object.set_xlim(0, numpy.max(lead_times_hours))
    else:
        axes_object.set_xlim(
            lead_times_hours[0] - 0.5,
            lead_times_hours[-1] + 0.5
        )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(model_metafile_name, gridsat_dir_name, prediction_file_name,
         show_violin_plots, confidence_level, min_latitude_deg_n,
         max_latitude_deg_n, min_longitude_deg_e, max_longitude_deg_e,
         first_init_time_string, last_init_time_string, output_dir_name):
    """Plots predictions for many storms, one map per time step.

    This is effectively the main method.

    :param model_metafile_name: See documentation at top of file.
    :param gridsat_dir_name: Same.
    :param prediction_file_name: Same.
    :param show_violin_plots: Same.
    :param confidence_level: Same.
    :param min_latitude_deg_n: Same.
    :param max_latitude_deg_n: Same.
    :param min_longitude_deg_e: Same.
    :param max_longitude_deg_e: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    if confidence_level <= 0 or show_violin_plots:
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

    print(min_longitude_deg_e)
    print(max_longitude_deg_e)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Find cyclones to plot.
    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

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

    good_indices = numpy.where(numpy.logical_and(
        prediction_dict[prediction_io.INIT_TIMES_KEY] >=
        init_times_unix_sec[0],
        prediction_dict[prediction_io.INIT_TIMES_KEY] <=
        init_times_unix_sec[-1]
    ))[0]

    prediction_dict = prediction_io.subset_by_index(
        prediction_dict=prediction_dict, desired_indices=good_indices
    )

    if longitude_positive_in_west:
        prediction_dict[prediction_io.STORM_LONGITUDES_KEY] = (
            lng_conversion.convert_lng_positive_in_west(
                prediction_dict[prediction_io.STORM_LONGITUDES_KEY]
            )
        )
    else:
        prediction_dict[prediction_io.STORM_LONGITUDES_KEY] = (
            lng_conversion.convert_lng_negative_in_west(
                prediction_dict[prediction_io.STORM_LONGITUDES_KEY]
            )
        )

    good_indices = numpy.where(numpy.logical_and(
        prediction_dict[prediction_io.STORM_LONGITUDES_KEY] >=
        min_longitude_deg_e,
        prediction_dict[prediction_io.STORM_LONGITUDES_KEY] <=
        max_longitude_deg_e
    ))[0]

    prediction_dict = prediction_io.subset_by_index(
        prediction_dict=prediction_dict, desired_indices=good_indices
    )

    good_indices = numpy.where(numpy.logical_and(
        prediction_dict[prediction_io.STORM_LATITUDES_KEY] >=
        min_latitude_deg_n,
        prediction_dict[prediction_io.STORM_LATITUDES_KEY] <=
        max_latitude_deg_n
    ))[0]

    prediction_dict = prediction_io.subset_by_index(
        prediction_dict=prediction_dict, desired_indices=good_indices
    )

    # Read model metadata and determine target variable.
    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
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

    # Do actual plotting.
    for this_time_unix_sec in init_times_unix_sec:
        example_indices_this_time = numpy.where(
            prediction_dict[prediction_io.INIT_TIMES_KEY] == this_time_unix_sec
        )[0]

        gridsat_file_name = _find_gridsat_file(
            directory_name=gridsat_dir_name,
            valid_time_unix_sec=this_time_unix_sec
        )

        print('Reading data from: "{0:s}"...'.format(gridsat_file_name))
        (
            brightness_temp_matrix_kelvins,
            grid_latitudes_deg_n,
            grid_longitudes_deg_e
        ) = _read_gridsat_file(
            netcdf_file_name=gridsat_file_name,
            min_latitude_deg_n=min_latitude_deg_n,
            max_latitude_deg_n=max_latitude_deg_n,
            min_longitude_deg_e=min_longitude_deg_e,
            max_longitude_deg_e=max_longitude_deg_e,
            longitude_positive_in_west=longitude_positive_in_west
        )

        print(brightness_temp_matrix_kelvins.shape)

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        plotting_utils.plot_borders(
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            axes_object=axes_object
        )
        satellite_plotting.plot_2d_grid(
            brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
            axes_object=axes_object,
            latitude_array_deg_n=grid_latitudes_deg_n,
            longitude_array_deg_e=grid_longitudes_deg_e,
            cbar_orientation_string=None
        )

        for i in range(len(example_indices_this_time)):
            j = example_indices_this_time[i]

            # Plot marker at TC center.
            this_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
                prediction_dict[prediction_io.STORM_LONGITUDES_KEY][j]
            )
            axes_object.plot(
                this_longitude_deg_e,
                prediction_dict[prediction_io.STORM_LATITUDES_KEY][j],
                linestyle='None', marker=TC_CENTER_MARKER_TYPE,
                markersize=TC_CENTER_MARKER_SIZE, markeredgewidth=0,
                markerfacecolor=TC_CENTER_MARKER_COLOUR,
                markeredgecolor=TC_CENTER_MARKER_COLOUR
            )

            # Plot cyclone ID near center.
            label_string = '{0:s}'.format(
                prediction_dict[prediction_io.CYCLONE_IDS_KEY][j][-2:]
            )

            this_latitude_deg_n = (
                prediction_dict[prediction_io.STORM_LATITUDES_KEY][j] + 1.
            )
            vertical_alignment_string = 'bottom'

            if this_latitude_deg_n > max_latitude_deg_n:
                this_latitude_deg_n = (
                    prediction_dict[prediction_io.STORM_LATITUDES_KEY][j] - 1.
                )
                vertical_alignment_string = 'top'

            axes_object.text(
                this_longitude_deg_e, this_latitude_deg_n, label_string,
                color=LABEL_COLOUR, fontsize=LABEL_FONT_SIZE,
                bbox=CYCLONE_ID_BOUNDING_BOX_DICT,
                horizontalalignment='center',
                verticalalignment=vertical_alignment_string,
                zorder=1e10
            )

        init_time_string = time_conversion.unix_sec_to_string(
            this_time_unix_sec, TIME_FORMAT
        )

        if show_violin_plots:
            if len(example_indices_this_time) == 0:
                graph_file_name = '{0:s}/{1:s}_graph0.jpg'.format(
                    output_dir_name, init_time_string
                )
                _plot_predictions_with_violin(
                    prediction_dict=prediction_dict, example_index=-1,
                    predict_td_to_ts=predict_td_to_ts,
                    output_file_name=graph_file_name
                )

                graph_file_names = [graph_file_name]
            else:
                graph_file_names = [None] * len(example_indices_this_time)

                for j in range(len(example_indices_this_time)):
                    graph_file_names[j] = '{0:s}/{1:s}_graph{2:d}.jpg'.format(
                        output_dir_name, init_time_string, j
                    )
                    _plot_predictions_with_violin(
                        prediction_dict=prediction_dict,
                        example_index=example_indices_this_time[j],
                        predict_td_to_ts=predict_td_to_ts,
                        output_file_name=graph_file_names[j]
                    )
        else:
            graph_file_name = '{0:s}/{1:s}_graph.jpg'.format(
                output_dir_name, init_time_string
            )
            _plot_predictions_no_violin(
                prediction_dict=prediction_dict,
                example_indices=example_indices_this_time,
                predict_td_to_ts=predict_td_to_ts,
                confidence_level=confidence_level,
                output_file_name=graph_file_name
            )

            graph_file_names = [graph_file_name]

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

        title_string = (
            'Forecasts and observations for {0:s}, init {1:s}'
        ).format(target_variable_string, init_time_string)

        axes_object.set_title(title_string)

        satellite_file_name = '{0:s}/{1:s}_satellite.jpg'.format(
            output_dir_name, init_time_string
        )
        print('Saving figure to file: "{0:s}"...'.format(satellite_file_name))
        figure_object.savefig(
            satellite_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        concat_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, init_time_string
        )

        print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
        imagemagick_utils.concatenate_images(
            input_file_names=[satellite_file_name] + graph_file_names,
            output_file_name=concat_file_name,
            num_panel_rows=1, num_panel_columns=len(graph_file_names) + 1
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_file_name, output_file_name=concat_file_name,
            output_size_pixels=CONCAT_FIGURE_SIZE_PX
        )

        os.remove(satellite_file_name)
        for this_file_name in graph_file_names:
            os.remove(this_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_metafile_name=getattr(INPUT_ARG_OBJECT, MODEL_METAFILE_ARG_NAME),
        gridsat_dir_name=getattr(INPUT_ARG_OBJECT, GRIDSAT_DIR_ARG_NAME),
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        show_violin_plots=getattr(INPUT_ARG_OBJECT, SHOW_VIOLIN_PLOTS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        min_latitude_deg_n=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_latitude_deg_n=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_longitude_deg_e=getattr(INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_longitude_deg_e=getattr(INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        first_init_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_init_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
