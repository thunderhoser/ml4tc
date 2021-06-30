"""Plots normalized values of SHIPS-based predictors."""

import os
import sys
import argparse
import numpy
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import gg_plotting_utils
import imagemagick_utils
import example_io
import ships_io
import example_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

DEFAULT_FONT_SIZE = 30
FORECAST_HOUR_FONT_SIZE = 8
LAG_TIME_FONT_SIZE = 12
PREDICTOR_FONT_SIZE = 7

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

MIN_COLOUR_VALUE = -3.
MAX_COLOUR_VALUE = 3.
COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

FIGURE_RESOLUTION_DPI = 600
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

EXAMPLE_FILE_ARG_NAME = 'input_norm_example_file_name'
FORECAST_PREDICTORS_ARG_NAME = 'forecast_predictor_names'
LAGGED_PREDICTORS_ARG_NAME = 'lagged_predictor_names'
INIT_TIMES_ARG_NAME = 'init_time_strings'
FIRST_TIME_ARG_NAME = 'first_init_time_string'
LAST_TIME_ARG_NAME = 'last_init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = (
    'Path to file with normalized learning examples for one cyclone.  Will be '
    'read by `example_io.read_file`.'
)
FORECAST_PREDICTORS_HELP_STRING = (
    'List with names of forecast SHIPS predictors to use.'
)
LAGGED_PREDICTORS_HELP_STRING = (
    'List with names of lagged SHIPS predictors to use.'
)
INIT_TIMES_HELP_STRING = (
    'List of initialization times (format "yyyy-mm-dd-HHMMSS").  SHIPS '
    'predictors will be plotted for each of these init times.'
)
FIRST_TIME_HELP_STRING = (
    '[used only if `{0:s}` is left alone] First init time (format '
    '"yyyy-mm-dd-HHMMSS").'
).format(INIT_TIMES_ARG_NAME)

LAST_TIME_HELP_STRING = (
    '[used only if `{0:s}` is left alone] Last init time (format '
    '"yyyy-mm-dd-HHMMSS").'
).format(INIT_TIMES_ARG_NAME)

OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Images will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FORECAST_PREDICTORS_ARG_NAME, type=str, nargs='+', required=False,
    default=neural_net.DEFAULT_SHIPS_PREDICTOR_NAMES_FORECAST,
    help=FORECAST_PREDICTORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAGGED_PREDICTORS_ARG_NAME, type=str, nargs='+', required=False,
    default=neural_net.DEFAULT_SHIPS_PREDICTOR_NAMES_LAGGED,
    help=LAGGED_PREDICTORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIMES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=INIT_TIMES_HELP_STRING
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


def _add_colour_bar(figure_file_name, colour_bar_file_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_bar_file_name: Path to output file.  Image with colour bar
        will be saved here.
    """

    this_image_matrix = Image.open(figure_file_name)
    figure_width_px, figure_height_px = this_image_matrix.size
    figure_width_inches = float(figure_width_px) / FIGURE_RESOLUTION_DPI
    figure_height_inches = float(figure_height_px) / FIGURE_RESOLUTION_DPI

    extra_figure_object, extra_axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )
    extra_axes_object.axis('off')

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=MIN_COLOUR_VALUE, vmax=MAX_COLOUR_VALUE, clip=False
    )
    dummy_values = numpy.array([MIN_COLOUR_VALUE, MAX_COLOUR_VALUE])

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=extra_axes_object, data_matrix=dummy_values,
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=1., font_size=DEFAULT_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    extra_figure_object.savefig(
        colour_bar_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(extra_figure_object)

    print('Concatenating colour bar to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[figure_file_name, colour_bar_file_name],
        output_file_name=figure_file_name,
        num_panel_rows=1, num_panel_columns=2,
        extra_args_string='-gravity Center'
    )

    os.remove(colour_bar_file_name)
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name
    )


def plot_fcst_predictors_one_init_time(
        example_table_xarray, init_time_index, predictor_indices,
        output_dir_name):
    """Plots forecast predictors for one initialization time.

    :param example_table_xarray: See doc for
        `plot_lagged_predictors_one_init_time`.
    :param init_time_index: Same.
    :param predictor_indices: Same.
    :param output_dir_name: Same.
    :return: output_file_name: Path to output file, where image was saved.
    """

    xt = example_table_xarray

    forecast_times_hours = numpy.round(
        xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values
    ).astype(int)

    predictor_matrix = numpy.transpose(
        xt[example_utils.SHIPS_PREDICTORS_FORECAST_KEY].values[
            init_time_index, :, predictor_indices
        ]
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    colour_norm_object = pyplot.Normalize(
        vmin=MIN_COLOUR_VALUE, vmax=MAX_COLOUR_VALUE
    )
    axes_object.imshow(
        predictor_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
        norm=colour_norm_object
    )

    y_tick_values = numpy.linspace(
        0, predictor_matrix.shape[0] - 1, num=predictor_matrix.shape[0],
        dtype=float
    )
    y_tick_labels = ['{0:d}'.format(t) for t in forecast_times_hours]
    pyplot.yticks(
        y_tick_values, y_tick_labels, fontsize=FORECAST_HOUR_FONT_SIZE
    )
    axes_object.set_ylabel(
        'Forecast time (hours)', fontsize=FORECAST_HOUR_FONT_SIZE
    )

    x_tick_values = numpy.linspace(
        0, predictor_matrix.shape[1] - 1, num=predictor_matrix.shape[1],
        dtype=float
    )
    x_tick_labels = (
        xt.coords[example_utils.SHIPS_PREDICTOR_FORECAST_DIM].values[
            predictor_indices
        ].tolist()
    )
    pyplot.xticks(
        x_tick_values, x_tick_labels, rotation=90., fontsize=PREDICTOR_FONT_SIZE
    )

    init_time_unix_sec = (
        xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values[init_time_index]
    )
    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    cyclone_id_string = xt[ships_io.CYCLONE_ID_KEY].values[init_time_index]
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    title_string = 'SHIPS predictors for {0:s} at {1:s}'.format(
        cyclone_id_string, init_time_string
    )
    axes_object.set_title(title_string)

    extensionless_file_name = '{0:s}/ships_{1:s}_{2:s}_forecast'.format(
        output_dir_name, cyclone_id_string, init_time_string
    )
    figure_file_name = '{0:s}.jpg'.format(extensionless_file_name)
    colour_bar_file_name = '{0:s}_cbar.jpg'.format(extensionless_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=figure_file_name)

    print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=figure_file_name,
        colour_bar_file_name=colour_bar_file_name
    )

    return figure_file_name


def plot_lagged_predictors_one_init_time(
        example_table_xarray, init_time_index, predictor_indices,
        output_dir_name):
    """Plots lagged predictors for one initialization time.

    :param example_table_xarray: xarray table in format returned by
        `example_io.read_file`.
    :param init_time_index: Index of initial time to plot.
    :param predictor_indices: 1-D numpy array with indices of predictors to
        plot.
    :param output_dir_name: Name of output directory.  Image will be saved here.
    :return: output_file_name: Path to output file, where image was saved.
    """

    xt = example_table_xarray

    lag_times_hours = xt.coords[example_utils.SHIPS_LAG_TIME_DIM].values
    predictor_matrix = numpy.transpose(
        xt[example_utils.SHIPS_PREDICTORS_LAGGED_KEY].values[
            init_time_index, :, predictor_indices
        ]
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    colour_norm_object = pyplot.Normalize(
        vmin=MIN_COLOUR_VALUE, vmax=MAX_COLOUR_VALUE
    )
    axes_object.imshow(
        predictor_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
        norm=colour_norm_object
    )

    y_tick_values = numpy.linspace(
        0, predictor_matrix.shape[0] - 1, num=predictor_matrix.shape[0],
        dtype=float
    )
    y_tick_labels = ['{0:.1f}'.format(t) for t in lag_times_hours]
    pyplot.yticks(y_tick_values, y_tick_labels, fontsize=LAG_TIME_FONT_SIZE)
    axes_object.set_ylabel('Lag time (hours)')

    x_tick_values = numpy.linspace(
        0, predictor_matrix.shape[1] - 1, num=predictor_matrix.shape[1],
        dtype=float
    )
    x_tick_labels = (
        xt.coords[example_utils.SHIPS_PREDICTOR_LAGGED_DIM].values[
            predictor_indices
        ].tolist()
    )
    pyplot.xticks(
        x_tick_values, x_tick_labels, rotation=90., fontsize=PREDICTOR_FONT_SIZE
    )

    init_time_unix_sec = (
        xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values[init_time_index]
    )
    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    cyclone_id_string = xt[ships_io.CYCLONE_ID_KEY].values[init_time_index]
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    title_string = 'SHIPS predictors for {0:s} at {1:s}'.format(
        cyclone_id_string, init_time_string
    )
    axes_object.set_title(title_string)

    extensionless_file_name = '{0:s}/ships_{1:s}_{2:s}_lagged'.format(
        output_dir_name, cyclone_id_string, init_time_string
    )
    figure_file_name = '{0:s}.jpg'.format(extensionless_file_name)
    colour_bar_file_name = '{0:s}_cbar.jpg'.format(extensionless_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=figure_file_name)

    print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=figure_file_name,
        colour_bar_file_name=colour_bar_file_name
    )

    return figure_file_name


def _run(norm_example_file_name, forecast_predictor_names,
         lagged_predictor_names, init_time_strings, first_init_time_string,
         last_init_time_string, output_dir_name):
    """Plots normalized values of SHIPS-based predictors.

    This is effectively the main method.

    :param norm_example_file_name: See documentation at top of file.
    :param forecast_predictor_names: Same.
    :param lagged_predictor_names: Same.
    :param init_time_strings: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if desired init times cannot be found.
    """

    print('Reading data from: "{0:s}"...'.format(norm_example_file_name))
    example_table_xarray = example_io.read_file(norm_example_file_name)

    all_forecast_predictor_names = (
        example_table_xarray.coords[
            example_utils.SHIPS_PREDICTOR_FORECAST_DIM
        ].values.tolist()
    )
    forecast_predictor_indices = numpy.array([
        all_forecast_predictor_names.index(n)
        for n in forecast_predictor_names
    ], dtype=int)

    all_lagged_predictor_names = (
        example_table_xarray.coords[
            example_utils.SHIPS_PREDICTOR_LAGGED_DIM
        ].values.tolist()
    )
    lagged_predictor_indices = numpy.array([
        all_lagged_predictor_names.index(n)
        for n in lagged_predictor_names
    ], dtype=int)

    all_init_times_unix_sec = (
        example_table_xarray.coords[example_utils.SHIPS_VALID_TIME_DIM].values
    )

    if len(init_time_strings) == 1 and init_time_strings[0] == '':
        first_init_time_unix_sec = time_conversion.string_to_unix_sec(
            first_init_time_string, TIME_FORMAT
        )
        last_init_time_unix_sec = time_conversion.string_to_unix_sec(
            last_init_time_string, TIME_FORMAT
        )
        time_indices = numpy.where(numpy.logical_and(
            all_init_times_unix_sec >= first_init_time_unix_sec,
            all_init_times_unix_sec <= last_init_time_unix_sec
        ))[0]

        if len(time_indices) == 0:
            error_string = (
                'Cannot find any init times in file "{0:s}" between {1:s} and '
                '{2:s}.'
            ).format(
                norm_example_file_name, first_init_time_string,
                last_init_time_string
            )

            raise ValueError(error_string)
    else:
        init_times_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, TIME_FORMAT)
            for t in init_time_strings
        ], dtype=int)

        time_indices = numpy.array([
            numpy.where(all_init_times_unix_sec == t)[0][0]
            for t in init_times_unix_sec
        ], dtype=int)

    for i in time_indices:
        plot_lagged_predictors_one_init_time(
            example_table_xarray=example_table_xarray,
            init_time_index=i, predictor_indices=lagged_predictor_indices,
            output_dir_name=output_dir_name
        )

        plot_fcst_predictors_one_init_time(
            example_table_xarray=example_table_xarray,
            init_time_index=i, predictor_indices=forecast_predictor_indices,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        norm_example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        forecast_predictor_names=getattr(
            INPUT_ARG_OBJECT, FORECAST_PREDICTORS_ARG_NAME
        ),
        lagged_predictor_names=getattr(
            INPUT_ARG_OBJECT, LAGGED_PREDICTORS_ARG_NAME
        ),
        init_time_strings=getattr(INPUT_ARG_OBJECT, INIT_TIMES_ARG_NAME),
        first_init_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_init_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
