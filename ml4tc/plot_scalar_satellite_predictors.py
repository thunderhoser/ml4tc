"""Plots normalized values of scalar satellite-based predictors."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import example_io
import example_utils
import satellite_utils
import neural_net

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
MIN_NORMALIZED_VALUE = -3.
MAX_NORMALIZED_VALUE = 3.

BAR_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
BAR_EDGE_COLOUR = numpy.full(3, 0.)
BAR_FONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
BAR_EDGE_WIDTH = 2.
BAR_FONT_SIZE = 20

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

DEFAULT_FONT_SIZE = 20
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

EXAMPLE_FILE_ARG_NAME = 'input_norm_example_file_name'
PREDICTORS_ARG_NAME = 'predictor_names'
VALID_TIMES_ARG_NAME = 'valid_time_strings'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = (
    'Path to file with normalized learning examples for one cyclone.  Will be '
    'read by `example_io.read_file`.'
)
PREDICTORS_HELP_STRING = 'List with names of scalar predictors to plot.'
VALID_TIMES_HELP_STRING = (
    'List of valid times (format "yyyy-mm-dd-HHMMSS").  Scalar predictors will '
    'be plotted for each of these valid times.'
)
FIRST_TIME_HELP_STRING = (
    '[used only if `{0:s}` is left alone] First valid time (format '
    '"yyyy-mm-dd-HHMMSS").'
).format(VALID_TIMES_ARG_NAME)

LAST_TIME_HELP_STRING = (
    '[used only if `{0:s}` is left alone] Last valid time (format '
    '"yyyy-mm-dd-HHMMSS").'
).format(VALID_TIMES_ARG_NAME)

OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Images will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTORS_ARG_NAME, type=str, nargs='+', required=False,
    default=neural_net.DEFAULT_SATELLITE_PREDICTOR_NAMES,
    help=PREDICTORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIMES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=VALID_TIMES_HELP_STRING
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


def plot_predictors_one_time(
        example_table_xarray, time_index, predictor_indices, output_dir_name,
        info_string=None):
    """Plots scalar satellite-based predictors for one valid time.

    :param example_table_xarray: xarray table in format returned by
        `example_io.read_file`.
    :param time_index: Index of valid time to plot.
    :param predictor_indices: 1-D numpy array with indices of predictors to
        plot.
    :param output_dir_name: Name of output directory.  Image will be saved here.
    :param info_string: Info string (to be appended to title).
    :return: output_file_name: Path to output file, where image was saved.
    """

    xt = example_table_xarray
    predictor_values = (
        xt[example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY].values[
            time_index, predictor_indices
        ]
    )

    num_predictors = len(predictor_values)
    y_coords = numpy.linspace(
        0, num_predictors - 1, num=num_predictors, dtype=float
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.barh(
        y_coords, predictor_values, color=BAR_FACE_COLOUR,
        edgecolor=BAR_EDGE_COLOUR, linewidth=BAR_EDGE_WIDTH
    )

    pyplot.yticks([], [])
    axes_object.set_xlim(MIN_NORMALIZED_VALUE, MAX_NORMALIZED_VALUE)

    predictor_names = xt.coords[
        example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
    ].values[predictor_indices].tolist()

    for j in range(num_predictors):
        axes_object.text(
            0, y_coords[j], predictor_names[j], color=BAR_FONT_COLOUR,
            horizontalalignment='center', verticalalignment='center',
            fontsize=BAR_FONT_SIZE, fontweight='bold'
        )

    valid_time_unix_sec = (
        xt.coords[example_utils.SATELLITE_TIME_DIM].values[time_index]
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )
    cyclone_id_string = xt[satellite_utils.CYCLONE_ID_KEY].values[time_index]
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    title_string = 'Satellite for {0:s} at {1:s}'.format(
        cyclone_id_string, valid_time_string
    )
    if info_string is not None:
        title_string += '; {0:s}'.format(info_string)

    axes_object.set_title(title_string)

    output_file_name = '{0:s}/scalar_satellite_{1:s}_{2:s}.jpg'.format(
        output_dir_name, cyclone_id_string, valid_time_string
    )
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return output_file_name


def _run(norm_example_file_name, predictor_names, valid_time_strings,
         first_time_string, last_time_string, output_dir_name):
    """Plots normalized values of scalar satellite-based predictors.

    This is effectively the main method.

    :param norm_example_file_name: See documentation at top of file.
    :param predictor_names: Same.
    :param valid_time_strings: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(norm_example_file_name))
    example_table_xarray = example_io.read_file(norm_example_file_name)

    all_predictor_names = (
        example_table_xarray.coords[
            example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
        ].values.tolist()
    )
    predictor_indices = numpy.array(
        [all_predictor_names.index(n) for n in predictor_names], dtype=int
    )

    all_valid_times_unix_sec = (
        example_table_xarray.coords[example_utils.SATELLITE_TIME_DIM].values
    )

    if len(valid_time_strings) == 1 and valid_time_strings[0] == '':
        first_valid_time_unix_sec = time_conversion.string_to_unix_sec(
            first_time_string, TIME_FORMAT
        )
        last_valid_time_unix_sec = time_conversion.string_to_unix_sec(
            last_time_string, TIME_FORMAT
        )
        time_indices = numpy.where(numpy.logical_and(
            all_valid_times_unix_sec >= first_valid_time_unix_sec,
            all_valid_times_unix_sec <= last_valid_time_unix_sec
        ))[0]

        if len(time_indices) == 0:
            error_string = (
                'Cannot find any valid times in file "{0:s}" between {1:s} and '
                '{2:s}.'
            ).format(
                norm_example_file_name, first_time_string, last_time_string
            )

            raise ValueError(error_string)
    else:
        valid_times_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, TIME_FORMAT)
            for t in valid_time_strings
        ], dtype=int)

        time_indices = numpy.array([
            numpy.where(all_valid_times_unix_sec == t)[0][0]
            for t in valid_times_unix_sec
        ], dtype=int)

    for i in time_indices:
        plot_predictors_one_time(
            example_table_xarray=example_table_xarray,
            time_index=i, predictor_indices=predictor_indices,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        norm_example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        predictor_names=getattr(INPUT_ARG_OBJECT, PREDICTORS_ARG_NAME),
        valid_time_strings=getattr(INPUT_ARG_OBJECT, VALID_TIMES_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
