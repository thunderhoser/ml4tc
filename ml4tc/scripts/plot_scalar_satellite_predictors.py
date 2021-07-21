"""Plots normalized values of scalar satellite-based predictors."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from ml4tc.io import example_io
from ml4tc.utils import general_utils
from ml4tc.utils import example_utils
from ml4tc.machine_learning import neural_net
from ml4tc.plotting import scalar_satellite_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
FIGURE_RESOLUTION_DPI = 300

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

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

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
        first_time_unix_sec = time_conversion.string_to_unix_sec(
            first_time_string, TIME_FORMAT
        )
        last_time_unix_sec = time_conversion.string_to_unix_sec(
            last_time_string, TIME_FORMAT
        )
        time_indices = general_utils.find_exact_times(
            actual_times_unix_sec=all_valid_times_unix_sec,
            first_desired_time_unix_sec=first_time_unix_sec,
            last_desired_time_unix_sec=last_time_unix_sec
        )
    else:
        valid_times_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, TIME_FORMAT)
            for t in valid_time_strings
        ], dtype=int)

        time_indices = general_utils.find_exact_times(
            actual_times_unix_sec=all_valid_times_unix_sec,
            desired_times_unix_sec=valid_times_unix_sec
        )

    for i in time_indices:
        figure_object, _, pathless_output_file_name = (
            scalar_satellite_plotting.plot_bar_graph_one_time(
                example_table_xarray=example_table_xarray, time_index=i,
                predictor_indices=predictor_indices
            )
        )

        output_file_name = '{0:s}/{1:s}'.format(
            output_dir_name, pathless_output_file_name
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
        norm_example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        predictor_names=getattr(INPUT_ARG_OBJECT, PREDICTORS_ARG_NAME),
        valid_time_strings=getattr(INPUT_ARG_OBJECT, VALID_TIMES_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
