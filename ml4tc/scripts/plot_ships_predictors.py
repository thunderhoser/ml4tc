"""Plots normalized values of SHIPS-based predictors."""

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
from ml4tc.plotting import plotting_utils
from ml4tc.plotting import ships_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
FIGURE_RESOLUTION_DPI = 300

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

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

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
        time_indices = general_utils.find_exact_times(
            actual_times_unix_sec=all_init_times_unix_sec,
            first_desired_time_unix_sec=first_init_time_unix_sec,
            last_desired_time_unix_sec=last_init_time_unix_sec
        )
    else:
        init_times_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, TIME_FORMAT)
            for t in init_time_strings
        ], dtype=int)

        time_indices = general_utils.find_exact_times(
            actual_times_unix_sec=all_init_times_unix_sec,
            desired_times_unix_sec=init_times_unix_sec
        )

    colour_norm_object = pyplot.Normalize(
        vmin=ships_plotting.MIN_NORMALIZED_VALUE,
        vmax=ships_plotting.MAX_NORMALIZED_VALUE
    )

    for i in time_indices:
        figure_object, _, pathless_file_name = (
            ships_plotting.plot_lagged_predictors_one_init_time(
                example_table_xarray=example_table_xarray, init_time_index=i,
                predictor_indices=lagged_predictor_indices
            )
        )

        extensionless_file_name = '.'.join(
            pathless_file_name.split('.')[:-1]
        )
        figure_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, extensionless_file_name
        )

        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        plotting_utils.add_colour_bar(
            figure_file_name=figure_file_name,
            colour_map_object=ships_plotting.COLOUR_MAP_OBJECT,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            font_size=ships_plotting.DEFAULT_FONT_SIZE,
            cbar_label_string='', tick_label_format_string='{0:.2g}'
        )

        figure_object, _, pathless_file_name = (
            ships_plotting.plot_fcst_predictors_one_init_time(
                example_table_xarray=example_table_xarray, init_time_index=i,
                predictor_indices=forecast_predictor_indices
            )
        )

        extensionless_file_name = '.'.join(
            pathless_file_name.split('.')[:-1]
        )
        figure_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, extensionless_file_name
        )

        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        plotting_utils.add_colour_bar(
            figure_file_name=figure_file_name,
            colour_map_object=ships_plotting.COLOUR_MAP_OBJECT,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            font_size=ships_plotting.DEFAULT_FONT_SIZE,
            cbar_label_string='', tick_label_format_string='{0:.2g}'
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
