"""Makes SHIPS-overview figure for 2023 RI (rapid intensification) paper."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4tc.io import example_io
from ml4tc.utils import example_utils
from ml4tc.utils import general_utils
from ml4tc.machine_learning import neural_net
from ml4tc.plotting import plotting_utils
from ml4tc.plotting import ships_plotting
from ml4tc.plotting import predictor_plotting

TIME_FORMAT = '%Y-%m-%d-%H'

MAX_FORECAST_HOUR = 24
SATELLITE_LAG_TIMES_HOURS = numpy.array([0, 1.5, 3, numpy.inf, numpy.nan])

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
RAPID_INTENSIFN_CUTOFF_M_S01 = 30 * KT_TO_METRES_PER_SECOND

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FONT_SIZE = 30
COLOUR_BAR_FONT_SIZE = 20

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

EXAMPLE_FILE_ARG_NAME = 'input_norm_example_file_name'
FORECAST_PREDICTORS_ARG_NAME = 'ships_forecast_predictor_names'
LAGGED_PREDICTORS_ARG_NAME = 'ships_lagged_predictor_names'
INIT_TIME_ARG_NAME = 'init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = (
    'Path to file with normalized learning examples for one cyclone.  Will be '
    'read by `example_io.read_file`.'
)
FORECAST_PREDICTORS_HELP_STRING = (
    'List with names of forecast (i.e., environmental and historical) SHIPS '
    'predictors.'
)
LAGGED_PREDICTORS_HELP_STRING = (
    'List with names of lagged (i.e., satellite-based) SHIPS predictors.'
)
INIT_TIME_HELP_STRING = 'Forecast-init time (format "yyyy-mm-dd-HH").'
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Images will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FORECAST_PREDICTORS_ARG_NAME, type=str, nargs='+', required=True,
    help=FORECAST_PREDICTORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAGGED_PREDICTORS_ARG_NAME, type=str, nargs='+', required=True,
    help=LAGGED_PREDICTORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_ARG_NAME, type=str, required=True,
    help=INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(norm_example_file_name, ships_forecast_predictor_names,
         ships_lagged_predictor_names, init_time_string, output_dir_name):
    """Makes SHIPS-overview figure for 2023 RI (rapid intensification) paper.

    This is effectively the main method.

    :param norm_example_file_name: See documentation at top of file.
    :param ships_forecast_predictor_names: Same.
    :param ships_lagged_predictor_names: Same.
    :param init_time_string: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    generator_option_dict = {
        neural_net.EXAMPLE_FILE_KEY: norm_example_file_name,
        neural_net.LEAD_TIMES_KEY: numpy.array([24], dtype=int),
        neural_net.SATELLITE_PREDICTORS_KEY: None,
        neural_net.SATELLITE_LAG_TIMES_KEY: numpy.array([0], dtype=int),
        neural_net.SHIPS_GOES_LAG_TIMES_KEY: numpy.array([0], dtype=int),
        neural_net.SHIPS_GOES_PREDICTORS_KEY: ships_lagged_predictor_names,
        neural_net.SHIPS_FORECAST_PREDICTORS_KEY:
            ships_forecast_predictor_names,
        neural_net.SHIPS_MAX_FORECAST_HOUR_KEY: MAX_FORECAST_HOUR,
        neural_net.NUM_POSITIVE_EXAMPLES_KEY: 2,
        neural_net.NUM_NEGATIVE_EXAMPLES_KEY: 2,
        neural_net.MAX_EXAMPLES_PER_CYCLONE_KEY: 2,
        neural_net.PREDICT_TD_TO_TS_KEY: False,
        neural_net.CLASS_CUTOFFS_KEY:
            numpy.array([RAPID_INTENSIFN_CUTOFF_M_S01]),
        neural_net.NUM_GRID_ROWS_KEY: 380,
        neural_net.NUM_GRID_COLUMNS_KEY: 540,
        neural_net.USE_TIME_DIFFS_KEY: False,
        neural_net.SATELLITE_TIME_TOLERANCE_KEY: 86400,
        neural_net.SATELLITE_MAX_MISSING_TIMES_KEY: 1,
        neural_net.SHIPS_TIME_TOLERANCE_KEY: 0,
        neural_net.SHIPS_MAX_MISSING_TIMES_KEY: 0,
        neural_net.USE_CLIMO_KEY: False,
        neural_net.DATA_AUG_NUM_TRANS_KEY: 0,
        neural_net.DATA_AUG_NUM_ROTATIONS_KEY: 0,
        neural_net.DATA_AUG_NUM_NOISINGS_KEY: 0
    }

    print('Reading data from: "{0:s}"...'.format(norm_example_file_name))
    example_table_xarray = example_io.read_file(norm_example_file_name)
    cyclone_id_string = (
        example_table_xarray[example_utils.SHIPS_CYCLONE_ID_KEY].values[0]
    )
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    data_dict = neural_net.create_inputs(generator_option_dict)
    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    all_init_times_unix_sec = data_dict[neural_net.INIT_TIMES_KEY]

    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )
    time_index = general_utils.find_exact_times(
        actual_times_unix_sec=all_init_times_unix_sec,
        desired_times_unix_sec=numpy.array([init_time_unix_sec], dtype=int)
    )[0]
    predictor_matrices = [
        None if m is None else m[[time_index], ...] for m in predictor_matrices
    ]

    forecast_hours = numpy.linspace(
        0, MAX_FORECAST_HOUR,
        num=int(numpy.round(MAX_FORECAST_HOUR / 6)) + 1, dtype=int
    )

    (
        figure_objects, axes_objects, _
    ) = predictor_plotting.plot_lagged_ships_one_example(
        predictor_matrices_one_example=predictor_matrices,
        model_metadata_dict=
        {neural_net.VALIDATION_OPTIONS_KEY: generator_option_dict},
        cyclone_id_string=cyclone_id_string,
        forecast_hours=forecast_hours,
        init_time_unix_sec=init_time_unix_sec
    )

    figure_object = figure_objects[0]
    axes_object = axes_objects[0]
    axes_object.set_title('Satellite-based SHIPS predictors')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')

    panel_file_names = [
        '{0:s}/satellite_based_predictors.jpg'.format(output_dir_name)
    ]
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.resize_image(
        input_file_name=panel_file_names[-1],
        output_file_name=panel_file_names[-1],
        output_size_pixels=PANEL_SIZE_PX
    )

    (
        figure_objects, axes_objects, _
    ) = predictor_plotting.plot_forecast_ships_one_example(
        predictor_matrices_one_example=predictor_matrices,
        model_metadata_dict=
        {neural_net.VALIDATION_OPTIONS_KEY: generator_option_dict},
        cyclone_id_string=cyclone_id_string,
        forecast_hours=forecast_hours,
        init_time_unix_sec=init_time_unix_sec
    )

    figure_object = figure_objects[0]
    axes_object = axes_objects[0]
    axes_object.set_title('Environmental and storm-based SHIPS predictors')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    panel_file_names.append(
        '{0:s}/enviro_and_hist_predictors.jpg'.format(output_dir_name)
    )
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.resize_image(
        input_file_name=panel_file_names[-1],
        output_file_name=panel_file_names[-1],
        output_size_pixels=PANEL_SIZE_PX
    )

    for this_file_name in panel_file_names:
        imagemagick_utils.resize_image(
            input_file_name=this_file_name, output_file_name=this_file_name,
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/ships_overview.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=1
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name, border_width_pixels=10
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    colour_norm_object = pyplot.Normalize(
        vmin=ships_plotting.MIN_NORMALIZED_VALUE,
        vmax=ships_plotting.MAX_NORMALIZED_VALUE
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=ships_plotting.COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='', tick_label_format_string='{0:.2g}'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        norm_example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        ships_forecast_predictor_names=getattr(
            INPUT_ARG_OBJECT, FORECAST_PREDICTORS_ARG_NAME
        ),
        ships_lagged_predictor_names=getattr(
            INPUT_ARG_OBJECT, LAGGED_PREDICTORS_ARG_NAME
        ),
        init_time_string=getattr(INPUT_ARG_OBJECT, INIT_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
