"""Plots satellite time series for 2023 RI (rapid intensification) paper."""

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
import imagemagick_utils
import example_io
import border_io
import normalization
import satellite_utils
import neural_net
import plotting_utils
import predictor_plotting
import satellite_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H'

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
RI_CUTOFFS_M_S01 = KT_TO_METRES_PER_SECOND * numpy.array([30.])

COLOUR_BAR_FONT_SIZE = 20

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

EXAMPLE_DIR_ARG_NAME = 'input_norm_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
INIT_TIME_ARG_NAME = 'init_time_string'
LAG_TIMES_ARG_NAME = 'lag_times_minutes'
PLOT_DIFFS_ARG_NAME = 'plot_temporal_diffs'
OUTPUT_DIR_ARG_NAME = 'output_figure_dir_name'

EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with normalized (and ideally rotated) examples.  The '
    'relevant file (for the given cyclone) will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file.  Will be read by `normalization.read_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Will plot satellite series for this cyclone at the given forecast-'
    'initialization time.'
)
INIT_TIME_HELP_STRING = (
    'Will plot satellite series for the given cyclone at this forecast-'
    'initialization time (format "yyyy-mm-dd-HH").'
)
LAG_TIMES_HELP_STRING = 'List of lag times.  Must include 0.'
PLOT_DIFFS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot temporal differences (raw values) at '
    'each non-zero lag time.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_ARG_NAME, type=str, required=True,
    help=INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAG_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=LAG_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_DIFFS_ARG_NAME, type=int, required=True,
    help=PLOT_DIFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(norm_example_dir_name, normalization_file_name, cyclone_id_string,
         init_time_string, lag_times_minutes, plot_temporal_diffs,
         output_dir_name):
    """Plots satellite time series for 2023 RI (rapid intensification) paper.

    This is effectively the main method.

    :param norm_example_dir_name: See documentation at top of file.
    :param normalization_file_name: Same.
    :param cyclone_id_string: Same.
    :param init_time_string: Same.
    :param lag_times_minutes: Same.
    :param plot_temporal_diffs: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )

    lag_times_minutes = numpy.sort(lag_times_minutes)[::-1]
    num_lag_times = len(lag_times_minutes)
    assert 0 in lag_times_minutes
    assert num_lag_times > 1

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    example_file_name = example_io.find_file(
        directory_name=norm_example_dir_name,
        cyclone_id_string=cyclone_id_string,
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_missing=True
    )

    option_dict = {
        neural_net.EXAMPLE_FILE_KEY: example_file_name,
        neural_net.LEAD_TIMES_KEY: numpy.array([24], dtype=int),
        neural_net.SATELLITE_PREDICTORS_KEY:
            [satellite_utils.BRIGHTNESS_TEMPERATURE_KEY],
        neural_net.SATELLITE_LAG_TIMES_KEY: lag_times_minutes,
        neural_net.SHIPS_GOES_PREDICTORS_KEY: None,
        neural_net.SHIPS_GOES_LAG_TIMES_KEY: None,
        neural_net.SHIPS_FORECAST_PREDICTORS_KEY: None,
        neural_net.SHIPS_MAX_FORECAST_HOUR_KEY: 24,
        neural_net.PREDICT_TD_TO_TS_KEY: False,
        neural_net.SATELLITE_TIME_TOLERANCE_KEY: 0,
        neural_net.SATELLITE_MAX_MISSING_TIMES_KEY: 0,
        neural_net.SHIPS_TIME_TOLERANCE_KEY: 0,
        neural_net.SHIPS_MAX_MISSING_TIMES_KEY: 0,
        neural_net.USE_CLIMO_KEY: False,
        neural_net.CLASS_CUTOFFS_KEY: RI_CUTOFFS_M_S01,
        neural_net.NUM_GRID_ROWS_KEY: None,
        neural_net.NUM_GRID_COLUMNS_KEY: None,
        neural_net.USE_TIME_DIFFS_KEY: False
    }

    data_dict = neural_net.create_inputs(option_dict)
    print(SEPARATOR_STRING)

    example_index = numpy.where(
        data_dict[neural_net.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]

    predictor_matrices_one_example = [
        None if p is None else p[[example_index], ...]
        for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]
    grid_latitude_matrix_deg_n = (
        data_dict[neural_net.GRID_LATITUDE_MATRIX_KEY][example_index, ...]
    )
    grid_longitude_matrix_deg_e = (
        data_dict[neural_net.GRID_LONGITUDE_MATRIX_KEY][example_index, ...]
    )

    (
        figure_objects, axes_objects, pathless_output_file_names
    ) = predictor_plotting.plot_brightness_temp_one_example(
        predictor_matrices_one_example=predictor_matrices_one_example,
        model_metadata_dict={neural_net.VALIDATION_OPTIONS_KEY: option_dict},
        cyclone_id_string=cyclone_id_string,
        init_time_unix_sec=init_time_unix_sec,
        grid_latitude_matrix_deg_n=grid_latitude_matrix_deg_n,
        grid_longitude_matrix_deg_e=grid_longitude_matrix_deg_e,
        normalization_table_xarray=normalization_table_xarray,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        plot_motion_arrow=True,
        plot_time_diffs_at_lags=plot_temporal_diffs
    )

    panel_file_names = [''] * num_lag_times

    for k in range(num_lag_times):
        panel_file_names[k] = '{0:s}/{1:s}'.format(
            output_dir_name, pathless_output_file_names[k]
        )

        print('Saving figure to file: "{0:s}"...'.format(panel_file_names[k]))
        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[k],
            output_file_name=panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    concat_figure_file_name = (
        '{0:s}/{1:s}_{2:s}_brightness_temp_concat.jpg'
    ).format(
        output_dir_name, cyclone_id_string, init_time_string
    )
    plotting_utils.concat_panels(
        panel_file_names=panel_file_names,
        concat_figure_file_name=concat_figure_file_name
    )

    if plot_temporal_diffs:
        this_cmap_object, this_cnorm_object = (
            satellite_plotting.get_diff_colour_scheme()
        )
        plotting_utils.add_colour_bar(
            figure_file_name=concat_figure_file_name,
            colour_map_object=this_cmap_object,
            colour_norm_object=this_cnorm_object,
            orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
            cbar_label_string='Brightness-temp difference (K)',
            tick_label_format_string='{0:d}'
        )

    this_cmap_object, this_cnorm_object = (
        satellite_plotting.get_colour_scheme()
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=this_cmap_object,
        colour_norm_object=this_cnorm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Brightness temp (K)',
        tick_label_format_string='{0:d}'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        norm_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        init_time_string=getattr(INPUT_ARG_OBJECT, INIT_TIME_ARG_NAME),
        lag_times_minutes=numpy.array(
            getattr(INPUT_ARG_OBJECT, LAG_TIMES_ARG_NAME), dtype=int
        ),
        plot_temporal_diffs=bool(
            getattr(INPUT_ARG_OBJECT, PLOT_DIFFS_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
