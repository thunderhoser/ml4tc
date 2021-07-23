"""Plots satellite images for one cyclone at the given times."""

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
import satellite_io
import border_io
import general_utils
import satellite_utils
import plotting_utils
import satellite_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

DEFAULT_FONT_SIZE = 20
FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

SATELLITE_FILE_ARG_NAME = 'input_satellite_file_name'
VALID_TIMES_ARG_NAME = 'valid_time_strings'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SATELLITE_FILE_HELP_STRING = (
    'Path to file with satellite data for one cyclone.  Will be read by '
    '`satellite_io.read_file`.'
)
VALID_TIMES_HELP_STRING = (
    'List of valid times (format "yyyy-mm-dd-HHMMSS").  The brightness-'
    'temperature map will be plotted for each of these valid times.'
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
    '--' + SATELLITE_FILE_ARG_NAME, type=str, required=True,
    help=SATELLITE_FILE_HELP_STRING
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


def plot_one_satellite_image(
        satellite_table_xarray, time_index, border_latitudes_deg_n,
        border_longitudes_deg_e, output_dir_name, info_string=None):
    """Plots one satellite image.

    P = number of points in border set

    :param satellite_table_xarray: xarray table in format returned by
        `satellite_io.read_file`.
    :param time_index: Index of time to plot.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param output_dir_name: Name of output directory.  Image will be saved here.
        If you do not want to save image right away, make this None.
    :param info_string: Info string (to be appended to title).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :return: output_file_name: Path to output file, where image was saved.  If
        `output_dir_name is None`, this will be a pathless file name.
    """

    t = satellite_table_xarray
    grid_latitudes_deg_n = (
        t[satellite_utils.GRID_LATITUDE_KEY].values[time_index, :]
    )
    grid_longitudes_deg_e = (
        t[satellite_utils.GRID_LONGITUDE_KEY].values[time_index, :]
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )
    brightness_temp_matrix_kelvins = (
        t[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY][time_index, ...].values
    )
    satellite_plotting.plot_2d_grid_regular(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object, latitudes_deg_n=grid_latitudes_deg_n,
        longitudes_deg_e=grid_longitudes_deg_e, font_size=DEFAULT_FONT_SIZE
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=grid_latitudes_deg_n,
        plot_longitudes_deg_e=grid_longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.,
        font_size=DEFAULT_FONT_SIZE
    )

    valid_time_unix_sec = (
        t.coords[satellite_utils.TIME_DIM].values[time_index]
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )
    cyclone_id_string = t[satellite_utils.CYCLONE_ID_KEY].values[time_index]
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    title_string = r'$T_b$ for {0:s} at {1:s}'.format(
        cyclone_id_string, valid_time_string
    )
    if info_string is not None:
        title_string += '; {0:s}'.format(info_string)

    axes_object.set_title(title_string)

    pathless_output_file_name = '{0:s}_{1:s}_brightness_temp.jpg'.format(
        cyclone_id_string, valid_time_string
    )

    if output_dir_name is None:
        return figure_object, axes_object, pathless_output_file_name

    output_file_name = '{0:s}/{1:s}'.format(
        output_dir_name, pathless_output_file_name
    )
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return None, None, output_file_name


def _run(satellite_file_name, valid_time_strings, first_valid_time_string,
         last_valid_time_string, output_dir_name):
    """Plots satellite images for one cyclone at the given times.

    This is effectively the main method.

    :param satellite_file_name: See documentation at top of file.
    :param valid_time_strings: Same.
    :param first_valid_time_string: Same.
    :param last_valid_time_string: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if desired times cannot be found.
    """

    print('Reading data from: "{0:s}"...'.format(satellite_file_name))
    satellite_table_xarray = satellite_io.read_file(satellite_file_name)
    all_valid_times_unix_sec = (
        satellite_table_xarray.coords[satellite_utils.TIME_DIM].values
    )

    if len(valid_time_strings) == 1 and valid_time_strings[0] == '':
        first_valid_time_unix_sec = time_conversion.string_to_unix_sec(
            first_valid_time_string, TIME_FORMAT
        )
        last_valid_time_unix_sec = time_conversion.string_to_unix_sec(
            last_valid_time_string, TIME_FORMAT
        )
        time_indices = general_utils.find_exact_times(
            actual_times_unix_sec=all_valid_times_unix_sec,
            first_desired_time_unix_sec=first_valid_time_unix_sec,
            last_desired_time_unix_sec=last_valid_time_unix_sec
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

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    for i in time_indices:
        plot_one_satellite_image(
            satellite_table_xarray=satellite_table_xarray, time_index=i,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        satellite_file_name=getattr(
            INPUT_ARG_OBJECT, SATELLITE_FILE_ARG_NAME
        ),
        valid_time_strings=getattr(INPUT_ARG_OBJECT, VALID_TIMES_ARG_NAME),
        first_valid_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_valid_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
