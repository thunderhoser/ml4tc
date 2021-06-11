"""Plots histogram for one satellite variable."""

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

import file_system_utils
import error_checking
import satellite_io
import satellite_utils

DEFAULT_FONT_SIZE = 20
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HISTOGRAM_EDGE_WIDTH = 1.5
HISTOGRAM_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIRST_YEAR_ARG_NAME = 'first_year'
LAST_YEAR_ARG_NAME = 'last_year'
VARIABLE_ARG_NAME = 'variable_name'
NUM_PIXELS_PER_TIME_ARG_NAME = 'num_pixels_per_time'
NUM_BINS_ARG_NAME = 'num_bins'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
YEAR_HELP_STRING = (
    'Will use satellite data for the period `{0:s}` to `{1:s}`.'
).format(FIRST_YEAR_ARG_NAME, LAST_YEAR_ARG_NAME)

VARIABLE_HELP_STRING = (
    'Name of satellite variable for which to plot histogram.  This must be the '
    'name used in the files.'
)
NUM_PIXELS_PER_TIME_HELP_STRING = (
    '[used only if variable is brightness temperature] Number of pixels to '
    'randomly sample at each time step.'
)
NUM_BINS_HELP_STRING = 'Number of bins in histogram.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  The histogram will be saved as an image here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_YEAR_ARG_NAME, type=int, required=True,
    help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_YEAR_ARG_NAME, type=int, required=True,
    help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VARIABLE_ARG_NAME, type=str, required=True,
    help=VARIABLE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PIXELS_PER_TIME_ARG_NAME, type=int, required=False, default=10,
    help=NUM_PIXELS_PER_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, required=False, default=100,
    help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _get_histogram(input_values, num_bins):
    """Computes histogram.

    B = number of bins

    :param input_values: 1-D numpy array of input values.
    :param num_bins: Number of bins.
    :return: lower_bin_edges: length-B numpy array of lower bin edges.
    :return: upper_bin_edges: length-B numpy array of upper bin edges.
    :return: frequencies: length-B numpy array of frequencies.
    """

    if numpy.all(numpy.isnan(input_values)):
        return (
            numpy.full(num_bins, numpy.nan),
            numpy.full(num_bins, numpy.nan),
            numpy.full(num_bins, numpy.nan)
        )

    bin_edges = numpy.linspace(
        numpy.nanpercentile(input_values, 0.5),
        numpy.nanpercentile(input_values, 99.5),
        num=num_bins + 1, dtype=float
    )
    bin_edges[0] = numpy.nanmin(input_values)
    bin_edges[-1] = numpy.nanmax(input_values)
    lower_bin_edges = bin_edges[:-1]
    upper_bin_edges = bin_edges[1:]

    frequencies = numpy.full(num_bins, numpy.nan)

    for k in range(num_bins):
        if k == num_bins - 1:
            good_flags = numpy.logical_and(
                input_values >= lower_bin_edges[k],
                input_values <= upper_bin_edges[k]
            )
        else:
            good_flags = numpy.logical_and(
                input_values >= lower_bin_edges[k],
                input_values < upper_bin_edges[k]
            )

        frequencies[k] = numpy.mean(good_flags)

    return lower_bin_edges, upper_bin_edges, frequencies


def _run(input_dir_name, first_year, last_year, variable_name,
         num_pixels_per_time, num_bins, output_file_name):
    """Plots histogram for one satellite variable.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_year: Same.
    :param last_year: Same.
    :param variable_name: Same.
    :param num_pixels_per_time: Same.
    :param num_bins: Same.
    :param output_file_name: Same.
    """

    if variable_name == satellite_utils.BRIGHTNESS_TEMPERATURE_KEY:
        error_checking.assert_is_greater(num_pixels_per_time, 0)
        error_checking.assert_is_leq(num_pixels_per_time, 100)
    else:
        num_pixels_per_time = None

    error_checking.assert_is_geq(last_year, first_year)

    cyclone_id_strings = satellite_io.find_cyclones(
        directory_name=input_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [satellite_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )

    good_indices = numpy.where(numpy.logical_and(
        cyclone_years >= first_year, cyclone_years <= last_year
    ))[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    cyclone_id_strings.sort()

    raw_values = numpy.array([], dtype=float)

    for this_cyclone_id_string in cyclone_id_strings:
        satellite_file_name = satellite_io.find_file(
            directory_name=input_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(satellite_file_name))
        satellite_table_xarray = satellite_io.read_file(satellite_file_name)

        if variable_name == satellite_utils.BRIGHTNESS_TEMPERATURE_KEY:
            brightness_temp_matrix_kelvins = (
                satellite_table_xarray[variable_name].values
            )
            num_times = brightness_temp_matrix_kelvins.shape[0]
            new_raw_values = numpy.random.choice(
                numpy.ravel(brightness_temp_matrix_kelvins),
                size=num_pixels_per_time * num_times, replace=False
            )
        else:
            new_raw_values = numpy.ravel(
                satellite_table_xarray[variable_name].values
            )

        raw_values = numpy.concatenate((raw_values, new_raw_values))

    print(SEPARATOR_STRING)

    lower_bin_edges, upper_bin_edges, frequencies = _get_histogram(
        input_values=raw_values, num_bins=num_bins
    )
    nan_frequency = numpy.mean(numpy.isnan(raw_values))
    del raw_values

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    x_values = numpy.linspace(0.5, 0.5 + num_bins - 1, num=num_bins)
    axes_object.bar(
        x=x_values, height=frequencies, width=1.,
        color=HISTOGRAM_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )

    axes_object.set_ylabel('Frequency of occurrence')

    title_string = 'Histogram for {0:s} (NaN frequency = {1:.4f})'.format(
        variable_name, nan_frequency
    )
    axes_object.set_title(title_string)

    x_tick_labels = [
        '[{0:.3e}, {1:.3e}]'.format(l, u)
        for l, u in zip(lower_bin_edges, upper_bin_edges)
    ]
    x_tick_labels[-1] = x_tick_labels[-1][:-1] + ')'

    label_freq = int(numpy.round(
        float(num_bins) / 20
    ))

    for k in range(num_bins):
        if numpy.mod(k, label_freq) == 0:
            continue

        x_tick_labels[k] = ' '

    axes_object.set_xticks(x_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_year=getattr(INPUT_ARG_OBJECT, FIRST_YEAR_ARG_NAME),
        last_year=getattr(INPUT_ARG_OBJECT, LAST_YEAR_ARG_NAME),
        variable_name=getattr(INPUT_ARG_OBJECT, VARIABLE_ARG_NAME),
        num_pixels_per_time=getattr(
            INPUT_ARG_OBJECT, NUM_PIXELS_PER_TIME_ARG_NAME
        ),
        num_bins=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
