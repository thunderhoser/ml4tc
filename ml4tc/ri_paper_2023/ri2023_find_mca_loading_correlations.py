"""Finds correlations between MCA loading and auxiliary variables.

MCA loading = standardized expansion coefficient for brightness temperature
(not Shapley value), based on maximum-covariance analysis

NOTE: Before running this script, you need to run
ri2023_find_mca_loading_vs_solar_time.py.
"""

import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4tc.io import extended_best_track_io as ebtrk_io
from ml4tc.utils import extended_best_track_utils as ebtrk_utils

HOURS_TO_SECONDS = 3600

TC_SIZE_VARIABLE_NAMES = [
    ebtrk_utils.OUTERMOST_RADIUS_KEY,
    ebtrk_utils.RADII_OF_34KT_WIND_KEY,
    ebtrk_utils.RADII_OF_50KT_WIND_KEY,
    ebtrk_utils.RADII_OF_64KT_WIND_KEY
]

SOLAR_TIME_NAME = 'solar_time_sec'
SHIFTED_SOLAR_TIME_NAME = 'shifted_solar_time_sec'
ABSOLUTE_SOLAR_TIME_NAME = 'absolute_solar_time_sec'

ALL_VARIABLE_NAMES = TC_SIZE_VARIABLE_NAMES + [
    SOLAR_TIME_NAME, SHIFTED_SOLAR_TIME_NAME, ABSOLUTE_SOLAR_TIME_NAME
]

ALL_VARIABLE_NAMES_FANCY = [
    'Radius of outermost\nclosed isobar',
    'Radius of 34-kt wind',
    'Radius of 50-kt wind',
    'Radius of 64-kt wind',
    'LST',
    'Shifted LST',
    'Absolute LST'
]

COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

DEFAULT_FONT_SIZE = 20
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

MODE4_FILE_ARG_NAME = 'input_mode4_file_name'
MODE5_FILE_ARG_NAME = 'input_mode5_file_name'
EBTRK_FILE_ARG_NAME = 'input_extended_best_track_file_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODE4_FILE_HELP_STRING = (
    'Path to file with MCA loading vs. local solar time for mode 4, created by '
    'ri2023_find_mca_loading_vs_solar_time.py.'
)
MODE5_FILE_HELP_STRING = 'Same as {0:s} but for mode 5.'.format(
    MODE4_FILE_ARG_NAME
)
EBTRK_FILE_HELP_STRING = (
    'Path to file with extended best-track data, including TC-size variables.  '
    'Will be read by `extended_best_track_io.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Figure (correlation matrix) will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODE4_FILE_ARG_NAME, type=str, required=True,
    help=MODE4_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODE5_FILE_ARG_NAME, type=str, required=True,
    help=MODE5_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EBTRK_FILE_ARG_NAME, type=str, required=True,
    help=EBTRK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _find_correlations_one_mode(mode_file_name, ebtrk_file_name):
    """Computes correlations for one mode.

    :param mode_file_name: Path to file with MCA loading vs. local solar time
        for one mode, created by ri2023_find_mca_loading_vs_solar_time.py.
    :param ebtrk_file_name: Path to file with extended best-track data,
        including TC-size variables.  Will be read by
        `extended_best_track_io.read_file`.
    :return: correlation_dict_raw_loadings: Dictionary, where each key is the
        name of one auxiliary variable X and the corresponding value is the
        Pearson correlation between MCA loading and X.
    :return: correlation_dict_absolute_loadings: Same but for absolute loadings.
    """

    print('Reading data from: "{0:s}"...'.format(mode_file_name))

    pickle_file_handle = open(mode_file_name, 'rb')
    cyclone_id_strings = pickle.load(pickle_file_handle)
    init_times_unix_sec = pickle.load(pickle_file_handle)
    predictor_expansion_coeffs = pickle.load(pickle_file_handle)
    solar_times_sec = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)

    ebtrk_times_unix_sec = (
        HOURS_TO_SECONDS * ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values
    )
    ebtrk_cyclone_id_strings = (
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
    )

    ebtrk_indices = numpy.array([
        numpy.where(numpy.logical_and(
            ebtrk_times_unix_sec == t, ebtrk_cyclone_id_strings == c
        ))[0][0]
        for t, c in zip(init_times_unix_sec, cyclone_id_strings)
    ], dtype=int)

    correlation_dict_raw_loadings = dict()
    correlation_dict_absolute_loadings = dict()

    for this_key in TC_SIZE_VARIABLE_NAMES:
        if this_key == ebtrk_utils.OUTERMOST_RADIUS_KEY:
            these_ebtrk_values = (
                ebtrk_table_xarray[this_key].values[ebtrk_indices]
            )
        else:
            these_ebtrk_values = numpy.nanmean(
                ebtrk_table_xarray[this_key].values[ebtrk_indices], axis=1
            )

        real_subindices = numpy.where(numpy.isfinite(these_ebtrk_values))[0]

        correlation_dict_raw_loadings[this_key] = numpy.corrcoef(
            these_ebtrk_values[real_subindices],
            predictor_expansion_coeffs[real_subindices]
        )[0, 1]

        correlation_dict_absolute_loadings[this_key] = numpy.corrcoef(
            these_ebtrk_values[real_subindices],
            numpy.absolute(predictor_expansion_coeffs[real_subindices])
        )[0, 1]

    shifted_solar_times_sec = solar_times_sec - 12 * HOURS_TO_SECONDS
    shifted_solar_times_sec[shifted_solar_times_sec < 0] += (
        24 * HOURS_TO_SECONDS
    )
    absolute_solar_times_sec = numpy.absolute(
        solar_times_sec - 12 * HOURS_TO_SECONDS
    )

    correlation_dict_raw_loadings[SOLAR_TIME_NAME] = numpy.corrcoef(
        solar_times_sec,
        predictor_expansion_coeffs
    )[0, 1]
    correlation_dict_absolute_loadings[SOLAR_TIME_NAME] = numpy.corrcoef(
        solar_times_sec,
        numpy.absolute(predictor_expansion_coeffs)
    )[0, 1]

    correlation_dict_raw_loadings[SHIFTED_SOLAR_TIME_NAME] = numpy.corrcoef(
        shifted_solar_times_sec,
        predictor_expansion_coeffs
    )[0, 1]
    correlation_dict_absolute_loadings[SHIFTED_SOLAR_TIME_NAME] = numpy.corrcoef(
        shifted_solar_times_sec,
        numpy.absolute(predictor_expansion_coeffs)
    )[0, 1]

    correlation_dict_raw_loadings[ABSOLUTE_SOLAR_TIME_NAME] = numpy.corrcoef(
        absolute_solar_times_sec,
        predictor_expansion_coeffs
    )[0, 1]
    correlation_dict_absolute_loadings[ABSOLUTE_SOLAR_TIME_NAME] = numpy.corrcoef(
        absolute_solar_times_sec,
        numpy.absolute(predictor_expansion_coeffs)
    )[0, 1]

    return correlation_dict_raw_loadings, correlation_dict_absolute_loadings


def _run(mode4_file_name, mode5_file_name, ebtrk_file_name, output_file_name):
    """Finds correlations between MCA loading and auxiliary variables.

    This is effectively the main method.

    :param mode4_file_name: See documentation at top of file.
    :param mode5_file_name: Same.
    :param ebtrk_file_name: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    (
        correlation_dict_mode4_raw_loadings,
        correlation_dict_mode4_absolute_loadings
    ) = _find_correlations_one_mode(
        mode_file_name=mode4_file_name,
        ebtrk_file_name=ebtrk_file_name
    )

    (
        correlation_dict_mode5_raw_loadings,
        correlation_dict_mode5_absolute_loadings
    ) = _find_correlations_one_mode(
        mode_file_name=mode5_file_name,
        ebtrk_file_name=ebtrk_file_name
    )

    num_variables = len(ALL_VARIABLE_NAMES)
    correlation_matrix = numpy.full((num_variables, 4), numpy.nan)

    for j in range(num_variables):
        correlation_matrix[j, 0] = correlation_dict_mode4_raw_loadings[
            ALL_VARIABLE_NAMES[j]
        ]
        correlation_matrix[j, 1] = correlation_dict_mode4_absolute_loadings[
            ALL_VARIABLE_NAMES[j]
        ]
        correlation_matrix[j, 2] = correlation_dict_mode5_raw_loadings[
            ALL_VARIABLE_NAMES[j]
        ]
        correlation_matrix[j, 3] = correlation_dict_mode5_absolute_loadings[
            ALL_VARIABLE_NAMES[j]
        ]

    y_tick_labels = ALL_VARIABLE_NAMES_FANCY
    x_tick_labels = [
        'Mode-4 EC', 'Absolute mode-4 EC', 'Mode-5 EC', 'Absolute mode-5 EC'
    ]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    min_colour_value = numpy.min(correlation_matrix)
    max_colour_value = numpy.max(correlation_matrix)
    median_colour_value = 0.5 * (min_colour_value + max_colour_value)
    axes_object.imshow(
        correlation_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    x_tick_values = numpy.linspace(
        0, len(x_tick_labels) - 1, num=len(x_tick_labels), dtype=int
    )
    y_tick_values = numpy.linspace(
        0, len(y_tick_labels) - 1, num=len(y_tick_labels), dtype=int
    )
    pyplot.xticks(x_tick_values, x_tick_labels, rotation=90.)
    pyplot.yticks(y_tick_values, y_tick_labels)

    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            axes_object.text(
                j, i,
                '{0:.2f}'.format(correlation_matrix[i, j]).replace('0.', '.'),
                fontsize=45,
                color=(
                    numpy.full(3, 0.)
                    if correlation_matrix[i, j] > median_colour_value
                    else numpy.full(3, 1.)
                ),
                horizontalalignment='center', verticalalignment='center'
            )

    gg_plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=correlation_matrix,
        colour_map_object=COLOUR_MAP_OBJECT,
        min_value=min_colour_value, max_value=max_colour_value,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=0.8, font_size=30
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        mode4_file_name=getattr(INPUT_ARG_OBJECT, MODE4_FILE_ARG_NAME),
        mode5_file_name=getattr(INPUT_ARG_OBJECT, MODE5_FILE_ARG_NAME),
        ebtrk_file_name=getattr(INPUT_ARG_OBJECT, EBTRK_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
