"""Plots PCA (principal-component analysis) for gridded Shapley values."""

import os
import sys
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_general_utils
import file_system_utils
import error_checking
import plotting_utils
import satellite_plotting
import run_pca_for_shapley_maps as run_pca

TOLERANCE = 1e-10
COLOUR_BAR_FONT_SIZE = 30

DEFAULT_FONT_SIZE = 20
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_pca_file_name'
NUM_MODES_ARG_NAME = 'num_modes_to_plot'
SHAPLEY_COLOUR_MAP_ARG_NAME = 'shapley_colour_map_name'
SHAPLEY_NUM_CONTOURS_ARG_NAME = 'shapley_half_num_contours'
SHAPLEY_MIN_PERCENTILE_ARG_NAME = 'shapley_min_colour_percentile'
SHAPLEY_MAX_PERCENTILE_ARG_NAME = 'shapley_max_colour_percentile'
PREDICTOR_COLOUR_MAP_ARG_NAME = 'predictor_colour_map_name'
PREDICTOR_MIN_PERCENTILE_ARG_NAME = 'predictor_min_colour_percentile'
PREDICTOR_MAX_PERCENTILE_ARG_NAME = 'predictor_max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing PCA results.  Will be read by xarray.'
)
NUM_MODES_HELP_STRING = (
    'Will plot the top K modes, where K = {0:s}.  For each mode, will plot '
    'both Shapley values and predictor values regressed onto the singular '
    'vector for said mode.'
).format(NUM_MODES_ARG_NAME)

SHAPLEY_COLOUR_MAP_HELP_STRING = (
    'Name of colour scheme for Shapley values.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
SHAPLEY_NUM_CONTOURS_HELP_STRING = (
    'Number of contours for Shapley values of each sign (positive or '
    'negative).  Total number of contours will be 2 * {0:s}.'
).format(SHAPLEY_NUM_CONTOURS_ARG_NAME)

SHAPLEY_MIN_PERCENTILE_HELP_STRING = (
    'Determines min value in colour scheme for Shapley values.  For each map '
    '(i.e., spatial grid), min value in colour scheme will be [q]th percentile '
    'of all Shapley values in map, where q = {0:s}, ranging from 0...100.'
).format(SHAPLEY_MIN_PERCENTILE_ARG_NAME)

SHAPLEY_MAX_PERCENTILE_HELP_STRING = (
    'Same as {0:s} but for max value in colour scheme.'
).format(SHAPLEY_MIN_PERCENTILE_ARG_NAME)

PREDICTOR_COLOUR_MAP_HELP_STRING = (
    'Same as {0:s} but for predictors.'
).format(SHAPLEY_COLOUR_MAP_ARG_NAME)

PREDICTOR_MIN_PERCENTILE_HELP_STRING = (
    'Same as {0:s} but for predictors.'
).format(SHAPLEY_MIN_PERCENTILE_ARG_NAME)

PREDICTOR_MAX_PERCENTILE_HELP_STRING = (
    'Same as {0:s} but for predictors.'
).format(SHAPLEY_MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_MODES_ARG_NAME, type=int, required=True,
    help=NUM_MODES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default='binary', help=SHAPLEY_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_NUM_CONTOURS_ARG_NAME, type=int, required=False,
    default=10, help=SHAPLEY_NUM_CONTOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_MIN_PERCENTILE_ARG_NAME, type=float, required=False,
    default=0.5, help=SHAPLEY_MIN_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99.5, help=SHAPLEY_MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default='seismic', help=PREDICTOR_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_MIN_PERCENTILE_ARG_NAME, type=float, required=False,
    default=0.5, help=PREDICTOR_MIN_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99.5, help=PREDICTOR_MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_mode(
        regressed_predictor_matrix, regressed_shapley_matrix, mode_index,
        explained_variance_fraction, shapley_colour_map_object,
        shapley_half_num_contours, shapley_min_colour_percentile,
        shapley_max_colour_percentile, predictor_colour_map_object,
        predictor_min_colour_percentile, predictor_max_colour_percentile,
        output_file_name):
    """Plots one mode.

    Specifically, for a given mode, plots predictor values and Shapley values
    regressed onto the singular vector for said mode.

    M = number of rows in grid
    N = number of columns in grid

    :param regressed_predictor_matrix: M-by-N numpy array of regressed predictor
        values.
    :param regressed_shapley_matrix: M-by-N numpy array of regressed Shapley
        values.
    :param mode_index: Zero-based mode index.  If k, this means that we are
        plotting the [k + 1]th mode.
    :param explained_variance_fraction: Fraction of variance explained by this
        mode.
    :param shapley_colour_map_object: See documentation at top of file.
    :param shapley_half_num_contours: Same.
    :param shapley_min_colour_percentile: Same.
    :param shapley_max_colour_percentile: Same.
    :param predictor_colour_map_object: Same.
    :param predictor_min_colour_percentile: Same.
    :param predictor_max_colour_percentile: Same.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    grid_latitudes_deg_n = numpy.linspace(
        -10, 10, num=regressed_shapley_matrix.shape[0], dtype=float
    )
    grid_longitudes_deg_e = numpy.linspace(
        -10, 10, num=regressed_shapley_matrix.shape[1], dtype=float
    )

    predictor_min_colour_value = numpy.percentile(
        regressed_predictor_matrix, predictor_min_colour_percentile
    )
    predictor_max_colour_value = numpy.percentile(
        regressed_predictor_matrix, predictor_max_colour_percentile
    )
    predictor_colour_norm_object = pyplot.Normalize(
        vmin=predictor_min_colour_value, vmax=predictor_max_colour_value
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    satellite_plotting.plot_2d_grid(
        brightness_temp_matrix_kelvins=regressed_predictor_matrix,
        axes_object=axes_object,
        latitude_array_deg_n=grid_latitudes_deg_n,
        longitude_array_deg_e=grid_longitudes_deg_e,
        plot_motion_arrow=True,
        colour_map_object=predictor_colour_map_object,
        colour_norm_object=predictor_colour_norm_object,
        cbar_orientation_string=None
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=numpy.ravel(grid_latitudes_deg_n),
        plot_longitudes_deg_e=numpy.ravel(grid_longitudes_deg_e),
        axes_object=axes_object, parallel_spacing_deg=2.,
        meridian_spacing_deg=2., font_size=DEFAULT_FONT_SIZE
    )

    shapley_min_colour_value = numpy.percentile(
        numpy.absolute(regressed_shapley_matrix), shapley_min_colour_percentile
    )
    shapley_max_colour_value = numpy.percentile(
        numpy.absolute(regressed_shapley_matrix), shapley_max_colour_percentile
    )
    shapley_colour_norm_object = pyplot.Normalize(
        vmin=shapley_min_colour_value, vmax=shapley_max_colour_value
    )

    satellite_plotting.plot_saliency(
        saliency_matrix=regressed_shapley_matrix,
        axes_object=axes_object,
        latitude_array_deg_n=grid_latitudes_deg_n,
        longitude_array_deg_e=grid_longitudes_deg_e,
        min_abs_contour_value=shapley_min_colour_value,
        max_abs_contour_value=shapley_max_colour_value,
        half_num_contours=shapley_half_num_contours,
        colour_map_object=shapley_colour_map_object
    )

    title_string = (
        r'$T_b$ and Shapley values regressed onto {0:d}th EOF'
    ).format(mode_index + 1)

    title_string += '\n(explained variance = {0:.1f}%)'.format(
        100 * explained_variance_fraction
    )
    axes_object.set_title(title_string)

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=predictor_colour_map_object,
        colour_norm_object=predictor_colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Brightness temperature',
        tick_label_format_string='{0:.2g}'
    )

    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=shapley_colour_map_object,
        colour_norm_object=shapley_colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Absolute Shapley value',
        tick_label_format_string='{0:.2g}'
    )


def _run(pca_file_name, num_modes_to_plot, shapley_colour_map_name,
         shapley_half_num_contours, shapley_min_colour_percentile,
         shapley_max_colour_percentile, predictor_colour_map_name,
         predictor_min_colour_percentile, predictor_max_colour_percentile,
         output_dir_name):
    """Plots PCA (principal-component analysis) for gridded Shapley values.

    This is effectively the main method.

    :param pca_file_name: See documentation at top of file.
    :param num_modes_to_plot: Same.
    :param shapley_colour_map_name: Same.
    :param shapley_half_num_contours: Same.
    :param shapley_min_colour_percentile: Same.
    :param shapley_max_colour_percentile: Same.
    :param predictor_colour_map_name: Same.
    :param predictor_min_colour_percentile: Same.
    :param predictor_max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_greater(num_modes_to_plot, 0)
    error_checking.assert_is_geq(shapley_half_num_contours, 5)
    error_checking.assert_is_geq(shapley_min_colour_percentile, 0.)
    error_checking.assert_is_leq(shapley_min_colour_percentile, 5.)
    error_checking.assert_is_geq(shapley_max_colour_percentile, 95.)
    error_checking.assert_is_leq(shapley_max_colour_percentile, 100.)

    error_checking.assert_is_geq(predictor_min_colour_percentile, 0.)
    error_checking.assert_is_leq(predictor_min_colour_percentile, 5.)
    error_checking.assert_is_geq(predictor_max_colour_percentile, 95.)
    error_checking.assert_is_leq(predictor_max_colour_percentile, 100.)

    shapley_colour_map_object = pyplot.get_cmap(shapley_colour_map_name)
    predictor_colour_map_object = pyplot.get_cmap(predictor_colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(pca_file_name))
    pca_table_xarray = xarray.open_dataset(pca_file_name)

    regressed_shapley_matrix = (
        pca_table_xarray[run_pca.REGRESSED_SHAPLEY_VALUE_KEY].values[
            :num_modes_to_plot, ...
        ]
    )
    regressed_predictor_matrix = (
        pca_table_xarray[run_pca.REGRESSED_PREDICTOR_KEY].values[
            :num_modes_to_plot, ...
        ]
    )
    eigenvalues = pca_table_xarray[run_pca.EIGENVALUE_KEY].values
    del pca_table_xarray

    num_modes_to_plot = regressed_shapley_matrix.shape[0]

    for i in range(num_modes_to_plot):
        regressed_shapley_matrix[i, ...] = (
            gg_general_utils.apply_gaussian_filter(
                input_matrix=regressed_shapley_matrix[i, ...],
                e_folding_radius_grid_cells=2.
            )
        )

        _plot_one_mode(
            regressed_predictor_matrix=regressed_predictor_matrix[i, ...],
            regressed_shapley_matrix=regressed_shapley_matrix[i, ...],
            mode_index=i,
            explained_variance_fraction=eigenvalues[i] / numpy.sum(eigenvalues),
            shapley_colour_map_object=shapley_colour_map_object,
            shapley_half_num_contours=shapley_half_num_contours,
            shapley_min_colour_percentile=shapley_min_colour_percentile,
            shapley_max_colour_percentile=shapley_max_colour_percentile,
            predictor_colour_map_object=predictor_colour_map_object,
            predictor_min_colour_percentile=predictor_min_colour_percentile,
            predictor_max_colour_percentile=predictor_max_colour_percentile,
            output_file_name=
            '{0:s}/regressed_shapley_and_predictors_mode{1:03d}.jpg'.format(
                output_dir_name, i + 1
            )
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        pca_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_modes_to_plot=getattr(INPUT_ARG_OBJECT, NUM_MODES_ARG_NAME),
        shapley_colour_map_name=getattr(
            INPUT_ARG_OBJECT, SHAPLEY_COLOUR_MAP_ARG_NAME
        ),
        shapley_half_num_contours=getattr(
            INPUT_ARG_OBJECT, SHAPLEY_NUM_CONTOURS_ARG_NAME
        ),
        shapley_min_colour_percentile=getattr(
            INPUT_ARG_OBJECT, SHAPLEY_MIN_PERCENTILE_ARG_NAME
        ),
        shapley_max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, SHAPLEY_MAX_PERCENTILE_ARG_NAME
        ),
        predictor_colour_map_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_COLOUR_MAP_ARG_NAME
        ),
        predictor_min_colour_percentile=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_MIN_PERCENTILE_ARG_NAME
        ),
        predictor_max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_MAX_PERCENTILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )