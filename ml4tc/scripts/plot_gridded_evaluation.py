"""Plots evaluation scores on grid."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import radar_plotting
from ml4tc.io import border_io
from ml4tc.io import prediction_io
from ml4tc.utils import evaluation
from ml4tc.plotting import plotting_utils

TOLERANCE = 1e-6
DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'

MAX_COLOUR_PERCENTILE = 99.

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 50
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
GRID_METAFILE_ARG_NAME = 'input_grid_metafile_name'
TOTAL_VALIDN_EVAL_FILE_ARG_NAME = 'input_total_validn_eval_file_name'
SEQ_COLOUR_MAP_ARG_NAME = 'sequential_colour_map_name'
DIV_COLOUR_MAP_ARG_NAME = 'diverging_colour_map_name'
BIAS_COLOUR_MAP_ARG_NAME = 'bias_colour_map_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with evaluation files (one for each grid cell).  Files '
    'will be found by `evaluation.find_file` and read by '
    '`evaluation.read_file`.'
)
GRID_METAFILE_HELP_STRING = (
    'Path to file with grid metadata.  Will be read by '
    '`prediction_io.read_grid_metafile`.'
)
TOTAL_VALIDN_EVAL_FILE_HELP_STRING = (
    'Path to evaluation file for total validation set.  Will be read by '
    '`evaluation.read_file` and used to determine best probability threshold.'
)
SEQ_COLOUR_MAP_HELP_STRING = (
    'Name of sequential colour map (must be accepted by '
    '`matplotlib.pyplot.get_cmap`).  Will be used for POD, success ratio, CSI, '
    'Brier score, and climatological event frequency.'
)
DIV_COLOUR_MAP_HELP_STRING = (
    'Name of diverging colour map (must be accepted by '
    '`matplotlib.pyplot.get_cmap`).  Will be used for Brier skill score.'
)
BIAS_COLOUR_MAP_HELP_STRING = (
    'Name of colour map for frequency bias (must be accepted by '
    '`matplotlib.pyplot.get_cmap`).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GRID_METAFILE_ARG_NAME, type=str, required=True,
    help=GRID_METAFILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TOTAL_VALIDN_EVAL_FILE_ARG_NAME, type=str, required=True,
    help=TOTAL_VALIDN_EVAL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SEQ_COLOUR_MAP_ARG_NAME, type=str, required=False, default='plasma',
    help=SEQ_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DIV_COLOUR_MAP_ARG_NAME, type=str, required=False, default='seismic',
    help=DIV_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BIAS_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default='seismic', help=BIAS_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_bias_colour_scheme(colour_map_name, max_colour_value):
    """Returns colour scheme for frequency bias.

    :param colour_map_name: Name of colour scheme (must be accepted by
        `matplotlib.pyplot.get_cmap`).
    :param max_colour_value: Max value in colour scheme.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    orig_colour_map_object = pyplot.get_cmap(colour_map_name)

    negative_values = numpy.linspace(0, 1, num=1001, dtype=float)
    positive_values = numpy.linspace(1, max_colour_value, num=1001, dtype=float)
    bias_values = numpy.concatenate((negative_values, positive_values))

    normalized_values = numpy.linspace(0, 1, num=len(bias_values), dtype=float)
    rgb_matrix = orig_colour_map_object(normalized_values)[:, :-1]

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        bias_values, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _plot_one_score(
        score_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e, colour_map_name,
        is_frequency_bias, is_bss, output_file_name,
        title_string=None, panel_letter=None):
    """Plots one score on 2-D georeferenced grid.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border set

    :param score_matrix: M-by-N numpy array of scores.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param colour_map_name: Name of colour scheme (must be accepted by
        `matplotlib.pyplot.get_cmap`).
    :param is_frequency_bias: Boolean flag.  If True, the score being plotted is
        frequency bias.
    :param is_bss: Boolean flag.  If True, Brier skill score is being plotted.
    :param output_file_name: Path to output file (figure will be saved here).
    :param title_string: Title (will be added above figure).  If you do not want
        a title, make this None.
    :param panel_letter: Panel letter.  For example, if the letter is "a", will
        add "(a)" at top-left of figure, assuming that it will eventually be a
        panel in a larger figure.  If you do not want a panel letter, make this
        None.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    if is_frequency_bias:
        this_offset = numpy.percentile(
            numpy.absolute(score_matrix[numpy.isfinite(score_matrix)] - 1.),
            MAX_COLOUR_PERCENTILE
        )
        min_colour_value = 0.
        max_colour_value = 1. + this_offset

        colour_map_object, colour_norm_object = _get_bias_colour_scheme(
            colour_map_name=colour_map_name, max_colour_value=max_colour_value
        )
    else:
        colour_map_object = pyplot.get_cmap(colour_map_name)

        if is_bss:
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(score_matrix), MAX_COLOUR_PERCENTILE
            )
            max_colour_value = min([max_colour_value, 1.])
            min_colour_value = -1 * max_colour_value
        else:
            max_colour_value = numpy.nanpercentile(
                score_matrix, MAX_COLOUR_PERCENTILE
            )
            min_colour_value = numpy.nanpercentile(
                score_matrix, 100. - MAX_COLOUR_PERCENTILE
            )

        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )

    radar_plotting.plot_latlng_grid(
        field_matrix=score_matrix, field_name=DUMMY_FIELD_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(grid_latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(grid_longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(grid_latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(grid_longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    if is_frequency_bias:
        colour_bar_object = gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=score_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical', extend_min=False, extend_max=True,
            font_size=FONT_SIZE
        )

        tick_values = colour_bar_object.get_ticks()
        tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    else:
        colour_bar_object = gg_plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=score_matrix,
            colour_map_object=colour_map_object,
            min_value=min_colour_value, max_value=max_colour_value,
            orientation_string='vertical',
            extend_min=is_bss or min_colour_value > TOLERANCE,
            extend_max=max_colour_value < 1. - TOLERANCE,
            font_size=FONT_SIZE
        )

        tick_values = colour_bar_object.get_ticks()
        tick_strings = ['{0:.2g}'.format(v) for v in tick_values]

    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=grid_latitudes_deg_n,
        plot_longitudes_deg_e=grid_longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2., font_size=FONT_SIZE
    )

    if title_string is not None:
        axes_object.set_title(title_string)

    if panel_letter is not None:
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(panel_letter)
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(evaluation_dir_name, grid_metafile_name, total_validn_eval_file_name,
         sequential_colour_map_name, diverging_colour_map_name,
         bias_colour_map_name, output_dir_name):
    """Plots evaluation scores on grid.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param grid_metafile_name: Same.
    :param total_validn_eval_file_name: Same.
    :param sequential_colour_map_name: Same.
    :param diverging_colour_map_name: Same.
    :param bias_colour_map_name: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(total_validn_eval_file_name))
    this_table_xarray = evaluation.read_file(total_validn_eval_file_name)
    probability_threshold = evaluation.find_best_threshold(this_table_xarray)

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read metadata for grid.
    print('Reading grid metadata from: "{0:s}"...'.format(grid_metafile_name))
    grid_latitudes_deg_n, grid_longitudes_deg_e = (
        prediction_io.read_grid_metafile(grid_metafile_name)
    )

    num_grid_rows = len(grid_latitudes_deg_n)
    num_grid_columns = len(grid_longitudes_deg_e)

    # Read evaluation files.
    dimensions = (num_grid_rows, num_grid_columns)
    brier_score_matrix = numpy.full(dimensions, numpy.nan)
    bss_matrix = numpy.full(dimensions, numpy.nan)
    event_freq_matrix = numpy.full(dimensions, numpy.nan)
    mean_prob_matrix = numpy.full(dimensions, numpy.nan)
    pod_matrix = numpy.full(dimensions, numpy.nan)
    far_matrix = numpy.full(dimensions, numpy.nan)
    frequency_bias_matrix = numpy.full(dimensions, numpy.nan)
    csi_matrix = numpy.full(dimensions, numpy.nan)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_file_name = evaluation.find_file(
                directory_name=evaluation_dir_name, grid_row=i, grid_column=j,
                raise_error_if_missing=False
            )

            if not os.path.isfile(this_file_name):
                continue

            print('Reading data from: "{0:s}"...'.format(this_file_name))
            et = evaluation.read_file(this_file_name)

            brier_score_matrix[i, j] = numpy.mean(
                et[evaluation.BRIER_SCORE_KEY].values
            )
            bss_matrix[i, j] = numpy.mean(
                et[evaluation.BRIER_SKILL_SCORE_KEY].values
            )

            event_freq_by_bin = numpy.nanmean(
                et[evaluation.MEAN_OBSERVATION_KEY].values, axis=1
            )
            event_freq_matrix[i, j] = numpy.average(
                event_freq_by_bin,
                weights=et[evaluation.EXAMPLE_COUNT_NO_BS_KEY].values
            )
            mean_prob_matrix[i, j] = numpy.average(
                et[evaluation.MEAN_PREDICTION_NO_BS_KEY].values,
                weights=et[evaluation.EXAMPLE_COUNT_NO_BS_KEY].values
            )

            all_prob_thresholds = (
                et.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values
            )
            threshold_index = numpy.argmin(numpy.absolute(
                all_prob_thresholds - probability_threshold
            ))
            threshold_diff = numpy.absolute(
                all_prob_thresholds[threshold_index] - probability_threshold
            )
            assert threshold_diff <= TOLERANCE

            pod_matrix[i, j] = numpy.mean(
                et[evaluation.POD_KEY].values[threshold_index, :]
            )
            far_matrix[i, j] = 1. - numpy.mean(
                et[evaluation.SUCCESS_RATIO_KEY].values[threshold_index, :]
            )
            frequency_bias_matrix[i, j] = numpy.mean(
                et[evaluation.FREQUENCY_BIAS_KEY].values[threshold_index, :]
            )
            csi_matrix[i, j] = numpy.mean(
                et[evaluation.CSI_KEY].values[threshold_index, :]
            )

    _plot_one_score(
        score_matrix=brier_score_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, is_bss=False,
        output_file_name='{0:s}/brier_score.jpg'.format(output_dir_name),
        title_string='Brier score'
    )

    _plot_one_score(
        score_matrix=bss_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=diverging_colour_map_name,
        is_frequency_bias=False, is_bss=True,
        output_file_name='{0:s}/brier_skill_score.jpg'.format(output_dir_name),
        title_string='Brier skill score'
    )

    _plot_one_score(
        score_matrix=event_freq_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, is_bss=False,
        output_file_name=
        '{0:s}/event_frequency.jpg'.format(output_dir_name),
        title_string='Label-based climatology'
    )

    _plot_one_score(
        score_matrix=mean_prob_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, is_bss=False,
        output_file_name=
        '{0:s}/mean_forecast_prob.jpg'.format(output_dir_name),
        title_string='Model-based climatology'
    )

    _plot_one_score(
        score_matrix=pod_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, is_bss=False,
        output_file_name='{0:s}/pod.jpg'.format(output_dir_name),
        title_string='Probability of detection'
    )

    _plot_one_score(
        score_matrix=far_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, is_bss=False,
        output_file_name='{0:s}/far.jpg'.format(output_dir_name),
        title_string='False-alarm ratio'
    )

    _plot_one_score(
        score_matrix=frequency_bias_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=bias_colour_map_name,
        is_frequency_bias=True, is_bss=False,
        output_file_name='{0:s}/frequency_bias.jpg'.format(output_dir_name),
        title_string='Frequency bias'
    )

    _plot_one_score(
        score_matrix=csi_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, is_bss=False,
        output_file_name='{0:s}/csi.jpg'.format(output_dir_name),
        title_string='Critical success index'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        grid_metafile_name=getattr(INPUT_ARG_OBJECT, GRID_METAFILE_ARG_NAME),
        total_validn_eval_file_name=getattr(
            INPUT_ARG_OBJECT, TOTAL_VALIDN_EVAL_FILE_ARG_NAME
        ),
        sequential_colour_map_name=getattr(
            INPUT_ARG_OBJECT, SEQ_COLOUR_MAP_ARG_NAME
        ),
        diverging_colour_map_name=getattr(
            INPUT_ARG_OBJECT, DIV_COLOUR_MAP_ARG_NAME
        ),
        bias_colour_map_name=getattr(
            INPUT_ARG_OBJECT, BIAS_COLOUR_MAP_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
