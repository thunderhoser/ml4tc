"""Plots evaluation scores on hyperparameter grid for Experiment 11."""

import os
import sys
import argparse
from PIL import Image
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import evaluation
import gg_plotting_utils
import imagemagick_utils
import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CHANNEL_COUNT_STRINGS = [
    '002-002-004-004-008-008-012-012-016-016',
    '004-004-008-008-012-012-016-016-024-024',
    '004-004-008-008-016-016-032-032-048-048',
    '004-004-008-008-016-016-032-032-064-064'
]

LAG_TIME_STRINGS_MINUTES = [
    '0', '0-30', '0-30-60', '0-30-60-90', '0-30-60-90-120',
    '0-30-60-90-120-150', '0-30-60-90-120-150-180',
    '0-30-60-90-120-150-180-210', '0-30-60-90-120-150-180-210-240',
    '0-30-60-90-120-150-180-210-240-270', '0-30-60-90-120-150-180-210-240-270-300',
    '0-30-60-90-120-150-180-210-240-270-300-330',
    '0-30-60-90-120-150-180-210-240-270-300-330-360',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420-450',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420-450-480',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420-450-480-510',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420-450-480-510-540',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420-450-480-510-540-570',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420-450-480-510-540-570-600',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420-450-480-510-540-570-600-630',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420-450-480-510-540-570-600-630-660',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420-450-480-510-540-570-600-630-660-690',
    '0-30-60-90-120-150-180-210-240-270-300-330-360-390-420-450-480-510-540-570-600-630-660-690-720',
    '0-60', '0-60-120', '0-60-120-180', '0-60-120-180-240',
    '0-60-120-180-240-300', '0-60-120-180-240-300-360',
    '0-60-120-180-240-300-360-420', '0-60-120-180-240-300-360-420-480',
    '0-60-120-180-240-300-360-420-480-540',
    '0-60-120-180-240-300-360-420-480-540-600',
    '0-60-120-180-240-300-360-420-480-540-600-660',
    '0-60-120-180-240-300-360-420-480-540-600-660-720',
    '0-90', '0-90-180', '0-90-180-270', '0-90-180-270-360',
    '0-90-180-270-360-450', '0-90-180-270-360-450-540',
    '0-90-180-270-360-450-540-630', '0-90-180-270-360-450-540-630-720',
    '0-120', '0-120-240', '0-120-240-360', '0-120-240-360-480',
    '0-120-240-360-480-600', '0-120-240-360-480-600-720',
    '0-180', '0-180-360', '0-180-360-540', '0-180-360-540-720',
    '0-240', '0-240-480', '0-240-480-720'
]

LAST_LAYER_CHANNEL_COUNTS = numpy.array([16, 24, 48, 64], dtype=int)
LAG_TIME_STRINGS_HOURS = [
    '0', '0, 0.5', '0, 0.5, 1', '0, 0.5, ..., 1.5', '0, 0.5, ..., 2',
    '0, 0.5, ..., 2.5', '0, 0.5, ..., 3', '0, 0.5, ..., 3.5', '0, 0.5, ..., 4',
    '0, 0.5, ..., 4.5', '0, 0.5, ..., 5', '0, 0.5, ..., 5.5', '0, 0.5, ..., 6',
    '0, 0.5, ..., 6.5', '0, 0.5, ..., 7', '0, 0.5, ..., 7.5', '0, 0.5, ..., 8',
    '0, 0.5, ..., 8.5', '0, 0.5, ..., 9', '0, 0.5, ..., 9.5', '0, 0.5, ..., 10',
    '0, 0.5, ..., 10.5', '0, 0.5, ..., 11', '0, 0.5, ..., 11.5', '0, 0.5, ..., 12',
    '0, 1', '0, 1, 2', '0, 1, ..., 3', '0, 1, ..., 4', '0, 1, ..., 5',
    '0, 1, ..., 6', '0, 1, ..., 7', '0, 1, ..., 8', '0, 1, ..., 9',
    '0, 1, ..., 10', '0, 1, ..., 11', '0, 1, ..., 12',
    '0, 1.5', '0, 1.5, 3', '0, 1.5, ..., 4.5', '0, 1.5, ..., 6',
    '0, 1.5, ..., 7.5', '0, 1.5, ..., 9', '0, 1.5, ..., 10.5', '0, 1.5, ..., 12',
    '0, 2', '0, 2, 4', '0, 2, ..., 6', '0, 2, ..., 8', '0, 2, ..., 10',
    '0, 2, ..., 12',
    '0, 3', '0, 3, 6', '0, 3, ..., 9', '0, 3, ..., 12',
    '0, 4', '0, 4, 8', '0, 4, ..., 12'
]

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.3
MARKER_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.2
SELECTED_MARKER_INDICES = numpy.array([-1, -1, -1], dtype=int)

DEFAULT_FONT_SIZE = 20
COLOUR_BAR_FONT_SIZE = 25

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

BIAS_COLOUR_MAP_NAME = 'seismic'
DEFAULT_COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')
BSS_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXPERIMENT_DIR_HELP_STRING = (
    'Name of top-level directory with models.  Evaluation scores will be found '
    'therein.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
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


def _plot_scores_2d(
        score_matrix, x_tick_labels, y_tick_labels, colour_map_object,
        colour_norm_object):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param colour_norm_object: Colour-normalizer (instance of
        `matplotlib.colors.Normalize` or similar).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.imshow(
        score_matrix, cmap=colour_map_object, origin='lower',
        norm=colour_norm_object
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )

    pyplot.xticks(x_tick_values, x_tick_labels, rotation=90.)
    pyplot.yticks(y_tick_values, y_tick_labels)

    return figure_object, axes_object


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.
    
    C = number of last-layer channel counts
    L = number of lag-time sets

    :param score_matrix: C-by-L numpy array of scores.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix) + 0.
    scores_1d[numpy.isnan(scores_1d)] = -numpy.inf
    sort_indices_1d = numpy.argsort(-scores_1d)

    i_sort_indices, j_sort_indices = numpy.unravel_index(
        sort_indices_1d, score_matrix.shape
    )

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        j = j_sort_indices[k]

        print((
            '{0:d}th-highest {1:s} = {2:.4g} ... num last-layer filters = '
            '{3:d} ... lag times = {4:s} h'
        ).format(
            k + 1, score_name, score_matrix[i, j],
            LAST_LAYER_CHANNEL_COUNTS[i], LAG_TIME_STRINGS_HOURS[j]
        ))


def _add_markers(figure_object, axes_object, best_marker_indices):
    """Adds markers to figure.

    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param best_marker_indices: length-2 numpy array of array indices for best
        model.
    """

    figure_width_px = (
        figure_object.get_size_inches()[0] * figure_object.dpi
    )
    marker_size_px = figure_width_px * (
        BEST_MARKER_SIZE_GRID_CELLS / len(LAG_TIME_STRINGS_MINUTES)
    )
    axes_object.plot(
        best_marker_indices[1], best_marker_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )

    if any(SELECTED_MARKER_INDICES < 0):
        return

    figure_width_px = (
        figure_object.get_size_inches()[0] * figure_object.dpi
    )
    marker_size_px = figure_width_px * (
        SELECTED_MARKER_SIZE_GRID_CELLS / len(LAG_TIME_STRINGS_MINUTES)
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )


def _add_colour_bar(
        figure_file_name, colour_map_object, colour_norm_object,
        temporary_dir_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param colour_norm_object: Colour-normalizer (instance of
        `matplotlib.colors.Normalize` or similar).
    :param temporary_dir_name: Name of temporary output directory.
    """

    this_image_matrix = Image.open(figure_file_name)
    figure_width_px, figure_height_px = this_image_matrix.size
    figure_width_inches = float(figure_width_px) / FIGURE_RESOLUTION_DPI
    figure_height_inches = float(figure_height_px) / FIGURE_RESOLUTION_DPI

    extra_figure_object, extra_axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )
    extra_axes_object.axis('off')

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=extra_axes_object,
        data_matrix=numpy.linspace(-1, 1, num=21, dtype=float),
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=1.25, font_size=COLOUR_BAR_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    extra_file_name = '{0:s}/extra_colour_bar.jpg'.format(temporary_dir_name)
    print('Saving colour bar to: "{0:s}"...'.format(extra_file_name))

    extra_figure_object.savefig(
        extra_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(extra_figure_object)

    print('Concatenating colour bar to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[figure_file_name, extra_file_name],
        output_file_name=figure_file_name,
        num_panel_rows=1, num_panel_columns=2,
        extra_args_string='-gravity Center'
    )

    os.remove(extra_file_name)
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name
    )


def _run(experiment_dir_name, output_dir_name):
    """Plots evaluation scores on hyperparameter grid for Experiment 11.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_channel_counts = len(LAST_LAYER_CHANNEL_COUNTS)
    num_lag_time_sets = len(LAG_TIME_STRINGS_HOURS)
    dimensions = (num_channel_counts, num_lag_time_sets)

    auc_matrix = numpy.full(dimensions, numpy.nan)
    aupd_matrix = numpy.full(dimensions, numpy.nan)
    bss_matrix = numpy.full(dimensions, numpy.nan)
    csi_matrix = numpy.full(dimensions, numpy.nan)
    frequency_bias_matrix = numpy.full(dimensions, numpy.nan)

    y_tick_labels = ['{0:d}'.format(n) for n in LAST_LAYER_CHANNEL_COUNTS]
    x_tick_labels = ['{0:s}'.format(s) for s in LAG_TIME_STRINGS_HOURS]
    y_axis_label = 'Num last-layer filters'
    x_axis_label = 'Lag times (hours)'

    for i in range(num_channel_counts):
        for j in range(num_lag_time_sets):
            this_score_file_name = (
                '{0:s}/channels={1:s}_lag-times-minutes={2:s}/'
                'validation_2005-2009/evaluation.nc'
            ).format(
                experiment_dir_name, CHANNEL_COUNT_STRINGS[i],
                LAG_TIME_STRINGS_MINUTES[j]
            )

            if not os.path.isfile(this_score_file_name):
                continue

            print('Reading data from: "{0:s}"...'.format(
                this_score_file_name
            ))
            t = evaluation.read_file(this_score_file_name)

            auc_matrix[i, j] = numpy.mean(t[evaluation.AUC_KEY].values)
            aupd_matrix[i, j] = numpy.mean(t[evaluation.AUPD_KEY].values)
            bss_matrix[i, j] = numpy.mean(
                t[evaluation.BRIER_SKILL_SCORE_KEY].values
            )

            these_csi_by_bootstrap_rep = numpy.mean(
                t[evaluation.CSI_KEY].values, axis=1
            )
            csi_matrix[i, j] = numpy.max(these_csi_by_bootstrap_rep)

            this_threshold_index = numpy.argmax(these_csi_by_bootstrap_rep)
            frequency_bias_matrix[i, j] = numpy.mean(
                t[evaluation.FREQUENCY_BIAS_KEY].values[
                    this_threshold_index, :
                ]
            )

    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=aupd_matrix, score_name='AUPD')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=auc_matrix, score_name='AUC')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=bss_matrix, score_name='BSS')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=csi_matrix, score_name='CSI')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=-numpy.absolute(1. - frequency_bias_matrix),
        score_name='negative frequency-bias offset'
    )
    print(SEPARATOR_STRING)

    this_index = numpy.nanargmax(numpy.ravel(auc_matrix))
    best_auc_indices = numpy.unravel_index(this_index, auc_matrix.shape)

    this_index = numpy.nanargmax(numpy.ravel(aupd_matrix))
    best_aupd_indices = numpy.unravel_index(this_index, aupd_matrix.shape)

    this_index = numpy.nanargmax(numpy.ravel(bss_matrix))
    best_bss_indices = numpy.unravel_index(this_index, bss_matrix.shape)

    this_index = numpy.nanargmax(numpy.ravel(csi_matrix))
    best_csi_indices = numpy.unravel_index(this_index, csi_matrix.shape)

    this_index = numpy.nanargmin(numpy.ravel(
        numpy.absolute(1. - frequency_bias_matrix)
    ))
    best_bias_indices = numpy.unravel_index(
        this_index, frequency_bias_matrix.shape
    )

    # Plot AUC.
    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(auc_matrix, 1),
        vmax=numpy.nanpercentile(auc_matrix, 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=auc_matrix,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object
    )
    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=best_auc_indices
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Area under ROC curve')

    output_file_name = '{0:s}/auc.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        temporary_dir_name=output_dir_name
    )

    # Plot AUPD.
    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(aupd_matrix, 1),
        vmax=numpy.nanpercentile(aupd_matrix, 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=aupd_matrix,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object
    )
    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=best_aupd_indices
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Area under performance diagram')

    output_file_name = '{0:s}/aupd.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        temporary_dir_name=output_dir_name
    )

    # Plot BSS.
    this_max_value = numpy.nanpercentile(numpy.absolute(bss_matrix), 99.)
    this_max_value = min([this_max_value, 1.])
    colour_norm_object = pyplot.Normalize(
        vmin=-1 * this_max_value, vmax=this_max_value
    )

    figure_object, axes_object = _plot_scores_2d(
        score_matrix=bss_matrix,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=BSS_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object
    )
    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=best_bss_indices
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Brier skill score')

    output_file_name = '{0:s}/bss.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=BSS_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        temporary_dir_name=output_dir_name
    )

    # Plot CSI.
    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(csi_matrix, 1),
        vmax=numpy.nanpercentile(csi_matrix, 99)
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=csi_matrix,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object
    )
    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=best_csi_indices
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Critical success index')

    output_file_name = '{0:s}/csi.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        temporary_dir_name=output_dir_name
    )

    # Plot frequency bias.
    this_offset = numpy.nanpercentile(
        numpy.absolute(1. - frequency_bias_matrix), 99.
    )
    colour_map_object, colour_norm_object = _get_bias_colour_scheme(
        colour_map_name=BIAS_COLOUR_MAP_NAME,
        max_colour_value=1. + this_offset
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=frequency_bias_matrix,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )
    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=best_bias_indices
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Frequency bias')

    output_file_name = '{0:s}/frequency_bias.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        temporary_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )