"""Plots evaluation metrics vs. hyperparams for QR Experiment 1 for Taiwan."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from scipy.stats import rankdata

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import evaluation
import uq_evaluation
import gg_plotting_utils
import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CENTRAL_LOSS_WEIGHTS = numpy.linspace(1, 10, num=10, dtype=float)

THESE_QUANTILE_LEVEL_SETS = [
    numpy.linspace(0.01, 0.99, num=99),
    numpy.linspace(0.01, 0.99, num=99)[::2],
    numpy.linspace(0.01, 0.99, num=99)[::3],
    numpy.linspace(0.01, 0.99, num=99)[::4],
    numpy.linspace(0.01, 0.99, num=99)[::5],
    numpy.linspace(0.01, 0.99, num=99)[::6],
    numpy.linspace(0.01, 0.99, num=99)[::7],
    numpy.linspace(0.01, 0.99, num=99)[::8],
    numpy.linspace(0.01, 0.99, num=99)[::9]
]
THESE_QUANTILE_LEVEL_SETS = [
    numpy.concatenate((
        numpy.array([0.025, 0.25, 0.5, 0.75, 0.975, 0.99]), s
    ))
    for s in THESE_QUANTILE_LEVEL_SETS
]
THESE_QUANTILE_LEVEL_SETS = [
    numpy.sort(numpy.unique(s)) for s in THESE_QUANTILE_LEVEL_SETS
]

QUANTILE_LEVEL_COUNTS = numpy.array(
    [len(s) for s in THESE_QUANTILE_LEVEL_SETS], dtype=int
)[::-1]

DEFAULT_FONT_SIZE = 20

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
        score_matrix, min_colour_value, max_colour_value, x_tick_labels,
        y_tick_labels, colour_map_object, colour_norm_object=None):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param colour_norm_object: Normalizer for colour scheme (instance of
        `matplotlib.pyplot.Normalize` or similar).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if colour_norm_object is None:
        if numpy.isnan(min_colour_value) or numpy.isnan(max_colour_value):
            min_colour_value = 0.
            max_colour_value = 1.

        axes_object.imshow(
            score_matrix, cmap=colour_map_object, origin='lower',
            vmin=min_colour_value, vmax=max_colour_value
        )

        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False
        )
    else:
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

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=0.8, font_size=DEFAULT_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    W = number of central-loss weights
    Q = number of quantile counts

    :param score_matrix: W-by-Q numpy array of scores.
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
            '{0:d}th-highest {1:s} = {2:.4g} ... '
            'central-loss weight = {3:.1f} ... '
            'number of quantile levels = {4:d} '
        ).format(
            k + 1, score_name, score_matrix[i, j],
            CENTRAL_LOSS_WEIGHTS[i], QUANTILE_LEVEL_COUNTS[j]
        ))


def _print_ranking_all_scores(
        auc_matrix, aupd_matrix, bss_matrix, csi_matrix, frequency_bias_matrix,
        ssrel_matrix, mean_predictive_stdev_matrix, monotonicity_fraction_matrix,
        rank_mainly_by_auc):
    """Prints ranking for all scores.

    W = number of central-loss weights
    Q = number of quantile counts

    :param auc_matrix: W-by-Q numpy array with AUC (area under ROC curve).
    :param aupd_matrix: Same but for AUPD (area under performance diagram).
    :param bss_matrix: Same but for Brier skill score.
    :param csi_matrix: Same but for critical success index.
    :param frequency_bias_matrix: Same but for frequency bias.
    :param ssrel_matrix: Same but for spread-skill reliability.
    :param mean_predictive_stdev_matrix: Same but for mean stdev of predictive
        distribution.
    :param monotonicity_fraction_matrix: Same but for monotonicity fraction.
    :param rank_mainly_by_auc: Boolean flag.  If True (False), will rank mainly
        by AUC (SSREL).
    """

    if rank_mainly_by_auc:
        these_scores = -1 * numpy.ravel(auc_matrix)
        these_scores[numpy.isnan(these_scores)] = -numpy.inf
    else:
        these_scores = numpy.ravel(ssrel_matrix)
        these_scores[numpy.isnan(these_scores)] = numpy.inf

    sort_indices_1d = numpy.argsort(these_scores)
    i_sort_indices, j_sort_indices = numpy.unravel_index(
        sort_indices_1d, auc_matrix.shape
    )

    these_scores = -1 * numpy.ravel(auc_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    auc_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), auc_matrix.shape
    )

    these_scores = -1 * numpy.ravel(aupd_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    aupd_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), aupd_matrix.shape
    )

    these_scores = -1 * numpy.ravel(bss_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    bss_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), bss_matrix.shape
    )

    these_scores = -1 * numpy.ravel(csi_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    csi_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), csi_matrix.shape
    )

    these_scores = numpy.ravel(numpy.absolute(1. - frequency_bias_matrix))
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    bias_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), frequency_bias_matrix.shape
    )

    these_scores = numpy.ravel(ssrel_matrix)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    ssrel_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), ssrel_matrix.shape
    )

    these_scores = -1 * numpy.ravel(mean_predictive_stdev_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    stdev_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        mean_predictive_stdev_matrix.shape
    )

    these_scores = -1 * numpy.ravel(monotonicity_fraction_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    mf_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        monotonicity_fraction_matrix.shape
    )

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        j = j_sort_indices[k]

        print((
            '{0:d}th-best model ... central-loss weight = {1:.1f} ... '
            'number of quantile levels = {2:d} ... '
            'AUC rank = {3:.1f} ... AUPD rank = {4:.1f} ... '
            'BSS rank = {5:.1f} ... CSI rank = {6:.1f} ... '
            'frequency-bias rank = {7:.1f} ... '
            'SSREL rank = {8:.1f} ... MF rank = {9:.1f} ... '
            'predictive-stdev rank = {10:.1f}'
        ).format(
            k + 1, CENTRAL_LOSS_WEIGHTS[i], QUANTILE_LEVEL_COUNTS[j],
            auc_rank_matrix[i, j], aupd_rank_matrix[i, j],
            bss_rank_matrix[i, j], csi_rank_matrix[i, j],
            bias_rank_matrix[i, j],
            ssrel_rank_matrix[i, j], mf_rank_matrix[i, j],
            stdev_rank_matrix[i, j]
        ))


def _run(experiment_dir_name, output_dir_name):
    """Plots evaluation metrics vs. hyperparams for QR Experiment 1 for Taiwan.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_central_loss_weights = len(CENTRAL_LOSS_WEIGHTS)
    num_quantile_level_counts = len(QUANTILE_LEVEL_COUNTS)
    dimensions = (num_central_loss_weights, num_quantile_level_counts)

    auc_matrix = numpy.full(dimensions, numpy.nan)
    aupd_matrix = numpy.full(dimensions, numpy.nan)
    bss_matrix = numpy.full(dimensions, numpy.nan)
    csi_matrix = numpy.full(dimensions, numpy.nan)
    frequency_bias_matrix = numpy.full(dimensions, numpy.nan)
    ssrel_matrix = numpy.full(dimensions, numpy.nan)
    mean_predictive_stdev_matrix = numpy.full(dimensions, numpy.nan)
    monotonicity_fraction_matrix = numpy.full(dimensions, numpy.nan)

    y_tick_labels = [
        '{0:d}'.format(int(numpy.round(w))) for w in CENTRAL_LOSS_WEIGHTS
    ]
    x_tick_labels = ['{0:d}'.format(c) for c in QUANTILE_LEVEL_COUNTS]

    y_axis_label = 'Weight for deterministic prediction'
    x_axis_label = 'Number of quantiles'

    for i in range(num_central_loss_weights):
        for j in range(num_quantile_level_counts):
            this_score_file_name = (
                '{0:s}/central-loss-weight={1:04.1f}_'
                'num-quantile-levels={2:03d}/'
                'validation_sans_uq/evaluation.nc'
            ).format(
                experiment_dir_name, CENTRAL_LOSS_WEIGHTS[i],
                QUANTILE_LEVEL_COUNTS[j]
            )

            if os.path.isfile(this_score_file_name):
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

            this_score_file_name = (
                '{0:s}/central-loss-weight={1:04.1f}_'
                'num-quantile-levels={2:03d}/'
                'validation_with_uq/spread_vs_skill.nc'
            ).format(
                experiment_dir_name, CENTRAL_LOSS_WEIGHTS[i],
                QUANTILE_LEVEL_COUNTS[j]
            )

            if os.path.isfile(this_score_file_name):
                print('Reading data from: "{0:s}"...'.format(
                    this_score_file_name
                ))
                result_dict = uq_evaluation.read_spread_vs_skill(
                    this_score_file_name
                )

                ssrel_matrix[i, j] = result_dict[
                    uq_evaluation.SPREAD_SKILL_RELIABILITY_KEY
                ]

                non_zero_indices = numpy.where(
                    result_dict[uq_evaluation.EXAMPLE_COUNTS_KEY] > 0
                )[0]
                mean_predictive_stdev_matrix[i, j] = numpy.average(
                    result_dict[uq_evaluation.MEAN_PREDICTION_STDEVS_KEY][
                        non_zero_indices
                    ],
                    weights=
                    result_dict[uq_evaluation.EXAMPLE_COUNTS_KEY][
                        non_zero_indices
                    ]
                )

            this_score_file_name = (
                '{0:s}/central-loss-weight={1:04.1f}_'
                'num-quantile-levels={2:03d}/'
                'validation_with_uq/discard_test.nc'
            ).format(
                experiment_dir_name, CENTRAL_LOSS_WEIGHTS[i],
                QUANTILE_LEVEL_COUNTS[j]
            )

            if os.path.isfile(this_score_file_name):
                print('Reading data from: "{0:s}"...'.format(
                    this_score_file_name
                ))
                monotonicity_fraction_matrix[i, j] = (
                    uq_evaluation.read_discard_results(this_score_file_name)[
                        uq_evaluation.MONOTONICITY_FRACTION_KEY
                    ]
                )

    print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        auc_matrix=auc_matrix, aupd_matrix=aupd_matrix, bss_matrix=bss_matrix,
        csi_matrix=csi_matrix, frequency_bias_matrix=frequency_bias_matrix,
        ssrel_matrix=ssrel_matrix,
        mean_predictive_stdev_matrix=mean_predictive_stdev_matrix,
        monotonicity_fraction_matrix=monotonicity_fraction_matrix,
        rank_mainly_by_auc=True
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        auc_matrix=auc_matrix, aupd_matrix=aupd_matrix, bss_matrix=bss_matrix,
        csi_matrix=csi_matrix, frequency_bias_matrix=frequency_bias_matrix,
        ssrel_matrix=ssrel_matrix,
        mean_predictive_stdev_matrix=mean_predictive_stdev_matrix,
        monotonicity_fraction_matrix=monotonicity_fraction_matrix,
        rank_mainly_by_auc=False
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=auc_matrix, score_name='AUC')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=aupd_matrix, score_name='AUPD')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=bss_matrix, score_name='BSS')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=csi_matrix, score_name='CSI')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=-numpy.absolute(1. - frequency_bias_matrix),
        score_name='negative deviation of bias from 1.0'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=-ssrel_matrix, score_name='negative SSREL'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=mean_predictive_stdev_matrix,
        score_name='mean predictive stdev'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=monotonicity_fraction_matrix,
        score_name='monotonicity fraction'
    )
    print(SEPARATOR_STRING)

    # Plot AUC.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=auc_matrix,
        min_colour_value=numpy.nanpercentile(auc_matrix, 1),
        max_colour_value=numpy.nanpercentile(auc_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Area under ROC curve')
    figure_file_name = '{0:s}/auc.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot AUPD.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=aupd_matrix,
        min_colour_value=numpy.nanpercentile(aupd_matrix, 1),
        max_colour_value=numpy.nanpercentile(aupd_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Area under performance diagram')
    figure_file_name = '{0:s}/aupd.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot BSS.
    this_max_value = numpy.nanpercentile(numpy.absolute(bss_matrix), 99.)
    this_max_value = min([this_max_value, 1.])
    this_min_value = -1 * this_max_value

    figure_object, axes_object = _plot_scores_2d(
        score_matrix=bss_matrix,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=BSS_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Brier skill score')
    figure_file_name = '{0:s}/bss.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot CSI.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=csi_matrix,
        min_colour_value=numpy.nanpercentile(csi_matrix, 1),
        max_colour_value=numpy.nanpercentile(csi_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Critical success index')
    figure_file_name = '{0:s}/csi.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot frequency bias.
    this_offset = numpy.nanpercentile(
        numpy.absolute(1. - frequency_bias_matrix), 99.
    )
    if numpy.isnan(this_offset):
        this_offset = 1.

    colour_map_object, colour_norm_object = _get_bias_colour_scheme(
        colour_map_name=BIAS_COLOUR_MAP_NAME,
        max_colour_value=1. + this_offset
    )
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=frequency_bias_matrix,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        min_colour_value=0., max_colour_value=1. + this_offset,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
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

    # Plot SSREL.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=ssrel_matrix,
        min_colour_value=numpy.nanpercentile(ssrel_matrix, 1),
        max_colour_value=numpy.nanpercentile(ssrel_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Spread-skill reliability')
    figure_file_name = '{0:s}/ssrel.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot mean predictive stdev.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=mean_predictive_stdev_matrix,
        min_colour_value=numpy.nanpercentile(mean_predictive_stdev_matrix, 1),
        max_colour_value=numpy.nanpercentile(mean_predictive_stdev_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Mean stdev of predictive distribution')
    figure_file_name = '{0:s}/mean_predictive_stdev.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot monotonicity fraction.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=monotonicity_fraction_matrix,
        min_colour_value=numpy.nanpercentile(monotonicity_fraction_matrix, 1),
        max_colour_value=numpy.nanpercentile(monotonicity_fraction_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Monotonicity fraction from discard test')
    figure_file_name = '{0:s}/monotonicity_fraction.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
