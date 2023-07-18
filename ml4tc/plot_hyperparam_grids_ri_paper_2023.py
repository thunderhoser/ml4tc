"""Plots hyperparameter grids for main experiment in 2023 RI paper."""

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

import file_system_utils
import imagemagick_utils
import evaluation
import uq_evaluation
import plotting_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FREQUENCY_BIAS_KEY = evaluation.FREQUENCY_BIAS_KEY
MEAN_PREDICTION_STDEVS_KEY = uq_evaluation.MEAN_PREDICTION_STDEVS_KEY
RMSE_VALUES_KEY = uq_evaluation.RMSE_VALUES_KEY
EXAMPLE_COUNTS_KEY = uq_evaluation.EXAMPLE_COUNTS_KEY

DROPOUT_RATES_AXIS1 = numpy.array([0.5, 0.6, 0.7, 0.8, 0.9])
CIRA_IR_LAG_TIME_COUNTS_AXIS2 = numpy.array([0, 1, 2, 3], dtype=int)
TEMPORAL_DIFF_FLAGS_AXIS3 = numpy.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool
)
SHIPS_ENVIRO_FLAGS_AXIS3 = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool
)
SHIPS_HISTORICAL_FLAGS_AXIS3 = numpy.array(
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool
)
SHIPS_SATELLITE_FLAGS_AXIS3 = numpy.array(
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=bool
)

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

NUM_PANEL_ROWS = 4
NUM_PANEL_COLUMNS = 4
PANEL_SIZE_PX = int(2.5e6)
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

    # colour_bar_object = gg_plotting_utils.plot_colour_bar(
    #     axes_object_or_matrix=axes_object,
    #     data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
    #     colour_map_object=colour_map_object,
    #     colour_norm_object=colour_norm_object,
    #     orientation_string='vertical', extend_min=False, extend_max=False,
    #     fraction_of_axis_length=0.8, font_size=DEFAULT_FONT_SIZE
    # )
    #
    # tick_values = colour_bar_object.get_ticks()
    # tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    # colour_bar_object.set_ticks(tick_values)
    # colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    D = number of dropout rates
    C = number of CIRA IR lag-time counts
    S = number of SHIPS-predictor combinations

    :param score_matrix: D-by-C-by-S numpy array of scores.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix) + 0.
    scores_1d[numpy.isnan(scores_1d)] = -numpy.inf
    sort_indices_1d = numpy.argsort(-scores_1d)
    
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, score_matrix.shape
    )

    for a in range(len(i_sort_indices)):
        i = i_sort_indices[a]
        j = j_sort_indices[a]
        k = k_sort_indices[a]
        
        these_ships_predictor_types = []
        if SHIPS_ENVIRO_FLAGS_AXIS3[k]:
            these_ships_predictor_types.append('enviro')
        if SHIPS_HISTORICAL_FLAGS_AXIS3[k]:
            these_ships_predictor_types.append('hist')
        if SHIPS_SATELLITE_FLAGS_AXIS3[k]:
            these_ships_predictor_types.append('GOES')
        
        if len(these_ships_predictor_types) == 0:
            this_ships_predictor_type_string = 'none'
        else:
            this_ships_predictor_type_string = ', '.join(
                these_ships_predictor_types
            )

        print((
            '{0:d}th-highest {1:s} = {2:.4g} ... dropout rate = {3:.1f} ... '
            'num CIRA IR lag times = {4:d} ... CIRA IR temporal diffs = {5:s} '
            '... SHIPS predictors = {6:s}'
        ).format(
            a + 1, score_name, score_matrix[i, j, k],
            DROPOUT_RATES_AXIS1[i],
            CIRA_IR_LAG_TIME_COUNTS_AXIS2[j],
            'yes' if TEMPORAL_DIFF_FLAGS_AXIS3[k] else 'no',
            this_ships_predictor_type_string
        ))

    print('\n')


def _print_ranking_all_scores(
        auc_matrix, aupd_matrix, bss_matrix, csi_matrix, frequency_bias_matrix,
        ssrel_matrix, ssrat_matrix, mean_predictive_stdev_matrix,
        monotonicity_fraction_matrix, rank_mainly_by_auc):
    """Prints ranking for all scores.

    D = number of dropout rates
    C = number of CIRA IR lag-time counts
    S = number of SHIPS-predictor combinations

    :param auc_matrix: D-by-C-by-S numpy array with AUC (area under ROC curve).
    :param aupd_matrix: Same but for AUPD (area under performance diagram).
    :param bss_matrix: Same but for Brier skill score.
    :param csi_matrix: Same but for critical success index.
    :param frequency_bias_matrix: Same but for frequency bias.
    :param ssrel_matrix: Same but for spread-skill reliability.
    :param ssrat_matrix: Same but for spread-skill ratio.
    :param mean_predictive_stdev_matrix: Same but for mean stdev of predictive
        distribution.
    :param monotonicity_fraction_matrix: Same but for monotonicity fraction.
    :param rank_mainly_by_auc: Boolean flag.  If True (False), will rank mainly
        by AUC (SSREL).
    """

    if rank_mainly_by_auc:
        these_scores = -1 * numpy.ravel(auc_matrix)
        these_scores[numpy.isnan(these_scores)] = numpy.inf
    else:
        these_scores = numpy.ravel(ssrel_matrix)
        these_scores[numpy.isnan(these_scores)] = numpy.inf

    sort_indices_1d = numpy.argsort(these_scores)

    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
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
        rankdata(these_scores, method='average'),
        frequency_bias_matrix.shape
    )

    these_scores = numpy.ravel(ssrel_matrix)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    ssrel_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), ssrel_matrix.shape
    )

    these_scores = numpy.ravel(numpy.absolute(1. - ssrat_matrix))
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    ssrat_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        ssrat_matrix.shape
    )

    these_scores = -1 * numpy.ravel(mean_predictive_stdev_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    spread_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        mean_predictive_stdev_matrix.shape
    )

    these_scores = -1 * numpy.ravel(monotonicity_fraction_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    mf_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        monotonicity_fraction_matrix.shape
    )

    for a in range(len(i_sort_indices)):
        i = i_sort_indices[a]
        j = j_sort_indices[a]
        k = k_sort_indices[a]

        these_ships_predictor_types = []
        if SHIPS_ENVIRO_FLAGS_AXIS3[k]:
            these_ships_predictor_types.append('enviro')
        if SHIPS_HISTORICAL_FLAGS_AXIS3[k]:
            these_ships_predictor_types.append('hist')
        if SHIPS_SATELLITE_FLAGS_AXIS3[k]:
            these_ships_predictor_types.append('GOES')

        if len(these_ships_predictor_types) == 0:
            this_ships_predictor_type_string = 'none'
        else:
            this_ships_predictor_type_string = ', '.join(
                these_ships_predictor_types
            )

        print((
            '{0:d}th-best model ... dropout rate = {1:.1f} ... '
            'num CIRA IR lag times = {2:d} ... use CIRA IR temporal diffs = '
            '{3:s} ... SHIPS predictors = {4:s} ... '
            'AUC rank = {5:.1f} ... AUPD rank = {6:.1f} ... '
            'BSS rank = {7:.1f} ... CSI rank = {8:.1f} ... '
            'frequency-bias rank = {9:.1f} ... '
            'SSREL rank = {10:.1f} ... SSRAT rank = {11:.1f} ... '
            'MF rank = {12:.1f} ... spread rank = {13:.1f}'
        ).format(
            a + 1,
            DROPOUT_RATES_AXIS1[i],
            CIRA_IR_LAG_TIME_COUNTS_AXIS2[j],
            'yes' if TEMPORAL_DIFF_FLAGS_AXIS3[k] else 'no',
            this_ships_predictor_type_string,
            auc_rank_matrix[i, j, k], aupd_rank_matrix[i, j, k],
            bss_rank_matrix[i, j, k], csi_rank_matrix[i, j, k],
            bias_rank_matrix[i, j, k],
            ssrel_rank_matrix[i, j, k], ssrat_rank_matrix[i, j, k],
            mf_rank_matrix[i, j, k], spread_rank_matrix[i, j, k]
        ))

    print('\n')


def _run(experiment_dir_name, top_output_dir_name):
    """Plots hyperparameter grids for main experiment in 2023 RI paper.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param top_output_dir_name: Same.
    """

    axis1_length = len(DROPOUT_RATES_AXIS1)
    axis2_length = len(CIRA_IR_LAG_TIME_COUNTS_AXIS2)
    axis3_length = len(SHIPS_SATELLITE_FLAGS_AXIS3)
    dimensions = (axis1_length, axis2_length, axis3_length)

    auc_matrix = numpy.full(dimensions, numpy.nan)
    aupd_matrix = numpy.full(dimensions, numpy.nan)
    bss_matrix = numpy.full(dimensions, numpy.nan)
    csi_matrix = numpy.full(dimensions, numpy.nan)
    frequency_bias_matrix = numpy.full(dimensions, numpy.nan)
    ssrel_matrix = numpy.full(dimensions, numpy.nan)
    ssrat_matrix = numpy.full(dimensions, numpy.nan)
    mean_predictive_stdev_matrix = numpy.full(dimensions, numpy.nan)
    monotonicity_fraction_matrix = numpy.full(dimensions, numpy.nan)

    y_tick_labels = [
        '{0:.1f}'.format(d) for d in DROPOUT_RATES_AXIS1
    ]
    x_tick_labels = [
        '{0:d}'.format(n) for n in CIRA_IR_LAG_TIME_COUNTS_AXIS2
    ]

    y_axis_label = 'Dropout rate'
    x_axis_label = 'Number of CIRA IR lag times'

    for i in range(axis1_length):
        for j in range(axis2_length):
            for k in range(axis3_length):
                if (
                        CIRA_IR_LAG_TIME_COUNTS_AXIS2[j] == 0
                        and not SHIPS_ENVIRO_FLAGS_AXIS3[k]
                        and not SHIPS_HISTORICAL_FLAGS_AXIS3[k]
                        and not SHIPS_SATELLITE_FLAGS_AXIS3[k]
                ):
                    continue

                # this_eval_file_name = (
                #     '{0:s}/{1:s}num-cira-ir-lag-times={2:d}_'
                #     'use-ships-enviro={3:d}_use-ships-historical={4:d}_'
                #     'use-ships-satellite={5:d}{6:s}{7:s}/'
                #     'validation/evaluation.nc'
                # ).format(
                #     experiment_dir_name,
                #     'more_dropout/' if DROPOUT_RATES_AXIS1[i] > 0.55 else '',
                #     CIRA_IR_LAG_TIME_COUNTS_AXIS2[j],
                #     int(SHIPS_ENVIRO_FLAGS_AXIS3[k]),
                #     int(SHIPS_HISTORICAL_FLAGS_AXIS3[k]),
                #     int(SHIPS_SATELLITE_FLAGS_AXIS3[k]),
                #     '',
                #     '_dense-dropout-rate={0:.1f}'.format(DROPOUT_RATES_AXIS1[i])
                #     if DROPOUT_RATES_AXIS1[i] > 0.55 else ''
                # )

                this_eval_file_name = (
                    '{0:s}/{1:s}num-cira-ir-lag-times={2:d}_'
                    'use-ships-enviro={3:d}_use-ships-historical={4:d}_'
                    'use-ships-satellite={5:d}_use-temporal-diffs={6:d}{7:s}/'
                    'validation/evaluation.nc'
                ).format(
                    experiment_dir_name,
                    'more_dropout/' if DROPOUT_RATES_AXIS1[i] > 0.55 else '',
                    CIRA_IR_LAG_TIME_COUNTS_AXIS2[j],
                    int(SHIPS_ENVIRO_FLAGS_AXIS3[k]),
                    int(SHIPS_HISTORICAL_FLAGS_AXIS3[k]),
                    int(SHIPS_SATELLITE_FLAGS_AXIS3[k]),
                    int(TEMPORAL_DIFF_FLAGS_AXIS3[k]),
                    '_dense-dropout-rate={0:.1f}'.format(DROPOUT_RATES_AXIS1[i])
                    if DROPOUT_RATES_AXIS1[i] > 0.55 else ''
                )

                print('Reading data from: "{0:s}"...'.format(
                    this_eval_file_name
                ))
                t = evaluation.read_file(this_eval_file_name)

                auc_matrix[i, j, k] = numpy.mean(t[evaluation.AUC_KEY].values)
                aupd_matrix[i, j, k] = numpy.mean(t[evaluation.AUPD_KEY].values)
                bss_matrix[i, j, k] = numpy.mean(
                    t[evaluation.BRIER_SKILL_SCORE_KEY].values
                )

                these_csi_by_prob_thres = numpy.mean(
                    t[evaluation.CSI_KEY].values, axis=1
                )
                csi_matrix[i, j, k] = numpy.max(these_csi_by_prob_thres)

                this_index = numpy.argmax(these_csi_by_prob_thres)
                frequency_bias_matrix[i, j, k] = numpy.mean(
                    t[FREQUENCY_BIAS_KEY].values[this_index, :]
                )

                this_eval_file_name = this_eval_file_name.replace(
                    '/evaluation.nc', '/spread_vs_skill.nc'
                )

                print('Reading data from: "{0:s}"...'.format(
                    this_eval_file_name
                ))
                result_dict = uq_evaluation.read_spread_vs_skill(
                    this_eval_file_name
                )

                ssrel_matrix[i, j, k] = result_dict[
                    uq_evaluation.SPREAD_SKILL_RELIABILITY_KEY
                ]
                ssrat_matrix[i, j, k] = result_dict[
                    uq_evaluation.SPREAD_SKILL_RELIABILITY_KEY
                ]

                non_zero_indices = numpy.where(
                    result_dict[EXAMPLE_COUNTS_KEY] > 0
                )[0]
                mean_predictive_stdev_matrix[i, j, k] = numpy.average(
                    result_dict[MEAN_PREDICTION_STDEVS_KEY][non_zero_indices],
                    weights=result_dict[EXAMPLE_COUNTS_KEY][non_zero_indices]
                )

                this_overall_rmse = numpy.average(
                    result_dict[RMSE_VALUES_KEY][non_zero_indices],
                    weights=result_dict[EXAMPLE_COUNTS_KEY][non_zero_indices]
                )
                ssrat_matrix[i, j, k] = (
                    mean_predictive_stdev_matrix[i, j, k] / this_overall_rmse
                )

                this_eval_file_name = this_eval_file_name.replace(
                    '/spread_vs_skill.nc', '/discard_test.nc'
                )

                print('Reading data from: "{0:s}"...'.format(
                    this_eval_file_name
                ))
                monotonicity_fraction_matrix[i, j, k] = (
                    uq_evaluation.read_discard_results(this_eval_file_name)[
                        uq_evaluation.MONOTONICITY_FRACTION_KEY
                    ]
                )

    print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        auc_matrix=auc_matrix, aupd_matrix=aupd_matrix, bss_matrix=bss_matrix,
        csi_matrix=csi_matrix, frequency_bias_matrix=frequency_bias_matrix,
        ssrel_matrix=ssrel_matrix, ssrat_matrix=ssrat_matrix,
        mean_predictive_stdev_matrix=mean_predictive_stdev_matrix,
        monotonicity_fraction_matrix=monotonicity_fraction_matrix,
        rank_mainly_by_auc=True
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        auc_matrix=auc_matrix, aupd_matrix=aupd_matrix, bss_matrix=bss_matrix,
        csi_matrix=csi_matrix, frequency_bias_matrix=frequency_bias_matrix,
        ssrel_matrix=ssrel_matrix, ssrat_matrix=ssrat_matrix,
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
        score_matrix=-numpy.absolute(1. - ssrat_matrix),
        score_name='negative deviation of SSRAT from 1.0'
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

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=top_output_dir_name
    )

    auc_panel_file_names = [''] * axis3_length
    aupd_panel_file_names = [''] * axis3_length
    bss_panel_file_names = [''] * axis3_length
    csi_panel_file_names = [''] * axis3_length
    freq_bias_panel_file_names = [''] * axis3_length
    ssrel_panel_file_names = [''] * axis3_length
    ssrat_panel_file_names = [''] * axis3_length
    spread_panel_file_names = [''] * axis3_length
    mf_panel_file_names = [''] * axis3_length

    cmap_object_by_score = []
    cnorm_object_by_score = []

    letter_label = None

    for k in range(axis3_length):
        these_ships_predictor_types = []
        if SHIPS_ENVIRO_FLAGS_AXIS3[k]:
            these_ships_predictor_types.append('environmental')
        if SHIPS_HISTORICAL_FLAGS_AXIS3[k]:
            these_ships_predictor_types.append('historical')
        if SHIPS_SATELLITE_FLAGS_AXIS3[k]:
            these_ships_predictor_types.append('GOES')

        if len(these_ships_predictor_types) == 0:
            this_ships_predictor_type_string = 'none'
        else:
            this_ships_predictor_type_string = ', '.join(
                these_ships_predictor_types
            )

        # Plot AUC.
        cmap_object_by_score.append(DEFAULT_COLOUR_MAP_OBJECT)
        cnorm_object_by_score.append(matplotlib.colors.Normalize(
            vmin=numpy.nanpercentile(auc_matrix, 5),
            vmax=numpy.nanpercentile(auc_matrix, 100),
            clip=False
        ))
        
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=auc_matrix[..., k],
            min_colour_value=numpy.nanpercentile(auc_matrix, 5),
            max_colour_value=numpy.nanpercentile(auc_matrix, 100),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=cmap_object_by_score[-1],
            colour_norm_object=cnorm_object_by_score[-1]
        )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        title_string = (
            '({0:s}) CIRA IR temporal diffs = {1:s}\nSHIPS predictors = {2:s}'
        ).format(
            letter_label,
            'yes' if TEMPORAL_DIFF_FLAGS_AXIS3[k] else 'no',
            this_ships_predictor_type_string
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        auc_panel_file_names[k] = '{0:s}/auc_panel{1:02d}.jpg'.format(
            top_output_dir_name, k
        )

        print('Saving figure to: "{0:s}"...'.format(auc_panel_file_names[k]))
        figure_object.savefig(
            auc_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot AUPD.
        cmap_object_by_score.append(DEFAULT_COLOUR_MAP_OBJECT)
        cnorm_object_by_score.append(matplotlib.colors.Normalize(
            vmin=numpy.nanpercentile(aupd_matrix, 5),
            vmax=numpy.nanpercentile(aupd_matrix, 100),
            clip=False
        ))
        
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=aupd_matrix[..., k],
            min_colour_value=numpy.nanpercentile(aupd_matrix, 5),
            max_colour_value=numpy.nanpercentile(aupd_matrix, 100),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=cmap_object_by_score[-1],
            colour_norm_object=cnorm_object_by_score[-1]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        aupd_panel_file_names[k] = '{0:s}/aupd_panel{1:02d}.jpg'.format(
            top_output_dir_name, k
        )

        print('Saving figure to: "{0:s}"...'.format(aupd_panel_file_names[k]))
        figure_object.savefig(
            aupd_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot BSS.
        this_max_value = numpy.nanpercentile(bss_matrix, 100)
        this_min_value = numpy.nanpercentile(bss_matrix, 5)
        this_min_value = max([this_min_value, -1.])

        if numpy.absolute(this_max_value) > numpy.absolute(this_min_value):
            this_min_value = -1 * this_max_value
        else:
            this_max_value = -1 * this_min_value

        cmap_object_by_score.append(BSS_COLOUR_MAP_OBJECT) 
        cnorm_object_by_score.append(matplotlib.colors.Normalize(
            vmin=this_min_value, vmax=this_max_value, clip=False
        ))

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=bss_matrix[..., k],
            min_colour_value=this_min_value,
            max_colour_value=this_max_value,
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=cmap_object_by_score[-1],
            colour_norm_object=cnorm_object_by_score[-1]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        bss_panel_file_names[k] = '{0:s}/bss_panel{1:02d}.jpg'.format(
            top_output_dir_name, k
        )

        print('Saving figure to: "{0:s}"...'.format(bss_panel_file_names[k]))
        figure_object.savefig(
            bss_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot CSI.
        cmap_object_by_score.append(DEFAULT_COLOUR_MAP_OBJECT)
        cnorm_object_by_score.append(matplotlib.colors.Normalize(
            vmin=numpy.nanpercentile(csi_matrix, 5),
            vmax=numpy.nanpercentile(csi_matrix, 100),
            clip=False
        ))
        
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=csi_matrix[..., k],
            min_colour_value=numpy.nanpercentile(csi_matrix, 5),
            max_colour_value=numpy.nanpercentile(csi_matrix, 100),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=cmap_object_by_score[-1],
            colour_norm_object=cnorm_object_by_score[-1]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        csi_panel_file_names[k] = '{0:s}/csi_panel{1:02d}.jpg'.format(
            top_output_dir_name, k
        )

        print('Saving figure to: "{0:s}"...'.format(csi_panel_file_names[k]))
        figure_object.savefig(
            csi_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot frequency bias.
        this_offset = numpy.nanpercentile(
            numpy.absolute(1. - frequency_bias_matrix), 95
        )
        if numpy.isnan(this_offset):
            this_offset = 1.

        this_cmap_object, this_cnorm_object = _get_bias_colour_scheme(
            colour_map_name=BIAS_COLOUR_MAP_NAME,
            max_colour_value=1. + this_offset
        )
        cmap_object_by_score.append(this_cmap_object)
        cnorm_object_by_score.append(this_cnorm_object)

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=frequency_bias_matrix[..., k],
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            min_colour_value=0., max_colour_value=1. + this_offset,
            colour_map_object=cmap_object_by_score[-1],
            colour_norm_object=cnorm_object_by_score[-1]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        freq_bias_panel_file_names[k] = (
            '{0:s}/freq_bias_panel{1:02d}.jpg'.format(top_output_dir_name, k)
        )

        print('Saving figure to: "{0:s}"...'.format(
            freq_bias_panel_file_names[k]
        ))
        figure_object.savefig(
            freq_bias_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot SSREL.
        cmap_object_by_score.append(DEFAULT_COLOUR_MAP_OBJECT)
        cnorm_object_by_score.append(matplotlib.colors.Normalize(
            vmin=numpy.nanpercentile(ssrel_matrix, 0),
            vmax=numpy.nanpercentile(ssrel_matrix, 95),
            clip=False
        ))
        
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=ssrel_matrix[..., k],
            min_colour_value=numpy.nanpercentile(ssrel_matrix, 0),
            max_colour_value=numpy.nanpercentile(ssrel_matrix, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=cmap_object_by_score[-1],
            colour_norm_object=cnorm_object_by_score[-1]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        ssrel_panel_file_names[k] = '{0:s}/ssrel_panel{1:02d}.jpg'.format(
            top_output_dir_name, k
        )

        print('Saving figure to: "{0:s}"...'.format(ssrel_panel_file_names[k]))
        figure_object.savefig(
            ssrel_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot SSRAT.
        this_offset = numpy.nanpercentile(
            numpy.absolute(1. - ssrat_matrix), 95
        )
        if numpy.isnan(this_offset):
            this_offset = 1.

        this_cmap_object, this_cnorm_object = _get_bias_colour_scheme(
            colour_map_name=BIAS_COLOUR_MAP_NAME,
            max_colour_value=1. + this_offset
        )
        cmap_object_by_score.append(this_cmap_object)
        cnorm_object_by_score.append(this_cnorm_object)

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=ssrat_matrix[..., k],
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            min_colour_value=0., max_colour_value=1. + this_offset,
            colour_map_object=cmap_object_by_score[-1],
            colour_norm_object=cnorm_object_by_score[-1]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        ssrat_panel_file_names[k] = '{0:s}/ssrat_panel{1:02d}.jpg'.format(
            top_output_dir_name, k
        )

        print('Saving figure to: "{0:s}"...'.format(ssrat_panel_file_names[k]))
        figure_object.savefig(
            ssrat_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot mean predictive stdev.
        cmap_object_by_score.append(DEFAULT_COLOUR_MAP_OBJECT)
        cnorm_object_by_score.append(matplotlib.colors.Normalize(
            vmin=numpy.nanpercentile(mean_predictive_stdev_matrix, 5),
            vmax=numpy.nanpercentile(mean_predictive_stdev_matrix, 100),
            clip=False
        ))
        
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=mean_predictive_stdev_matrix[..., k],
            min_colour_value=
            numpy.nanpercentile(mean_predictive_stdev_matrix, 5),
            max_colour_value=
            numpy.nanpercentile(mean_predictive_stdev_matrix, 100),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=cmap_object_by_score[-1],
            colour_norm_object=cnorm_object_by_score[-1]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        spread_panel_file_names[k] = '{0:s}/spread_panel{1:02d}.jpg'.format(
            top_output_dir_name, k
        )

        print('Saving figure to: "{0:s}"...'.format(spread_panel_file_names[k]))
        figure_object.savefig(
            spread_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot monotonicity fraction.
        cmap_object_by_score.append(DEFAULT_COLOUR_MAP_OBJECT)
        cnorm_object_by_score.append(matplotlib.colors.Normalize(
            vmin=numpy.nanpercentile(monotonicity_fraction_matrix, 5),
            vmax=numpy.nanpercentile(monotonicity_fraction_matrix, 100),
            clip=False
        ))
        
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=monotonicity_fraction_matrix[..., k],
            min_colour_value=
            numpy.nanpercentile(monotonicity_fraction_matrix, 5),
            max_colour_value=
            numpy.nanpercentile(monotonicity_fraction_matrix, 100),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=cmap_object_by_score[-1],
            colour_norm_object=cnorm_object_by_score[-1]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        mf_panel_file_names[k] = '{0:s}/mono_fraction_panel{1:02d}.jpg'.format(
            top_output_dir_name, k
        )

        print('Saving figure to: "{0:s}"...'.format(mf_panel_file_names[k]))
        figure_object.savefig(
            mf_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    concat_file_name_by_score = [
        '{0:s}/auc.jpg'.format(top_output_dir_name),
        '{0:s}/aupd.jpg'.format(top_output_dir_name),
        '{0:s}/bss.jpg'.format(top_output_dir_name),
        '{0:s}/csi.jpg'.format(top_output_dir_name),
        '{0:s}/freq_bias.jpg'.format(top_output_dir_name),
        '{0:s}/ssrel.jpg'.format(top_output_dir_name),
        '{0:s}/ssrat.jpg'.format(top_output_dir_name),
        '{0:s}/spread.jpg'.format(top_output_dir_name),
        '{0:s}/mono_fraction.jpg'.format(top_output_dir_name)
    ]

    cbar_label_string_by_score = [
        'Area under ROC curve (AUC)',
        'Area under performance diagram (AUPD)',
        'Brier skill score (BSS)',
        'Critical success index (CSI)',
        'Frequency bias',
        'Spread-skill reliability (SSREL)',
        'Spread-skill ratio (SSRAT)',
        'Mean spread (ensemble stdev)',
        'Monotonicity fraction (MF)'
    ]

    panel_file_names_by_score = [
        auc_panel_file_names, aupd_panel_file_names, bss_panel_file_names,
        csi_panel_file_names, freq_bias_panel_file_names,
        ssrel_panel_file_names, ssrat_panel_file_names,
        spread_panel_file_names, mf_panel_file_names
    ]

    for m in range(len(concat_file_name_by_score)):
        for this_file_name in panel_file_names_by_score[m]:
            imagemagick_utils.resize_image(
                input_file_name=this_file_name, output_file_name=this_file_name,
                output_size_pixels=PANEL_SIZE_PX
            )

        print('Concatenating figures to: "{0:s}"...'.format(
            concat_file_name_by_score[m]
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=panel_file_names_by_score[m],
            output_file_name=concat_file_name_by_score[m],
            num_panel_rows=NUM_PANEL_ROWS,
            num_panel_columns=NUM_PANEL_COLUMNS,
            border_width_pixels=25
        )
        plotting_utils.add_colour_bar(
            figure_file_name=concat_file_name_by_score[m],
            colour_map_object=cmap_object_by_score[m],
            colour_norm_object=cnorm_object_by_score[m],
            orientation_string='vertical', font_size=30,
            cbar_label_string=cbar_label_string_by_score[m],
            tick_label_format_string='{0:.2f}'
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
