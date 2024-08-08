"""Plots model-comparison figure.

This will be a 3-panel figure, where each panel is a bar graph comparing 3
evaluation scores across 4 models.  The evaluation scores: AUC, AUPD, BSS.
The models: NN with everything, NN without SHIPS, NN without CIRA IR, and a
single baseline model (DTOPS or SHIPS-RII or SHIPS consensus).
"""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4tc.utils import evaluation

BASELINE_DESCRIPTION_STRINGS = ['basic', 'consensus', 'dtops']
BASELINE_DESCRIPTION_STRINGS_FANCY = ['SHIPS-RII', 'SHIPS consensus', 'DTOPS']

ERROR_BAR_DICT = {
    'ecolor': numpy.full(3, 0.),
    'elinewidth': 2,
    'capsize': 10,
    'capthick': 2
}

LINE_COLOURS_4MODELS = [
    numpy.array([27, 158, 119], dtype=float) / 255,
    numpy.array([217, 95, 2], dtype=float) / 255,
    numpy.array([117, 112, 179], dtype=float) / 255,
    numpy.full(3, 152. / 255)
]

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

NN_MODEL_DIRS_ARG_NAME = 'input_nn_model_dir_names'
NN_MODEL_DESCRIPTIONS_ARG_NAME = 'nn_model_description_strings'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

NN_MODEL_DIRS_HELP_STRING = (
    'List of input directories, one for each selected NN.  Each directory '
    'should be the top-level directory for the given NN.  This script will '
    'find results on the real-time testing data, matched with baseline models, '
    'and using isotonic regression.'
)
NN_MODEL_DESCRIPTIONS_HELP_STRING = (
    'List of NN descriptions, one per input directory.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Error bars will show this confidence level (ranging from 0...1), based on '
    'bootstrapping included in the evaluation files.'
)
OUTPUT_DIR_HELP_STRING = 'Path to output directory.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NN_MODEL_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=NN_MODEL_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NN_MODEL_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=NN_MODEL_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=True,
    help=CONFIDENCE_LEVEL_HELP_STRING
)


def _plot_comparison_1baseline(
        top_nn_model_dir_names, nn_model_description_strings, confidence_level,
        output_dir_name, baseline_index):
    """Plots comparison figure for one baseline model.

    :param top_nn_model_dir_names: See documentation at top of this script.
    :param nn_model_description_strings: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    :param baseline_index: Index of baseline model to include in figure.
    :return: output_file_name: Path to output file, where figure was saved.
    """

    baseline_description_string = BASELINE_DESCRIPTION_STRINGS[baseline_index]
    num_nn_models = len(nn_model_description_strings)
    num_models = num_nn_models + 1

    auc_matrix = None
    aupd_matrix = None
    bss_matrix = None

    for j in range(num_nn_models):
        this_eval_file_name = (
            '{0:s}/real_time_testing_matched_with_ships/'
            'isotonic_regression/cnn_evaluation_1000reps_cf_{1:s}.nc'
        ).format(top_nn_model_dir_names[j], baseline_description_string)

        print('Reading data from: "{0:s}"...'.format(this_eval_file_name))
        this_eval_table_xarray = evaluation.read_file(this_eval_file_name)

        if auc_matrix is None:
            num_bootstrap_reps = len(
                this_eval_table_xarray[evaluation.AUC_KEY].values
            )
            these_dim = (num_models, num_bootstrap_reps)

            auc_matrix = numpy.full(these_dim, numpy.nan)
            aupd_matrix = numpy.full(these_dim, numpy.nan)
            bss_matrix = numpy.full(these_dim, numpy.nan)

        auc_matrix[j, :] = this_eval_table_xarray[evaluation.AUC_KEY].values
        aupd_matrix[j, :] = (
            this_eval_table_xarray[evaluation.AUPD_KEY].values
        )
        bss_matrix[j, :] = (
            this_eval_table_xarray[evaluation.BRIER_SKILL_SCORE_KEY].values
        )

        if j > 0:
            continue

        this_eval_file_name = (
            '{0:s}/real_time_testing_matched_with_ships/'
            'isotonic_regression/ships_evaluation_1000reps_{1:s}.nc'
        ).format(top_nn_model_dir_names[j], baseline_description_string)

        print('Reading data from: "{0:s}"...'.format(this_eval_file_name))
        this_eval_table_xarray = evaluation.read_file(this_eval_file_name)

        auc_matrix[-1, :] = (
            this_eval_table_xarray[evaluation.AUC_KEY].values
        )
        aupd_matrix[-1, :] = (
            this_eval_table_xarray[evaluation.AUPD_KEY].values
        )
        bss_matrix[-1, :] = (
            this_eval_table_xarray[evaluation.BRIER_SKILL_SCORE_KEY].values
        )

    mean_values = numpy.concatenate([
        numpy.mean(auc_matrix, axis=1),
        numpy.mean(aupd_matrix, axis=1),
        numpy.mean(bss_matrix, axis=1)
    ])

    min_percentile = 50. * (1. - confidence_level)
    lower_bounds = numpy.concatenate([
        numpy.percentile(auc_matrix, min_percentile, axis=1),
        numpy.percentile(aupd_matrix, min_percentile, axis=1),
        numpy.percentile(bss_matrix, min_percentile, axis=1)
    ])

    max_percentile = 50. * (1. + confidence_level)
    upper_bounds = numpy.concatenate([
        numpy.percentile(auc_matrix, max_percentile, axis=1),
        numpy.percentile(aupd_matrix, max_percentile, axis=1),
        numpy.percentile(bss_matrix, max_percentile, axis=1)
    ])

    model_indices = numpy.linspace(
        0, num_models - 1, num=num_models, dtype=float
    )
    model_description_strings = (
        nn_model_description_strings +
        [BASELINE_DESCRIPTION_STRINGS_FANCY[baseline_index]]
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    legend_handles = [None] * num_models
    x_min = 1.
    x_max = -1.
    bar_width = 1. / (num_models + 1)
    num_metrics = 3

    for k in range(num_metrics):
        these_mean_values = mean_values[k * num_models:(k + 1) * num_models]
        these_lower_bounds = lower_bounds[k * num_models:(k + 1) * num_models]
        these_upper_bounds = upper_bounds[k * num_models:(k + 1) * num_models]
        this_error_bar_width_matrix = numpy.vstack([
            these_mean_values - these_lower_bounds,
            these_upper_bounds - these_mean_values
        ])

        for j in range(num_models):
            this_x = k + j * bar_width
            x_min = min([x_min, this_x])
            x_max = max([x_max, this_x])

            legend_handles[j] = axes_object.bar(
                x=this_x, height=these_mean_values[j], width=bar_width,
                color=LINE_COLOURS_4MODELS[j],
                yerr=this_error_bar_width_matrix[:, [j]],
                error_kw=ERROR_BAR_DICT
            )

    x_min -= 0.75 * bar_width
    x_max += 0.75 * bar_width
    axes_object.set_xlim([x_min, x_max])

    metric_indices = numpy.linspace(
        0, num_metrics - 1, num=num_metrics, dtype=float
    )
    x_tick_values = numpy.mean(model_indices * bar_width) + metric_indices
    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(['AUC', 'AUPD', 'BSS'])

    axes_object.set_title('NN comparison with {0:s}'.format(
        model_description_strings[-1]
    ))
    axes_object.legend(
        legend_handles, model_description_strings, loc='upper right'
    )

    panel_letter = chr(ord('a') + baseline_index)
    gg_plotting_utils.label_axes(
        axes_object=axes_object,
        label_string='({0:s})'.format(panel_letter)
    )

    output_file_name = '{0:s}/comparison_with_{1:s}.jpg'.format(
        output_dir_name, BASELINE_DESCRIPTION_STRINGS[baseline_index]
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return output_file_name


def _run(top_nn_model_dir_names, nn_model_description_strings, confidence_level,
         output_dir_name):
    """Plots model-comparison figure.

    This is effectively the main method.

    :param top_nn_model_dir_names: See documentation at top of this script.
    :param nn_model_description_strings: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(confidence_level, 0.8)
    error_checking.assert_is_less_than(confidence_level, 1.)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # TODO(thunderhoser): The code could be generalized to handle any number of
    # NN models by simply adding colours as an input arg, rather than forcing
    # the use of `LINE_COLOURS_4MODELS`.
    num_nn_models = len(top_nn_model_dir_names)
    assert num_nn_models == 3

    assert len(nn_model_description_strings) == num_nn_models
    nn_model_description_strings = [
        s.replace('_', ' ') for s in nn_model_description_strings
    ]

    num_baseline_models = len(BASELINE_DESCRIPTION_STRINGS)
    panel_file_names = [''] * num_baseline_models

    for i in range(num_baseline_models):
        panel_file_names[i] = _plot_comparison_1baseline(
            top_nn_model_dir_names=top_nn_model_dir_names,
            nn_model_description_strings=nn_model_description_strings,
            confidence_level=confidence_level,
            output_dir_name=output_dir_name,
            baseline_index=i
        )
        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[i],
            output_file_name=panel_file_names[i],
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/comparison_with_all.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2,
        num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_nn_model_dir_names=getattr(
            INPUT_ARG_OBJECT, NN_MODEL_DIRS_ARG_NAME
        ),
        nn_model_description_strings=getattr(
            INPUT_ARG_OBJECT, NN_MODEL_DESCRIPTIONS_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
