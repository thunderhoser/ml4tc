"""Plots model evaluation."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tc.utils import evaluation
from ml4tc.plotting import evaluation_plotting

# TODO(thunderhoser): Currently this script works only for binary
# classification.

MARKER_TYPE = 'o'
MARKER_SIZE = 25
MARKER_EDGE_WIDTH = 0

TITLE_FONT_SIZE = 30

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

EVALUATION_FILE_ARG_NAME = 'input_evaluation_file_name'
EVAL_FILE_FOR_BEST_THRES_ARG_NAME = 'input_eval_file_name_for_best_thres'
MODEL_DESCRIPTION_ARG_NAME = 'model_description_string'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EVALUATION_FILE_HELP_STRING = (
    'Path to file with evaluation results.  Will be read by '
    '`evaluation.read_file`.'
)
EVAL_FILE_FOR_BEST_THRES_HELP_STRING = (
    'Same as {0:s}, except this file (containing validation data, hopefully) '
    'will be used only to determine the best probability threshold.  If you '
    'want to determine the best probability threshold from {0:s} instead, '
    'leave this argument alone.'
).format(EVALUATION_FILE_ARG_NAME)

MODEL_DESCRIPTION_HELP_STRING = (
    'Model description, for use in figure titles.  If you want plain figure '
    'titles (like just "ROC curve" and "Performance diagram"), leave this '
    'argument alone.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Level (in range 0...1) for confidence intervals based on bootstrapping.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EVALUATION_FILE_ARG_NAME, type=str, required=True,
    help=EVALUATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EVAL_FILE_FOR_BEST_THRES_ARG_NAME, type=str, required=False,
    default='', help=EVAL_FILE_FOR_BEST_THRES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DESCRIPTION_ARG_NAME, type=str, required=False, default='',
    help=MODEL_DESCRIPTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(evaluation_file_name, eval_file_name_for_best_thres,
         model_description_string, confidence_level, output_dir_name):
    """Plots model evaluation.

    This is effectively the main method.

    :param evaluation_file_name: See documentation at top of file.
    :param eval_file_name_for_best_thres: Same.
    :param model_description_string: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    if eval_file_name_for_best_thres == '':
        eval_file_name_for_best_thres = None
    else:
        assert 'testing' not in eval_file_name_for_best_thres

    if model_description_string == '':
        model_description_string = None

    model_description_string = (
        '' if model_description_string is None
        else ' for {0:s}'.format(model_description_string)
    )

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_leq(confidence_level, 1.)
    min_percentile = 50. * (1. - confidence_level)
    max_percentile = 50. * (1. + confidence_level)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
    evaluation_table_xarray = evaluation.read_file(evaluation_file_name)
    et = evaluation_table_xarray

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    evaluation_plotting.plot_roc_curve(
        axes_object=axes_object,
        pod_matrix=numpy.transpose(et[evaluation.POD_KEY].values),
        pofd_matrix=numpy.transpose(et[evaluation.POFD_KEY].values),
        confidence_level=confidence_level
    )

    best_current_prob_threshold = evaluation.find_best_threshold(
        evaluation_table_xarray=evaluation_table_xarray,
        maximize_peirce_score=False
    )
    best_current_threshold_index = numpy.argmin(numpy.absolute(
        et.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values -
        best_current_prob_threshold
    ))

    if eval_file_name_for_best_thres is None:
        best_validn_threshold_index = None
    else:
        print('Reading data from: "{0:s}"...'.format(
            eval_file_name_for_best_thres
        ))
        this_eval_table_xarray = evaluation.read_file(
            eval_file_name_for_best_thres
        )
        best_validn_prob_threshold = evaluation.find_best_threshold(
            evaluation_table_xarray=this_eval_table_xarray,
            maximize_peirce_score=False
        )
        best_validn_threshold_index = numpy.argmin(numpy.absolute(
            et.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values -
            best_validn_prob_threshold
        ))

    # best_x = numpy.mean(et[evaluation.POFD_KEY].values[best_threshold_index, :])
    # best_y = numpy.mean(et[evaluation.POD_KEY].values[best_threshold_index, :])
    # axes_object.plot(
    #     best_x, best_y, linestyle='None', marker=MARKER_TYPE,
    #     markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
    #     markerfacecolor=evaluation_plotting.ROC_CURVE_COLOUR,
    #     markeredgecolor=evaluation_plotting.ROC_CURVE_COLOUR
    # )

    auc_values = et[evaluation.AUC_KEY].values
    num_bootstrap_reps = len(auc_values)

    if num_bootstrap_reps == 1:
        title_string = 'ROC curve{0:s}\nAUC = {1:.3f}'.format(
            model_description_string, auc_values[0]
        )
    else:
        title_string = 'ROC curve{0:s}\nAUC = [{1:.3f}, {2:.3f}]'.format(
            model_description_string,
            numpy.percentile(auc_values, min_percentile),
            numpy.percentile(auc_values, max_percentile)
        )

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
    print(title_string)

    figure_file_name = '{0:s}/roc_curve.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    evaluation_plotting.plot_performance_diagram(
        axes_object=axes_object,
        pod_matrix=numpy.transpose(et[evaluation.POD_KEY].values),
        success_ratio_matrix=
        numpy.transpose(et[evaluation.SUCCESS_RATIO_KEY].values),
        confidence_level=confidence_level
    )

    aupd_values = et[evaluation.AUPD_KEY].values

    best_prob_threshold = evaluation.find_best_threshold(
        evaluation_table_xarray=evaluation_table_xarray,
        maximize_peirce_score=False
    )

    if eval_file_name_for_best_thres is not None:
        pass

    best_current_x = numpy.mean(
        et[evaluation.SUCCESS_RATIO_KEY].values[best_current_threshold_index, :]
    )
    best_current_y = numpy.mean(
        et[evaluation.POD_KEY].values[best_current_threshold_index, :]
    )
    axes_object.plot(
        best_current_x, best_current_y, linestyle='None', marker=MARKER_TYPE,
        markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
        markerfacecolor=evaluation_plotting.ROC_CURVE_COLOUR,
        markeredgecolor=evaluation_plotting.ROC_CURVE_COLOUR
    )
    axes_object.text(
        best_current_x + 0.01, best_current_y + 0.01, 'T',
        color=evaluation_plotting.ROC_CURVE_COLOUR,
        fontsize=40, fontweight='bold',
        horizontalalignment='left', verticalalignment='bottom'
    )

    if best_validn_threshold_index is None:
        this_index = best_current_threshold_index
    else:
        this_index = best_validn_threshold_index

        best_validation_x = numpy.mean(
            et[evaluation.SUCCESS_RATIO_KEY].values[
                best_validn_threshold_index, :
            ]
        )
        best_validation_y = numpy.mean(
            et[evaluation.POD_KEY].values[best_validn_threshold_index, :]
        )
        axes_object.plot(
            best_validation_x, best_validation_y, linestyle='None',
            marker=MARKER_TYPE,
            markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
            markerfacecolor=evaluation_plotting.ROC_CURVE_COLOUR,
            markeredgecolor=evaluation_plotting.ROC_CURVE_COLOUR
        )
        axes_object.text(
            best_validation_x + 0.01, best_validation_y + 0.01, 'V',
            color=evaluation_plotting.ROC_CURVE_COLOUR,
            fontsize=40, fontweight='bold',
            horizontalalignment='left', verticalalignment='bottom'
        )

    csi_values = et[evaluation.CSI_KEY].values[this_index, :]
    freq_bias_values = et[evaluation.FREQUENCY_BIAS_KEY].values[this_index, :]

    if num_bootstrap_reps == 1:
        title_string = (
            'Performance diagram{0:s}\n'
            'AUPD = {1:.3f}; CSI* = {2:.3f}; FB* = {3:.3f}'
        ).format(
            model_description_string,
            aupd_values[0], csi_values[0], freq_bias_values[0]
        )
    else:
        title_string = (
            'Performance diagram{0:s}\nAUPD = [{1:.3f}, {2:.3f}];\n'
            'CSI* = [{3:.3f}, {4:.3f}]; FB* = [{5:.3f}, {6:.3f}]'
        ).format(
            model_description_string,
            numpy.percentile(aupd_values, min_percentile),
            numpy.percentile(aupd_values, max_percentile),
            numpy.percentile(csi_values, min_percentile),
            numpy.percentile(csi_values, max_percentile),
            numpy.percentile(freq_bias_values, min_percentile),
            numpy.percentile(freq_bias_values, max_percentile),
        )

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
    print(title_string)

    figure_file_name = '{0:s}/performance_diagram.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    evaluation_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_prediction_matrix=
        numpy.transpose(et[evaluation.MEAN_PREDICTION_KEY].values),
        mean_observation_matrix=
        numpy.transpose(et[evaluation.MEAN_OBSERVATION_KEY].values),
        example_counts=et[evaluation.EXAMPLE_COUNT_NO_BS_KEY].values,
        mean_value_in_training=et.attrs[evaluation.TRAINING_EVENT_FREQ_KEY],
        min_value_to_plot=0., max_value_to_plot=1.,
        confidence_level=confidence_level
    )

    brier_scores = et[evaluation.BRIER_SCORE_KEY].values
    bss_values = et[evaluation.BRIER_SKILL_SCORE_KEY].values

    if num_bootstrap_reps == 1:
        title_string = (
            'Attributes diagram{0:s}\n'
            'BS = {1:.3f}; BSS = {2:.3f}'
        ).format(
            model_description_string, brier_scores[0], bss_values[0]
        )
    else:
        title_string = (
            'Attributes diagram{0:s}\n'
            'BS = [{1:.3f}, {2:.3f}]; BSS = [{3:.3f}, {4:.3f}]'
        ).format(
            model_description_string,
            numpy.percentile(brier_scores, min_percentile),
            numpy.percentile(brier_scores, max_percentile),
            numpy.percentile(bss_values, min_percentile),
            numpy.percentile(bss_values, max_percentile)
        )

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
    print(title_string)

    figure_file_name = '{0:s}/attributes_diagram.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_file_name=getattr(
            INPUT_ARG_OBJECT, EVALUATION_FILE_ARG_NAME
        ),
        eval_file_name_for_best_thres=getattr(
            INPUT_ARG_OBJECT, EVAL_FILE_FOR_BEST_THRES_ARG_NAME
        ),
        model_description_string=getattr(
            INPUT_ARG_OBJECT, MODEL_DESCRIPTION_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
