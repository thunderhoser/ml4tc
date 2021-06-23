"""Plots model evaluation."""

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
import evaluation
import evaluation_plotting

# TODO(thunderhoser): Currently this script works only for binary
# classification.

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

EVALUATION_FILE_ARG_NAME = 'input_evaluation_file_name'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EVALUATION_FILE_HELP_STRING = (
    'Path to file with evaluation results.  Will be read by '
    '`evaluation.read_file`.'
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
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(evaluation_file_name, confidence_level, output_dir_name):
    """Plots model evaluation.

    This is effectively the main method.

    :param evaluation_file_name: See documentation at top of file.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

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

    auc_values = et[evaluation.AUC_KEY].values
    num_bootstrap_reps = len(auc_values)

    if num_bootstrap_reps == 1:
        title_string = 'AUC = {0:.3f}'.format(auc_values[0])
    else:
        title_string = 'AUC = [{0:.3f}, {1:.3f}]'.format(
            numpy.percentile(auc_values, min_percentile),
            numpy.percentile(auc_values, max_percentile)
        )

    axes_object.set_title(title_string)
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
    max_csi_values = numpy.max(et[evaluation.CSI_KEY].values, axis=1)

    if num_bootstrap_reps == 1:
        title_string = 'AUPD = {0:.3f} ... max CSI = {1:.3f}'.format(
            aupd_values[0], max_csi_values[0]
        )
    else:
        title_string = (
            'AUPD = [{0:.3f}, {1:.3f}] ... max CSI = [{2:.3f}, {3:.3f}]'
        ).format(
            numpy.percentile(aupd_values, min_percentile),
            numpy.percentile(aupd_values, max_percentile),
            numpy.percentile(max_csi_values, min_percentile),
            numpy.percentile(max_csi_values, max_percentile)
        )

    axes_object.set_title(title_string)
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
    reliabilities = et[evaluation.RELIABILITY_KEY].values
    resolutions = et[evaluation.RESOLUTION_KEY].values

    if num_bootstrap_reps == 1:
        title_string = (
            'BS = {0:.3f} ... BSS = {1:.3f} ... REL = {2:.3f} ... RES = {3:.3f}'
        ).format(
            brier_scores[0], bss_values[0], reliabilities[0], resolutions[0]
        )
    else:
        title_string = (
            'BS = [{0:.3f}, {1:.3f}] ... BSS = [{2:.3f}, {3:.3f}] ... '
            'REL = [{4:.3f}, {5:.3f}] ... RES = [{6:.3f}, {7:.3f}]'
        ).format(
            numpy.percentile(brier_scores, min_percentile),
            numpy.percentile(brier_scores, max_percentile),
            numpy.percentile(bss_values, min_percentile),
            numpy.percentile(bss_values, max_percentile),
            numpy.percentile(reliabilities, min_percentile),
            numpy.percentile(reliabilities, max_percentile),
            numpy.percentile(resolutions, min_percentile),
            numpy.percentile(resolutions, max_percentile)
        )

    axes_object.set_title(title_string)
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
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
