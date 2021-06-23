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

import gg_model_evaluation as gg_model_eval
import file_system_utils
import prediction_io
import evaluation_plotting

# TODO(thunderhoser): Currently this script works only for binary
# classification.

NUM_PROB_THRESHOLDS = 1001
NUM_RELIABILITY_BINS = 20
EVENT_FREQ_IN_TRAINING = 0.03

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to file with predicted and actual values.  Will be read by '
    '`prediction_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_name, output_dir_name):
    """Plots model evaluation.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    target_classes = prediction_dict[prediction_io.TARGET_CLASSES_KEY]
    forecast_prob_matrix = prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY]
    num_classes = forecast_prob_matrix.shape[1]
    assert num_classes == 2

    forecast_probabilities = forecast_prob_matrix[:, 1]
    print(len(target_classes))
    print(numpy.sum(target_classes == 1))

    probability_thresholds = gg_model_eval.get_binarization_thresholds(
        threshold_arg=NUM_PROB_THRESHOLDS
    )

    pofd_values, pod_values = gg_model_eval.get_points_in_roc_curve(
        forecast_probabilities=forecast_probabilities,
        observed_labels=target_classes, threshold_arg=probability_thresholds
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    evaluation_plotting.plot_roc_curve(
        axes_object=axes_object,
        pod_matrix=numpy.expand_dims(pod_values, axis=0),
        pofd_matrix=numpy.expand_dims(pofd_values, axis=0)
    )

    figure_file_name = '{0:s}/roc_curve.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    success_ratios, pod_values = (
        gg_model_eval.get_points_in_performance_diagram(
            forecast_probabilities=forecast_probabilities,
            observed_labels=target_classes, threshold_arg=probability_thresholds
        )
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    evaluation_plotting.plot_performance_diagram(
        axes_object=axes_object,
        pod_matrix=numpy.expand_dims(pod_values, axis=0),
        success_ratio_matrix=numpy.expand_dims(success_ratios, axis=0)
    )

    figure_file_name = '{0:s}/performance_diagram.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    mean_predictions, mean_observations, example_counts = (
        gg_model_eval.get_points_in_reliability_curve(
            forecast_probabilities=forecast_probabilities,
            observed_labels=target_classes,
            num_forecast_bins=NUM_RELIABILITY_BINS
        )
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    evaluation_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_prediction_matrix=numpy.expand_dims(mean_predictions, axis=0),
        mean_observation_matrix=numpy.expand_dims(mean_observations, axis=0),
        example_counts=example_counts,
        mean_value_in_training=EVENT_FREQ_IN_TRAINING,
        min_value_to_plot=0., max_value_to_plot=1.
    )

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
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
