"""Evaluates model predictions."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
LEAD_TIMES_ARG_NAME = 'lead_times_hours'
EVENT_FREQ_ARG_NAME = 'event_freq_in_training'
NUM_PROB_THRESHOLDS_ARG_NAME = 'num_prob_thresholds'
NUM_RELIABILITY_BINS_ARG_NAME = 'num_reliability_bins'
NUM_BOOTSTRAP_REPS_ARG_NAME = 'num_bootstrap_reps'
OUTPUT_FILE_ARG_NAME = 'output_eval_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and target values.  Will be '
    'read by `prediction_io.read_file`.'
)
LEAD_TIMES_HELP_STRING = (
    'List of lead times.  Will evaluate model aggregated over these lead '
    'times.  If you want to aggregate over all lead times, leave this argument '
    'alone.'
)
EVENT_FREQ_HELP_STRING = (
    'Event frequency in training data (ranging from 0...1).'
)
NUM_PROB_THRESHOLDS_HELP_STRING = (
    'Number of probability thresholds for deterministic predictions.'
)
NUM_RELIABILITY_BINS_HELP_STRING = 'Number of bins for reliability curves.'
NUM_BOOTSTRAP_REPS_HELP_STRING = 'Number of replicates for bootstrapping.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Evaluation results will be written here by '
    '`evaluation.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=LEAD_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EVENT_FREQ_ARG_NAME, type=float, required=True,
    help=EVENT_FREQ_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PROB_THRESHOLDS_ARG_NAME, type=int, required=False,
    default=evaluation.DEFAULT_NUM_PROB_THRESHOLDS,
    help=NUM_PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_RELIABILITY_BINS_ARG_NAME, type=int, required=False,
    default=evaluation.DEFAULT_NUM_RELIABILITY_BINS,
    help=NUM_RELIABILITY_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_REPS_ARG_NAME, type=int, required=False,
    default=evaluation.DEFAULT_NUM_BOOTSTRAP_REPS,
    help=NUM_BOOTSTRAP_REPS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_name, lead_times_hours, event_freq_in_training,
         num_prob_thresholds, num_reliability_bins, num_bootstrap_reps,
         output_file_name):
    """Evaluates model predictions.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param lead_times_hours: Same.
    :param event_freq_in_training: Same.
    :param num_prob_thresholds: Same.
    :param num_reliability_bins: Same.
    :param num_bootstrap_reps: Same.
    :param output_file_name: Same.
    """

    if len(lead_times_hours) == 1 and lead_times_hours[0] <= 0:
        lead_times_hours = None

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    if lead_times_hours is not None:
        prediction_dict = prediction_io.subset_by_lead_time(
            prediction_dict=prediction_dict, lead_times_hours=lead_times_hours
        )

    num_lead_times = len(prediction_dict[prediction_io.LEAD_TIMES_KEY])
    cyclone_id_strings = prediction_dict[prediction_io.CYCLONE_IDS_KEY]
    init_times_unix_sec = prediction_dict[prediction_io.INIT_TIMES_KEY]

    if num_lead_times > 1:
        cyclone_id_strings = numpy.tile(
            numpy.array(cyclone_id_strings), reps=num_lead_times
        )
        cyclone_id_strings = cyclone_id_strings.tolist()
        init_times_unix_sec = numpy.tile(
            init_times_unix_sec, reps=num_lead_times
        )

    forecast_prob_matrix = prediction_io.get_mean_predictions(prediction_dict)
    forecast_probabilities = numpy.ravel(forecast_prob_matrix)
    target_classes = numpy.ravel(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY]
    )

    good_indices = numpy.where(numpy.isfinite(forecast_probabilities))[0]

    print(SEPARATOR_STRING)

    evaluation_table_xarray = evaluation.evaluate_model_binary(
        forecast_probabilities=forecast_probabilities[good_indices],
        target_classes=target_classes[good_indices],
        event_freq_in_training=event_freq_in_training,
        cyclone_id_strings=[cyclone_id_strings[k] for k in good_indices],
        init_times_unix_sec=init_times_unix_sec[good_indices],
        model_file_name=prediction_dict[prediction_io.MODEL_FILE_KEY],
        num_prob_thresholds=num_prob_thresholds,
        num_reliability_bins=num_reliability_bins,
        num_bootstrap_reps=num_bootstrap_reps
    )

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    evaluation.write_file(
        evaluation_table_xarray=evaluation_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        lead_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, LEAD_TIMES_ARG_NAME), dtype=int
        ),
        event_freq_in_training=getattr(INPUT_ARG_OBJECT, EVENT_FREQ_ARG_NAME),
        num_prob_thresholds=getattr(
            INPUT_ARG_OBJECT, NUM_PROB_THRESHOLDS_ARG_NAME
        ),
        num_reliability_bins=getattr(
            INPUT_ARG_OBJECT, NUM_RELIABILITY_BINS_ARG_NAME
        ),
        num_bootstrap_reps=getattr(
            INPUT_ARG_OBJECT, NUM_BOOTSTRAP_REPS_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
