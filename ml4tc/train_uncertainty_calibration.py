"""Trains uncertainty-calibration model."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import uncertainty_calibration

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
BIN_EDGES_ARG_NAME = 'bin_edge_prediction_stdevs'
MODEL_FILE_ARG_NAME = 'output_model_file_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to input file, containing predictions to be calibrated.  Will be '
    'read by `prediction_io.read_file`.'
)
BIN_EDGES_HELP_STRING = (
    'List of bin cutoffs -- ranging from (0, 1) -- each a standard deviation '
    'of the predictive distribution.  This script will automatically use 0 and '
    '1 as the lowest and highest cutoffs.'
)
MODEL_FILE_HELP_STRING = (
    'Path to output file.  Model will be written here by '
    '`uncertainty_calibration.write_model`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BIN_EDGES_ARG_NAME, type=float, nargs='+', required=True,
    help=BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)


def _run(prediction_file_name, bin_edge_prediction_stdevs,
         output_file_name):
    """Trains uncertainty-calibration model.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param bin_edge_prediction_stdevs: Same.
    :param output_file_name: Same.
    :raises: ValueError: if predictions in `input_prediction_file_name` already
        have calibrated uncertainty.
    """

    print('Reading original predictions from: "{0:s}"...'.format(
        prediction_file_name
    ))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    if (
            prediction_dict[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
            is not None
    ):
        raise ValueError(
            'Input predictions already have calibrated uncertainty.'
        )

    bin_edge_prediction_stdevs, stdev_inflation_factors = (
        uncertainty_calibration.train_model(
            prediction_dict=prediction_dict,
            bin_edge_prediction_stdevs=bin_edge_prediction_stdevs
        )
    )

    print('Writing uncertainty-calibration model to: "{0:s}"...'.format(
        output_file_name
    ))
    uncertainty_calibration.write_model(
        netcdf_file_name=output_file_name,
        bin_edge_prediction_stdevs=bin_edge_prediction_stdevs,
        stdev_inflation_factors=stdev_inflation_factors
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        bin_edge_prediction_stdevs=numpy.array(
            getattr(INPUT_ARG_OBJECT, BIN_EDGES_ARG_NAME), dtype=float
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME)
    )
