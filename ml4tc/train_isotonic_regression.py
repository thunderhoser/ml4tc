"""Trains isotonic-regression model."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import isotonic_regression

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
MODEL_FILE_ARG_NAME = 'output_model_file_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to input file, containing predictions to be calibrated.  Will be '
    'read by `prediction_io.read_file`.'
)
MODEL_FILE_HELP_STRING = (
    'Path to output file.  Model will be written here by '
    '`isotonic_regression.write_model`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)


def _run(prediction_file_name, output_file_name):
    """Trains isotonic-regression model.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param output_file_name: Same.
    :raises: ValueError: if predictions in `input_prediction_file_name` already
        have bias correction.
    """

    print('Reading original predictions from: "{0:s}"...'.format(
        prediction_file_name
    ))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    if (
            prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY]
            is not None
    ):
        raise ValueError('Input predictions already have bias correction.')

    model_object = isotonic_regression.train_model(prediction_dict)

    print('Writing isotonic-regression model to: "{0:s}"...'.format(
        output_file_name
    ))
    isotonic_regression.write_file(
        dill_file_name=output_file_name, model_object=model_object
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME)
    )
