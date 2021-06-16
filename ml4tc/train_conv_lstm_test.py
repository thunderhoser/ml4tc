"""Trains test conv-LSTM model."""

import os
import sys
import numpy
import tensorflow.keras as tf_keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net

TEMPLATE_FILE_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/ml4tc_models/conv_lstm_test/'
    'conv_lstm_test_template.h5'
)
OUTPUT_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/ml4tc_models/conv_lstm_test'
EXAMPLE_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_project/'
    'learning_examples/imputed/normalized'
)

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
TRAINING_YEARS = numpy.concatenate((
    numpy.linspace(1993, 2004, num=12, dtype=int),
    numpy.linspace(2015, 2019, num=5, dtype=int)
))
VALIDATION_YEARS = numpy.linspace(2005, 2009, num=5, dtype=int)

TRAINING_OPTION_DICT = {
    neural_net.EXAMPLE_DIRECTORY_KEY: EXAMPLE_DIR_NAME,
    neural_net.YEARS_KEY: TRAINING_YEARS,
    neural_net.LEAD_TIME_KEY: 24,
    neural_net.SATELLITE_LAG_TIMES_KEY:
        numpy.array([0, 60, 120, 180], dtype=int),
    neural_net.SHIPS_LAG_TIMES_KEY: numpy.array([0, 6, 12, 18], dtype=int),
    neural_net.NUM_EXAMPLES_PER_BATCH_KEY: 32,
    neural_net.MAX_EXAMPLES_PER_CYCLONE_KEY: 4,
    neural_net.CLASS_CUTOFFS_KEY: numpy.array([25 * KT_TO_METRES_PER_SECOND])
}

VALIDATION_OPTION_DICT = {
    neural_net.EXAMPLE_DIRECTORY_KEY: EXAMPLE_DIR_NAME,
    neural_net.YEARS_KEY: VALIDATION_YEARS
}


def _run():
    """Trains test CNN.

    This is effectively the main method.
    """

    model_object = tf_keras.models.load_model(TEMPLATE_FILE_NAME)

    neural_net.train_model(
        model_object=model_object, output_dir_name=OUTPUT_DIR_NAME,
        num_epochs=100,
        num_training_batches_per_epoch=32,
        training_option_dict=TRAINING_OPTION_DICT,
        num_validation_batches_per_epoch=16,
        validation_option_dict=VALIDATION_OPTION_DICT,
    )


if __name__ == '__main__':
    _run()
