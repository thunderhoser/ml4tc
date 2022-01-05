"""Makes template for test conv-LSTM model."""

import os
import sys
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import architecture_utils
import conv_lstm_architecture

OPTION_DICT_GRIDDED_SAT = {
    conv_lstm_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([480, 640, 4, 1], dtype=int),
    conv_lstm_architecture.NUM_LAYERS_BY_BLOCK_KEY: numpy.full(7, 2, dtype=int),
    conv_lstm_architecture.NUM_CHANNELS_KEY: numpy.array(
        [8, 8, 16, 16, 24, 24, 32, 32, 48, 48, 64, 64, 128, 128], dtype=int
    ),
    conv_lstm_architecture.DROPOUT_RATES_KEY: numpy.full(14, 0.),
    conv_lstm_architecture.KEEP_TIME_DIMENSION_KEY: True,
    conv_lstm_architecture.ACTIVATION_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    conv_lstm_architecture.ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    conv_lstm_architecture.L2_WEIGHT_KEY: 1e-6,
    conv_lstm_architecture.USE_BATCH_NORM_KEY: True
}

OPTION_DICT_UNGRIDDED_SAT = {
    conv_lstm_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([4, 16], dtype=int),
    conv_lstm_architecture.NUM_CHANNELS_KEY: numpy.array([50, 100], dtype=int),
    conv_lstm_architecture.DROPOUT_RATES_KEY: numpy.array([0.25, 0.25]),
    conv_lstm_architecture.ACTIVATION_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    conv_lstm_architecture.ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    conv_lstm_architecture.L2_WEIGHT_KEY: 0.,
    conv_lstm_architecture.USE_BATCH_NORM_KEY: True
}

OPTION_DICT_SHIPS = {
    conv_lstm_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([4, 2943], dtype=int),
    conv_lstm_architecture.NUM_CHANNELS_KEY: numpy.array([300, 600], dtype=int),
    conv_lstm_architecture.DROPOUT_RATES_KEY: numpy.array([0.25, 0.25]),
    conv_lstm_architecture.ACTIVATION_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    conv_lstm_architecture.ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    conv_lstm_architecture.L2_WEIGHT_KEY: 0.,
    conv_lstm_architecture.USE_BATCH_NORM_KEY: True
}

OPTION_DICT_DENSE = {
    conv_lstm_architecture.NUM_NEURONS_KEY:
        numpy.array([543, 112, 23, 5, 1], dtype=int),
    conv_lstm_architecture.DROPOUT_RATES_KEY:
        numpy.array([0.25, 0.25, 0.25, 0.25, 0]),
    conv_lstm_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    conv_lstm_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    conv_lstm_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.SIGMOID_FUNCTION_STRING,
    conv_lstm_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    conv_lstm_architecture.L2_WEIGHT_KEY: 0.,
    conv_lstm_architecture.USE_BATCH_NORM_KEY: True
}

OUTPUT_FILE_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/ml4tc_models/conv_lstm_test/'
    'conv_lstm_test_template.h5'
)


def _run():
    """Makes template for test conv-LSTM model.

    This is effectively the main method.
    """

    # file_system_utils.mkdir_recursive_if_necessary(file_name=OUTPUT_FILE_NAME)

    model_object = conv_lstm_architecture.create_model(
        option_dict_gridded_sat=OPTION_DICT_GRIDDED_SAT,
        option_dict_ungridded_sat=OPTION_DICT_UNGRIDDED_SAT,
        option_dict_ships=OPTION_DICT_SHIPS,
        option_dict_dense=OPTION_DICT_DENSE,
        loss_function=keras.losses.binary_crossentropy,
        metric_functions=[keras.losses.binary_crossentropy]
    )

    print('Writing model to: "{0:s}"...'.format(OUTPUT_FILE_NAME))
    model_object.save(
        filepath=OUTPUT_FILE_NAME, overwrite=True, include_optimizer=True
    )


if __name__ == '__main__':
    _run()
