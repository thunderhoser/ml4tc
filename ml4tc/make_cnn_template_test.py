"""Makes template for test CNN."""

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
import cnn_architecture

OPTION_DICT_GRIDDED_SAT = {
    cnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([480, 640, 4, 1], dtype=int),
    cnn_architecture.NUM_LAYERS_BY_BLOCK_KEY: numpy.full(7, 2, dtype=int),
    cnn_architecture.NUM_CHANNELS_KEY: numpy.array(
        [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512], dtype=int
    ),
    cnn_architecture.DROPOUT_RATES_KEY: numpy.full(14, 0.),
    cnn_architecture.ACTIVATION_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 1e-6,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

OPTION_DICT_UNGRIDDED_SAT = {
    cnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([4, 16], dtype=int),
    cnn_architecture.NUM_NEURONS_KEY: numpy.array([50, 100], dtype=int),
    cnn_architecture.DROPOUT_RATES_KEY: numpy.array([0.25, 0.25]),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 0.,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

OPTION_DICT_SHIPS = {
    cnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([4, 2943], dtype=int),
    cnn_architecture.NUM_NEURONS_KEY: numpy.array([500, 1000], dtype=int),
    cnn_architecture.DROPOUT_RATES_KEY: numpy.array([0.25, 0.25]),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 0.,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

OPTION_DICT_DENSE = {
    cnn_architecture.NUM_NEURONS_KEY:
        numpy.array([1428, 232, 38, 6, 1], dtype=int),
    cnn_architecture.DROPOUT_RATES_KEY:
        numpy.array([0.25, 0.25, 0.25, 0.25, 0]),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.SIGMOID_FUNCTION_STRING,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    cnn_architecture.L2_WEIGHT_KEY: 0.,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

OUTPUT_FILE_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/ml4tc_models/cnn_test/cnn_test_template.h5'
)


def _run():
    """Makes template for test CNN.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=OUTPUT_FILE_NAME)

    model_object = cnn_architecture.create_model(
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
