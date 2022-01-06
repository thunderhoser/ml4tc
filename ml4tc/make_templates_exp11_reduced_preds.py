"""Makes conv-LSTM templates for Experiment 11 with reduced predictor set."""

import os
import sys
import copy
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import architecture_utils
import neural_net
import conv_lstm_architecture

BASE_OPTION_DICT_GRIDDED_SAT = {
    # conv_lstm_architecture.INPUT_DIMENSIONS_KEY:
    #     numpy.array([380, 540, 4, 1], dtype=int),
    conv_lstm_architecture.NUM_LAYERS_BY_BLOCK_KEY: numpy.full(5, 2, dtype=int),
    # conv_lstm_architecture.NUM_CHANNELS_KEY: numpy.array(
    #     [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256], dtype=int
    # ),
    conv_lstm_architecture.DROPOUT_RATES_KEY: numpy.full(10, 0.),
    conv_lstm_architecture.KEEP_TIME_DIMENSION_KEY: True,
    conv_lstm_architecture.ACTIVATION_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    conv_lstm_architecture.ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    conv_lstm_architecture.L2_WEIGHT_KEY: 1e-3,
    conv_lstm_architecture.USE_BATCH_NORM_KEY: True
}

BASE_OPTION_DICT_SHIPS = {
    conv_lstm_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([5, 12], dtype=int),
    conv_lstm_architecture.NUM_CHANNELS_KEY: numpy.array([10, 50], dtype=int),
    conv_lstm_architecture.DROPOUT_RATES_KEY: numpy.array([0.5, 0.5]),
    conv_lstm_architecture.ACTIVATION_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    conv_lstm_architecture.ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    conv_lstm_architecture.L2_WEIGHT_KEY: 0.,
    conv_lstm_architecture.USE_BATCH_NORM_KEY: True
}

BASE_OPTION_DICT_DENSE = {
    # conv_lstm_architecture.NUM_NEURONS_KEY:
    #     numpy.array([1428, 232, 38, 6, 1], dtype=int),
    conv_lstm_architecture.DROPOUT_RATES_KEY:
        numpy.array([0.5, 0.5, 0.5, 0.5, 0]),
    conv_lstm_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    conv_lstm_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    conv_lstm_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.SIGMOID_FUNCTION_STRING,
    conv_lstm_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    conv_lstm_architecture.L2_WEIGHT_KEY: 0.,
    conv_lstm_architecture.USE_BATCH_NORM_KEY: True
}

CHANNEL_COUNTS_ARRAY = [
    numpy.array([4, 4, 8, 8, 12, 12, 16, 16, 24, 24, 32, 32], dtype=int),
    numpy.array([4, 4, 8, 8, 16, 16, 32, 32, 48, 48, 64, 64], dtype=int),
    numpy.array([4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 96, 96], dtype=int),
    numpy.array([2, 2, 4, 4, 6, 6, 8, 8, 12, 12, 16, 16], dtype=int)
]

LAG_TIME_COUNTS = numpy.linspace(1, 25, num=25, dtype=int)

LOSS_FUNCTION = keras.losses.binary_crossentropy
METRIC_FUNCTION_LIST = [LOSS_FUNCTION] + list(neural_net.METRIC_DICT.values())

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_models/'
    'experiment11_reduced_preds/templates'
)


def _run():
    """Makes conv-LSTM templates for Experiment 11 with reduced predictor set.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    for i in range(len(CHANNEL_COUNTS_ARRAY)):
        for j in range(len(LAG_TIME_COUNTS)):
            option_dict_gridded_sat = copy.deepcopy(
                BASE_OPTION_DICT_GRIDDED_SAT
            )
            option_dict_ships = copy.deepcopy(BASE_OPTION_DICT_SHIPS)
            option_dict_dense = copy.deepcopy(BASE_OPTION_DICT_DENSE)

            option_dict_gridded_sat.update({
                conv_lstm_architecture.INPUT_DIMENSIONS_KEY:
                    numpy.array([190, 270, LAG_TIME_COUNTS[j], 1], dtype=int),
                conv_lstm_architecture.NUM_CHANNELS_KEY:
                    CHANNEL_COUNTS_ARRAY[i]
            })

            option_dict_dense[conv_lstm_architecture.NUM_NEURONS_KEY] = (
                architecture_utils.get_dense_layer_dimensions(
                    num_input_units=40 * CHANNEL_COUNTS_ARRAY[i][-1] + 50,
                    num_classes=2, num_dense_layers=5, for_classification=True
                )[1]
            )

            model_object = conv_lstm_architecture.create_model(
                option_dict_gridded_sat=option_dict_gridded_sat,
                option_dict_ungridded_sat=None,
                option_dict_ships=option_dict_ships,
                option_dict_dense=option_dict_dense,
                loss_function=LOSS_FUNCTION,
                metric_functions=METRIC_FUNCTION_LIST
            )

            channel_counts_string = '-'.join([
                '{0:03d}'.format(c) for c in CHANNEL_COUNTS_ARRAY[i]
            ])

            output_file_name = (
                '{0:s}/model_channels={1:s}_num-lag-times={2:02d}.h5'
            ).format(
                OUTPUT_DIR_NAME, channel_counts_string, LAG_TIME_COUNTS[j]
            )

            print('Writing model to: "{0:s}"...'.format(output_file_name))
            model_object.save(
                filepath=output_file_name, overwrite=True,
                include_optimizer=True
            )


if __name__ == '__main__':
    _run()
