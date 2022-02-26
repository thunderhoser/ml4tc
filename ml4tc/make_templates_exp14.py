"""Makes CNN templates for Experiment 14, with no satellite images."""

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
import cnn_architecture

NUM_SHIPS_CHANNELS = 16 * 5 + 131 * 5

BASE_OPTION_DICT_SHIPS = {
    cnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([5, NUM_SHIPS_CHANNELS], dtype=int),
    cnn_architecture.NUM_NEURONS_KEY:
        numpy.array([1000, 1000, 1000], dtype=int),
    cnn_architecture.DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5]),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 0.,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

BASE_OPTION_DICT_UNGRIDDED_SAT = {
    # cnn_architecture.INPUT_DIMENSIONS_KEY:
    #     numpy.array([4, 16], dtype=int),
    cnn_architecture.NUM_NEURONS_KEY: numpy.array([100, 100, 100], dtype=int),
    cnn_architecture.DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5]),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 0.,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

BASE_OPTION_DICT_DENSE = {
    # cnn_architecture.NUM_NEURONS_KEY:
    #     numpy.array([1428, 232, 38, 6, 1], dtype=int),
    cnn_architecture.DROPOUT_RATES_KEY:
        numpy.array([0.5, 0.5, 0.5, 0.5, 0]),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.SIGMOID_FUNCTION_STRING,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    cnn_architecture.L2_WEIGHT_KEY: 0.,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

LAG_TIME_COUNTS = numpy.linspace(1, 25, num=25, dtype=int)

LOSS_FUNCTION = keras.losses.binary_crossentropy
METRIC_FUNCTION_LIST = [LOSS_FUNCTION] + list(neural_net.METRIC_DICT.values())

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_models/'
    'experiment14_no_images/templates'
)


def _run():
    """Makes CNN templates for Experiment 14, with no satellite images.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    for j in range(len(LAG_TIME_COUNTS)):
        option_dict_ungridded_sat = copy.deepcopy(
            BASE_OPTION_DICT_UNGRIDDED_SAT
        )
        option_dict_ships = copy.deepcopy(BASE_OPTION_DICT_SHIPS)
        option_dict_dense = copy.deepcopy(BASE_OPTION_DICT_DENSE)

        option_dict_ungridded_sat.update({
            cnn_architecture.INPUT_DIMENSIONS_KEY:
                numpy.array([LAG_TIME_COUNTS[j], 14], dtype=int)
        })

        option_dict_dense[cnn_architecture.NUM_NEURONS_KEY] = (
            architecture_utils.get_dense_layer_dimensions(
                num_input_units=1100, num_classes=2, num_dense_layers=5,
                for_classification=True
            )[1]
        )

        model_object = cnn_architecture.create_model(
            option_dict_gridded_sat=None,
            option_dict_ungridded_sat=option_dict_ungridded_sat,
            option_dict_ships=option_dict_ships,
            option_dict_dense=option_dict_dense,
            loss_function=LOSS_FUNCTION,
            metric_functions=METRIC_FUNCTION_LIST
        )

        output_file_name = '{0:s}/model_num-lag-times={1:02d}.h5'.format(
            OUTPUT_DIR_NAME, LAG_TIME_COUNTS[j]
        )

        print('Writing model to: "{0:s}"...'.format(output_file_name))
        model_object.save(
            filepath=output_file_name, overwrite=True,
            include_optimizer=True
        )


if __name__ == '__main__':
    _run()
