"""Makes CNN templates for BNN Experiment 1 for rapid intensification."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import architecture_utils
import neural_net
import cnn_architecture
import cnn_architecture_bayesian as bcnn_architecture

ENSEMBLE_SIZE = 100

OPTION_DICT_GRIDDED_SAT = {
    cnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([380, 540, 1, 1], dtype=int),
    cnn_architecture.NUM_LAYERS_BY_BLOCK_KEY: numpy.full(6, 2, dtype=int),
    cnn_architecture.NUM_CHANNELS_KEY: numpy.array(
        [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256], dtype=int
    ),
    cnn_architecture.DROPOUT_RATES_KEY: numpy.full(12, 0.),
    cnn_architecture.DROPOUT_MC_FLAGS_KEY: numpy.full(12, 0, dtype=bool),
    cnn_architecture.ACTIVATION_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 1e-7,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

OPTION_DICT_SHIPS = {
    cnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([5, 70], dtype=int),
    cnn_architecture.NUM_NEURONS_KEY: numpy.array([500, 1000], dtype=int),
    cnn_architecture.DROPOUT_RATES_KEY: numpy.full(2, 0.5),
    cnn_architecture.DROPOUT_MC_FLAGS_KEY: numpy.full(2, 0, dtype=bool),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 0.,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

NO_UNC_STRING = bcnn_architecture.POINT_ESTIMATE_TYPE_STRING
FLIPOUT_STRING = bcnn_architecture.FLIPOUT_TYPE_STRING
REPARAM_STRING = bcnn_architecture.REPARAMETERIZATION_TYPE_STRING

BASE_OPTION_DICT_DENSE = {
    # bcnn_architecture.NUM_NEURONS_KEY:
    #     numpy.array([1428, 232, 38, 6, 1], dtype=int),
    # bcnn_architecture.DROPOUT_RATES_KEY:
    #     numpy.array([0.5, 0.5, 0.5, 0.5, 0]),
    bcnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    bcnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    bcnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.SIGMOID_FUNCTION_STRING,
    bcnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    bcnn_architecture.USE_BATCH_NORM_KEY: True,
    bcnn_architecture.LAST_DROPOUT_BEFORE_ACTIV_KEY: True,
    bcnn_architecture.KL_SCALING_FACTOR_KEY: 1e-5,
    # bcnn_architecture.LAYER_TYPES_KEY: [
    #     NO_UNC_STRING, NO_UNC_STRING,
    #     FLIPOUT_STRING, FLIPOUT_STRING, FLIPOUT_STRING
    # ],
    bcnn_architecture.ENSEMBLE_SIZE_KEY: ENSEMBLE_SIZE
}

DROPOUT_RATES_2D_LIST = [
    numpy.array([0.5, 0.7, 0.7, 0.5, 0.0]),
    numpy.array([0.5, 0.9, 0.5, 0.3, 0.0]),
    numpy.array([0.5, 0.3, 0.3, 0.9, 0.0]),
    numpy.array([0.5, 0.5, 0.5, 0.1, 0.0]),
    numpy.array([0.5, 0.3, 0.5, 0.9, 0.0]),
    numpy.array([0.5, 0.9, 0.1, 0.5, 0.0]),
    numpy.array([0.5, 0.1, 0.5, 0.3, 0.0])
]

DROPOUT_RATE_ABBREVS = [
    '0.700-0.700-0.500',
    '0.900-0.500-0.300',
    '0.300-0.300-0.900',
    '0.500-0.500-0.100',
    '0.300-0.500-0.900',
    '0.900-0.100-0.500',
    '0.100-0.500-0.300'
]

# LAYER_TYPE_STRINGS_2D_LIST = [
#     [NO_UNC_STRING, NO_UNC_STRING, FLIPOUT_STRING],
#     [NO_UNC_STRING, NO_UNC_STRING, REPARAM_STRING],
#     [NO_UNC_STRING, FLIPOUT_STRING, FLIPOUT_STRING],
#     [NO_UNC_STRING, FLIPOUT_STRING, REPARAM_STRING],
#     [NO_UNC_STRING, REPARAM_STRING, FLIPOUT_STRING],
#     [NO_UNC_STRING, REPARAM_STRING, REPARAM_STRING],
#     [FLIPOUT_STRING, FLIPOUT_STRING, FLIPOUT_STRING],
#     [FLIPOUT_STRING, FLIPOUT_STRING, REPARAM_STRING],
#     [FLIPOUT_STRING, REPARAM_STRING, FLIPOUT_STRING],
#     [FLIPOUT_STRING, REPARAM_STRING, REPARAM_STRING],
#     [REPARAM_STRING, FLIPOUT_STRING, FLIPOUT_STRING],
#     [REPARAM_STRING, FLIPOUT_STRING, REPARAM_STRING],
#     [REPARAM_STRING, REPARAM_STRING, FLIPOUT_STRING],
#     [REPARAM_STRING, REPARAM_STRING, REPARAM_STRING]
# ]

LAYER_TYPE_STRINGS_2D_LIST = [
    [NO_UNC_STRING, NO_UNC_STRING, FLIPOUT_STRING],
    [NO_UNC_STRING, NO_UNC_STRING, REPARAM_STRING],
    [NO_UNC_STRING, FLIPOUT_STRING, FLIPOUT_STRING],
    [NO_UNC_STRING, REPARAM_STRING, REPARAM_STRING],
    [FLIPOUT_STRING, FLIPOUT_STRING, FLIPOUT_STRING],
    [REPARAM_STRING, REPARAM_STRING, REPARAM_STRING]
]

for m in range(len(LAYER_TYPE_STRINGS_2D_LIST)):
    LAYER_TYPE_STRINGS_2D_LIST[m].insert(0, NO_UNC_STRING)
    LAYER_TYPE_STRINGS_2D_LIST[m].insert(0, NO_UNC_STRING)

LAYER_TYPE_ABBREVS = [
    'point-point-flipout', 'point-point-reparam',
    'point-flipout-flipout', 'point-reparam-reparam',
    'flipout-flipout-flipout', 'reparam-reparam-reparam'
]

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_models/bnn_experiment01/'
    'templates'
)


def _run():
    """Makes CNN templates for BNN Experiment 1 for rapid intensification.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    num_dropout_rate_sets = len(DROPOUT_RATES_2D_LIST)
    num_layer_type_sets = len(LAYER_TYPE_STRINGS_2D_LIST)

    for i in range(num_layer_type_sets):
        for j in range(num_dropout_rate_sets):
            d_sat = OPTION_DICT_GRIDDED_SAT
            d_ships = OPTION_DICT_SHIPS
            num_flattened_features = (
                40 * d_sat[cnn_architecture.NUM_CHANNELS_KEY][-1] +
                d_ships[cnn_architecture.NUM_NEURONS_KEY][-1]
            )

            exp_neuron_counts = architecture_utils.get_dense_layer_dimensions(
                num_input_units=num_flattened_features,
                num_classes=2, num_dense_layers=5,
                for_classification=True
            )[1]
            neuron_counts = numpy.linspace(
                1, exp_neuron_counts[0], num=5, dtype=int
            )[::-1]
            neuron_counts[-1] = ENSEMBLE_SIZE

            option_dict_dense = copy.deepcopy(BASE_OPTION_DICT_DENSE)
            option_dict_dense[bcnn_architecture.NUM_NEURONS_KEY] = neuron_counts
            option_dict_dense[bcnn_architecture.DROPOUT_RATES_KEY] = (
                DROPOUT_RATES_2D_LIST[j]
            )
            option_dict_dense[bcnn_architecture.LAYER_TYPES_KEY] = (
                LAYER_TYPE_STRINGS_2D_LIST[i]
            )

            model_object = bcnn_architecture.create_crps_model_ri(
                option_dict_gridded_sat=OPTION_DICT_GRIDDED_SAT,
                option_dict_ungridded_sat=None,
                option_dict_ships=OPTION_DICT_SHIPS,
                option_dict_dense=option_dict_dense
            )

            model_file_name = (
                '{0:s}/layer-types={1:s}_dropout-rates={2:s}/model.h5'
            ).format(
                OUTPUT_DIR_NAME, LAYER_TYPE_ABBREVS[i], DROPOUT_RATE_ABBREVS[j]
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=model_file_name
            )

            print('Writing model to: "{0:s}"...'.format(model_file_name))
            model_object.save(
                filepath=model_file_name, overwrite=True,
                include_optimizer=True
            )

            model_metafile_name = neural_net.find_metafile(
                model_file_name=model_file_name,
                raise_error_if_missing=False
            )

            dummy_option_dict = neural_net.DEFAULT_GENERATOR_OPTION_DICT
            dummy_option_dict[neural_net.PREDICT_TD_TO_TS_KEY] = False
            dummy_option_dict[neural_net.LEAD_TIMES_KEY] = numpy.array(
                [24], dtype=int
            )
            dummy_option_dict[neural_net.SATELLITE_LAG_TIMES_KEY] = None

            print('Writing metadata to: "{0:s}"...'.format(
                model_metafile_name
            ))
            neural_net._write_metafile(
                pickle_file_name=model_metafile_name, num_epochs=100,
                use_crps_loss=True,
                quantile_levels=None, central_loss_function_weight=None,
                num_training_batches_per_epoch=100,
                training_option_dict=dummy_option_dict,
                num_validation_batches_per_epoch=100,
                validation_option_dict=dummy_option_dict,
                do_early_stopping=True, plateau_lr_multiplier=0.6,
                bnn_architecture_dict={
                    'option_dict_gridded_sat': OPTION_DICT_GRIDDED_SAT,
                    'option_dict_ungridded_sat': None,
                    'option_dict_ships': OPTION_DICT_SHIPS,
                    'option_dict_dense': option_dict_dense
                }
            )


if __name__ == '__main__':
    _run()
