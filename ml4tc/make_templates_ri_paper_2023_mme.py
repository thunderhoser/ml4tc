"""Makes CNN templates for multi-model ensembles in 2023 RI paper."""

import os
import sys
import copy
import numpy
from keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import architecture_utils
import neural_net
import cnn_architecture

BASE_OPTION_DICT_GRIDDED_SAT = {
    # cnn_architecture.INPUT_DIMENSIONS_KEY:
    #     numpy.array([380, 540, 3, 1], dtype=int),
    cnn_architecture.NUM_LAYERS_BY_BLOCK_KEY: numpy.full(6, 2, dtype=int),
    cnn_architecture.NUM_CHANNELS_KEY: numpy.array(
        [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256], dtype=int
    ),
    cnn_architecture.DROPOUT_RATES_KEY: numpy.full(12, 0.),
    cnn_architecture.DROPOUT_MC_FLAGS_KEY: numpy.full(12, 0, dtype=bool),
    cnn_architecture.ACTIVATION_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 0.,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

BASE_OPTION_DICT_SHIPS = {
    # cnn_architecture.INPUT_DIMENSIONS_KEY:
    #     numpy.array([5, 65], dtype=int),
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

BASE_OPTION_DICT_DENSE = {
    # cnn_architecture.NUM_NEURONS_KEY:
    #     numpy.array([1428, 232, 38, 6, 1], dtype=int),
    # cnn_architecture.DROPOUT_RATES_KEY:
    #     numpy.array([0.5, 0.5, 0.5, 0.5, 0]),
    cnn_architecture.DROPOUT_MC_FLAGS_KEY: numpy.full(5, 0, dtype=bool),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.SIGMOID_FUNCTION_STRING,
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    cnn_architecture.L2_WEIGHT_KEY: 0.,
    cnn_architecture.USE_BATCH_NORM_KEY: True,
    cnn_architecture.LAST_DROPOUT_BEFORE_ACTIV_KEY: True
}

OUTER_ENSEMBLE_SIZE = 50
INNER_ENSEMBLE_SIZE = 100
NUM_DENSE_LAYERS = 5
NUM_SHIPS_GOES_PREDICTORS = 16
NUM_SHIPS_GOES_LAG_TIMES = 5
NUM_SHIPS_ENVIRO_PREDICTORS = 11
NUM_SHIPS_HIST_PREDICTORS = 2
NUM_SHIPS_FORECAST_HOURS = 5

CIRA_IR_LAG_TIME_COUNTS = numpy.array([0, 1, 1], dtype=int)
USE_SHIPS_ENVIRO_FLAGS = numpy.array([1, 0, 1], dtype=bool)
USE_SHIPS_HISTORICAL_FLAGS = numpy.array([1, 0, 1], dtype=bool)
USE_SHIPS_SATELLITE_FLAGS = numpy.array([1, 0, 1], dtype=bool)
DROPOUT_RATES = numpy.array([0.8, 0.7, 0.8])

METRIC_FUNCTION_LIST = list(neural_net.METRIC_DICT.values())

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_models/'
    'ri_paper_2023_experiment/multimodel_ensembles/templates'
)


def _run():
    """Makes CNN templates for multi-model ensembles in 2023 RI paper.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    for i in range(len(CIRA_IR_LAG_TIME_COUNTS)):
        d = BASE_OPTION_DICT_GRIDDED_SAT
        num_flattened_features = 40 * d[cnn_architecture.NUM_CHANNELS_KEY][-1]

        if (
                USE_SHIPS_ENVIRO_FLAGS[i] or
                USE_SHIPS_HISTORICAL_FLAGS[i] or
                USE_SHIPS_SATELLITE_FLAGS[i]
        ):
            use_some_ships_data = True
            num_flattened_features += BASE_OPTION_DICT_SHIPS[
                cnn_architecture.NUM_NEURONS_KEY
            ][-1]
        else:
            use_some_ships_data = False

        if not use_some_ships_data and CIRA_IR_LAG_TIME_COUNTS[i] == 0:
            continue

        neuron_counts = architecture_utils.get_dense_layer_dimensions(
            num_input_units=num_flattened_features,
            num_classes=2, num_dense_layers=NUM_DENSE_LAYERS,
            for_classification=True
        )[1]

        neuron_counts = architecture_utils.get_dense_layer_dimensions(
            num_input_units=neuron_counts[0],
            num_classes=INNER_ENSEMBLE_SIZE,
            num_dense_layers=NUM_DENSE_LAYERS - 1,
            for_classification=False
        )[0]

        neuron_counts = numpy.concatenate((
            neuron_counts,
            numpy.array([INNER_ENSEMBLE_SIZE], dtype=int)
        ))

        option_dict_dense = copy.deepcopy(BASE_OPTION_DICT_DENSE)
        option_dict_dense[cnn_architecture.NUM_NEURONS_KEY] = neuron_counts
        option_dict_dense[cnn_architecture.DROPOUT_RATES_KEY] = numpy.full(
            NUM_DENSE_LAYERS, DROPOUT_RATES[i]
        )
        option_dict_dense[cnn_architecture.DROPOUT_RATES_KEY][-1] = 0.

        if CIRA_IR_LAG_TIME_COUNTS[i] > 0:
            option_dict_gridded_sat = copy.deepcopy(
                BASE_OPTION_DICT_GRIDDED_SAT
            )

            option_dict_gridded_sat[
                cnn_architecture.INPUT_DIMENSIONS_KEY
            ] = numpy.array(
                [380, 540, CIRA_IR_LAG_TIME_COUNTS[i], 1], dtype=int
            )
        else:
            option_dict_gridded_sat = None

        num_scalar_ships_predictors = (
            int(USE_SHIPS_ENVIRO_FLAGS[i]) *
            NUM_SHIPS_ENVIRO_PREDICTORS * NUM_SHIPS_FORECAST_HOURS +
            int(USE_SHIPS_HISTORICAL_FLAGS[i]) *
            NUM_SHIPS_HIST_PREDICTORS * NUM_SHIPS_FORECAST_HOURS +
            int(USE_SHIPS_SATELLITE_FLAGS[i]) *
            NUM_SHIPS_GOES_PREDICTORS * NUM_SHIPS_GOES_LAG_TIMES
        )

        if num_scalar_ships_predictors > 0:
            option_dict_ships = copy.deepcopy(BASE_OPTION_DICT_SHIPS)
            option_dict_ships[
                cnn_architecture.INPUT_DIMENSIONS_KEY
            ] = numpy.array([num_scalar_ships_predictors], dtype=int)
        else:
            option_dict_ships = None

        for j in range(OUTER_ENSEMBLE_SIZE):
            model_object = cnn_architecture.create_crps_model_ri(
                option_dict_gridded_sat=option_dict_gridded_sat,
                option_dict_ungridded_sat=None,
                option_dict_ships=option_dict_ships,
                option_dict_dense=option_dict_dense,
                num_estimates=INNER_ENSEMBLE_SIZE
            )

            print(model_object.get_layer(index=5).name)
            print(K.eval(model_object.get_layer(index=5).weights[0]))

            model_file_name = (
                '{0:s}/num-cira-ir-lag-times={1:d}_use-ships-enviro={2:d}_'
                'use-ships-historical={3:d}_use-ships-satellite={4:d}_'
                'dense-dropout-rate={5:.1f}/ensemble_member_{6:02d}/model.h5'
            ).format(
                OUTPUT_DIR_NAME,
                CIRA_IR_LAG_TIME_COUNTS[i],
                USE_SHIPS_ENVIRO_FLAGS[i],
                USE_SHIPS_HISTORICAL_FLAGS[i],
                USE_SHIPS_SATELLITE_FLAGS[i],
                DROPOUT_RATES[i],
                j + 1
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=model_file_name
            )
            print('Writing model to: "{0:s}"...'.format(model_file_name))
            model_object.save(
                filepath=model_file_name, overwrite=True, include_optimizer=True
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

            print('Writing metadata to: "{0:s}"...'.format(model_metafile_name))
            neural_net._write_metafile(
                pickle_file_name=model_metafile_name, num_epochs=100,
                use_crps_loss=True,
                quantile_levels=None, central_loss_function_weight=None,
                num_training_batches_per_epoch=100,
                training_option_dict=dummy_option_dict,
                num_validation_batches_per_epoch=100,
                validation_option_dict=dummy_option_dict,
                do_early_stopping=True, plateau_lr_multiplier=0.6,
                bnn_architecture_dict=None
            )


if __name__ == '__main__':
    _run()
