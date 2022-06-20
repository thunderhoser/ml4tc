"""Makes CNN templates for Quantile-regression Exp 2 for Taiwan problem."""

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

LEAD_TIMES_HOURS = numpy.linspace(6, 168, num=28, dtype=int)

BASE_OPTION_DICT_GRIDDED_SAT = {
    cnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([380, 540, 2, 1], dtype=int),
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
    cnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([5, 289], dtype=int),
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
    cnn_architecture.DROPOUT_RATES_KEY:
        numpy.array([0.5, 0.5, 0.5, 0.5, 0]),
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

CENTRAL_LOSS_WEIGHTS = numpy.linspace(1, 10, num=10, dtype=float)

QUANTILE_LEVELS = numpy.array([
    0.005, 0.995, 0.025, 0.975, 0.05, 0.95, 0.1, 0.9, 0.25, 0.5, 0.75,
    0.0125, 0.9875, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8
])

QUANTILE_LEVELS = numpy.sort(QUANTILE_LEVELS)

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_models/'
    'qr_experiment02_td-to-ts/templates'
)


def _run():
    """Makes CNN templates for Quantile-regression Exp 2 for Taiwan problem.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    for i in range(len(CENTRAL_LOSS_WEIGHTS)):
        option_dict_ships = copy.deepcopy(BASE_OPTION_DICT_SHIPS)
        option_dict_dense = copy.deepcopy(BASE_OPTION_DICT_DENSE)

        d = BASE_OPTION_DICT_GRIDDED_SAT
        num_flattened_features = (
            option_dict_ships[cnn_architecture.NUM_NEURONS_KEY][-1] +
            40 * d[cnn_architecture.NUM_CHANNELS_KEY][-1]
        )

        exp_neuron_counts = architecture_utils.get_dense_layer_dimensions(
            num_input_units=num_flattened_features,
            num_classes=2, num_dense_layers=5,
            for_classification=True
        )[1]

        neuron_counts = numpy.linspace(
            1, exp_neuron_counts[0], num=5, dtype=int
        )[::-1]

        option_dict_dense[cnn_architecture.NUM_NEURONS_KEY] = neuron_counts

        model_object = cnn_architecture.create_qr_model_td_to_ts_new(
            option_dict_gridded_sat=BASE_OPTION_DICT_GRIDDED_SAT,
            option_dict_ungridded_sat=None,
            option_dict_ships=option_dict_ships,
            option_dict_dense=option_dict_dense,
            quantile_levels=QUANTILE_LEVELS,
            central_loss_weight=CENTRAL_LOSS_WEIGHTS[i],
            num_lead_times=len(LEAD_TIMES_HOURS)
        )

        model_file_name = (
            '{0:s}/central-loss-weight={1:04.1f}/model.h5'
        ).format(
            OUTPUT_DIR_NAME, CENTRAL_LOSS_WEIGHTS[i]
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
        dummy_option_dict[neural_net.PREDICT_TD_TO_TS_KEY] = True
        dummy_option_dict[neural_net.LEAD_TIMES_KEY] = LEAD_TIMES_HOURS
        dummy_option_dict[neural_net.SATELLITE_LAG_TIMES_KEY] = None

        print('Writing metadata to: "{0:s}"...'.format(
            model_metafile_name
        ))
        neural_net._write_metafile(
            pickle_file_name=model_metafile_name, num_epochs=100,
            quantile_levels=QUANTILE_LEVELS,
            central_loss_function_weight=CENTRAL_LOSS_WEIGHTS[i],
            num_training_batches_per_epoch=100,
            training_option_dict=dummy_option_dict,
            num_validation_batches_per_epoch=100,
            validation_option_dict=dummy_option_dict,
            do_early_stopping=True, plateau_lr_multiplier=0.6
        )


if __name__ == '__main__':
    _run()
