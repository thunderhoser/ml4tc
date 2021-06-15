"""Methods for building CNN."""

import numpy
import keras
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils

INPUT_DIMENSIONS_KEY = 'input_dimensions'
NUM_LAYERS_BY_BLOCK_KEY = 'num_layers_by_block'
NUM_CHANNELS_KEY = 'num_channels_by_layer'
DROPOUT_RATES_KEY = 'dropout_rate_by_layer'
KEEP_TIME_DIMENSION_KEY = 'keep_time_dimension'
ACTIVATION_FUNCTION_KEY = 'activation_function_name'
ACTIVATION_FUNCTION_ALPHA_KEY = 'activation_function_alpha'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'

NUM_NEURONS_KEY = 'num_neurons_by_layer'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'

DEFAULT_OPTION_DICT_GRIDDED_SAT = {
    NUM_LAYERS_BY_BLOCK_KEY: numpy.array([2, 2, 2, 2, 2, 2, 2], dtype=int),
    NUM_CHANNELS_KEY: numpy.array(
        [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512], dtype=int
    ),
    DROPOUT_RATES_KEY: numpy.full(14, 0.),
    ACTIVATION_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True
}

DEFAULT_OPTION_DICT_UNGRIDDED_SAT = {
    NUM_NEURONS_KEY: numpy.array([50, 100], dtype=int),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True
}

DEFAULT_OPTION_DICT_SHIPS = {
    NUM_NEURONS_KEY: numpy.array([300, 600], dtype=int),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True
}

DEFAULT_OPTION_DICT_DENSE = {
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True
}


def _create_layers_gridded_sat(option_dict):
    """Creates layers for gridded satellite data.

    B = number of conv blocks
    C = total number of conv layers

    :param option_dict: Dictionary with the following keys.
    option_dict['input_dimensions']:
        length-4 numpy array with input dimensions for gridded satellite data:
        (num_grid_rows, num_grid_columns, num_lag_times, num_channels).
    option_dict['num_conv_layers_by_block']: length-B numpy array with number of
        conv layers for each block.
    option_dict['num_channels_by_conv_layer']: length-C numpy array with number
        of channels for each conv layer.
    option_dict['dropout_rate_by_conv_layer']: length-C numpy array with dropout
        rate for each conv layer.  Use number <= 0 for no dropout.
    option_dict['activation_function_name']: Name of activation function for all
        conv layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['activation_function_alpha']: Alpha (slope parameter) for
        activation function for all conv layers.  Applies only to ReLU and eLU.
    option_dict['l2_weight']: Weight for L_2 regularization in conv layers.
    option_dict['use_batch_normalization']: Boolean flag.  If True, will use
        batch normalization after each conv layer.

    :return: input_layer_object: Input layer for gridded satellite data
        (instance of `keras.layers.Input`).
    :return: last_layer_object: Last layer for processing only gridded satellite
        data (instance of `keras.layers`).
    """

    # Check input args.
    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_layers_by_block = option_dict[NUM_LAYERS_BY_BLOCK_KEY]
    num_channels_by_layer = option_dict[NUM_CHANNELS_KEY]
    dropout_rate_by_layer = option_dict[DROPOUT_RATES_KEY]
    activation_function_name = option_dict[ACTIVATION_FUNCTION_KEY]
    activation_function_alpha = option_dict[ACTIVATION_FUNCTION_ALPHA_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    error_checking.assert_is_numpy_array(
        input_dimensions, exact_dimensions=numpy.array([4], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(input_dimensions)
    error_checking.assert_is_greater_numpy_array(input_dimensions, 0)

    error_checking.assert_is_numpy_array(num_layers_by_block, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(num_layers_by_block)
    error_checking.assert_is_geq_numpy_array(num_layers_by_block, 1)

    num_blocks = len(num_layers_by_block)
    num_layers = numpy.sum(num_layers_by_block)
    error_checking.assert_is_geq(num_blocks, 6)
    error_checking.assert_is_leq(num_blocks, 7)

    error_checking.assert_is_numpy_array(
        num_channels_by_layer,
        exact_dimensions=numpy.array([num_layers], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(num_channels_by_layer)
    error_checking.assert_is_geq_numpy_array(num_channels_by_layer, 1)

    error_checking.assert_is_numpy_array(
        dropout_rate_by_layer,
        exact_dimensions=numpy.array([num_layers], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(dropout_rate_by_layer, 1.)

    error_checking.assert_is_geq(l2_weight, 0.)
    error_checking.assert_is_boolean(use_batch_normalization)

    # Do actual stuff.
    input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist())
    )

    num_lag_times = input_dimensions[2]
    num_channels = input_dimensions[3]
    new_dimensions = (
        tuple(input_dimensions[:2].tolist()) + (num_lag_times * num_channels,)
    )
    layer_object = keras.layers.Reshape(target_shape=new_dimensions)(
        input_layer_object
    )

    regularization_func = architecture_utils.get_weight_regularizer(
        l2_weight=l2_weight
    )

    num_blocks = len(num_layers_by_block)
    k = -1

    for i in range(num_blocks):
        for _ in range(num_layers_by_block[i]):
            k += 1

            layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_layer[k],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularization_func
            )(layer_object)

            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_name,
                alpha_for_relu=activation_function_alpha,
                alpha_for_elu=activation_function_alpha
            )(layer_object)

            if dropout_rate_by_layer[k] > 0:
                layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=dropout_rate_by_layer[k]
                )(layer_object)

            if use_batch_normalization:
                layer_object = architecture_utils.get_batch_norm_layer()(
                    layer_object
                )

        layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=architecture_utils.MAX_POOLING_STRING
        )(layer_object)

    layer_object = architecture_utils.get_flattening_layer()(layer_object)

    return input_layer_object, layer_object


def create_dense_layers(input_layer_object, option_dict):
    """Creates dense layers.

    D = number of dense layers

    :param input_layer_object: Input to first dense layer (instance of
        `keras.layers`).
    :param option_dict: Dictionary with the following keys.
    option_dict['num_layers']: Number of dense layers.
    option_dict['num_neurons_by_layer']: length-D numpy array with number of
        output neurons for each dense layer.
    option_dict['dropout_rate_by_layer']: length-D numpy array with dropout rate
        for each dense layer.  Use number <= 0 for no dropout.
    option_dict['inner_activ_function_name']: Name of activation function for
        all inner (non-output) dense layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['inner_activ_function_alpha']: Alpha (slope parameter) for
        activation function for all inner (non-output) dense layers.  Applies
        only to ReLU and eLU.
    option_dict['output_activ_function_name']: Name of activation function for
        output layer.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['output_activ_function_alpha']: Alpha (slope parameter) for
        activation function for output layer.  Applies only to ReLU and eLU.
    option_dict['l2_weight']: Weight for L_2 regularization in dense layers.
    option_dict['use_batch_normalization']: Boolean flag.  If True, will use
        batch normalization after each inner (non-output) dense layer.

    :return: last_layer_object: Last layer (instance of `keras.layers`).
    """

    # Check input args.
    num_neurons_by_layer = option_dict[NUM_NEURONS_KEY]
    dropout_rate_by_layer = option_dict[DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    error_checking.assert_is_numpy_array(
        num_neurons_by_layer, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(num_neurons_by_layer)
    error_checking.assert_is_geq_numpy_array(num_neurons_by_layer, 1)

    num_layers = len(num_neurons_by_layer)
    error_checking.assert_is_numpy_array(
        dropout_rate_by_layer,
        exact_dimensions=numpy.array([num_layers], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(dropout_rate_by_layer, 1.)

    error_checking.assert_is_geq(l2_weight, 0.)
    error_checking.assert_is_boolean(use_batch_normalization)

    # Do actual stuff.
    regularization_func = architecture_utils.get_weight_regularizer(
        l2_weight=l2_weight
    )

    num_layers = len(num_neurons_by_layer)
    layer_object = None

    for i in range(num_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_neurons_by_layer[i],
            weight_regularizer=regularization_func
        )(this_input_layer_object)

        if i == num_layers - 1:
            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=output_activ_function_name,
                alpha_for_relu=output_activ_function_alpha,
                alpha_for_elu=output_activ_function_alpha
            )(layer_object)
        else:
            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(layer_object)

        if dropout_rate_by_layer[i] > 0:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rate_by_layer[i]
            )(layer_object)

        if use_batch_normalization and i != num_layers - 1:
            layer_object = architecture_utils.get_batch_norm_layer()(
                layer_object
            )

    return layer_object


def create_model(
        option_dict_gridded_sat, option_dict_ungridded_sat, option_dict_ships,
        option_dict_dense, loss_function, metric_functions):
    """Creates CNN.

    :param option_dict_gridded_sat: See doc for `_create_layers_gridded_sat`.
    :param option_dict_ungridded_sat: See doc for `create_dense_layers`.
    :param option_dict_ships: Same.
    :param option_dict_dense: Same.
    :param loss_function: Loss function.
    :param metric_functions: 1-D list of metric functions.
    :return: model_object: Untrained CNN (instance of `keras.models.Model`).
    """

    option_dict_gridded_sat_orig = option_dict_gridded_sat.copy()
    option_dict_gridded_sat = DEFAULT_OPTION_DICT_GRIDDED_SAT.copy()
    option_dict_gridded_sat.update(option_dict_gridded_sat_orig)

    option_dict_ungridded_sat_orig = option_dict_ungridded_sat.copy()
    option_dict_ungridded_sat = DEFAULT_OPTION_DICT_UNGRIDDED_SAT.copy()
    option_dict_ungridded_sat.update(option_dict_ungridded_sat_orig)

    option_dict_ships_orig = option_dict_ships.copy()
    option_dict_ships = DEFAULT_OPTION_DICT_SHIPS.copy()
    option_dict_ships.update(option_dict_ships_orig)

    option_dict_dense_orig = option_dict_dense.copy()
    option_dict_dense = DEFAULT_OPTION_DICT_DENSE.copy()
    option_dict_dense.update(option_dict_dense_orig)

    input_dimensions_ungridded_sat = (
        option_dict_ungridded_sat[INPUT_DIMENSIONS_KEY]
    )
    input_layer_object_ungridded_sat = keras.layers.Input(
        shape=tuple(input_dimensions_ungridded_sat.tolist())
    )
    new_dimensions = (numpy.prod(input_dimensions_ungridded_sat),)
    ungridded_sat_layer_object = keras.layers.Reshape(
        target_shape=new_dimensions
    )(input_layer_object_ungridded_sat)

    input_dimensions_ships = option_dict_ships[INPUT_DIMENSIONS_KEY]
    input_layer_object_ships = keras.layers.Input(
        shape=tuple(input_dimensions_ships.tolist())
    )
    new_dimensions = (numpy.prod(input_dimensions_ships),)
    ships_layer_object = keras.layers.Reshape(target_shape=new_dimensions)(
        input_layer_object_ships
    )

    input_layer_object_gridded_sat, gridded_sat_layer_object = (
        _create_layers_gridded_sat(option_dict_gridded_sat)
    )
    ungridded_sat_layer_object = create_dense_layers(
        input_layer_object=ungridded_sat_layer_object,
        option_dict=option_dict_ungridded_sat
    )
    ships_layer_object = create_dense_layers(
        input_layer_object=ships_layer_object,
        option_dict=option_dict_ships
    )

    if option_dict_ungridded_sat[USE_BATCH_NORM_KEY]:
        ungridded_sat_layer_object = architecture_utils.get_batch_norm_layer()(
            ungridded_sat_layer_object
        )
    if option_dict_ships[USE_BATCH_NORM_KEY]:
        ships_layer_object = architecture_utils.get_batch_norm_layer()(
            ships_layer_object
        )

    layer_object = keras.layers.concatenate([
        gridded_sat_layer_object, ungridded_sat_layer_object,
        ships_layer_object
    ])
    layer_object = create_dense_layers(
        input_layer_object=layer_object, option_dict=option_dict_dense
    )

    input_layer_objects = [
        input_layer_object_gridded_sat, input_layer_object_ungridded_sat,
        input_layer_object_ships
    ]
    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=layer_object
    )

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=metric_functions
    )
    model_object.summary()

    return model_object
