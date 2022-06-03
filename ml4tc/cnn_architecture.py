"""Methods for building CNN."""

import copy
import numpy
import keras
from keras.layers import Add
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4tc.machine_learning import custom_losses

INPUT_DIMENSIONS_KEY = 'input_dimensions'
NUM_LAYERS_BY_BLOCK_KEY = 'num_layers_by_block'
NUM_CHANNELS_KEY = 'num_channels_by_layer'
DROPOUT_RATES_KEY = 'dropout_rate_by_layer'
DROPOUT_MC_FLAGS_KEY = 'dropout_mc_flag_by_layer'
ACTIVATION_FUNCTION_KEY = 'activation_function_name'
ACTIVATION_FUNCTION_ALPHA_KEY = 'activation_function_alpha'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'

NUM_NEURONS_KEY = 'num_neurons_by_layer'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
LAST_DROPOUT_BEFORE_ACTIV_KEY = 'last_dropout_before_activation'

DEFAULT_OPTION_DICT_GRIDDED_SAT = {
    NUM_LAYERS_BY_BLOCK_KEY: numpy.array([2, 2, 2, 2, 2, 2, 2], dtype=int),
    NUM_CHANNELS_KEY: numpy.array(
        [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512], dtype=int
    ),
    DROPOUT_RATES_KEY: numpy.full(14, 0.),
    DROPOUT_MC_FLAGS_KEY: numpy.full(14, 0, dtype=bool),
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
    DROPOUT_RATES_KEY: numpy.full(2, 0.),
    DROPOUT_MC_FLAGS_KEY: numpy.full(2, 0, dtype=bool),
    USE_BATCH_NORM_KEY: True,
    LAST_DROPOUT_BEFORE_ACTIV_KEY: False
}

DEFAULT_OPTION_DICT_SHIPS = {
    NUM_NEURONS_KEY: numpy.array([300, 600], dtype=int),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    DROPOUT_RATES_KEY: numpy.full(2, 0.),
    DROPOUT_MC_FLAGS_KEY: numpy.full(2, 0, dtype=bool),
    USE_BATCH_NORM_KEY: True,
    LAST_DROPOUT_BEFORE_ACTIV_KEY: False
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
    option_dict['num_layers_by_block']: length-B numpy array with number of
        conv layers for each block.
    option_dict['num_channels_by_layer']: length-C numpy array with number
        of channels for each conv layer.
    option_dict['dropout_rate_by_layer']: length-C numpy array with dropout
        rate for each conv layer.  Usenumber <= 0 for no dropout.
    option_dict['dropout_mc_flag_by_layer']: length-C numpy array with Boolean
        flag for each conv layer, indicating whether or not to use Monte Carlo
        dropout (i.e., leave dropout on at inference time).
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
    dropout_mc_flag_by_layer = option_dict[DROPOUT_MC_FLAGS_KEY]
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

    these_dim = numpy.array([num_layers], dtype=int)
    error_checking.assert_is_numpy_array(
        dropout_rate_by_layer, exact_dimensions=these_dim
    )
    error_checking.assert_is_leq_numpy_array(dropout_rate_by_layer, 1.)

    error_checking.assert_is_numpy_array(
        dropout_mc_flag_by_layer, exact_dimensions=these_dim
    )
    error_checking.assert_is_boolean_numpy_array(dropout_mc_flag_by_layer)

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
                this_mc_flag = bool(dropout_mc_flag_by_layer[k])

                layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=dropout_rate_by_layer[k]
                )(layer_object, training=this_mc_flag)

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
    option_dict['dropout_mc_flag_by_layer']: length-D numpy array with Boolean
        flag for each conv layer, indicating whether or not to use Monte Carlo
        dropout.
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
    option_dict['last_dropout_before_activation']: Boolean flag.  If True
        (False), will apply dropout to last layer before (after) activation
        function.  Obviously, if there is no dropout for the last layer, this is
        a moot point.

    :return: last_layer_object: Last layer (instance of `keras.layers`).
    """

    # Check input args.
    num_neurons_by_layer = option_dict[NUM_NEURONS_KEY]
    dropout_rate_by_layer = option_dict[DROPOUT_RATES_KEY]
    dropout_mc_flag_by_layer = option_dict[DROPOUT_MC_FLAGS_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    last_dropout_before_activation = option_dict[LAST_DROPOUT_BEFORE_ACTIV_KEY]

    error_checking.assert_is_numpy_array(
        num_neurons_by_layer, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(num_neurons_by_layer)
    error_checking.assert_is_geq_numpy_array(num_neurons_by_layer, 1)

    num_layers = len(num_neurons_by_layer)
    these_dim = numpy.array([num_layers], dtype=int)

    error_checking.assert_is_numpy_array(
        dropout_rate_by_layer, exact_dimensions=these_dim
    )
    error_checking.assert_is_leq_numpy_array(dropout_rate_by_layer, 1.)

    error_checking.assert_is_numpy_array(
        dropout_mc_flag_by_layer, exact_dimensions=these_dim
    )
    error_checking.assert_is_boolean_numpy_array(dropout_mc_flag_by_layer)

    error_checking.assert_is_geq(l2_weight, 0.)
    error_checking.assert_is_boolean(use_batch_normalization)
    error_checking.assert_is_boolean(last_dropout_before_activation)

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
            use_dropout_now = (
                dropout_rate_by_layer[i] > 0 and
                last_dropout_before_activation
            )
        else:
            use_dropout_now = False

        if use_dropout_now:
            this_mc_flag = bool(dropout_mc_flag_by_layer[i])

            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rate_by_layer[i]
            )(layer_object, training=this_mc_flag)

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

        if i == num_layers - 1:
            use_dropout_now = (
                dropout_rate_by_layer[i] > 0 and
                not last_dropout_before_activation
            )
        else:
            use_dropout_now = dropout_rate_by_layer[i] > 0

        if use_dropout_now:
            this_mc_flag = bool(dropout_mc_flag_by_layer[i])

            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rate_by_layer[i]
            )(layer_object, training=this_mc_flag)

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
        If you do not want to use gridded satellite data, make this None.
    :param option_dict_ungridded_sat: See doc for `create_dense_layers`.  If you
        do not want to use ungridded satellite data, make this None.
    :param option_dict_ships: See doc for `create_dense_layers`.  If you do not
        want to use SHIPS data, make this None.
    :param option_dict_dense: See doc for `create_dense_layers`.
    :param loss_function: Loss function.
    :param metric_functions: 1-D list of metric functions.
    :return: model_object: Untrained CNN (instance of `keras.models.Model`).
    """

    input_layer_objects = []
    flattening_layer_objects = []

    if option_dict_gridded_sat is not None:
        option_dict_gridded_sat_orig = option_dict_gridded_sat.copy()
        option_dict_gridded_sat = DEFAULT_OPTION_DICT_GRIDDED_SAT.copy()
        option_dict_gridded_sat.update(option_dict_gridded_sat_orig)

        this_input_layer_object, this_flattening_layer_object = (
            _create_layers_gridded_sat(option_dict_gridded_sat)
        )

        input_layer_objects.append(this_input_layer_object)
        flattening_layer_objects.append(this_flattening_layer_object)

    if option_dict_ungridded_sat is not None:
        option_dict_ungridded_sat_orig = option_dict_ungridded_sat.copy()
        option_dict_ungridded_sat = DEFAULT_OPTION_DICT_UNGRIDDED_SAT.copy()
        option_dict_ungridded_sat.update(option_dict_ungridded_sat_orig)

        input_dimensions = option_dict_ungridded_sat[INPUT_DIMENSIONS_KEY]
        this_input_layer_object = keras.layers.Input(
            shape=tuple(input_dimensions.tolist())
        )
        new_dimensions = (numpy.prod(input_dimensions),)
        this_layer_object = keras.layers.Reshape(target_shape=new_dimensions)(
            this_input_layer_object
        )

        this_flattening_layer_object = create_dense_layers(
            input_layer_object=this_layer_object,
            option_dict=option_dict_ungridded_sat
        )

        if option_dict_ungridded_sat[USE_BATCH_NORM_KEY]:
            this_flattening_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    this_flattening_layer_object
                )
            )

        input_layer_objects.append(this_input_layer_object)
        flattening_layer_objects.append(this_flattening_layer_object)

    if option_dict_ships is not None:
        option_dict_ships_orig = option_dict_ships.copy()
        option_dict_ships = DEFAULT_OPTION_DICT_SHIPS.copy()
        option_dict_ships.update(option_dict_ships_orig)

        input_dimensions = option_dict_ships[INPUT_DIMENSIONS_KEY]
        this_input_layer_object = keras.layers.Input(
            shape=tuple(input_dimensions.tolist())
        )
        new_dimensions = (numpy.prod(input_dimensions),)
        this_layer_object = keras.layers.Reshape(target_shape=new_dimensions)(
            this_input_layer_object
        )

        this_flattening_layer_object = create_dense_layers(
            input_layer_object=this_layer_object,
            option_dict=option_dict_ships
        )

        if option_dict_ships[USE_BATCH_NORM_KEY]:
            this_flattening_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    this_flattening_layer_object
                )
            )

        input_layer_objects.append(this_input_layer_object)
        flattening_layer_objects.append(this_flattening_layer_object)

    option_dict_dense_orig = option_dict_dense.copy()
    option_dict_dense = DEFAULT_OPTION_DICT_DENSE.copy()
    option_dict_dense.update(option_dict_dense_orig)

    if len(flattening_layer_objects) > 1:
        layer_object = keras.layers.concatenate(flattening_layer_objects)
    else:
        layer_object = flattening_layer_objects[0]

    layer_object = create_dense_layers(
        input_layer_object=layer_object, option_dict=option_dict_dense
    )
    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=layer_object
    )

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=metric_functions
    )
    model_object.summary()

    return model_object


def create_quantile_regression_model(
        option_dict_gridded_sat, option_dict_ungridded_sat, option_dict_ships,
        option_dict_dense, quantile_levels):
    """Creates CNN for quantile regression.

    :param option_dict_gridded_sat: See doc for `create_model`.
    :param option_dict_ungridded_sat: Same.
    :param option_dict_ships: Same.
    :param option_dict_dense: Same.
    :param quantile_levels: 1-D numpy array of quantile levels, ranging from
        (0, 1).
    :return: model_object: Untrained CNN (instance of `keras.models.Model`).
    """

    error_checking.assert_is_numpy_array(quantile_levels, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(quantile_levels, 0.)
    error_checking.assert_is_less_than_numpy_array(quantile_levels, 1.)
    quantile_levels = numpy.sort(quantile_levels)

    input_layer_objects = []
    flattening_layer_objects = []

    if option_dict_gridded_sat is not None:
        option_dict_gridded_sat_orig = option_dict_gridded_sat.copy()
        option_dict_gridded_sat = DEFAULT_OPTION_DICT_GRIDDED_SAT.copy()
        option_dict_gridded_sat.update(option_dict_gridded_sat_orig)

        option_dict_gridded_sat[DROPOUT_MC_FLAGS_KEY][:] = False

        this_input_layer_object, this_flattening_layer_object = (
            _create_layers_gridded_sat(option_dict_gridded_sat)
        )

        input_layer_objects.append(this_input_layer_object)
        flattening_layer_objects.append(this_flattening_layer_object)

    if option_dict_ungridded_sat is not None:
        option_dict_ungridded_sat_orig = option_dict_ungridded_sat.copy()
        option_dict_ungridded_sat = DEFAULT_OPTION_DICT_UNGRIDDED_SAT.copy()
        option_dict_ungridded_sat.update(option_dict_ungridded_sat_orig)

        option_dict_ungridded_sat[DROPOUT_MC_FLAGS_KEY][:] = False

        input_dimensions = option_dict_ungridded_sat[INPUT_DIMENSIONS_KEY]
        this_input_layer_object = keras.layers.Input(
            shape=tuple(input_dimensions.tolist())
        )
        new_dimensions = (numpy.prod(input_dimensions),)
        this_layer_object = keras.layers.Reshape(target_shape=new_dimensions)(
            this_input_layer_object
        )

        this_flattening_layer_object = create_dense_layers(
            input_layer_object=this_layer_object,
            option_dict=option_dict_ungridded_sat
        )

        if option_dict_ungridded_sat[USE_BATCH_NORM_KEY]:
            this_flattening_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    this_flattening_layer_object
                )
            )

        input_layer_objects.append(this_input_layer_object)
        flattening_layer_objects.append(this_flattening_layer_object)

    if option_dict_ships is not None:
        option_dict_ships_orig = option_dict_ships.copy()
        option_dict_ships = DEFAULT_OPTION_DICT_SHIPS.copy()
        option_dict_ships.update(option_dict_ships_orig)

        option_dict_ships[DROPOUT_MC_FLAGS_KEY][:] = False

        input_dimensions = option_dict_ships[INPUT_DIMENSIONS_KEY]
        this_input_layer_object = keras.layers.Input(
            shape=tuple(input_dimensions.tolist())
        )
        new_dimensions = (numpy.prod(input_dimensions),)
        this_layer_object = keras.layers.Reshape(target_shape=new_dimensions)(
            this_input_layer_object
        )

        this_flattening_layer_object = create_dense_layers(
            input_layer_object=this_layer_object,
            option_dict=option_dict_ships
        )

        if option_dict_ships[USE_BATCH_NORM_KEY]:
            this_flattening_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    this_flattening_layer_object
                )
            )

        input_layer_objects.append(this_input_layer_object)
        flattening_layer_objects.append(this_flattening_layer_object)

    option_dict_dense_orig = option_dict_dense.copy()
    option_dict_dense = DEFAULT_OPTION_DICT_DENSE.copy()
    option_dict_dense.update(option_dict_dense_orig)

    option_dict_dense[DROPOUT_MC_FLAGS_KEY][:] = False

    num_output_neurons = option_dict_dense[NUM_NEURONS_KEY][-1] + 0
    output_activ_function_name = copy.deepcopy(
        option_dict_dense[OUTPUT_ACTIV_FUNCTION_KEY]
    )
    output_activ_function_alpha = copy.deepcopy(
        option_dict_dense[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    )

    option_dict_dense[NUM_NEURONS_KEY] = (
        option_dict_dense[NUM_NEURONS_KEY][:-1]
    )
    option_dict_dense[DROPOUT_RATES_KEY] = (
        option_dict_dense[DROPOUT_RATES_KEY][:-1]
    )
    option_dict_dense[DROPOUT_MC_FLAGS_KEY] = (
        option_dict_dense[DROPOUT_MC_FLAGS_KEY][:-1]
    )
    option_dict_dense[OUTPUT_ACTIV_FUNCTION_KEY] = (
        option_dict_dense[INNER_ACTIV_FUNCTION_KEY]
    )
    option_dict_dense[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY] = (
        option_dict_dense[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    )

    if len(flattening_layer_objects) > 1:
        current_layer_object = keras.layers.concatenate(
            flattening_layer_objects
        )
    else:
        current_layer_object = flattening_layer_objects[0]

    current_layer_object = create_dense_layers(
        input_layer_object=current_layer_object, option_dict=option_dict_dense
    )

    num_output_channels = len(quantile_levels) + 1
    output_layer_names = [
        'quantile_output{0:03d}'.format(k) for k in range(num_output_channels)
    ]
    output_layer_names[0] = 'central_output'

    pre_activn_out_layers = [None] * num_output_channels
    output_layers = [None] * num_output_channels
    loss_dict = {}

    regularization_func = architecture_utils.get_weight_regularizer(
        l2_weight=option_dict_dense[L2_WEIGHT_KEY]
    )

    for k in range(num_output_channels):
        pre_activn_out_layers[k] = architecture_utils.get_dense_layer(
            num_output_units=num_output_neurons,
            weight_regularizer=regularization_func
        )(current_layer_object)

        if k > 1:
            pre_activn_out_layers[k] = architecture_utils.get_activation_layer(
                activation_function_string=
                architecture_utils.RELU_FUNCTION_STRING,
                alpha_for_relu=0., alpha_for_elu=0.
            )(pre_activn_out_layers[k])

            pre_activn_out_layers[k] = Add()(
                [pre_activn_out_layers[k - 1], pre_activn_out_layers[k]]
            )

        output_layers[k] = architecture_utils.get_activation_layer(
            activation_function_string=output_activ_function_name,
            alpha_for_relu=output_activ_function_alpha,
            alpha_for_elu=output_activ_function_alpha,
            layer_name=output_layer_names[k]
        )(pre_activn_out_layers[k])

        if k == 0:
            loss_dict[output_layer_names[k]] = keras.losses.binary_crossentropy
        else:
            loss_dict[output_layer_names[k]] = custom_losses.quantile_loss(
                quantile_level=quantile_levels[k - 1]
            )

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=output_layers
    )
    model_object.compile(
        loss=loss_dict, optimizer=keras.optimizers.Adam()
    )
    model_object.summary()

    return model_object
