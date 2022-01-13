"""Methods for building convolutional LSTM."""

import numpy
import keras
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4tc.machine_learning import cnn_architecture

INPUT_DIMENSIONS_KEY = 'input_dimensions'
NUM_LAYERS_BY_BLOCK_KEY = 'num_layers_by_block'
NUM_CHANNELS_KEY = 'num_channels_by_layer'
DROPOUT_RATES_KEY = 'dropout_rate_by_layer'
KEEP_TIME_DIMENSION_KEY = 'keep_time_dimension'
ACTIVATION_FUNCTION_KEY = 'activation_function_name'
ACTIVATION_FUNCTION_ALPHA_KEY = 'activation_function_alpha'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'

NUM_NEURONS_KEY = cnn_architecture.NUM_NEURONS_KEY
INNER_ACTIV_FUNCTION_KEY = cnn_architecture.INNER_ACTIV_FUNCTION_KEY
INNER_ACTIV_FUNCTION_ALPHA_KEY = cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY
OUTPUT_ACTIV_FUNCTION_KEY = cnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = (
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY
)

DEFAULT_OPTION_DICT_GRIDDED_SAT = {
    NUM_LAYERS_BY_BLOCK_KEY: numpy.array([2, 2, 2, 2, 2, 2, 2], dtype=int),
    NUM_CHANNELS_KEY: numpy.array(
        [8, 8, 16, 16, 24, 24, 32, 32, 48, 48, 64, 64, 128, 128], dtype=int
    ),
    DROPOUT_RATES_KEY: numpy.full(14, 0.),
    ACTIVATION_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True
}

DEFAULT_OPTION_DICT_UNGRIDDED_SAT = {
    NUM_CHANNELS_KEY: numpy.array([100], dtype=int),
    ACTIVATION_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True
}

DEFAULT_OPTION_DICT_SHIPS = {
    NUM_CHANNELS_KEY: numpy.array([1000], dtype=int),
    ACTIVATION_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    ACTIVATION_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True
}

DEFAULT_OPTION_DICT_DENSE = {
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True
}

# TODO(thunderhoser): Do I also need to pass normal (non-recurrent) activation
# and dropout rate to methods that create an LSTM layer?


def _get_lstm_layer(
        num_output_units, recurrent_activation_func_or_name,
        recurrent_dropout_rate, regularization_func, return_sequences):
    """Creates simple LSTM layer (with no convolution).

    :param num_output_units: Number of output units.
    :param recurrent_activation_func_or_name: Activation function for recurrent
        step (may be passed as a function or string).
    :param recurrent_dropout_rate: Dropout rate for recurrent step.
    :param regularization_func: Regularization function (will be used for main
        kernel weights, recurrent-kernel weights, and biases).  If you do not
        want regularization, make this None.
    :param return_sequences: Boolean flag.  If True (False), layer will return
        full sequence (last output in output sequence).
    :return: layer_object: Instance of `keras.layers.LSTM`.
    """

    error_checking.assert_is_integer(num_output_units)
    error_checking.assert_is_geq(num_output_units, 1)
    error_checking.assert_is_boolean(return_sequences)
    error_checking.assert_is_less_than(recurrent_dropout_rate, 1.)
    if not recurrent_dropout_rate > 0.:
        recurrent_dropout_rate = 0.

    return keras.layers.LSTM(
        units=num_output_units, activation=None, use_bias=True,
        recurrent_activation=recurrent_activation_func_or_name,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        recurrent_initializer='orthogonal',
        unit_forget_bias=True,
        kernel_regularizer=regularization_func,
        recurrent_regularizer=regularization_func,
        bias_regularizer=regularization_func,
        activity_regularizer=None,
        return_sequences=return_sequences,
        dropout=0., recurrent_dropout=recurrent_dropout_rate
    )


def _get_2d_conv_lstm_layer(
        num_kernel_rows, num_kernel_columns, num_rows_per_stride,
        num_columns_per_stride, num_filters, use_padding,
        recurrent_activation_func_or_name,
        recurrent_dropout_rate, regularization_func, return_sequences):
    """Creates LSTM layer with 2-D convolution.

    :param num_kernel_rows: Number of spatial rows in convolutional filters.
    :param num_kernel_columns: Number of spatial columns in convolutional
        filters.
    :param num_rows_per_stride: Number of spatial rows per filter stride.
    :param num_columns_per_stride: Number of spatial columns per filter stride.
    :param num_filters: Number of convolutional filters.
    :param use_padding: Boolean flag.  If True (False), will (not) pad edges
        after convolution.
    :param recurrent_activation_func_or_name: See doc for `_get_lstm_layer`.
    :param recurrent_dropout_rate: Same.
    :param regularization_func: Same.
    :param return_sequences: Same.
    :return: layer_object: Instance of `keras.layers.ConvLSTM2D`.
    """

    error_checking.assert_is_integer(num_kernel_rows)
    error_checking.assert_is_geq(num_kernel_rows, 1)
    error_checking.assert_is_integer(num_kernel_columns)
    error_checking.assert_is_geq(num_kernel_columns, 1)
    error_checking.assert_is_integer(num_rows_per_stride)
    error_checking.assert_is_geq(num_rows_per_stride, 1)
    error_checking.assert_is_integer(num_columns_per_stride)
    error_checking.assert_is_geq(num_columns_per_stride, 1)
    error_checking.assert_is_integer(num_filters)
    error_checking.assert_is_geq(num_filters, 1)
    error_checking.assert_is_boolean(use_padding)
    error_checking.assert_is_boolean(return_sequences)

    error_checking.assert_is_less_than(recurrent_dropout_rate, 1.)
    if not recurrent_dropout_rate > 0.:
        recurrent_dropout_rate = 0.

    return keras.layers.ConvLSTM2D(
        filters=num_filters,
        kernel_size=(num_kernel_rows, num_kernel_columns),
        strides=(num_rows_per_stride, num_columns_per_stride),
        padding='same' if use_padding else 'valid',
        data_format='channels_last', dilation_rate=(1, 1),
        activation=None, use_bias=True,
        recurrent_activation=recurrent_activation_func_or_name,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        kernel_regularizer=regularization_func,
        recurrent_regularizer=regularization_func,
        bias_regularizer=regularization_func,
        return_sequences=return_sequences,
        dropout=0.0, recurrent_dropout=recurrent_dropout_rate
    )


def _create_layers_gridded_sat(option_dict):
    """Creates layers for gridded satellite data.

    B = number of conv blocks
    C = total number of conv layers

    :param option_dict: Dictionary with the following keys.
    option_dict['input_dimensions']:
        length-4 numpy array with input dimensions:
        (num_grid_rows, num_grid_columns, num_lag_times, num_channels).
    option_dict['num_layers_by_block']: length-B numpy array with number of
        conv layers for each block.
    option_dict['num_channels_by_layer']: length-C numpy array with number of
        channels for each conv layer.
    option_dict['dropout_rate_by_layer']: length-C numpy array with dropout rate
        for each conv layer.  Use number <= 0 for no dropout.
    option_dict['keep_time_dimension']: Boolean flag.  If True, will keep time
        dimension until the end and repeat conv-LSTM layers until the end.  If
        False, will remove time dimension after first conv-LSTM layer, then do
        conv without LSTM.
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
    keep_time_dimension = option_dict[KEEP_TIME_DIMENSION_KEY]
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
    error_checking.assert_is_geq(num_blocks, 2)

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

    error_checking.assert_is_boolean(keep_time_dimension)
    error_checking.assert_is_geq(l2_weight, 0.)
    error_checking.assert_is_boolean(use_batch_normalization)

    # Do actual stuff.
    input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist())
    )
    layer_object = keras.layers.Permute(dims=(3, 1, 2, 4))(input_layer_object)

    regularization_func = architecture_utils.get_weight_regularizer(
        l2_weight=l2_weight
    )

    num_blocks = len(num_layers_by_block)
    num_layers = numpy.sum(num_layers_by_block)
    k = -1
    return_sequences = True

    for i in range(num_blocks):
        for _ in range(num_layers_by_block[i]):
            k += 1
            return_sequences = keep_time_dimension and k != num_layers - 1

            if keep_time_dimension or k == 0:
                layer_object = _get_2d_conv_lstm_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=num_channels_by_layer[k], use_padding=True,
                    recurrent_activation_func_or_name='hard_sigmoid',
                    recurrent_dropout_rate=0.,
                    regularization_func=regularization_func,
                    return_sequences=return_sequences
                )(layer_object)
            else:
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

        if return_sequences:
            layer_object = architecture_utils.get_3d_pooling_layer(
                num_rows_in_window=1, num_columns_in_window=2,
                num_heights_in_window=2,
                num_rows_per_stride=1, num_columns_per_stride=2,
                num_heights_per_stride=2,
                pooling_type_string=architecture_utils.MAX_POOLING_STRING
            )(layer_object)
        else:
            layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=2, num_columns_in_window=2,
                num_rows_per_stride=2, num_columns_per_stride=2,
                pooling_type_string=architecture_utils.MAX_POOLING_STRING
            )(layer_object)

    layer_object = architecture_utils.get_flattening_layer()(layer_object)

    return input_layer_object, layer_object


def _create_layers_ungridded(option_dict):
    """Creates layers for ungridded data.

    L = number of LSTM layers

    :param option_dict: Dictionary with the following keys.
    option_dict['input_dimensions']:
        length-2 numpy array with input dimensions:
        (num_lag_times, num_channels).
    option_dict['num_channels_by_layer']: length-L numpy array with number of
        channels for each LSTM layer.
    option_dict['dropout_rate_by_layer']: length-L numpy array with dropout rate
        for each LSTM layer.  Use number <= 0 for no dropout.
    option_dict['activation_function_name']: Name of activation function for all
        LSTM layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['activation_function_alpha']: Alpha (slope parameter) for
        activation function for all LSTM layers.  Applies only to ReLU and eLU.
    option_dict['l2_weight']: Weight for L_2 regularization in LSTM layers.
    option_dict['use_batch_normalization']: Boolean flag.  If True, will use
        batch normalization after each LSTM layer.

    :return: input_layer_object: Input layer for ungridded data (instance of
        `keras.layers.Input`).
    :return: last_layer_object: Last layer for processing only this set of
        ungridded data (instance of `keras.layers`).
    """

    # Check input args.
    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_channels_by_layer = option_dict[NUM_CHANNELS_KEY]
    dropout_rate_by_layer = option_dict[DROPOUT_RATES_KEY]
    activation_function_name = option_dict[ACTIVATION_FUNCTION_KEY]
    activation_function_alpha = option_dict[ACTIVATION_FUNCTION_ALPHA_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    error_checking.assert_is_numpy_array(
        input_dimensions, exact_dimensions=numpy.array([2], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(input_dimensions)
    error_checking.assert_is_greater_numpy_array(input_dimensions, 0)

    error_checking.assert_is_numpy_array(
        num_channels_by_layer, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(num_channels_by_layer)
    error_checking.assert_is_geq_numpy_array(num_channels_by_layer, 1)

    num_layers = len(num_channels_by_layer)
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
    regularization_func = architecture_utils.get_weight_regularizer(
        l2_weight=l2_weight
    )

    num_layers = len(num_channels_by_layer)
    layer_object = None

    for i in range(num_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = _get_lstm_layer(
            num_output_units=num_channels_by_layer[i],
            recurrent_activation_func_or_name='hard_sigmoid',
            recurrent_dropout_rate=0.,
            regularization_func=regularization_func,
            return_sequences=i != num_layers - 1
        )(this_input_layer_object)

        layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_name,
            alpha_for_relu=activation_function_alpha,
            alpha_for_elu=activation_function_alpha
        )(layer_object)

        if dropout_rate_by_layer[i] > 0:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rate_by_layer[i]
            )(layer_object)

        if use_batch_normalization:
            layer_object = architecture_utils.get_batch_norm_layer()(
                layer_object
            )

    layer_object = architecture_utils.get_flattening_layer()(layer_object)

    return input_layer_object, layer_object


def create_model(
        option_dict_gridded_sat, option_dict_ungridded_sat, option_dict_ships,
        option_dict_dense, loss_function, metric_functions):
    """Creates conv-LSTM model.

    :param option_dict_gridded_sat: See doc for `_create_layers_gridded_sat`.
        If you do not want to use gridded satellite data, make this None.
    :param option_dict_ungridded_sat: See doc for `_create_layers_ungridded`.
        If you do not want to use ungridded satellite data, make this None.
    :param option_dict_ships: See doc for `_create_layers_ungridded`.  If you do
        not want to use SHIPS data, make this None.
    :param option_dict_dense: See doc for
        `cnn_architecture.create_dense_layers`.
    :param loss_function: Loss function.
    :param metric_functions: 1-D list of metric functions.
    :return: model_object: Untrained conv-LSTM model (instance of
        `keras.models.Model`).
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

        this_input_layer_object, this_flattening_layer_object = (
            _create_layers_ungridded(option_dict_ungridded_sat)
        )

        input_layer_objects.append(this_input_layer_object)
        flattening_layer_objects.append(this_flattening_layer_object)

    if option_dict_ships is not None:
        option_dict_ships_orig = option_dict_ships.copy()
        option_dict_ships = DEFAULT_OPTION_DICT_SHIPS.copy()
        option_dict_ships.update(option_dict_ships_orig)

        this_input_layer_object, this_flattening_layer_object = (
            _create_layers_ungridded(option_dict_ships)
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

    layer_object = cnn_architecture.create_dense_layers(
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
