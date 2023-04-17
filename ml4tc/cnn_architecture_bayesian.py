"""Methods for building Bayesian CNN."""

import os
import sys
import numpy
import keras
import keras.layers
import tensorflow_probability as tf_prob
from tensorflow_probability.python.distributions import \
    kullback_leibler as kl_lib

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import cnn_architecture
import custom_losses

POINT_ESTIMATE_TYPE_STRING = 'point_estimate'
FLIPOUT_TYPE_STRING = 'flipout'
REPARAMETERIZATION_TYPE_STRING = 'reparameterization'
VALID_LAYER_TYPE_STRINGS = [
    POINT_ESTIMATE_TYPE_STRING, FLIPOUT_TYPE_STRING,
    REPARAMETERIZATION_TYPE_STRING
]

INPUT_DIMENSIONS_KEY = cnn_architecture.INPUT_DIMENSIONS_KEY

DEFAULT_OPTION_DICT_GRIDDED_SAT = (
    cnn_architecture.DEFAULT_OPTION_DICT_GRIDDED_SAT
)
DEFAULT_OPTION_DICT_UNGRIDDED_SAT = (
    cnn_architecture.DEFAULT_OPTION_DICT_UNGRIDDED_SAT
)
DEFAULT_OPTION_DICT_SHIPS = cnn_architecture.DEFAULT_OPTION_DICT_SHIPS

NUM_NEURONS_KEY = cnn_architecture.NUM_NEURONS_KEY
DROPOUT_RATES_KEY = cnn_architecture.DROPOUT_RATES_KEY
INNER_ACTIV_FUNCTION_KEY = cnn_architecture.INNER_ACTIV_FUNCTION_KEY
INNER_ACTIV_FUNCTION_ALPHA_KEY = cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY
OUTPUT_ACTIV_FUNCTION_KEY = cnn_architecture.OUTPUT_ACTIV_FUNCTION_KEY
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = (
    cnn_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY
)
USE_BATCH_NORM_KEY = cnn_architecture.USE_BATCH_NORM_KEY
LAST_DROPOUT_BEFORE_ACTIV_KEY = cnn_architecture.LAST_DROPOUT_BEFORE_ACTIV_KEY

KL_SCALING_FACTOR_KEY = 'kl_divergence_scaling_factor'
LAYER_TYPES_KEY = 'layer_type_strings'
ENSEMBLE_SIZE_KEY = 'ensemble_size'


def _check_layer_type(layer_type_string):
    """Ensures that layer type is valid.

    :param layer_type_string: Layer type (must be in list
        VALID_LAYER_TYPE_STRINGS).
    :raises ValueError: if `layer_type_string not in VALID_LAYER_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(layer_type_string)

    if layer_type_string not in VALID_LAYER_TYPE_STRINGS:
        error_string = (
            'Valid layer types (listed below) do not include "{0:s}":\n{1:s}'
        ).format(layer_type_string, str(VALID_LAYER_TYPE_STRINGS))

        raise ValueError(error_string)


def _get_dense_layer(
        previous_layer_object, layer_type_string, num_output_neurons,
        layer_name, kl_divergence_scaling_factor):
    """Creates dense layer.

    :param previous_layer_object: Previous layer (instance of
        `keras.layers.Layer` or similar).
    :param layer_type_string: See documentation for `_check_layer_type`.
    :param num_output_neurons: Number of output neurons.
    :param layer_name: Layer name (string).
    :param kl_divergence_scaling_factor: Scaling factor for Kullback-Leibler
        divergence.
    :return: layer_object: New conv layer (instance of `keras.layers.Dense` or
        similar).
    """

    _check_layer_type(layer_type_string)

    if layer_type_string == POINT_ESTIMATE_TYPE_STRING:
        return architecture_utils.get_dense_layer(
            num_output_units=num_output_neurons,
            layer_name=layer_name
        )(previous_layer_object)

    if layer_type_string == FLIPOUT_TYPE_STRING:
        return tf_prob.layers.DenseFlipout(
            units=num_output_neurons,
            activation=None,
            kernel_divergence_fn=(
                lambda q, p, ignore:
                kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
            ),
            bias_divergence_fn=(
                lambda q, p, ignore:
                kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
            )
        )(previous_layer_object)

    return tf_prob.layers.DenseReparameterization(
        units=num_output_neurons,
        activation=None,
        kernel_divergence_fn=(
            lambda q, p, ignore:
            kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
        ),
        bias_divergence_fn=(
            lambda q, p, ignore:
            kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
        )
    )(previous_layer_object)


def _create_dense_layers(input_layer_object, option_dict):
    """Creates dense layers.

    D = number of layers

    :param input_layer_object: Input to first dense layer (instance of
        `keras.layers`).
    :param option_dict: Dictionary with the following keys.
    option_dict["num_neurons_by_layer"]: See doc for
        `cnn_architecture.create_dense_layers`.
    option_dict["dropout_rate_by_layer"]: Same.
    option_dict["inner_activ_function_name"]: Same.
    option_dict["inner_activ_function_alpha"]: Same.
    option_dict["output_activ_function_name"]: Same.
    option_dict["output_activ_function_alpha"]: Same.
    option_dict["use_batch_normalization"]: Same.
    option_dict["last_dropout_before_activation"]: Same.
    option_dict["kl_divergence_scaling_factor"]: Scaling factor for
        Kullback-Leibler divergence.
    option_dict["layer_type_strings"]: length-D list of layer types (each must
        be accepted by `_check_layer_type`).
    option_dict["ensemble_size"]: Number of ensemble members (to be constrained
        with CRPS loss function).

    :return: last_layer_object: Last layer (instance of `keras.layers`).
    """

    # Check input args.
    num_neurons_by_layer = option_dict[NUM_NEURONS_KEY]
    dropout_rate_by_layer = option_dict[DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    last_dropout_before_activation = option_dict[LAST_DROPOUT_BEFORE_ACTIV_KEY]

    kl_divergence_scaling_factor = option_dict[KL_SCALING_FACTOR_KEY]
    layer_type_strings = option_dict[LAYER_TYPES_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]

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

    error_checking.assert_is_boolean(use_batch_normalization)
    error_checking.assert_is_boolean(last_dropout_before_activation)

    error_checking.assert_is_numpy_array(
        numpy.array(layer_type_strings), exact_dimensions=these_dim
    )
    for s in layer_type_strings:
        _check_layer_type(s)

    error_checking.assert_is_greater(kl_divergence_scaling_factor, 0.)
    error_checking.assert_is_less_than(kl_divergence_scaling_factor, 1.)
    error_checking.assert_is_integer(ensemble_size)
    error_checking.assert_is_geq(ensemble_size, 1)

    # Do actual stuff.
    num_layers = len(num_neurons_by_layer)
    layer_object = None

    for i in range(num_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = _get_dense_layer(
            previous_layer_object=this_input_layer_object,
            layer_type_string=layer_type_strings[i],
            num_output_neurons=num_neurons_by_layer[i],
            kl_divergence_scaling_factor=kl_divergence_scaling_factor,
            layer_name=None
        )

        if i == num_layers - 1:
            use_dropout_now = (
                dropout_rate_by_layer[i] > 0 and last_dropout_before_activation
            )
        else:
            use_dropout_now = False

        if use_dropout_now:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rate_by_layer[i]
            )(layer_object)

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
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rate_by_layer[i]
            )(layer_object)

        if use_batch_normalization and i != num_layers - 1:
            layer_object = architecture_utils.get_batch_norm_layer()(
                layer_object
            )

    layer_object = keras.layers.Reshape(
        target_shape=(1, ensemble_size)
    )(layer_object)

    return layer_object


def create_crps_model_ri(
        option_dict_gridded_sat, option_dict_ungridded_sat, option_dict_ships,
        option_dict_dense):
    """Creates Bayesian CNN with CRPS loss function for RI prediction.

    :param option_dict_gridded_sat: See doc for `cnn_architecture.create_model`.
    :param option_dict_ungridded_sat: Same.
    :param option_dict_ships: Same.
    :param option_dict_dense: See doc for `_create_dense_layers`.
    :return: model_object: Untrained CNN (instance of `keras.models.Model`).
    """

    input_layer_objects = []
    flattening_layer_objects = []

    if option_dict_gridded_sat is not None:
        option_dict_gridded_sat_orig = option_dict_gridded_sat.copy()
        option_dict_gridded_sat = DEFAULT_OPTION_DICT_GRIDDED_SAT.copy()
        option_dict_gridded_sat.update(option_dict_gridded_sat_orig)

        this_input_layer_object, this_flattening_layer_object = (
            cnn_architecture.create_layers_gridded_sat(option_dict_gridded_sat)
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

        this_flattening_layer_object = cnn_architecture.create_dense_layers(
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

        this_flattening_layer_object = cnn_architecture.create_dense_layers(
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

    if len(flattening_layer_objects) > 1:
        current_layer_object = keras.layers.concatenate(
            flattening_layer_objects
        )
    else:
        current_layer_object = flattening_layer_objects[0]

    output_layer_object = _create_dense_layers(
        input_layer_object=current_layer_object, option_dict=option_dict_dense
    )

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=output_layer_object
    )
    model_object.compile(
        loss=custom_losses.crps_loss(), optimizer=keras.optimizers.Adam()
    )
    model_object.summary()

    return model_object
