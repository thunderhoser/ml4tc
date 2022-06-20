"""Custom loss functions for Keras models."""
import numpy
import tensorflow
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking


def _log2(input_tensor):
    """Computes logarithm in base 2.

    :param input_tensor: Keras tensor.
    :return: logarithm_tensor: Keras tensor with the same shape as
        `input_tensor`.
    """

    return K.log(K.maximum(input_tensor, 1e-6)) / K.log(2.)


def quantile_loss(quantile_level):
    """Quantile loss function.

    :param quantile_level: Quantile level.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_greater(quantile_level, 0.)
    error_checking.assert_is_less_than(quantile_level, 1.)

    def loss(target_tensor, prediction_tensor):
        """Computes quantile loss.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Quantile loss.
        """

        return K.mean(
            K.maximum(
                quantile_level * (target_tensor - prediction_tensor),
                (quantile_level - 1) * (target_tensor - prediction_tensor)
            )
        )

    return loss


def quantile_loss_one_variable(quantile_level, variable_index):
    """Quantile loss function for one variable only.

    :param quantile_level: Quantile level.
    :param variable_index: Index of target variable used in this loss function.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_greater(quantile_level, 0.)
    error_checking.assert_is_less_than(quantile_level, 1.)
    error_checking.assert_is_integer(variable_index)
    error_checking.assert_is_geq(variable_index, 0)

    def loss(target_tensor, prediction_tensor):
        """Computes quantile loss.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Quantile loss.
        """

        this_target_tensor = target_tensor[:, variable_index]
        this_prediction_tensor = tensorflow.squeeze(prediction_tensor)

        return K.mean(
            K.maximum(
                quantile_level * (this_target_tensor - this_prediction_tensor),
                (quantile_level - 1) *
                (this_target_tensor - this_prediction_tensor)
            )
        )

    return loss


def quantile_loss_3d_output(quantile_level, quantile_index):
    """Computes quantile loss for one quantile level with 3-D output.

    "3-D output" means that the prediction tensor has three axes: examples, then
    lead times, then predictions.  So dimensions are E x L x (1 + Q), where
    Q = number of quantile levels.

    :param quantile_level: Quantile level.
    :param quantile_index: Index of quantile level for use in this loss
        function.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_greater(quantile_level, 0.)
    error_checking.assert_is_less_than(quantile_level, 1.)
    error_checking.assert_is_integer(quantile_index)
    error_checking.assert_is_geq(quantile_index, 1)

    def loss(target_tensor, prediction_tensor):
        """Computes quantile loss for one quantile level with 3-D output.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Quantile loss.
        """

        this_prediction_tensor = prediction_tensor[..., quantile_index + 1]

        return K.mean(
            K.maximum(
                quantile_level * (target_tensor - this_prediction_tensor),
                (quantile_level - 1) * (target_tensor - this_prediction_tensor)
            )
        )

    return loss


def cross_entropy_3d_output():
    """Computes cross-entropy for central prediction with 3-D output.

    "3-D output" means that the prediction tensor has three axes: examples, then
    lead times, then predictions.  So dimensions are E x L x S, with the first
    index of the last axis representing the central prediction.

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes cross-entropy for central prediction with 3-D output.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Quantile loss.
        """

        this_prediction_tensor = prediction_tensor[..., 0]

        xentropy_tensor = (
            target_tensor * _log2(this_prediction_tensor) +
            (1. - target_tensor) * _log2(1. - this_prediction_tensor)
        )

        return -K.mean(xentropy_tensor)

    return loss


def quantile_loss_plus_xentropy_3d_output(quantile_levels):
    """Computes total loss with 3-D output.

    "3-D output" means that the prediction tensor has three axes: examples, then
    lead times, then predictions.  So dimensions are E x L x (1 + Q), where
    Q = number of quantile levels.

    :param quantile_levels: length-Q numpy array of quantile levels.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(quantile_levels, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(quantile_levels, 0.)
    error_checking.assert_is_less_than_numpy_array(quantile_levels, 1.)

    quantile_levels = numpy.expand_dims(quantile_levels, axis=0)
    quantile_levels = numpy.expand_dims(quantile_levels, axis=0)

    def loss(target_tensor, prediction_tensor):
        """Computes total loss with 3-D output.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Quantile loss.
        """

        this_target_tensor = K.expand_dims(target_tensor, axis=-1)
        this_prediction_tensor = prediction_tensor[..., 1:]

        return K.mean(
            K.maximum(
                quantile_levels * (this_target_tensor - this_prediction_tensor),
                (quantile_levels - 1) *
                (this_target_tensor - this_prediction_tensor)
            )
        )

    return loss
