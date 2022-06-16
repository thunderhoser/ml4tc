"""Custom loss functions for Keras models."""

from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking


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
        this_prediction_tensor = prediction_tensor[:, variable_index]

        return K.mean(
            K.maximum(
                quantile_level * (this_target_tensor - this_prediction_tensor),
                (quantile_level - 1) *
                (this_target_tensor - this_prediction_tensor)
            )
        )

    return loss
