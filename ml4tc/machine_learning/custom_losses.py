"""Custom loss functions for Keras models."""

from tensorflow.keras import backend as K


def quantile_loss(quantile_level):
    """Quantile loss function.

    :param quantile_level: Quantile level.
    :return: loss: Loss function (defined below).
    """

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
