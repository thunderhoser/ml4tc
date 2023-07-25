"""Isotonic regression."""

import os
import dill
import numpy
from sklearn.isotonic import IsotonicRegression
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import prediction_io


def train_model(prediction_dict):
    """Trains isotonic-regression model.

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :return: model_object: Trained model (instance of
        `sklearn.isotonic.IsotonicRegression`).
    """

    mean_forecast_prob_matrix = prediction_io.get_mean_predictions(
        prediction_dict
    )
    assert mean_forecast_prob_matrix.shape[1] == 1

    mean_forecast_probs = mean_forecast_prob_matrix[:, 0]
    target_classes = prediction_dict[prediction_io.TARGET_MATRIX_KEY][:, 0]

    model_object = IsotonicRegression(
        increasing=True, out_of_bounds='clip', y_min=0., y_max=1.
    )
    model_object.fit(X=mean_forecast_probs, y=target_classes)

    return model_object


def apply_model(prediction_dict, model_object):
    """Applies isotonic-regression model.

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param model_object: See output doc for `train_model`.
    :return: prediction_dict: Same as input but with different predictions.
    """

    forecast_prob_matrix_4d = prediction_dict[
        prediction_io.PROBABILITY_MATRIX_KEY
    ]
    assert forecast_prob_matrix_4d.shape[2] == 1

    forecast_prob_matrix_2d = forecast_prob_matrix_4d[:, 1, 0, :]
    num_examples = forecast_prob_matrix_2d.shape[0]
    ensemble_size = forecast_prob_matrix_2d.shape[-1]

    forecast_prob_matrix_2d = numpy.reshape(
        model_object.predict(numpy.ravel(forecast_prob_matrix_2d)),
        (num_examples, ensemble_size)
    )

    assert not numpy.any(forecast_prob_matrix_2d < 0.)
    assert not numpy.any(forecast_prob_matrix_2d > 1.)

    forecast_prob_matrix_3d = numpy.expand_dims(
        forecast_prob_matrix_2d, axis=-2
    )
    forecast_prob_matrix_3d = numpy.concatenate(
        (1. - forecast_prob_matrix_3d, forecast_prob_matrix_3d),
        axis=1
    )
    forecast_prob_matrix_4d = numpy.expand_dims(
        forecast_prob_matrix_3d, axis=-2
    )
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        forecast_prob_matrix_4d
    )

    return prediction_dict


def find_file(model_dir_name, raise_error_if_missing=True):
    """Finds Dill file with isotonic-regression model.

    :param model_dir_name: Name of directory.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: dill_file_name: Path to Dill file with model.
    """

    error_checking.assert_is_string(model_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    dill_file_name = '{0:s}/isotonic_regression.dill'.format(model_dir_name)

    if raise_error_if_missing and not os.path.isfile(dill_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            dill_file_name
        )
        raise ValueError(error_string)

    return dill_file_name


def write_file(dill_file_name, model_object):
    """Writes isotonic-regression model to Dill file.

    :param dill_file_name: Path to output file.
    :param model_object: See doc for `train_model`.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(model_object, dill_file_handle)
    dill_file_handle.close()


def read_file(dill_file_name):
    """Reads isotonic-regression model from Dill file.

    :param dill_file_name: Path to input file.
    :return: model_object: See doc for `train_model`.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    model_object = dill.load(dill_file_handle)
    dill_file_handle.close()

    return model_object
