"""Methods for calibrating uncertainty estimates."""

import os
import sys
import numpy
import netCDF4
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import uq_evaluation

BIN_DIMENSION_KEY = 'bin'
BIN_EDGE_DIMENSION_KEY = 'bin_edge'
STDEV_INFLATION_FACTORS_KEY = 'stdev_inflation_factors'
BIN_EDGE_STDEVS_KEY = 'bin_edge_prediction_stdevs'


def _check_predictions(prediction_dict):
    """Error-checks predictions (to be used for training or to be calibrated).

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :raises: ValueError: if predictions were created with quantile regression or
        with no uncertainty at all.
    """

    quantile_levels = prediction_dict[prediction_io.QUANTILE_LEVELS_KEY]
    num_prediction_sets = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape[-1]
    )

    if quantile_levels is not None:
        raise ValueError(
            'Uncertainty calibration cannot be done for models with quantile '
            'regression (yet -- maybe I will get more creative in the future).'
        )

    if num_prediction_sets == 1:
        raise ValueError(
            'Uncertainty calibration cannot be done for models with only one '
            'prediction set (no uncertainty quantification)!'
        )


def _check_model(bin_edge_prediction_stdevs, stdev_inflation_factors):
    """Error-checks uncertainty-calibration model.

    :param bin_edge_prediction_stdevs: See output doc for `train_model`.
    :param stdev_inflation_factors: Same.
    :raises: AssertionError: if first bin edge != 0 or last bin edge !=
        infinity.
    """

    error_checking.assert_is_numpy_array(
        bin_edge_prediction_stdevs, num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(
        bin_edge_prediction_stdevs[1:], 0.
    )
    error_checking.assert_is_less_than_numpy_array(
        bin_edge_prediction_stdevs[:-1], 1.
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(bin_edge_prediction_stdevs), 0.
    )

    assert numpy.isclose(bin_edge_prediction_stdevs[0], 0, atol=1e-6)
    assert numpy.isinf(bin_edge_prediction_stdevs[-1])

    num_bins = len(bin_edge_prediction_stdevs) - 1
    error_checking.assert_is_numpy_array(
        stdev_inflation_factors,
        exact_dimensions=numpy.array([num_bins], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(stdev_inflation_factors, 0.)


def train_model(prediction_dict, bin_edge_prediction_stdevs):
    """Trains uncertainty-calibration model.

    B = number of bins

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param bin_edge_prediction_stdevs: length-(B - 1) numpy array of bin
        cutoffs.  Each is a standard deviation for the predictive distribution.
        Ultimately, there will be B + 1 edges; this method will use 0 as the
        lowest edge and 1 as the highest edge.
    :return: bin_edge_prediction_stdevs: length-(B + 1) numpy array, where the
        [i]th and [i + 1]th entries are the edges for the [i]th bin.
    :return: stdev_inflation_factors: length-B numpy array of inflation factors
        for standard deviation.
    """

    _check_predictions(prediction_dict)

    spread_skill_result_dict = uq_evaluation.get_spread_vs_skill(
        prediction_dict=prediction_dict,
        bin_edge_prediction_stdevs=bin_edge_prediction_stdevs,
        use_median=False, use_fancy_quantile_method_for_stdev=False
    )

    bin_edge_prediction_stdevs = (
        spread_skill_result_dict[uq_evaluation.BIN_EDGE_PREDICTION_STDEVS_KEY]
    )
    stdev_inflation_factors = (
        spread_skill_result_dict[uq_evaluation.RMSE_VALUES_KEY] /
        spread_skill_result_dict[uq_evaluation.MEAN_PREDICTION_STDEVS_KEY]
    )

    nan_flags = numpy.isnan(stdev_inflation_factors)
    if not numpy.any(nan_flags):
        return bin_edge_prediction_stdevs, stdev_inflation_factors

    nan_indices = numpy.where(nan_flags)[0]
    real_indices = numpy.where(numpy.invert(nan_flags))[0]
    bin_center_prediction_stdevs = 0.5 * (
        bin_edge_prediction_stdevs[:-1] + bin_edge_prediction_stdevs[1:]
    )

    interp_object = interp1d(
        x=bin_center_prediction_stdevs[real_indices],
        y=stdev_inflation_factors[real_indices],
        kind='linear', bounds_error=False, assume_sorted=True,
        fill_value=(
            stdev_inflation_factors[real_indices[0]],
            stdev_inflation_factors[real_indices[-1]]
        )
    )

    stdev_inflation_factors[nan_indices] = interp_object(
        bin_center_prediction_stdevs[nan_indices]
    )

    return bin_edge_prediction_stdevs, stdev_inflation_factors


def apply_model(prediction_dict, bin_edge_prediction_stdevs,
                stdev_inflation_factors):
    """Applies trained uncertainty-calibration model.

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`, containing predictions to calibrate.
    :param bin_edge_prediction_stdevs: See output doc for `train_model`.
    :param stdev_inflation_factors: Same.
    :return: prediction_dict: Same but with calibrated predictions.
    """

    _check_predictions(prediction_dict)
    _check_model(
        bin_edge_prediction_stdevs=bin_edge_prediction_stdevs,
        stdev_inflation_factors=stdev_inflation_factors
    )

    # Do actual stuff.
    num_prediction_sets = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape[-1]
    )

    stdev_probability_matrix = prediction_io.get_predictive_stdevs(
        prediction_dict=prediction_dict, use_fancy_quantile_method=False
    )
    mean_probability_matrix = prediction_io.get_mean_predictions(
        prediction_dict
    )
    mean_probability_matrix = numpy.repeat(
        numpy.expand_dims(mean_probability_matrix, -1),
        axis=-1, repeats=num_prediction_sets
    )

    full_probability_matrix = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][:, 1, ...]
    )
    full_probability_matrix_new = numpy.full(
        full_probability_matrix.shape, numpy.nan
    )
    num_bins = len(bin_edge_prediction_stdevs) - 1

    for k in range(num_bins):
        these_indices = numpy.where(numpy.logical_and(
            stdev_probability_matrix >= bin_edge_prediction_stdevs[k],
            stdev_probability_matrix < bin_edge_prediction_stdevs[k + 1]
        ))

        fpmo = full_probability_matrix
        full_probability_matrix_new[these_indices] = (
            mean_probability_matrix[these_indices] +
            stdev_inflation_factors[k] *
            (fpmo[these_indices] - mean_probability_matrix[these_indices])
        )

    assert not numpy.any(numpy.isnan(full_probability_matrix_new))

    # Make predictions increase with lead time.
    diff_matrix = numpy.diff(full_probability_matrix_new, axis=-2)
    diff_matrix = numpy.maximum(diff_matrix, 0.)

    full_probability_matrix_new = numpy.concatenate((
        full_probability_matrix_new[:, 0, :],
        full_probability_matrix_new[:, 0, :] + diff_matrix
    ), axis=-2)

    full_probability_matrix_new = numpy.maximum(full_probability_matrix_new, 0.)
    full_probability_matrix_new = numpy.minimum(full_probability_matrix_new, 1.)

    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = numpy.stack(
        (1. - full_probability_matrix_new, full_probability_matrix_new), axis=-3
    )

    # Nudge mean predictions back to what they were before calibration.
    mean_probability_matrix_new = prediction_io.get_mean_predictions(
        prediction_dict
    )
    mean_probability_matrix_new = numpy.repeat(
        numpy.expand_dims(mean_probability_matrix_new, -1),
        axis=-1, repeats=num_prediction_sets
    )
    full_probability_matrix_new = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][:, 1, ...]
    )

    full_probability_matrix_new = (
        full_probability_matrix_new -
        (mean_probability_matrix_new - mean_probability_matrix)
    )
    full_probability_matrix_new = numpy.maximum(full_probability_matrix_new, 0.)
    full_probability_matrix_new = numpy.minimum(full_probability_matrix_new, 1.)

    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = numpy.stack(
        (1. - full_probability_matrix_new, full_probability_matrix_new), axis=-3
    )

    return prediction_dict


def write_model(netcdf_file_name, bin_edge_prediction_stdevs,
                stdev_inflation_factors):
    """Writes uncertainty-calibration model to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param bin_edge_prediction_stdevs: See output doc for `train_model`.
    :param stdev_inflation_factors: Same.
    """

    _check_model(
        bin_edge_prediction_stdevs=bin_edge_prediction_stdevs,
        stdev_inflation_factors=stdev_inflation_factors
    )

    # Write file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    num_bins = len(bin_edge_prediction_stdevs) - 1

    dataset_object.createDimension(BIN_DIMENSION_KEY, num_bins)
    dataset_object.createDimension(BIN_EDGE_DIMENSION_KEY, num_bins + 1)

    dataset_object.createVariable(
        STDEV_INFLATION_FACTORS_KEY, datatype=numpy.float32,
        dimensions=BIN_DIMENSION_KEY
    )
    dataset_object.variables[STDEV_INFLATION_FACTORS_KEY][:] = (
        stdev_inflation_factors
    )

    dataset_object.createVariable(
        BIN_EDGE_STDEVS_KEY, datatype=numpy.float32,
        dimensions=BIN_EDGE_DIMENSION_KEY
    )
    dataset_object.variables[BIN_EDGE_STDEVS_KEY][:] = (
        bin_edge_prediction_stdevs
    )

    dataset_object.close()


def read_model(netcdf_file_name):
    """Reads uncertainty-calibration model from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: bin_edge_prediction_stdevs: See output doc for `train_model`.
    :return: stdev_inflation_factors: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    bin_edge_prediction_stdevs = (
        dataset_object.variables[BIN_EDGE_STDEVS_KEY][:]
    )
    stdev_inflation_factors = (
        dataset_object.variables[STDEV_INFLATION_FACTORS_KEY][:]
    )

    dataset_object.close()
    return bin_edge_prediction_stdevs, stdev_inflation_factors
