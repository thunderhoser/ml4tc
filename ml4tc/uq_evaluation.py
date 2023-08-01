"""Evaluation methods for uncertainty quantification (UQ)."""

import os
import sys
import numpy
import netCDF4
from scipy.integrate import simps
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io

PROB_LEVELS_TO_INTEG_NOT_QR = numpy.linspace(0, 1, num=101, dtype=float)
PROB_LEVELS_TO_INTEG_FOR_QR = numpy.linspace(0, 1, num=41, dtype=float)
NUM_EXAMPLES_PER_BATCH = 100

ERROR_FUNCTION_KEY = 'error_function_name'
UNCERTAINTY_FUNCTION_KEY = 'uncertainty_function_name'

DISCARD_FRACTION_DIM_KEY = 'discard_fraction'
DISCARD_FRACTIONS_KEY = 'discard_fractions'
ERROR_VALUES_KEY = 'error_values'
EXAMPLE_FRACTIONS_KEY = 'example_fractions'
MEAN_CENTRAL_PREDICTIONS_KEY = 'mean_central_predictions'
MEAN_TARGET_VALUES_KEY = 'mean_target_values'
MONOTONICITY_FRACTION_KEY = 'monotonicity_fraction'

USE_MEDIAN_KEY = 'use_median'
USE_FANCY_QUANTILES_KEY = 'use_fancy_quantile_method_for_stdev'

SPREAD_SKILL_BIN_DIM_KEY = 'bin'
SPREAD_SKILL_BIN_EDGE_DIM_KEY = 'bin_edge'
MEAN_PREDICTION_STDEVS_KEY = 'mean_prediction_stdevs'
BIN_EDGE_PREDICTION_STDEVS_KEY = 'bin_edge_prediction_stdevs'
RMSE_VALUES_KEY = 'rmse_values'
EXAMPLE_COUNTS_KEY = 'example_counts'
SPREAD_SKILL_RELIABILITY_KEY = 'spread_skill_reliability'
SPREAD_SKILL_RATIO_KEY = 'spread_skill_ratio'


def _log2(input_array):
    """Computes logarithm in base 2.

    :param input_array: numpy array.
    :return: logarithm_array: numpy array with the same shape as `input_array`.
    """

    return numpy.log2(numpy.maximum(input_array, 1e-6))


def _get_squared_errors(prediction_dict, use_median):
    """Returns squared errors.

    E = number of examples
    L = number of lead times.

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param use_median: Boolean flag.  If True (False), will use median (mean) of
        each predictive distribution.
    :return: squared_error_matrix: E-by-L numpy array of squared errors.
    """

    if use_median:
        forecast_prob_matrix = prediction_io.get_median_predictions(
            prediction_dict
        )
    else:
        forecast_prob_matrix = prediction_io.get_mean_predictions(
            prediction_dict
        )

    target_class_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY]
    return (forecast_prob_matrix - target_class_matrix) ** 2


def _get_crps_monte_carlo(prediction_dict):
    """Computes CRPS for model with Monte Carlo dropout.

    CRPS = continuous ranked probability score

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :return: crps_value: CRPS (scalar).
    """

    forecast_prob_matrix = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][:, 1, ...]
    )
    target_class_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY]
    num_examples = forecast_prob_matrix.shape[0]

    crps_numerator = 0.
    crps_denominator = 0.

    for i in range(0, num_examples, NUM_EXAMPLES_PER_BATCH):
        print('Have computed CRPS for {0:d} of {1:d} examples...'.format(
            i, num_examples
        ))

        first_index = i
        last_index = min([
            i + NUM_EXAMPLES_PER_BATCH, num_examples
        ])

        cdf_matrix = numpy.stack([
            numpy.mean(
                forecast_prob_matrix[first_index:last_index, ...] <= p, axis=-1
            )
            for p in PROB_LEVELS_TO_INTEG_NOT_QR
        ], axis=-1)

        this_target_matrix = numpy.expand_dims(
            target_class_matrix[first_index:last_index, :], axis=-1
        )
        integrated_cdf_matrix = simps(
            y=(cdf_matrix + (this_target_matrix - 1)) ** 2,
            x=PROB_LEVELS_TO_INTEG_NOT_QR, axis=-1
        )
        crps_numerator += numpy.sum(integrated_cdf_matrix)
        crps_denominator += integrated_cdf_matrix.size

    print('Have computed CRPS for all {0:d} examples!'.format(num_examples))

    return crps_numerator / crps_denominator


def _get_crps_quantile_regression_1batch(
        prediction_dict, first_example_index, last_example_index):
    """Computes CRPS for quantile-regression model on one batch of examples.

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param first_example_index: Array index of first example in batch.
    :param last_example_index: Array index of last example in batch.
    :return: crps_numerator: Numerator of CRPS.
    :return: crps_denominator: Denominator of CRPS.
    """

    forecast_prob_matrix = prediction_dict[
        prediction_io.PROBABILITY_MATRIX_KEY
    ][first_example_index:last_example_index, 1, ...]

    target_class_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY][
        first_example_index:last_example_index, ...
    ]
    target_class_matrix = numpy.expand_dims(target_class_matrix, axis=-1)

    num_examples = forecast_prob_matrix.shape[0]
    num_lead_times = forecast_prob_matrix.shape[1]
    cdf_matrix = numpy.full(
        (num_examples, num_lead_times, len(PROB_LEVELS_TO_INTEG_FOR_QR)),
        numpy.nan
    )

    for i in range(num_examples):
        for j in range(num_lead_times):
            interp_object = interp1d(
                x=forecast_prob_matrix[i, j, 1:],
                y=prediction_dict[prediction_io.QUANTILE_LEVELS_KEY],
                kind='linear', bounds_error=False, assume_sorted=True,
                fill_value='extrapolate'
            )
            cdf_matrix[i, j, :] = interp_object(PROB_LEVELS_TO_INTEG_FOR_QR)

    cdf_matrix = numpy.maximum(cdf_matrix, 0.)
    cdf_matrix = numpy.minimum(cdf_matrix, 1.)
    cdf_matrix[..., -1] = 1.
    cdf_matrix[..., 0] = 0.

    integrated_cdf_matrix = simps(
        y=(cdf_matrix + (target_class_matrix - 1)) ** 2,
        x=PROB_LEVELS_TO_INTEG_FOR_QR, axis=-1
    )

    return numpy.sum(integrated_cdf_matrix), integrated_cdf_matrix.size


def _get_crps_quantile_regression(prediction_dict):
    """Computes CRPS for model with quantile regression.

    CRPS = continuous ranked probability score

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :return: crps_value: CRPS (scalar).
    """

    num_examples = prediction_dict[prediction_io.TARGET_MATRIX_KEY].shape[0]
    crps_numerator = 0.
    crps_denominator = 0.

    for i in range(0, num_examples, NUM_EXAMPLES_PER_BATCH):
        print('Have computed CRPS for {0:d} of {1:d} examples...'.format(
            i, num_examples
        ))

        this_numerator, this_denominator = _get_crps_quantile_regression_1batch(
            prediction_dict=prediction_dict, first_example_index=i,
            last_example_index=min([i + NUM_EXAMPLES_PER_BATCH, num_examples])
        )

        crps_numerator += this_numerator
        crps_denominator += this_denominator

    print('Have computed CRPS for all {0:d} examples!'.format(num_examples))

    return crps_numerator / crps_denominator


def get_xentropy_error_function(use_median):
    """Creates error function to compute cross-entropy.

    :param use_median: Boolean flag.  If True (False), will use median (mean) of
        each predictive distribution.
    :return: error_function: Function handle.
    """

    def error_function(prediction_dict, use_flag_matrix):
        """Computes cross-entropy.

        E = number of examples
        L = number of lead times

        :param prediction_dict: Dictionary in format returned by
            `prediction_io.read_file`.
        :param use_flag_matrix: E-by-L numpy array of Boolean flags, indicating
            which examples to use.
        :return: cross_entropy: Cross-entropy (scalar).
        :raises: ValueError: if there are more than 2 classes.
        """

        num_classes = (
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape[1]
        )
        if num_classes > 2:
            raise ValueError('Cannot do this with more than 2 classes.')

        expected_dim = numpy.array(
            prediction_dict[prediction_io.TARGET_MATRIX_KEY].shape, dtype=int
        )
        error_checking.assert_is_numpy_array(
            use_flag_matrix, exact_dimensions=expected_dim
        )
        error_checking.assert_is_boolean_numpy_array(use_flag_matrix)

        if use_median:
            forecast_probs = prediction_io.get_median_predictions(
                prediction_dict
            )[use_flag_matrix == True]
        else:
            forecast_probs = prediction_io.get_mean_predictions(
                prediction_dict
            )[use_flag_matrix == True]

        target_values = prediction_dict[prediction_io.TARGET_MATRIX_KEY][
            use_flag_matrix == True
        ]

        return -numpy.mean(
            target_values * _log2(forecast_probs) +
            (1. - target_values) * _log2(1. - forecast_probs)
        )

    return error_function


def get_stdev_uncertainty_function(use_fancy_quantile_method):
    """Creates function to compute stdev of predictive distribution.

    :param use_fancy_quantile_method: See doc for
        `prediction_io.get_predictive_stdevs`.
    :return: uncertainty_function: Function handle.
    """

    def uncertainty_function(prediction_dict):
        """Computes stdev of predictive distribution for each example.

        E = number of examples

        :param prediction_dict: Dictionary in format returned by
            `prediction_io.read_file`.
        :return: prob_stdevs: length-E numpy array with standard deviations of
            forecast probabilities.
        """

        return prediction_io.get_predictive_stdevs(
            prediction_dict=prediction_dict,
            use_fancy_quantile_method=use_fancy_quantile_method
        )

    return uncertainty_function


def get_crps(prediction_dict):
    """Computes continuous ranked probability score (CRPS).

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :return: crps_value: CRPS (scalar).
    :raises: ValueError: if there are more than 2 classes.
    """

    num_classes = prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape[1]
    if num_classes > 2:
        raise ValueError('Cannot do this with more than 2 classes.')

    if prediction_dict[prediction_io.QUANTILE_LEVELS_KEY] is None:
        return _get_crps_monte_carlo(prediction_dict)

    return _get_crps_quantile_regression(prediction_dict)


def run_discard_test(
        prediction_dict, discard_fractions, error_function,
        uncertainty_function, use_median, is_error_pos_oriented):
    """Runs the discard test.

    F = number of discard fractions
    E = number of examples
    L = number of lead times.

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param discard_fractions: length-(F - 1) numpy array of discard fractions,
        ranging from (0, 1).  This method will use 0 as the lowest discard
        fraction.

    :param error_function: Function with the following inputs and outputs...
    Input: prediction_dict: See above.
    Input: use_flag_matrix: E-by-L numpy array of Boolean flags, indicating
        which data points to use.
    Output: error_value: Scalar value of error metric.

    :param uncertainty_function: Function with the following inputs and
        outputs...
    Input: prediction_dict: See above.
    Output: uncertainty_matrix: E-by-L numpy array with values of uncertainty
        metric.  The metric must be oriented so that higher value = more
        uncertainty.

    :param use_median: Boolean flag.  If True (False), central predictions will
        be medians (means).
    :param is_error_pos_oriented: Boolean flag.  If True (False), error function
        is positively (negatively) oriented.

    :return: result_dict: Dictionary with the following keys.
    result_dict['discard_fractions']: length-F numpy array of discard fractions,
        sorted in increasing order.
    result_dict['error_values']: length-F numpy array of corresponding error
        values.
    result_dict['example_fractions']: length-F numpy array with fraction of
        examples left after each discard.
    result_dict['mean_central_predictions']: length-F numpy array, where the
        [i]th entry is the mean central (mean or median) prediction for the
        [i]th discard fraction.
    result_dict['mean_target_values']: length-F numpy array, where the [i]th
        entry is the mean target value for the [i]th discard fraction.
    result_dict['monotonicity_fraction']: Monotonicity fraction.  This is the
        fraction of times that the error function improves when discard
        fraction is increased.

    :raises: ValueError: if there are more than 2 classes.
    """

    num_classes = prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape[1]
    if num_classes > 2:
        raise ValueError('Cannot do this with more than 2 classes.')

    # Check input args.
    error_checking.assert_is_boolean(use_median)
    error_checking.assert_is_boolean(is_error_pos_oriented)

    error_checking.assert_is_numpy_array(discard_fractions, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(discard_fractions, 0.)
    error_checking.assert_is_less_than_numpy_array(discard_fractions, 1.)

    discard_fractions = numpy.concatenate((
        numpy.array([0.]),
        discard_fractions
    ))

    num_fractions = len(discard_fractions)
    assert num_fractions >= 2

    # Do actual stuff.
    uncertainty_matrix = uncertainty_function(prediction_dict)

    if use_median:
        central_prediction_matrix = prediction_io.get_median_predictions(
            prediction_dict
        )
    else:
        central_prediction_matrix = prediction_io.get_mean_predictions(
            prediction_dict
        )

    discard_fractions = numpy.sort(discard_fractions)
    error_values = numpy.full(num_fractions, numpy.nan)
    example_fractions = numpy.full(num_fractions, numpy.nan)
    mean_central_predictions = numpy.full(num_fractions, numpy.nan)
    mean_target_values = numpy.full(num_fractions, numpy.nan)
    use_flag_matrix = numpy.full(uncertainty_matrix.shape, 1, dtype=bool)

    for k in range(num_fractions):
        this_percentile_level = 100 * (1 - discard_fractions[k])
        this_inverted_mask = (
            uncertainty_matrix >
            numpy.percentile(uncertainty_matrix, this_percentile_level)
        )
        use_flag_matrix[this_inverted_mask] = False

        example_fractions[k] = numpy.mean(use_flag_matrix)

        error_values[k] = error_function(prediction_dict, use_flag_matrix)
        mean_central_predictions[k] = numpy.mean(
            central_prediction_matrix[use_flag_matrix == True]
        )
        mean_target_values[k] = numpy.mean(
            prediction_dict[prediction_io.TARGET_MATRIX_KEY][
                use_flag_matrix == True
            ]
        )

    if is_error_pos_oriented:
        monotonicity_fraction = numpy.mean(numpy.diff(error_values) > 0)
    else:
        monotonicity_fraction = numpy.mean(numpy.diff(error_values) < 0)

    return {
        DISCARD_FRACTIONS_KEY: discard_fractions,
        ERROR_VALUES_KEY: error_values,
        EXAMPLE_FRACTIONS_KEY: example_fractions,
        MEAN_CENTRAL_PREDICTIONS_KEY: mean_central_predictions,
        MEAN_TARGET_VALUES_KEY: mean_target_values,
        MONOTONICITY_FRACTION_KEY: monotonicity_fraction
    }


def get_spread_vs_skill(
        prediction_dict, bin_edge_prediction_stdevs, use_median,
        use_fancy_quantile_method_for_stdev):
    """Computes model spread vs. model skill.

    B = number of bins

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param bin_edge_prediction_stdevs: length-(B - 1) numpy array of bin
        cutoffs.  Each is a standard deviation for the predictive distribution.
        Ultimately, there will be B + 1 edges; this method will use 0 as the
        lowest edge and 1 as the highest edge.
    :param use_median: Boolean flag.  If True (False), will use median (mean) of
        each predictive distribution.
    :param use_fancy_quantile_method_for_stdev: See doc for
        `prediction_io.get_predictive_stdevs`.
    :return: result_dict: Dictionary with the following keys.
    result_dict['mean_prediction_stdevs']: length-B numpy array, where the [i]th
        entry is the mean standard deviation of predictive distributions in the
        [i]th bin.
    result_dict['bin_edge_prediction_stdevs']: length-(B + 1) numpy array,
        where the [i]th and [i + 1]th entries are the edges for the [i]th bin.
    result_dict['rmse_values']: length-B numpy array, where the [i]th
        entry is the root mean squared error of central (mean or median)
        predictions in the [i]th bin.
    result_dict['spread_skill_reliability']: Spread-skill reliability (SSREL).
    result_dict['spread_skill_ratio']: Spread-skill ratio (SSRAT).
    result_dict['example_counts']: length-B numpy array of corresponding example
        counts.
    result_dict['mean_central_predictions']: length-B numpy array, where the
        [i]th entry is the mean central (mean or median) prediction for the
        [i]th bin.
    result_dict['mean_target_values']: length-B numpy array, where the [i]th
        entry is the mean target value for the [i]th bin.
    """

    # Check input args.
    error_checking.assert_is_boolean(use_median)

    error_checking.assert_is_numpy_array(
        bin_edge_prediction_stdevs, num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(bin_edge_prediction_stdevs, 0.)
    error_checking.assert_is_less_than_numpy_array(
        bin_edge_prediction_stdevs, 1.
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(bin_edge_prediction_stdevs), 0.
    )

    bin_edge_prediction_stdevs = numpy.concatenate((
        numpy.array([0.]),
        bin_edge_prediction_stdevs,
        numpy.array([numpy.inf])
    ))

    num_bins = len(bin_edge_prediction_stdevs) - 1
    assert num_bins >= 2

    num_prediction_sets = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape[3]
    )
    assert num_prediction_sets > 2

    # Do actual stuff.
    if use_median:
        central_prediction_matrix = prediction_io.get_median_predictions(
            prediction_dict
        )
    else:
        central_prediction_matrix = prediction_io.get_mean_predictions(
            prediction_dict
        )

    predictive_stdev_matrix = prediction_io.get_predictive_stdevs(
        prediction_dict=prediction_dict,
        use_fancy_quantile_method=use_fancy_quantile_method_for_stdev
    )
    squared_error_matrix = _get_squared_errors(
        prediction_dict=prediction_dict, use_median=use_median
    )

    mean_prediction_stdevs = numpy.full(num_bins, numpy.nan)
    rmse_values = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, 0, dtype=int)
    mean_central_predictions = numpy.full(num_bins, numpy.nan)
    mean_target_values = numpy.full(num_bins, numpy.nan)

    for k in range(num_bins):
        these_indices = numpy.where(numpy.logical_and(
            predictive_stdev_matrix >= bin_edge_prediction_stdevs[k],
            predictive_stdev_matrix < bin_edge_prediction_stdevs[k + 1]
        ))

        mean_prediction_stdevs[k] = numpy.sqrt(numpy.mean(
            predictive_stdev_matrix[these_indices] ** 2
        ))
        rmse_values[k] = numpy.sqrt(numpy.mean(
            squared_error_matrix[these_indices]
        ))
        example_counts[k] = len(these_indices[0])
        mean_central_predictions[k] = numpy.mean(
            central_prediction_matrix[these_indices]
        )
        mean_target_values[k] = numpy.mean(
            prediction_dict[prediction_io.TARGET_MATRIX_KEY][these_indices]
        )

    these_diffs = numpy.absolute(mean_prediction_stdevs - rmse_values)
    these_diffs[numpy.isnan(these_diffs)] = 0.
    spread_skill_reliability = numpy.average(
        these_diffs, weights=example_counts
    )

    non_zero_indices = numpy.where(example_counts > 0)[0]

    this_numer = numpy.sqrt(numpy.average(
        mean_prediction_stdevs[non_zero_indices] ** 2,
        weights=example_counts[non_zero_indices]
    ))
    this_denom = numpy.sqrt(numpy.average(
        rmse_values[non_zero_indices] ** 2,
        weights=example_counts[non_zero_indices]
    ))
    spread_skill_ratio = this_numer / this_denom

    return {
        MEAN_PREDICTION_STDEVS_KEY: mean_prediction_stdevs,
        BIN_EDGE_PREDICTION_STDEVS_KEY: bin_edge_prediction_stdevs,
        RMSE_VALUES_KEY: rmse_values,
        SPREAD_SKILL_RELIABILITY_KEY: spread_skill_reliability,
        SPREAD_SKILL_RATIO_KEY: spread_skill_ratio,
        EXAMPLE_COUNTS_KEY: example_counts,
        MEAN_CENTRAL_PREDICTIONS_KEY: mean_central_predictions,
        MEAN_TARGET_VALUES_KEY: mean_target_values
    }


def write_discard_results(
        netcdf_file_name, result_dict, error_function_name,
        uncertainty_function_name, use_fancy_quantile_method_for_stdev):
    """Writes results of discard test to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param result_dict: Dictionary returned by `run_discard_test`.
    :param error_function_name: Name of error function (string).  This will be
        used later for plotting.
    :param uncertainty_function_name: Name of uncertainty function (string).
        This will be used later for plotting.
    :param use_fancy_quantile_method_for_stdev: See doc for
        `compute_spread_vs_skill`.
    """

    # Check input args.
    error_checking.assert_is_string(error_function_name)
    error_checking.assert_is_string(uncertainty_function_name)
    error_checking.assert_is_boolean(use_fancy_quantile_method_for_stdev)

    # Write file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(ERROR_FUNCTION_KEY, error_function_name)
    dataset_object.setncattr(
        UNCERTAINTY_FUNCTION_KEY, uncertainty_function_name
    )
    dataset_object.setncattr(
        USE_FANCY_QUANTILES_KEY, int(use_fancy_quantile_method_for_stdev)
    )
    dataset_object.setncattr(
        MONOTONICITY_FRACTION_KEY, result_dict[MONOTONICITY_FRACTION_KEY]
    )

    num_fractions = len(result_dict[DISCARD_FRACTIONS_KEY])
    dataset_object.createDimension(DISCARD_FRACTION_DIM_KEY, num_fractions)

    for this_key in [
            DISCARD_FRACTIONS_KEY, ERROR_VALUES_KEY,
            MEAN_CENTRAL_PREDICTIONS_KEY, MEAN_TARGET_VALUES_KEY,
            EXAMPLE_FRACTIONS_KEY
    ]:
        dataset_object.createVariable(
            this_key, datatype=numpy.float64,
            dimensions=DISCARD_FRACTION_DIM_KEY
        )
        dataset_object.variables[this_key][:] = result_dict[this_key]

    dataset_object.close()


def read_discard_results(netcdf_file_name):
    """Reads results of discard test from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_dict: Dictionary in format created by `run_discard_test`,
        plus the following extra keys.
    result_dict['error_function_name']: Name of error metric used in test.
    result_dict['uncertainty_function_name']: Name of uncertainty metric used in
        test.
    result_dict['use_fancy_quantile_method_for_stdev']: See doc for
        `compute_spread_vs_skill`.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    result_dict = {
        ERROR_FUNCTION_KEY: str(getattr(dataset_object, ERROR_FUNCTION_KEY)),
        UNCERTAINTY_FUNCTION_KEY: str(
            getattr(dataset_object, UNCERTAINTY_FUNCTION_KEY)
        ),
        MONOTONICITY_FRACTION_KEY: float(
            getattr(dataset_object, MONOTONICITY_FRACTION_KEY)
        )
    }

    try:
        result_dict[USE_FANCY_QUANTILES_KEY] = bool(
            getattr(dataset_object, USE_FANCY_QUANTILES_KEY)
        )
    except:
        result_dict[USE_FANCY_QUANTILES_KEY] = False

    for this_key in [
            DISCARD_FRACTIONS_KEY, ERROR_VALUES_KEY,
            MEAN_CENTRAL_PREDICTIONS_KEY, MEAN_TARGET_VALUES_KEY,
            EXAMPLE_FRACTIONS_KEY
    ]:
        result_dict[this_key] = numpy.array(
            dataset_object.variables[this_key][:], dtype=float
        )

    dataset_object.close()
    return result_dict


def write_spread_vs_skill(
        netcdf_file_name, result_dict, use_median,
        use_fancy_quantile_method_for_stdev):
    """Writes spread vs. skill to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param result_dict: Dictionary created by `get_spread_vs_skill`.
    :param use_median: Same.
    :param use_fancy_quantile_method_for_stdev: Same.
    """

    # Check input args.
    error_checking.assert_is_boolean(use_median)
    error_checking.assert_is_boolean(use_fancy_quantile_method_for_stdev)

    # Write file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(USE_MEDIAN_KEY, int(use_median))
    dataset_object.setncattr(
        USE_FANCY_QUANTILES_KEY, int(use_fancy_quantile_method_for_stdev)
    )
    dataset_object.setncattr(
        SPREAD_SKILL_RELIABILITY_KEY, result_dict[SPREAD_SKILL_RELIABILITY_KEY]
    )
    dataset_object.setncattr(
        SPREAD_SKILL_RATIO_KEY, result_dict[SPREAD_SKILL_RATIO_KEY]
    )

    num_bins = len(result_dict[MEAN_PREDICTION_STDEVS_KEY])
    dataset_object.createDimension(SPREAD_SKILL_BIN_DIM_KEY, num_bins)
    dataset_object.createDimension(SPREAD_SKILL_BIN_EDGE_DIM_KEY, num_bins + 1)

    for this_key in [
            MEAN_PREDICTION_STDEVS_KEY, RMSE_VALUES_KEY,
            MEAN_CENTRAL_PREDICTIONS_KEY, MEAN_TARGET_VALUES_KEY
    ]:
        dataset_object.createVariable(
            this_key, datatype=numpy.float64,
            dimensions=SPREAD_SKILL_BIN_DIM_KEY
        )
        dataset_object.variables[this_key][:] = result_dict[this_key]

    dataset_object.createVariable(
        BIN_EDGE_PREDICTION_STDEVS_KEY, datatype=numpy.float64,
        dimensions=SPREAD_SKILL_BIN_EDGE_DIM_KEY
    )
    dataset_object.variables[BIN_EDGE_PREDICTION_STDEVS_KEY][:] = (
        result_dict[BIN_EDGE_PREDICTION_STDEVS_KEY]
    )

    for this_key in [EXAMPLE_COUNTS_KEY]:
        dataset_object.createVariable(
            this_key, datatype=numpy.int32,
            dimensions=SPREAD_SKILL_BIN_DIM_KEY
        )
        dataset_object.variables[this_key][:] = result_dict[this_key]

    dataset_object.close()


def read_spread_vs_skill(netcdf_file_name):
    """Reads spread vs. skill from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_dict: Dictionary in format created by `get_spread_vs_skill`,
        plus the following extra keys.
    result_dict['use_median']: Boolean flag.  If True (False), used median
        (mean) to define central prediction.
    result_dict['use_fancy_quantile_method_for_stdev']: See doc for
        `get_spread_vs_skill`.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    result_dict = {
        USE_MEDIAN_KEY: bool(getattr(dataset_object, USE_MEDIAN_KEY)),
        SPREAD_SKILL_RELIABILITY_KEY:
            float(getattr(dataset_object, SPREAD_SKILL_RELIABILITY_KEY))
    }

    try:
        result_dict[USE_FANCY_QUANTILES_KEY] = bool(
            getattr(dataset_object, USE_FANCY_QUANTILES_KEY)
        )
    except:
        result_dict[USE_FANCY_QUANTILES_KEY] = False

    for this_key in [
            MEAN_PREDICTION_STDEVS_KEY, BIN_EDGE_PREDICTION_STDEVS_KEY,
            RMSE_VALUES_KEY, MEAN_CENTRAL_PREDICTIONS_KEY,
            MEAN_TARGET_VALUES_KEY
    ]:
        result_dict[this_key] = numpy.array(
            dataset_object.variables[this_key][:], dtype=float
        )

    for this_key in [EXAMPLE_COUNTS_KEY]:
        result_dict[this_key] = numpy.array(
            dataset_object.variables[this_key][:], dtype=int
        )

    if hasattr(dataset_object, SPREAD_SKILL_RATIO_KEY):
        result_dict[SPREAD_SKILL_RATIO_KEY] = float(
            getattr(dataset_object, SPREAD_SKILL_RATIO_KEY)
        )
    else:
        non_zero_indices = numpy.where(result_dict[EXAMPLE_COUNTS_KEY] > 0)[0]

        this_numer = numpy.sqrt(numpy.average(
            result_dict[MEAN_PREDICTION_STDEVS_KEY][non_zero_indices] ** 2,
            weights=result_dict[EXAMPLE_COUNTS_KEY][non_zero_indices]
        ))
        this_denom = numpy.sqrt(numpy.average(
            result_dict[RMSE_VALUES_KEY][non_zero_indices] ** 2,
            weights=result_dict[EXAMPLE_COUNTS_KEY][non_zero_indices]
        ))
        result_dict[SPREAD_SKILL_RATIO_KEY] = this_numer / this_denom

    dataset_object.close()
    return result_dict
