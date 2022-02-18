"""IO and helper methods for permutation-based importance test."""

import os
import sys
import copy
import numpy
import netCDF4
from sklearn.metrics import roc_auc_score as sklearn_auc

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import satellite_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DEFAULT_NUM_BOOTSTRAP_REPS = 1000

THREE_PREDICTOR_MATRICES_KEY = 'three_predictor_matrices'
PERMUTED_FLAGS_KEY = 'permuted_flag_arrays'
PERMUTED_MATRICES_KEY = 'permuted_matrix_indices'
PERMUTED_VARIABLES_KEY = 'permuted_variable_indices'
PERMUTED_COSTS_KEY = 'permuted_cost_matrix'
DEPERMUTED_MATRICES_KEY = 'depermuted_matrix_indices'
DEPERMUTED_VARIABLES_KEY = 'depermuted_variable_indices'
DEPERMUTED_COSTS_KEY = 'depermuted_cost_matrix'

ORIGINAL_COST_KEY = 'orig_cost_estimates'
BEST_PREDICTORS_KEY = 'best_predictor_names'
BEST_COSTS_KEY = 'best_cost_matrix'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_COSTS_KEY = 'step1_cost_matrix'
BACKWARDS_FLAG_KEY = 'is_backwards_test'

PREDICTOR_DIM_KEY = 'predictor'
STEP_DIM_KEY = 'step'
BOOTSTRAP_REPLICATE_DIM_KEY = 'bootstrap_replicate'
PREDICTOR_CHAR_DIM_KEY = 'predictor_name_char'


def _check_args(three_predictor_matrices, target_array, num_bootstrap_reps,
                num_steps):
    """Checks arguments for run_forward_test or run_backwards_test.

    :param three_predictor_matrices: See doc for `run_forward_test` or
        `run_backwards_test`.
    :param target_array: Same.
    :param num_bootstrap_reps: Same.
    :param num_steps: Same.
    :return: num_bootstrap_reps: Same.
    :return: num_steps: Same.
    """

    if num_steps is None:
        num_steps = int(1e10)

    error_checking.assert_is_integer(num_steps)
    error_checking.assert_is_greater(num_steps, 0)

    error_checking.assert_is_list(three_predictor_matrices)
    assert len(three_predictor_matrices) == 3

    num_matrices = len(three_predictor_matrices)
    num_examples = -1

    for i in range(num_matrices):
        if three_predictor_matrices[i] is None:
            continue

        error_checking.assert_is_numpy_array_without_nan(
            three_predictor_matrices[i]
        )
        if num_examples == -1:
            num_examples = three_predictor_matrices[i].shape[0]

        expected_dim = numpy.array(
            (num_examples,) + three_predictor_matrices[i].shape[1:], dtype=int
        )
        error_checking.assert_is_numpy_array(
            three_predictor_matrices[i], exact_dimensions=expected_dim
        )

    error_checking.assert_is_integer_numpy_array(target_array)
    error_checking.assert_is_geq_numpy_array(target_array, 0)
    error_checking.assert_is_leq_numpy_array(target_array, 1)
    expected_dim = numpy.array(
        (num_examples,) + target_array.shape[1:], dtype=int
    )
    error_checking.assert_is_numpy_array(
        target_array, exact_dimensions=expected_dim
    )

    error_checking.assert_is_integer(num_bootstrap_reps)
    num_bootstrap_reps = numpy.maximum(num_bootstrap_reps, 1)

    return num_bootstrap_reps, num_steps


def _permute_values(
        predictor_matrix, predictor_type_enum, variable_index,
        model_metadata_dict=None, model_lag_time_index=None,
        permuted_value_matrix=None):
    """Permutes values of one predictor variable across all examples.

    :param predictor_matrix: See doc for `run_forward_test_one_step`.
    :param predictor_type_enum: Predictor type (integer).
    :param variable_index: Will permute the [k]th predictor variable, where k =
        `variable_index`.
    :param model_metadata_dict: [used only if predictor type is SHIPS]
        Dictionary returned by `neural_net.read_metafile`.
    :param model_lag_time_index: Will permute predictor only for the [j]th lag
        time, where j = `model_lag_time_index`.  To permute the predictor for
        all times, leave this argument as None.
    :param permuted_value_matrix: numpy array of permuted values to replace
        original ones.  This matrix must have the same shape as the submatrix
        being replaced.  If None, values will be permuted on the fly.
    :return: predictor_matrix: Same as input but with desired values permuted.
    :return: permuted_value_matrix: See input doc.  If input was None, this will
        be a new array created on the fly.  If input was specified, this will be
        the same as input.
    """

    if permuted_value_matrix is None:
        random_indices = numpy.random.permutation(predictor_matrix.shape[0])
    else:
        random_indices = []

    if predictor_type_enum != 2:
        if model_lag_time_index is None:
            if permuted_value_matrix is None:
                permuted_value_matrix = predictor_matrix[..., variable_index][
                    random_indices, ...
                ]

            predictor_matrix[..., variable_index] = permuted_value_matrix
        else:
            if permuted_value_matrix is None:
                permuted_value_matrix = (
                    predictor_matrix[..., model_lag_time_index, variable_index][
                        random_indices, ...
                    ]
                )

            predictor_matrix[..., model_lag_time_index, variable_index] = (
                permuted_value_matrix
            )

        return predictor_matrix, permuted_value_matrix

    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    t = training_option_dict

    if t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY] is None:
        num_lagged_predictors = 0
    else:
        num_lagged_predictors = len(t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY])

    if t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY] is None:
        num_forecast_predictors = 0
    else:
        num_forecast_predictors = len(
            t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
        )

    max_forecast_hour = t[neural_net.SHIPS_MAX_FORECAST_HOUR_KEY]
    num_builtin_lag_times = len(t[neural_net.SHIPS_BUILTIN_LAG_TIMES_KEY])

    lagged_predictor_matrix_4d, forecast_predictor_matrix_4d = (
        neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=predictor_matrix,
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=num_builtin_lag_times,
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=int(numpy.round(max_forecast_hour / 6)) + 1
        )
    )

    if model_lag_time_index is None:
        if variable_index < num_lagged_predictors:
            if permuted_value_matrix is None:
                permuted_value_matrix = lagged_predictor_matrix_4d[
                    ..., variable_index
                ][random_indices, ...]

            lagged_predictor_matrix_4d[..., variable_index] = (
                permuted_value_matrix
            )
        else:
            if permuted_value_matrix is None:
                permuted_value_matrix = forecast_predictor_matrix_4d[
                    ..., variable_index - num_lagged_predictors
                ][random_indices, ...]

            forecast_predictor_matrix_4d[
                ..., variable_index - num_lagged_predictors
            ] = permuted_value_matrix
    else:
        if variable_index < num_lagged_predictors:
            if permuted_value_matrix is None:
                permuted_value_matrix = lagged_predictor_matrix_4d[
                    :, model_lag_time_index, :, variable_index
                ][random_indices, ...]

            lagged_predictor_matrix_4d[
                :, model_lag_time_index, :, variable_index
            ] = permuted_value_matrix
        else:
            if permuted_value_matrix is None:
                permuted_value_matrix = forecast_predictor_matrix_4d[
                    :, model_lag_time_index, :,
                    variable_index - num_lagged_predictors
                ][random_indices, ...]

            forecast_predictor_matrix_4d[
                :, model_lag_time_index, :,
                variable_index - num_lagged_predictors
            ] = permuted_value_matrix

    predictor_matrix = neural_net.ships_predictors_4d_to_3d(
        lagged_predictor_matrix_4d=lagged_predictor_matrix_4d,
        forecast_predictor_matrix_4d=forecast_predictor_matrix_4d
    )

    return predictor_matrix, permuted_value_matrix


def _depermute_values(
        predictor_matrix, clean_predictor_matrix, predictor_type_enum,
        variable_index, model_metadata_dict=None, model_lag_time_index=None):
    """Depermutes (cleans up) values of one predictor var across all examples.

    E = number of examples

    :param predictor_matrix: See doc for `_permute_values`.
    :param clean_predictor_matrix: Clean version of `predictor_matrix`, with no
        values permuted.
    :param predictor_type_enum: Same.
    :param variable_index: Same.
    :param model_metadata_dict: Same.
    :param model_lag_time_index: Same.
    :return: predictor_matrix: Same.
    """

    if predictor_type_enum != 2:
        if model_lag_time_index is None:
            predictor_matrix[..., variable_index] = (
                clean_predictor_matrix[..., variable_index]
            )
        else:
            predictor_matrix[..., model_lag_time_index, variable_index] = (
                clean_predictor_matrix[
                    ..., model_lag_time_index, variable_index
                ]
            )

        return predictor_matrix

    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    t = training_option_dict

    if t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY] is None:
        num_lagged_predictors = 0
    else:
        num_lagged_predictors = len(t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY])

    if t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY] is None:
        num_forecast_predictors = 0
    else:
        num_forecast_predictors = len(
            t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
        )

    max_forecast_hour = t[neural_net.SHIPS_MAX_FORECAST_HOUR_KEY]
    num_builtin_lag_times = len(t[neural_net.SHIPS_BUILTIN_LAG_TIMES_KEY])

    lagged_predictor_matrix_4d, forecast_predictor_matrix_4d = (
        neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=predictor_matrix,
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=num_builtin_lag_times,
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=int(numpy.round(max_forecast_hour / 6)) + 1
        )
    )

    clean_lagged_predictor_matrix_4d, clean_forecast_predictor_matrix_4d = (
        neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=clean_predictor_matrix,
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=num_builtin_lag_times,
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=int(numpy.round(max_forecast_hour / 6)) + 1
        )
    )

    if model_lag_time_index is None:
        if variable_index < num_lagged_predictors:
            lagged_predictor_matrix_4d[..., variable_index] = (
                clean_lagged_predictor_matrix_4d[..., variable_index]
            )
        else:
            forecast_predictor_matrix_4d[
                ..., variable_index - num_lagged_predictors
            ] = (
                clean_forecast_predictor_matrix_4d[
                    ..., variable_index - num_lagged_predictors
                ]
            )
    else:
        i = model_lag_time_index
        j = variable_index

        if variable_index < num_lagged_predictors:
            lagged_predictor_matrix_4d[:, i, :, j] = (
                clean_lagged_predictor_matrix_4d[:, i, :, j]
            )
        else:
            forecast_predictor_matrix_4d[:, i, :, j - num_lagged_predictors] = (
                clean_forecast_predictor_matrix_4d[
                    :, i, :, j - num_lagged_predictors
                ]
            )

    predictor_matrix = neural_net.ships_predictors_4d_to_3d(
        lagged_predictor_matrix_4d=lagged_predictor_matrix_4d,
        forecast_predictor_matrix_4d=forecast_predictor_matrix_4d
    )

    return predictor_matrix


def _bootstrap_cost(target_array, forecast_prob_array, cost_function,
                    num_replicates):
    """Uses bootstrapping to estimate cost.

    E = number of examples
    K = number of classes

    :param target_array: If K > 2, this is an E-by-K numpy array of integers
        (0 or 1), indicating true classes.  If K = 2, this is a length-E numpy
        array of integers (0 or 1).
    :param forecast_prob_array: If K > 2, this is an E-by-K numpy array of
        class probabilities.  If K = 2, this is a length-E numpy array of
        positive-class probabilities.
    :param cost_function: Cost function.  Must be negatively oriented (i.e.,
        lower is better), with the following inputs and outputs.
    Input: target_array: See above.
    Input: forecast_prob_array: See above.
    Output: cost: Scalar value.

    :param num_replicates: Number of bootstrap replicates (i.e., number of times
        to estimate cost).
    :return: cost_estimates: length-B numpy array of cost estimates, where B =
        number of bootstrap replicates.
    """

    cost_estimates = numpy.full(num_replicates, numpy.nan)

    if num_replicates == 1:
        cost_estimates[0] = cost_function(target_array, forecast_prob_array)
    else:
        num_examples = target_array.shape[0]
        example_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )

        for k in range(num_replicates):
            these_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=True
            )
            cost_estimates[k] = cost_function(
                target_array[these_indices, ...],
                forecast_prob_array[these_indices, ...]
            )

    print('Average cost estimate over {0:d} replicates = {1:f}'.format(
        num_replicates, numpy.mean(cost_estimates)
    ))

    return cost_estimates


def _predictor_indices_to_metadata(model_metadata_dict, one_step_result_dict):
    """Converts predictor indices to metadata (name and lag time).

    N = number of permutations or depermutations

    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    :param one_step_result_dict: Dictionary created by
        `_run_forward_test_one_step` or `_run_backwards_test_one_step`.
    :return: predictor_names: length-N list of predictor names, in the order
        that they were (de)permuted.
    """

    # TODO(thunderhoser): Incorporate lag times.

    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    t = training_option_dict

    num_matrices = 3
    predictor_names_by_matrix = [[]] * num_matrices

    for i in range(num_matrices):
        if i == 0:
            if t[neural_net.SATELLITE_LAG_TIMES_KEY] is None:
                continue

            predictor_names_by_matrix[i] = [
                satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
            ]
        elif i == 1:
            if t[neural_net.SATELLITE_PREDICTORS_KEY] is None:
                continue

            predictor_names_by_matrix[i] = (
                t[neural_net.SATELLITE_PREDICTORS_KEY]
            )
        else:
            if t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY] is not None:
                predictor_names_by_matrix[i] += (
                    t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
                )

            if t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY] is not None:
                predictor_names_by_matrix[i] += (
                    t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
                )

    if PERMUTED_MATRICES_KEY in one_step_result_dict:
        matrix_indices = one_step_result_dict[PERMUTED_MATRICES_KEY]
        variable_indices = one_step_result_dict[PERMUTED_VARIABLES_KEY]
    else:
        matrix_indices = one_step_result_dict[DEPERMUTED_MATRICES_KEY]
        variable_indices = one_step_result_dict[DEPERMUTED_VARIABLES_KEY]

    predictor_names = []
    for i, j in zip(matrix_indices, variable_indices):
        predictor_names.append(predictor_names_by_matrix[i][j])

    return predictor_names


def _make_prediction_function(model_object):
    """Creates prediction function for neural net.

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :return: prediction_function: Function defined below.
    """

    def prediction_function(three_predictor_matrices):
        """Prediction function itself.

        E = number of examples
        K = number of classes

        :param three_predictor_matrices: See doc for `run_forward_test`.
        :return: forecast_prob_array: If K > 2, this is an E-by-K numpy array of
            class probabilities.  If K = 2, this is a length-E numpy array of
            positive-class probabilities.
        """

        print('\n')
        forecast_prob_array = neural_net.apply_model(
            model_object=model_object,
            predictor_matrices=[
                m for m in three_predictor_matrices if m is not None
            ],
            num_examples_per_batch=32, verbose=True
        )
        print('\n')

        return forecast_prob_array

    return prediction_function


def _run_forward_test_one_step(
        three_predictor_matrices, target_array, model_metadata_dict,
        prediction_function, cost_function, permuted_flag_arrays,
        num_bootstrap_reps):
    """Runs one step of the forward permutation test.

    I = number of input (predictor) matrices for model
    E = number of examples
    C = number of channels
    B = number of replicates for bootstrapping
    K = number of classes

    :param three_predictor_matrices: See doc for `run_forward_test`.
    :param target_array: Same.
    :param model_metadata_dict: Same.
    :param prediction_function: Function with the following inputs and outputs.
    Input: three_predictor_matrices: See above.
    Output: forecast_prob_array: If K > 2, this is an E-by-K numpy array of
        class probabilities.  If K = 2, this is a length-E numpy array of
        positive-class probabilities.

    :param cost_function: See doc for `run_forward_test`.
    :param permuted_flag_arrays: length-I list of Boolean numpy arrays.  The
        [i]th array has length P_i, where P_i is the number of predictor
        variables in the [i]th matrix.  If the [j]th variable in the [i]th
        matrix is currently permuted, then permuted_flag_arrays[i][j] = True.
        Otherwise, permuted_flag_arrays[i][j] = False.

    :param num_bootstrap_reps: Number of replicates for bootstrapping.
    :return: result_dict: Dictionary with the following keys, where P = number
        of permutations done in this step.
    result_dict['three_predictor_matrices']: Same as input but with more values
        permuted.
    result_dict['permuted_flag_arrays']: Same as input but with more `True`
        flags.
    result_dict['permuted_matrix_indices']: length-P numpy array with matrix
        indices for predictors permuted.
    result_dict['permuted_variable_indices']: length-P numpy array with variable
        indices for predictors permuted.
    result_dict['permuted_cost_matrix']: P-by-B numpy array of costs after
        permutation.
    """

    # TODO(thunderhoser): Allow to permute by lag time, not just by variable.

    if all([numpy.all(a) for a in permuted_flag_arrays]):
        return None

    permuted_matrix_indices = []
    permuted_variable_indices = []
    permuted_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    best_cost = -numpy.inf
    best_matrix_index = -1
    best_variable_index = -1
    best_permuted_value_matrix = None

    num_matrices = len(three_predictor_matrices)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    for i in range(num_matrices):
        t = training_option_dict

        if i == 0:
            num_variables = three_predictor_matrices[i].shape[-1]
        elif i == 1:
            if t[neural_net.SATELLITE_PREDICTORS_KEY] is None:
                continue

            num_variables = len(t[neural_net.SATELLITE_PREDICTORS_KEY])
        else:
            num_variables = 0
            if t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY] is not None:
                num_variables += len(t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY])
            if t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY] is not None:
                num_variables += len(
                    t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
                )

            if num_variables == 0:
                continue

        for j in range(num_variables):
            if permuted_flag_arrays[i][j]:
                continue

            permuted_matrix_indices.append(i)
            permuted_variable_indices.append(j)

            print((
                'Permuting {0:d}th of {1:d} variables in {2:d}th of {3:d} '
                'matrices...'
            ).format(
                j + 1, num_variables, i + 1, num_matrices
            ))

            this_predictor_matrix, this_permuted_value_matrix = _permute_values(
                predictor_matrix=three_predictor_matrices[i] + 0.,
                predictor_type_enum=i,
                variable_index=j, model_metadata_dict=model_metadata_dict
            )
            these_predictor_matrices = [
                this_predictor_matrix if k == i else three_predictor_matrices[k]
                for k in range(num_matrices)
            ]
            this_forecast_prob_array = prediction_function(
                these_predictor_matrices
            )
            this_cost_matrix = _bootstrap_cost(
                target_array=target_array,
                forecast_prob_array=this_forecast_prob_array,
                cost_function=cost_function, num_replicates=num_bootstrap_reps
            )

            this_cost_matrix = numpy.reshape(
                this_cost_matrix, (1, this_cost_matrix.size)
            )
            permuted_cost_matrix = numpy.concatenate(
                (permuted_cost_matrix, this_cost_matrix), axis=0
            )
            this_average_cost = numpy.mean(permuted_cost_matrix[-1, :])
            if this_average_cost < best_cost:
                continue

            best_cost = this_average_cost + 0.
            best_matrix_index = i
            best_variable_index = j
            best_permuted_value_matrix = this_permuted_value_matrix + 0.

    this_predictor_matrix = _permute_values(
        predictor_matrix=three_predictor_matrices[best_matrix_index],
        predictor_type_enum=best_matrix_index,
        variable_index=best_variable_index,
        model_metadata_dict=model_metadata_dict,
        permuted_value_matrix=best_permuted_value_matrix
    )[0]
    three_predictor_matrices = [
        this_predictor_matrix if k == best_matrix_index
        else three_predictor_matrices[k]
        for k in range(num_matrices)
    ]

    print((
        'Best predictor = {0:d}th variable in {1:d}th matrix (cost = {2:.4f})'
    ).format(
        best_variable_index + 1, best_matrix_index + 1, best_cost
    ))

    permuted_flag_arrays[best_matrix_index][best_variable_index] = True
    permuted_matrix_indices = numpy.array(permuted_matrix_indices, dtype=int)
    permuted_variable_indices = numpy.array(
        permuted_variable_indices, dtype=int
    )

    return {
        THREE_PREDICTOR_MATRICES_KEY: three_predictor_matrices,
        PERMUTED_FLAGS_KEY: permuted_flag_arrays,
        PERMUTED_MATRICES_KEY: permuted_matrix_indices,
        PERMUTED_VARIABLES_KEY: permuted_variable_indices,
        PERMUTED_COSTS_KEY: permuted_cost_matrix
    }


def _run_backwards_test_one_step(
        three_predictor_matrices, clean_predictor_matrices, target_array,
        model_metadata_dict, prediction_function, cost_function,
        permuted_flag_arrays, num_bootstrap_reps):
    """Runs one step of the backwards permutation test.

    :param three_predictor_matrices: See doc for `_run_forward_test_one_step`.
    :param clean_predictor_matrices: Clean version of
        `three_predictor_matrices`, with no permutation.
    :param target_array: See doc for `_run_forward_test_one_step`.
    :param model_metadata_dict: Same.
    :param prediction_function: Same.
    :param cost_function: Same.
    :param permuted_flag_arrays: Same.
    :param num_bootstrap_reps: Same.
    :return: result_dict: Dictionary with the following keys, where P = number
        of depermutations done in this step.
    result_dict['three_predictor_matrices']: Same as input but with fewer values
        permuted.
    result_dict['permuted_flag_arrays']: Same as input but with more `False`
        flags.
    result_dict['depermuted_matrix_indices']: length-P numpy array with matrix
        indices for predictors depermuted.
    result_dict['depermuted_variable_indices']: length-P numpy array with
        variable indices for predictors depermuted.
    result_dict['depermuted_cost_matrix']: P-by-B numpy array of costs after
        depermutation.
    """

    if not any([numpy.any(a) for a in permuted_flag_arrays]):
        return None

    depermuted_matrix_indices = []
    depermuted_variable_indices = []
    depermuted_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    best_cost = numpy.inf
    best_matrix_index = -1
    best_variable_index = -1

    num_matrices = len(three_predictor_matrices)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    for i in range(num_matrices):
        t = training_option_dict

        if i == 0:
            num_variables = three_predictor_matrices[i].shape[-1]
        elif i == 1:
            if t[neural_net.SATELLITE_PREDICTORS_KEY] is None:
                continue

            num_variables = len(t[neural_net.SATELLITE_PREDICTORS_KEY])
        else:
            num_variables = 0
            if t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY] is not None:
                num_variables += len(t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY])
            if t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY] is not None:
                num_variables += len(
                    t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
                )

            if num_variables == 0:
                continue

        for j in range(num_variables):
            if not permuted_flag_arrays[i][j]:
                continue

            depermuted_matrix_indices.append(i)
            depermuted_variable_indices.append(j)

            print((
                'Depermuting (cleaning up) {0:d}th of {1:d} variables in '
                '{2:d}th of {3:d} matrices...'
            ).format(
                j + 1, num_variables, i + 1, num_matrices
            ))

            this_predictor_matrix = _depermute_values(
                predictor_matrix=three_predictor_matrices[i] + 0.,
                clean_predictor_matrix=clean_predictor_matrices[i],
                predictor_type_enum=i, variable_index=j,
                model_metadata_dict=model_metadata_dict
            )
            these_predictor_matrices = [
                this_predictor_matrix if k == i else three_predictor_matrices[k]
                for k in range(num_matrices)
            ]
            this_forecast_prob_array = prediction_function(
                these_predictor_matrices
            )
            this_cost_matrix = _bootstrap_cost(
                target_array=target_array,
                forecast_prob_array=this_forecast_prob_array,
                cost_function=cost_function, num_replicates=num_bootstrap_reps
            )

            this_cost_matrix = numpy.reshape(
                this_cost_matrix, (1, this_cost_matrix.size)
            )
            depermuted_cost_matrix = numpy.concatenate(
                (depermuted_cost_matrix, this_cost_matrix), axis=0
            )
            this_average_cost = numpy.mean(depermuted_cost_matrix[-1, :])
            if this_average_cost > best_cost:
                continue

            best_cost = this_average_cost + 0.
            best_matrix_index = i
            best_variable_index = j

    this_predictor_matrix = _depermute_values(
        predictor_matrix=three_predictor_matrices[best_matrix_index],
        clean_predictor_matrix=clean_predictor_matrices[best_matrix_index],
        predictor_type_enum=best_matrix_index,
        variable_index=best_variable_index,
        model_metadata_dict=model_metadata_dict
    )
    three_predictor_matrices = [
        this_predictor_matrix if k == best_matrix_index
        else three_predictor_matrices[k]
        for k in range(num_matrices)
    ]

    print((
        'Best predictor = {0:d}th variable in {1:d}th matrix (cost = {2:.4f})'
    ).format(
        best_variable_index + 1, best_matrix_index + 1, best_cost
    ))

    permuted_flag_arrays[best_matrix_index][best_variable_index] = True
    depermuted_matrix_indices = numpy.array(
        depermuted_matrix_indices, dtype=int
    )
    depermuted_variable_indices = numpy.array(
        depermuted_variable_indices, dtype=int
    )

    return {
        THREE_PREDICTOR_MATRICES_KEY: three_predictor_matrices,
        PERMUTED_FLAGS_KEY: permuted_flag_arrays,
        DEPERMUTED_MATRICES_KEY: depermuted_matrix_indices,
        DEPERMUTED_VARIABLES_KEY: depermuted_variable_indices,
        DEPERMUTED_COSTS_KEY: depermuted_cost_matrix
    }


def make_auc_cost_function():
    """Creates cost function involving AUC (area under ROC curve).

    :return: cost_function: Function (see below).
    """

    def cost_function(target_values, forecast_probs):
        """Actual cost function.

        E = number of examples

        :param target_values: length-E numpy array of integers (0 or 1).
        :param forecast_probs: length-E numpy array of positive-class
            probabilities.
        :return: cost: Negative AUC.
        """

        error_checking.assert_is_numpy_array(target_values, num_dimensions=1)
        error_checking.assert_is_numpy_array(forecast_probs, num_dimensions=1)

        return -1 * sklearn_auc(y_true=target_values, y_score=forecast_probs)

    return cost_function


def run_forward_test(
        three_predictor_matrices, target_array, model_object,
        model_metadata_dict, cost_function,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS, num_steps=None):
    """Runs forward version of permutation test (both single- and multi-pass).

    E = number of examples
    K = number of classes
    B = number of replicates for bootstrapping
    N = number of predictors (either physical variables or
        lag-time/physical-variable pairs) available to permute

    :param three_predictor_matrices: See output doc for
        `neural_net.create_inputs`.
    :param target_array: If K > 2, this is an E-by-K numpy array of integers
        (0 or 1), indicating true classes.  If K = 2, this is a length-E numpy
        array of integers (0 or 1).
    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`, containing metadata for trained model.

    :param cost_function: Cost function.  Must be negatively oriented (i.e.,
        lower is better), with the following inputs and outputs.
    Input: target_array: See above.
    Input: forecast_prob_array: If K > 2, this is an E-by-K numpy array of
        class probabilities.  If K = 2, this is a length-E numpy array of
        positive-class probabilities.
    Output: cost: Scalar value.

    :param num_bootstrap_reps: Number of replicates for bootstrapping.
    :param num_steps: Number of steps to carry out.  Will keep going until N
        predictors are permanently permuted, where N = `num_steps`.  If None,
        will keep going until all predictors are permuted.
    :return: result_dict: Dictionary with the following keys.
    result_dict['orig_cost_estimates']: length-B numpy array with estimates of
        original cost (before permutation).
    result_dict['best_predictor_names']: length-N list with best predictor at
        each step.
    result_dict['best_lag_times_sec']: length-N numpy array of corresponding
        lag times.  This may be None.
    result_dict['best_cost_matrix']: N-by-B numpy array of costs after
        permutation at each step.
    result_dict['step1_predictor_names']: length-N list with predictors in order
        that they were permuted in step 1.
    result_dict['step1_lag_times_sec']: length-N numpy array of corresponding
        lag times.  This may be None.
    result_dict['step1_cost_matrix']: N-by-B numpy array of costs after
        permutation in step 1.
    result_dict['is_backwards_test']: Boolean flag (always False for this
        method).
    """

    # TODO(thunderhoser): Allow split by lag time.

    num_bootstrap_reps, num_steps = _check_args(
        three_predictor_matrices=three_predictor_matrices,
        target_array=target_array,
        num_bootstrap_reps=num_bootstrap_reps, num_steps=num_steps
    )
    num_matrices = len(three_predictor_matrices)
    prediction_function = _make_prediction_function(model_object)

    # Find original cost (before permutation).
    print('Finding original cost (before permutation)...')
    forecast_prob_array = prediction_function(three_predictor_matrices)

    orig_cost_estimates = _bootstrap_cost(
        target_array=target_array, forecast_prob_array=forecast_prob_array,
        cost_function=cost_function, num_replicates=num_bootstrap_reps
    )

    # Housekeeping.
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    permuted_flag_arrays = [numpy.array([], dtype=bool)] * num_matrices

    for i in range(num_matrices):
        t = training_option_dict

        if i == 0:
            num_variables = three_predictor_matrices[i].shape[-1]
        elif i == 1:
            if t[neural_net.SATELLITE_PREDICTORS_KEY] is None:
                continue

            num_variables = len(t[neural_net.SATELLITE_PREDICTORS_KEY])
        else:
            num_variables = 0
            if t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY] is not None:
                num_variables += len(t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY])
            if t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY] is not None:
                num_variables += len(
                    t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
                )

            if num_variables == 0:
                continue

        permuted_flag_arrays[i] = numpy.full(num_variables, 0, dtype=bool)

    best_predictor_names = []
    best_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)
    step1_predictor_names = None
    step1_cost_matrix = None

    step_num = 0

    # Do actual stuff.
    while True:
        print(SEPARATOR_STRING)
        step_num += 1

        if step_num > num_steps:
            break

        this_result_dict = _run_forward_test_one_step(
            three_predictor_matrices=three_predictor_matrices,
            target_array=target_array,
            model_metadata_dict=model_metadata_dict,
            prediction_function=prediction_function,
            cost_function=cost_function,
            permuted_flag_arrays=permuted_flag_arrays,
            num_bootstrap_reps=num_bootstrap_reps
        )

        if this_result_dict is None:
            break

        three_predictor_matrices = (
            this_result_dict[THREE_PREDICTOR_MATRICES_KEY]
        )
        permuted_flag_arrays = this_result_dict[PERMUTED_FLAGS_KEY]

        these_predictor_names = _predictor_indices_to_metadata(
            model_metadata_dict=model_metadata_dict,
            one_step_result_dict=this_result_dict
        )

        this_best_index = numpy.argmax(
            numpy.mean(this_result_dict[PERMUTED_COSTS_KEY], axis=1)
        )
        best_predictor_names.append(these_predictor_names[this_best_index])
        best_cost_matrix = numpy.concatenate((
            best_cost_matrix,
            this_result_dict[PERMUTED_COSTS_KEY][[this_best_index], :]
        ), axis=0)

        print((
            'Best predictor at {0:d}th step = {1:s} ... cost = {2:.4f}'
        ).format(
            step_num, best_predictor_names[-1],
            numpy.mean(best_cost_matrix[-1, :])
        ))

        if step_num != 1:
            continue

        step1_predictor_names = copy.deepcopy(these_predictor_names)
        step1_cost_matrix = this_result_dict[PERMUTED_COSTS_KEY] + 0.

    return {
        ORIGINAL_COST_KEY: orig_cost_estimates,
        BEST_PREDICTORS_KEY: best_predictor_names,
        BEST_COSTS_KEY: best_cost_matrix,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_COSTS_KEY: step1_cost_matrix,
        BACKWARDS_FLAG_KEY: False
    }


def run_backwards_test(
        three_predictor_matrices, target_array, model_object,
        model_metadata_dict, cost_function,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS, num_steps=None):
    """Runs backwards version of permutation test (both single- and multi-pass).

    E = number of examples
    N = number of predictors (either physical variables or
        lag-time/physical-variable pairs) available to permute

    :param three_predictor_matrices: See doc for `run_forward_test`.
    :param target_array: Same.
    :param model_object: Same.
    :param model_metadata_dict: Same.
    :param cost_function: Same.
    :param num_bootstrap_reps: Same.
    :param num_steps: Number of steps to carry out.  Will keep going until N
        predictors are permanently depermuted, where N = `num_steps`.  If None,
        will keep going until all predictors are depermuted.
    :return: result_dict: Dictionary with the following keys.
    result_dict['orig_cost_estimates']: length-B numpy array with estimates of
        original cost (before *de*permutation).
    result_dict['best_predictor_names']: length-N list with best predictor at
        each step.
    result_dict['best_lag_times_sec']: length-N numpy array of corresponding
        lag times.  This may be None.
    result_dict['best_cost_matrix']: N-by-B numpy array of costs after
        *de*permutation at each step.
    result_dict['step1_predictor_names']: length-N list with predictors in order
        that they were *de*permuted in step 1.
    result_dict['step1_lag_times_sec']: length-N numpy array of corresponding
        lag times.  This may be None.
    result_dict['step1_cost_matrix']: N-by-B numpy array of costs after
        *de*permutation in step 1.
    result_dict['is_backwards_test']: Boolean flag (always True for this
        method).
    """

    # TODO(thunderhoser): Allow split by lag time.

    num_bootstrap_reps, num_steps = _check_args(
        three_predictor_matrices=three_predictor_matrices,
        target_array=target_array,
        num_bootstrap_reps=num_bootstrap_reps, num_steps=num_steps
    )
    num_matrices = len(three_predictor_matrices)
    prediction_function = _make_prediction_function(model_object)

    # Housekeeping.
    clean_predictor_matrices = copy.deepcopy(three_predictor_matrices)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    permuted_flag_arrays = [numpy.array([], dtype=bool)] * num_matrices

    for i in range(num_matrices):
        t = training_option_dict

        if i == 0:
            num_variables = three_predictor_matrices[i].shape[-1]
        elif i == 1:
            if t[neural_net.SATELLITE_PREDICTORS_KEY] is None:
                continue

            num_variables = len(t[neural_net.SATELLITE_PREDICTORS_KEY])
        else:
            num_variables = 0
            if t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY] is not None:
                num_variables += len(t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY])
            if t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY] is not None:
                num_variables += len(
                    t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
                )

            if num_variables == 0:
                continue

        permuted_flag_arrays[i] = numpy.full(num_variables, 1, dtype=bool)

        for j in range(num_variables):
            three_predictor_matrices[i] = _permute_values(
                predictor_matrix=three_predictor_matrices[i],
                predictor_type_enum=i, variable_index=j,
                model_metadata_dict=model_metadata_dict,
                model_lag_time_index=None
            )[0]

    # Find original cost (before *de*permutation).
    print('Finding original cost (before *de*permutation)...')
    forecast_prob_array = prediction_function(three_predictor_matrices)

    orig_cost_estimates = _bootstrap_cost(
        target_array=target_array, forecast_prob_array=forecast_prob_array,
        cost_function=cost_function, num_replicates=num_bootstrap_reps
    )

    # Do actual stuff.
    best_predictor_names = []
    best_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)
    step1_predictor_names = None
    step1_cost_matrix = None

    step_num = 0

    while True:
        print(SEPARATOR_STRING)
        step_num += 1

        if step_num > num_steps:
            break

        this_result_dict = _run_backwards_test_one_step(
            three_predictor_matrices=three_predictor_matrices,
            clean_predictor_matrices=clean_predictor_matrices,
            target_array=target_array,
            model_metadata_dict=model_metadata_dict,
            prediction_function=prediction_function,
            cost_function=cost_function,
            permuted_flag_arrays=permuted_flag_arrays,
            num_bootstrap_reps=num_bootstrap_reps
        )

        if this_result_dict is None:
            break

        three_predictor_matrices = (
            this_result_dict[THREE_PREDICTOR_MATRICES_KEY]
        )
        permuted_flag_arrays = this_result_dict[PERMUTED_FLAGS_KEY]

        these_predictor_names = _predictor_indices_to_metadata(
            model_metadata_dict=model_metadata_dict,
            one_step_result_dict=this_result_dict
        )

        this_best_index = numpy.argmin(
            numpy.mean(this_result_dict[DEPERMUTED_COSTS_KEY], axis=1)
        )
        best_predictor_names.append(these_predictor_names[this_best_index])
        best_cost_matrix = numpy.concatenate((
            best_cost_matrix,
            this_result_dict[DEPERMUTED_COSTS_KEY][[this_best_index], :]
        ), axis=0)

        print((
            'Best predictor at {0:d}th step = {1:s} ... cost = {2:.4f}'
        ).format(
            step_num, best_predictor_names[-1],
            numpy.mean(best_cost_matrix[-1, :])
        ))

        if step_num != 1:
            continue

        step1_predictor_names = copy.deepcopy(these_predictor_names)
        step1_cost_matrix = this_result_dict[DEPERMUTED_COSTS_KEY] + 0.

    return {
        ORIGINAL_COST_KEY: orig_cost_estimates,
        BEST_PREDICTORS_KEY: best_predictor_names,
        BEST_COSTS_KEY: best_cost_matrix,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_COSTS_KEY: step1_cost_matrix,
        BACKWARDS_FLAG_KEY: True
    }


def write_file(result_dict, netcdf_file_name):
    """Writes results of permutation test to NetCDF file.

    :param result_dict: Dictionary created by `run_forward_test` or
        `run_backwards_test`.
    :param netcdf_file_name: Path to output file.
    """

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(
        BACKWARDS_FLAG_KEY, int(result_dict[BACKWARDS_FLAG_KEY])
    )

    num_predictors = result_dict[STEP1_COSTS_KEY].shape[0]
    num_steps = result_dict[BEST_COSTS_KEY].shape[0]
    num_bootstrap_reps = result_dict[BEST_COSTS_KEY].shape[1]

    dataset_object.createDimension(PREDICTOR_DIM_KEY, num_predictors)
    dataset_object.createDimension(STEP_DIM_KEY, num_steps)
    dataset_object.createDimension(
        BOOTSTRAP_REPLICATE_DIM_KEY, num_bootstrap_reps
    )

    best_predictor_names = result_dict[BEST_PREDICTORS_KEY]
    step1_predictor_names = result_dict[STEP1_PREDICTORS_KEY]
    num_predictor_chars = numpy.max(numpy.array([
        len(n) for n in best_predictor_names + step1_predictor_names
    ]))

    dataset_object.createDimension(PREDICTOR_CHAR_DIM_KEY, num_predictor_chars)

    this_string_format = 'S{0:d}'.format(num_predictor_chars)
    best_predictor_names_char_array = netCDF4.stringtochar(numpy.array(
        best_predictor_names, dtype=this_string_format
    ))

    dataset_object.createVariable(
        BEST_PREDICTORS_KEY, datatype='S1',
        dimensions=(STEP_DIM_KEY, PREDICTOR_CHAR_DIM_KEY)
    )
    dataset_object.variables[BEST_PREDICTORS_KEY][:] = numpy.array(
        best_predictor_names_char_array
    )

    # if result_dict[BEST_LAG_TIMES_KEY] is not None:
    #     best_lag_times_sec = result_dict[BEST_LAG_TIMES_KEY] + 0
    #     best_lag_times_sec[numpy.isnan(best_lag_times_sec)] = -1
    #     best_lag_times_sec = numpy.round(best_lag_times_sec).astype(int)
    #
    #     dataset_object.createVariable(
    #         BEST_LAG_TIMES_KEY, datatype=numpy.int32,
    #         dimensions=STEP_DIM_KEY
    #     )
    #     dataset_object.variables[BEST_LAG_TIMES_KEY][:] = best_lag_times_sec

    dataset_object.createVariable(
        BEST_COSTS_KEY, datatype=numpy.float32,
        dimensions=(STEP_DIM_KEY, BOOTSTRAP_REPLICATE_DIM_KEY)
    )
    dataset_object.variables[BEST_COSTS_KEY][:] = result_dict[BEST_COSTS_KEY]

    this_string_format = 'S{0:d}'.format(num_predictor_chars)
    step1_predictor_names_char_array = netCDF4.stringtochar(numpy.array(
        step1_predictor_names, dtype=this_string_format
    ))

    dataset_object.createVariable(
        STEP1_PREDICTORS_KEY, datatype='S1',
        dimensions=(PREDICTOR_DIM_KEY, PREDICTOR_CHAR_DIM_KEY)
    )
    dataset_object.variables[STEP1_PREDICTORS_KEY][:] = numpy.array(
        step1_predictor_names_char_array
    )

    # if result_dict[STEP1_LAG_TIMES_KEY] is not None:
    #     step1_lag_times_sec = result_dict[STEP1_LAG_TIMES_KEY] + 0
    #     step1_lag_times_sec[numpy.isnan(step1_lag_times_sec)] = -1
    #     step1_lag_times_sec = numpy.round(step1_lag_times_sec).astype(int)
    #
    #     dataset_object.createVariable(
    #         STEP1_LAG_TIMES_KEY, datatype=numpy.int32,
    #         dimensions=PREDICTOR_DIM_KEY
    #     )
    #     dataset_object.variables[STEP1_LAG_TIMES_KEY][:] = step1_lag_times_sec

    dataset_object.createVariable(
        STEP1_COSTS_KEY, datatype=numpy.float32,
        dimensions=(PREDICTOR_DIM_KEY, BOOTSTRAP_REPLICATE_DIM_KEY)
    )
    dataset_object.variables[STEP1_COSTS_KEY][:] = result_dict[STEP1_COSTS_KEY]

    dataset_object.createVariable(
        ORIGINAL_COST_KEY, datatype=numpy.float32,
        dimensions=BOOTSTRAP_REPLICATE_DIM_KEY
    )
    dataset_object.variables[ORIGINAL_COST_KEY][:] = (
        result_dict[ORIGINAL_COST_KEY]
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads results of permutation test from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_dict: See doc for `run_forward_test` or
        `run_backwards_test`.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    result_dict = {
        ORIGINAL_COST_KEY: dataset_object.variables[ORIGINAL_COST_KEY][:],
        BEST_PREDICTORS_KEY: [
            str(n) for n in netCDF4.chartostring(
                dataset_object.variables[BEST_PREDICTORS_KEY][:]
            )
        ],
        BEST_COSTS_KEY: dataset_object.variables[BEST_COSTS_KEY][:],
        STEP1_PREDICTORS_KEY: [
            str(n) for n in netCDF4.chartostring(
                dataset_object.variables[STEP1_PREDICTORS_KEY][:]
            )
        ],
        STEP1_COSTS_KEY: dataset_object.variables[STEP1_COSTS_KEY][:],
        BACKWARDS_FLAG_KEY: bool(getattr(dataset_object, BACKWARDS_FLAG_KEY))
    }

    dataset_object.close()

    # if BEST_LAG_TIMES_KEY in result_dict:
    #     result_dict[BEST_LAG_TIMES_KEY] = (
    #         dataset_object.variables[BEST_LAG_TIMES_KEY][:].astype(float)
    #     )
    #
    #     result_dict[BEST_LAG_TIMES_KEY][
    #         result_dict[BEST_LAG_TIMES_KEY] < 0
    #     ] = numpy.nan
    # else:
    #     result_dict[BEST_LAG_TIMES_KEY] = None
    #
    # if STEP1_LAG_TIMES_KEY in result_dict:
    #     result_dict[STEP1_LAG_TIMES_KEY] = (
    #         dataset_object.variables[STEP1_LAG_TIMES_KEY][:].astype(float)
    #     )
    #
    #     result_dict[STEP1_LAG_TIMES_KEY][
    #         result_dict[STEP1_LAG_TIMES_KEY] < 0
    #     ] = numpy.nan
    # else:
    #     result_dict[STEP1_LAG_TIMES_KEY] = None

    return result_dict
