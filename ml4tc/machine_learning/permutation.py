"""IO and helper methods for permutation-based importance test."""

import numpy
from sklearn.metrics import roc_auc_score as sklearn_auc
from gewittergefahr.gg_utils import error_checking
from ml4tc.machine_learning import neural_net

NUM_BUILTIN_SHIPS_LAG_TIMES = 4
NUM_SHIPS_FORECAST_HOURS = 23

GRIDDED_SATELLITE_ENUM = neural_net.GRIDDED_SATELLITE_ENUM
UNGRIDDED_SATELLITE_ENUM = neural_net.UNGRIDDED_SATELLITE_ENUM
SHIPS_ENUM = neural_net.SHIPS_ENUM


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

    if predictor_type_enum != SHIPS_ENUM:
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
    num_lagged_predictors = len(
        training_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
    )
    num_forecast_predictors = len(
        training_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
    )

    lagged_predictor_matrix_4d, forecast_predictor_matrix_4d = (
        neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=predictor_matrix,
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=NUM_BUILTIN_SHIPS_LAG_TIMES,
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=NUM_SHIPS_FORECAST_HOURS
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
                    ..., variable_index
                ][random_indices, ...]

            forecast_predictor_matrix_4d[..., variable_index] = (
                permuted_value_matrix
            )
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
                    :, model_lag_time_index, :, variable_index
                ][random_indices, ...]

            forecast_predictor_matrix_4d[
                :, model_lag_time_index, :, variable_index
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

    if predictor_type_enum != SHIPS_ENUM:
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
    num_lagged_predictors = len(
        training_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
    )
    num_forecast_predictors = len(
        training_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
    )

    lagged_predictor_matrix_4d, forecast_predictor_matrix_4d = (
        neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=predictor_matrix,
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=NUM_BUILTIN_SHIPS_LAG_TIMES,
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=NUM_SHIPS_FORECAST_HOURS
        )
    )

    clean_lagged_predictor_matrix_4d, clean_forecast_predictor_matrix_4d = (
        neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=clean_predictor_matrix,
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=NUM_BUILTIN_SHIPS_LAG_TIMES,
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=NUM_SHIPS_FORECAST_HOURS
        )
    )

    if model_lag_time_index is None:
        if variable_index < num_lagged_predictors:
            lagged_predictor_matrix_4d[..., variable_index] = (
                clean_lagged_predictor_matrix_4d[..., variable_index]
            )
        else:
            forecast_predictor_matrix_4d[..., variable_index] = (
                clean_forecast_predictor_matrix_4d[..., variable_index]
            )
    else:
        i = model_lag_time_index
        j = variable_index

        if variable_index < num_lagged_predictors:
            lagged_predictor_matrix_4d[:, i, :, j] = (
                clean_lagged_predictor_matrix_4d[:, i, :, j]
            )
        else:
            forecast_predictor_matrix_4d[:, i, :, j] = (
                clean_forecast_predictor_matrix_4d[:, i, :, j]
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


def _make_prediction_function(model_object):
    """Creates prediction function for neural net.

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :return: prediction_function: Function defined below.
    """

    def prediction_function(predictor_matrices):
        """Prediction function itself.

        E = number of examples
        K = number of classes

        :param predictor_matrices: See doc for `run_forward_test`.
        :return: forecast_prob_array: If K > 2, this is an E-by-K numpy array of
            class probabilities.  If K = 2, this is a length-E numpy array of
            positive-class probabilities.
        """

        return neural_net.apply_model(
            model_object=model_object, predictor_matrices=predictor_matrices,
            num_examples_per_batch=32, verbose=True
        )

    return prediction_function


def _run_forward_test_one_step(
        predictor_matrices, target_array, model_metadata_dict,
        prediction_function, cost_function, permuted_flag_arrays,
        num_bootstrap_reps):
    """Runs one step of the forward permutation test.

    I = number of input (predictor) matrices for model
    E = number of examples
    C = number of channels
    B = number of replicates for bootstrapping
    K = number of classes

    :param predictor_matrices: See doc for `run_forward_test`.
    :param target_array: Same.
    :param model_metadata_dict: Same.
    :param prediction_function: Function with the following inputs and outputs.
    Input: predictor_matrices: See above.
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
    result_dict['predictor_matrices']: Same as input but with more values
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

    # TODO(thunderhoser): Check args?

    if all([numpy.all(a) for a in permuted_flag_arrays]):
        return None

    permuted_matrix_indices = []
    permuted_variable_indices = []
    permuted_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    best_cost = -numpy.inf
    best_matrix_index = -1
    best_variable_index = -1
    best_permuted_value_matrix = None

    num_matrices = len(predictor_matrices)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    for i in range(num_matrices):
        if i == GRIDDED_SATELLITE_ENUM:
            num_variables = predictor_matrices[i].shape[-1]
        elif i == UNGRIDDED_SATELLITE_ENUM:
            num_variables = len(
                training_option_dict[neural_net.SATELLITE_PREDICTORS_KEY]
            )
        else:
            t = training_option_dict
            num_variables = (
                len(t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]) +
                len(t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY])
            )

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
                predictor_matrix=predictor_matrices[i] + 0.,
                predictor_type_enum=i,
                variable_index=j, model_metadata_dict=model_metadata_dict
            )
            these_predictor_matrices = [
                this_predictor_matrix if k == i else predictor_matrices[k]
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
        predictor_matrix=predictor_matrices[best_matrix_index],
        predictor_type_enum=best_matrix_index,
        variable_index=best_variable_index,
        model_metadata_dict=model_metadata_dict,
        permuted_value_matrix=best_permuted_value_matrix
    )
    predictor_matrices = [
        this_predictor_matrix if k == best_matrix_index
        else predictor_matrices[k]
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
        'predictor_matrices': predictor_matrices,
        'permuted_flag_arrays': permuted_flag_arrays,
        'permuted_matrix_indices': permuted_matrix_indices,
        'permuted_variable_indices': permuted_variable_indices,
        'permuted_cost_matrix': permuted_cost_matrix
    }


def _run_backwards_test_one_step(
        predictor_matrices, clean_predictor_matrices, target_array,
        model_metadata_dict, prediction_function, cost_function,
        permuted_flag_arrays, num_bootstrap_reps):
    """Runs one step of the backwards permutation test.

    :param predictor_matrices: See doc for `_run_forward_test_one_step`.
    :param clean_predictor_matrices: Clean version of `predictor_matrices`, with
        no permutation.
    :param target_array: See doc for `_run_forward_test_one_step`.
    :param model_metadata_dict: Same.
    :param prediction_function: Same.
    :param cost_function: Same.
    :param permuted_flag_arrays: Same.
    :param num_bootstrap_reps: Same.
    :return: result_dict: Dictionary with the following keys, where P = number
        of depermutations done in this step.
    result_dict['predictor_matrices']: Same as input but with fewer values
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

    num_matrices = len(predictor_matrices)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    for i in range(num_matrices):
        if i == GRIDDED_SATELLITE_ENUM:
            num_variables = predictor_matrices[i].shape[-1]
        elif i == UNGRIDDED_SATELLITE_ENUM:
            num_variables = len(
                training_option_dict[neural_net.SATELLITE_PREDICTORS_KEY]
            )
        else:
            t = training_option_dict
            num_variables = (
                len(t[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]) +
                len(t[neural_net.SHIPS_PREDICTORS_FORECAST_KEY])
            )

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
                predictor_matrix=predictor_matrices[i] + 0.,
                clean_predictor_matrix=clean_predictor_matrices[i],
                predictor_type_enum=i, variable_index=j,
                model_metadata_dict=model_metadata_dict
            )
            these_predictor_matrices = [
                this_predictor_matrix if k == i else predictor_matrices[k]
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
        predictor_matrix=predictor_matrices[best_matrix_index],
        clean_predictor_matrix=clean_predictor_matrices[best_matrix_index],
        predictor_type_enum=best_matrix_index,
        variable_index=best_variable_index,
        model_metadata_dict=model_metadata_dict
    )
    predictor_matrices = [
        this_predictor_matrix if k == best_matrix_index
        else predictor_matrices[k]
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
        'predictor_matrices': predictor_matrices,
        'permuted_flag_arrays': permuted_flag_arrays,
        'depermuted_matrix_indices': depermuted_matrix_indices,
        'depermuted_variable_indices': depermuted_variable_indices,
        'depermuted_cost_matrix': depermuted_cost_matrix
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
