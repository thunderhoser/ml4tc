"""Model evaluation."""

import os
import sys
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_model_evaluation as gg_model_eval
import file_system_utils
import error_checking
import prediction_io
import satellite_utils

DEFAULT_NUM_PROB_THRESHOLDS = 1001
DEFAULT_NUM_RELIABILITY_BINS = 20
DEFAULT_NUM_BOOTSTRAP_REPS = 1000

EXAMPLE_DIM = 'example_index'
PROBABILITY_THRESHOLD_DIM = 'probability_threshold'
RELIABILITY_BIN_DIM = 'reliability_bin'
BOOTSTRAP_REPLICATE_DIM = 'bootstrap_replicate'

MODEL_FILE_KEY = 'model_file_name'
TRAINING_EVENT_FREQ_KEY = 'training_event_frequency'

CYCLONE_ID_KEY = 'cyclone_id_string'
INIT_TIME_KEY = 'init_time_unix_sec'

NUM_TRUE_POSITIVES_KEY = 'num_true_positives'
NUM_FALSE_POSITIVES_KEY = 'num_false_positives'
NUM_FALSE_NEGATIVES_KEY = 'num_false_negatives'
NUM_TRUE_NEGATIVES_KEY = 'num_true_negatives'

MEAN_PREDICTION_KEY = 'mean_prediction'
MEAN_OBSERVATION_KEY = 'mean_observation'
MEAN_PREDICTION_NO_BS_KEY = 'mean_prediction_no_bootstrap'
EXAMPLE_COUNT_NO_BS_KEY = 'example_count_no_bootstrap'

AUC_KEY = 'area_under_roc_curve'
AUPD_KEY = 'area_under_perf_diagram'

POD_KEY = 'probability_of_detection'
POFD_KEY = 'probability_of_false_detection'
SUCCESS_RATIO_KEY = 'success_ratio'
FOCN_KEY = 'frequency_of_correct_nulls'
ACCURACY_KEY = 'accuracy'
CSI_KEY = 'critical_success_index'
FREQUENCY_BIAS_KEY = 'frequency_bias'

BRIER_SKILL_SCORE_KEY = 'brier_skill_score'
BRIER_SCORE_KEY = 'brier_score'
RELIABILITY_KEY = 'reliability'
RESOLUTION_KEY = 'resolution'


def _get_binary_scores_one_replicate(
        evaluation_table_xarray, forecast_probabilities, target_classes,
        event_freq_in_training, num_bootstrap_reps, bootstrap_rep_index):
    """Evaluates binary-classification model for one bootstrap replicate.

    :param evaluation_table_xarray: xarray table in format returned by
        `evaluate_model_binary`.
    :param forecast_probabilities: See doc for `evaluate_model_binary`.
    :param target_classes: Same.
    :param event_freq_in_training: Same.
    :param num_bootstrap_reps: Total number of bootstrap replicates.
    :param bootstrap_rep_index: Current index.  Will compute scores for [k]th
        bootstrap replicate, where k = `bootstrap_rep_index`.
    :return: evaluation_table_xarray: Same as input but with scores for [k]th
        bootstrap replicate filled.
    """

    k = bootstrap_rep_index

    num_examples = len(evaluation_table_xarray.coords[EXAMPLE_DIM].values)
    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )
    if num_bootstrap_reps > 1:
        example_indices = numpy.random.choice(
            example_indices, size=num_examples, replace=True
        )

    probability_thresholds = (
        evaluation_table_xarray.coords[PROBABILITY_THRESHOLD_DIM].values
    )
    num_prob_thresholds = len(probability_thresholds)

    for j in range(num_prob_thresholds):
        these_labels = (
            forecast_probabilities[example_indices] >= probability_thresholds[j]
        ).astype(int)

        this_contingency_dict = gg_model_eval.get_contingency_table(
            forecast_labels=these_labels,
            observed_labels=target_classes[example_indices]
        )

        evaluation_table_xarray[NUM_TRUE_POSITIVES_KEY].values[j, k] = (
            this_contingency_dict[gg_model_eval.NUM_TRUE_POSITIVES_KEY]
        )
        evaluation_table_xarray[NUM_FALSE_POSITIVES_KEY].values[j, k] = (
            this_contingency_dict[gg_model_eval.NUM_FALSE_POSITIVES_KEY]
        )
        evaluation_table_xarray[NUM_FALSE_NEGATIVES_KEY].values[j, k] = (
            this_contingency_dict[gg_model_eval.NUM_FALSE_NEGATIVES_KEY]
        )
        evaluation_table_xarray[NUM_TRUE_NEGATIVES_KEY].values[j, k] = (
            this_contingency_dict[gg_model_eval.NUM_TRUE_NEGATIVES_KEY]
        )
        evaluation_table_xarray[POD_KEY].values[j, k] = (
            gg_model_eval.get_pod(this_contingency_dict)
        )
        evaluation_table_xarray[POFD_KEY].values[j, k] = (
            gg_model_eval.get_pofd(this_contingency_dict)
        )
        evaluation_table_xarray[SUCCESS_RATIO_KEY].values[j, k] = (
            gg_model_eval.get_success_ratio(this_contingency_dict)
        )
        evaluation_table_xarray[FOCN_KEY].values[j, k] = (
            gg_model_eval.get_focn(this_contingency_dict)
        )
        evaluation_table_xarray[ACCURACY_KEY].values[j, k] = (
            gg_model_eval.get_accuracy(this_contingency_dict)
        )
        evaluation_table_xarray[CSI_KEY].values[j, k] = (
            gg_model_eval.get_csi(this_contingency_dict)
        )
        evaluation_table_xarray[FREQUENCY_BIAS_KEY].values[j, k] = (
            gg_model_eval.get_frequency_bias(this_contingency_dict)
        )

    evaluation_table_xarray[AUC_KEY].values[k] = (
        gg_model_eval.get_area_under_roc_curve(
            pod_by_threshold=evaluation_table_xarray[POD_KEY].values[:, k],
            pofd_by_threshold=evaluation_table_xarray[POFD_KEY].values[:, k]
        )
    )

    evaluation_table_xarray[AUPD_KEY].values[k] = (
        gg_model_eval.get_area_under_perf_diagram(
            pod_by_threshold=evaluation_table_xarray[POD_KEY].values[:, k],
            success_ratio_by_threshold=
            evaluation_table_xarray[SUCCESS_RATIO_KEY].values[:, k]
        )
    )

    num_reliability_bins = len(
        evaluation_table_xarray.coords[RELIABILITY_BIN_DIM].values
    )

    (
        evaluation_table_xarray[MEAN_PREDICTION_KEY].values[:, k],
        evaluation_table_xarray[MEAN_OBSERVATION_KEY].values[:, k],
        example_counts
    ) = gg_model_eval.get_points_in_reliability_curve(
        forecast_probabilities=forecast_probabilities[example_indices],
        observed_labels=target_classes[example_indices],
        num_forecast_bins=num_reliability_bins
    )

    bss_dict = gg_model_eval.get_brier_skill_score(
        mean_forecast_prob_by_bin=
        evaluation_table_xarray[MEAN_PREDICTION_KEY].values[:, k],
        mean_observed_label_by_bin=
        evaluation_table_xarray[MEAN_OBSERVATION_KEY].values[:, k],
        num_examples_by_bin=example_counts,
        climatology=event_freq_in_training
    )

    evaluation_table_xarray[BRIER_SKILL_SCORE_KEY].values[k] = (
        bss_dict[gg_model_eval.BSS_KEY]
    )
    evaluation_table_xarray[BRIER_SCORE_KEY].values[k] = (
        bss_dict[gg_model_eval.BRIER_SCORE_KEY]
    )
    evaluation_table_xarray[RELIABILITY_KEY].values[k] = (
        bss_dict[gg_model_eval.RELIABILITY_KEY]
    )
    evaluation_table_xarray[RESOLUTION_KEY].values[k] = (
        bss_dict[gg_model_eval.RESOLUTION_KEY]
    )

    return evaluation_table_xarray


def evaluate_model_binary(
        forecast_probabilities, target_classes, event_freq_in_training,
        cyclone_id_strings, init_times_unix_sec, model_file_name,
        num_prob_thresholds=DEFAULT_NUM_PROB_THRESHOLDS,
        num_reliability_bins=DEFAULT_NUM_RELIABILITY_BINS,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Evaluates model for binary classification.

    E = number of examples

    :param forecast_probabilities: length-E numpy array of forecast event
        (class = 1) probabilities.
    :param target_classes: length-E numpy array of true classes
        (integers in 0...1).
    :param event_freq_in_training: Event frequency in training data.
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param init_times_unix_sec: length-E numpy array of forecast-initialization
        times.
    :param model_file_name: Path to model that generated predictions.
    :param num_prob_thresholds: Number of probability thresholds for
        deterministic predictions.
    :param num_reliability_bins: Number of bins for reliability curves.
    :param num_bootstrap_reps: Number of replicates for bootstrapping.
    :return: evaluation_table_xarray: xarray table with results.  Variable names
        and dimensions should make this table self-explanatory.
    """

    error_checking.assert_is_geq_numpy_array(forecast_probabilities, 0.)
    error_checking.assert_is_leq_numpy_array(forecast_probabilities, 1.)
    error_checking.assert_is_numpy_array(
        forecast_probabilities, num_dimensions=1
    )

    error_checking.assert_is_integer_numpy_array(target_classes)
    error_checking.assert_is_geq_numpy_array(target_classes, 0)
    error_checking.assert_is_leq_numpy_array(target_classes, 1)

    num_examples = len(forecast_probabilities)
    expected_dim = numpy.array([num_examples], dtype=int)
    error_checking.assert_is_numpy_array(
        target_classes, exact_dimensions=expected_dim
    )

    error_checking.assert_is_geq(event_freq_in_training, 0.)
    error_checking.assert_is_leq(event_freq_in_training, 1.)
    error_checking.assert_is_integer(num_prob_thresholds)
    error_checking.assert_is_geq(num_prob_thresholds, 10)
    error_checking.assert_is_integer(num_reliability_bins)
    error_checking.assert_is_geq(num_reliability_bins, 10)
    error_checking.assert_is_integer(num_bootstrap_reps)
    error_checking.assert_is_geq(num_bootstrap_reps, 1)

    error_checking.assert_is_numpy_array(
        numpy.array(cyclone_id_strings), exact_dimensions=expected_dim
    )
    for this_id_string in cyclone_id_strings:
        _ = satellite_utils.parse_cyclone_id(this_id_string)

    error_checking.assert_is_integer_numpy_array(init_times_unix_sec)
    error_checking.assert_is_numpy_array(
        init_times_unix_sec, exact_dimensions=expected_dim
    )
    error_checking.assert_is_string(model_file_name)

    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )
    probability_thresholds = gg_model_eval.get_binarization_thresholds(
        threshold_arg=num_prob_thresholds
    )
    num_prob_thresholds = len(probability_thresholds)
    reliability_bin_indices = numpy.linspace(
        0, num_reliability_bins - 1, num=num_reliability_bins, dtype=int
    )
    bootstrap_rep_indices = numpy.linspace(
        0, num_bootstrap_reps - 1, num=num_bootstrap_reps, dtype=int
    )

    metadata_dict = {
        EXAMPLE_DIM: example_indices,
        PROBABILITY_THRESHOLD_DIM: probability_thresholds,
        RELIABILITY_BIN_DIM: reliability_bin_indices,
        BOOTSTRAP_REPLICATE_DIM: bootstrap_rep_indices
    }

    these_dim = (PROBABILITY_THRESHOLD_DIM, BOOTSTRAP_REPLICATE_DIM)
    this_integer_array = numpy.full(
        (num_prob_thresholds, num_bootstrap_reps), 0, dtype=int
    )
    this_float_array = numpy.full(
        (num_prob_thresholds, num_bootstrap_reps), numpy.nan
    )
    main_data_dict = {
        NUM_TRUE_POSITIVES_KEY: (these_dim, this_integer_array + 0),
        NUM_FALSE_POSITIVES_KEY: (these_dim, this_integer_array + 0),
        NUM_FALSE_NEGATIVES_KEY: (these_dim, this_integer_array + 0),
        NUM_TRUE_NEGATIVES_KEY: (these_dim, this_integer_array + 0),
        POD_KEY: (these_dim, this_float_array + 0.),
        POFD_KEY: (these_dim, this_float_array + 0.),
        SUCCESS_RATIO_KEY: (these_dim, this_float_array + 0.),
        FOCN_KEY: (these_dim, this_float_array + 0.),
        ACCURACY_KEY: (these_dim, this_float_array + 0.),
        CSI_KEY: (these_dim, this_float_array + 0.),
        FREQUENCY_BIAS_KEY: (these_dim, this_float_array + 0.)
    }

    these_dim = (RELIABILITY_BIN_DIM, BOOTSTRAP_REPLICATE_DIM)
    this_array = numpy.full(
        (num_reliability_bins, num_bootstrap_reps), numpy.nan
    )
    main_data_dict.update({
        MEAN_PREDICTION_KEY: (these_dim, this_array + 0.),
        MEAN_OBSERVATION_KEY: (these_dim, this_array + 0.)
    })

    these_dim = (RELIABILITY_BIN_DIM,)
    this_integer_array = numpy.full(num_reliability_bins, -1, dtype=int)
    this_float_array = numpy.full(num_reliability_bins, numpy.nan)
    main_data_dict.update({
        MEAN_PREDICTION_NO_BS_KEY: (these_dim, this_float_array + 0.),
        EXAMPLE_COUNT_NO_BS_KEY: (these_dim, this_integer_array + 0)
    })

    these_dim = (BOOTSTRAP_REPLICATE_DIM,)
    this_array = numpy.full(num_bootstrap_reps, numpy.nan)
    main_data_dict.update({
        AUC_KEY: (these_dim, this_array + 0.),
        AUPD_KEY: (these_dim, this_array + 0),
        BRIER_SKILL_SCORE_KEY: (these_dim, this_array + 0),
        BRIER_SCORE_KEY: (these_dim, this_array + 0),
        RELIABILITY_KEY: (these_dim, this_array + 0),
        RESOLUTION_KEY: (these_dim, this_array + 0)
    })

    main_data_dict.update({
        CYCLONE_ID_KEY: ((EXAMPLE_DIM,), cyclone_id_strings),
        INIT_TIME_KEY: ((EXAMPLE_DIM,), init_times_unix_sec)
    })

    evaluation_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    evaluation_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    evaluation_table_xarray.attrs[TRAINING_EVENT_FREQ_KEY] = (
        event_freq_in_training
    )

    (
        evaluation_table_xarray[MEAN_PREDICTION_NO_BS_KEY].values,
        _,
        evaluation_table_xarray[EXAMPLE_COUNT_NO_BS_KEY].values
    ) = gg_model_eval.get_points_in_reliability_curve(
        forecast_probabilities=forecast_probabilities,
        observed_labels=target_classes,
        num_forecast_bins=num_reliability_bins
    )

    for k in range(num_bootstrap_reps):
        print((
            'Computing scores for {0:d}th of {1:d} bootstrap replicates...'
        ).format(
            k + 1, num_bootstrap_reps
        ))

        evaluation_table_xarray = _get_binary_scores_one_replicate(
            evaluation_table_xarray=evaluation_table_xarray,
            forecast_probabilities=forecast_probabilities,
            target_classes=target_classes,
            event_freq_in_training=event_freq_in_training,
            num_bootstrap_reps=num_bootstrap_reps, bootstrap_rep_index=k
        )

    return evaluation_table_xarray


def find_best_threshold(evaluation_table_xarray):
    """Finds best probability threshold (that which maximizes CSI).

    :param evaluation_table_xarray: xarray table in format created by
        `evaluate_model_binary`.
    :return: best_threshold: Best probability threshold.
    """

    csi_by_threshold = numpy.mean(
        evaluation_table_xarray[CSI_KEY].values, axis=1
    )
    best_index = numpy.argmax(csi_by_threshold)
    best_threshold = evaluation_table_xarray.coords[
        PROBABILITY_THRESHOLD_DIM
    ].values[best_index]

    num_true_positives = numpy.mean(
        evaluation_table_xarray[NUM_TRUE_POSITIVES_KEY].values[best_index, :]
    )
    num_false_positives = numpy.mean(
        evaluation_table_xarray[NUM_FALSE_POSITIVES_KEY].values[best_index, :]
    )
    num_false_negatives = numpy.mean(
        evaluation_table_xarray[NUM_FALSE_NEGATIVES_KEY].values[best_index, :]
    )
    num_true_negatives = numpy.mean(
        evaluation_table_xarray[NUM_TRUE_NEGATIVES_KEY].values[best_index, :]
    )

    print((
        'Best CSI = {0:.3f} ... prob threshold = {1:.3f} ... a, b, c, d at '
        'same threshold = {2:.1f}, {3:.1f}, {4:.1f}, {5:.1f}'
    ).format(
        csi_by_threshold[best_index], best_threshold, num_true_positives,
        num_false_positives, num_false_negatives, num_true_negatives
    ))

    return best_threshold


def read_file(netcdf_file_name):
    """Reads evaluation results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: evaluation_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(evaluation_table_xarray, netcdf_file_name):
    """Writes evaluation results to NetCDF file.

    :param evaluation_table_xarray: xarray table in format returned by
        `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    evaluation_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def find_file(
        directory_name, month=None, basin_id_string=None,
        grid_row=None, grid_column=None, raise_error_if_missing=True):
    """Finds NetCDF file with evaluation results.

    :param directory_name: See doc for `prediction_io.find_file`.
    :param month: Same.
    :param basin_id_string: Same.
    :param grid_row: Same.
    :param grid_column: Same.
    :param raise_error_if_missing: Same.
    :return: evaluation_file_name: File path.
    """

    prediction_file_name = prediction_io.find_file(
        directory_name=directory_name,
        month=month, basin_id_string=basin_id_string,
        grid_row=grid_row, grid_column=grid_column,
        raise_error_if_missing=raise_error_if_missing
    )

    pathless_file_name = os.path.split(prediction_file_name)[-1].replace(
        'predictions', 'evaluation'
    )

    return '{0:s}/{1:s}'.format(
        os.path.split(prediction_file_name)[0],
        pathless_file_name
    )
