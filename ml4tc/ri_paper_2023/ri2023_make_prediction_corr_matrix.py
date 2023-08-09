"""Makes and plots prediction-correlation matrix.

The models in the correlation matrix will be the selected NNs and the three
baselines (SHIPS-RII, SHIPS consensus, and DTOPS).
"""

import argparse
import warnings
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from ml4tc.io import prediction_io

MAX_DISTANCE_DEG = 1.

BASELINE_DESCRIPTION_STRINGS = ['basis', 'consensus', 'dtops']
BASELINE_DESCRIPTION_STRINGS_FANCY = ['SHIPS-RII', 'SHIPS consensus', 'DTOPS']

NN_MODEL_DIRS_ARG_NAME = 'input_nn_model_dir_names'
NN_MODEL_DESCRIPTIONS_ARG_NAME = 'nn_model_description_strings'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

NN_MODEL_DIRS_HELP_STRING = (
    'List of input directories, one for each selected NN.  Each directory '
    'should be the top-level directory for the given NN.  This script will '
    'find results on the testing data, either matched with the SHIPS baselines '
    'or not (as appropriate), and using isotonic regression.'
)
NN_MODEL_DESCRIPTIONS_HELP_STRING = (
    'List of NN descriptions, one per input directory.'
)
OUTPUT_DIR_HELP_STRING = 'Path to output directory.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NN_MODEL_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=NN_MODEL_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NN_MODEL_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=NN_MODEL_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _match_examples(first_prediction_dict, second_prediction_dict):
    """Matches examples between two prediction sets.

    E = number of examples matched

    :param first_prediction_dict: Dictionary read by `prediction_io.read_file`.
    :param second_prediction_dict: Same but for second model.
    :return: first_indices: length-E numpy array of indices in first dictionary.
    :return: second_indices: length-E numpy array of indices in second
        dictionary.
    """

    first_init_times_unix_sec = (
        first_prediction_dict[prediction_io.INIT_TIMES_KEY]
    )
    first_latitudes_deg_n = (
        first_prediction_dict[prediction_io.STORM_LATITUDES_KEY]
    )
    first_pos_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        first_prediction_dict[prediction_io.STORM_LONGITUDES_KEY]
    )
    first_neg_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        first_prediction_dict[prediction_io.STORM_LONGITUDES_KEY]
    )

    second_init_times_unix_sec = (
        second_prediction_dict[prediction_io.INIT_TIMES_KEY]
    )
    second_latitudes_deg_n = (
        second_prediction_dict[prediction_io.STORM_LATITUDES_KEY]
    )
    second_pos_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        second_prediction_dict[prediction_io.STORM_LONGITUDES_KEY]
    )
    second_neg_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        second_prediction_dict[prediction_io.STORM_LONGITUDES_KEY]
    )

    num_second_examples = len(second_init_times_unix_sec)
    second_to_first_indices = numpy.full(num_second_examples, -1, dtype=int)

    for i in range(num_second_examples):
        js = numpy.where(
            first_init_times_unix_sec == second_init_times_unix_sec[i]
        )[0]

        if len(js) == 0:
            warning_string = (
                'POTENTIAL ERROR: Cannot find time match for CNN cyclone '
                '{0:s} at {1:s}.'
            ).format(
                second_prediction_dict[prediction_io.CYCLONE_IDS_KEY][i],
                time_conversion.unix_sec_to_string(
                    second_init_times_unix_sec[i], '%Y-%m-%d-%H%M%S'
                )
            )

            warnings.warn(warning_string)
            continue

        first_distances_deg = numpy.sqrt(
            (second_latitudes_deg_n[i] - first_latitudes_deg_n[js]) ** 2 +
            (second_pos_longitudes_deg_e[i] - first_pos_longitudes_deg_e[js]) ** 2
        )
        second_distances_deg = numpy.sqrt(
            (second_latitudes_deg_n[i] - first_latitudes_deg_n[js]) ** 2 +
            (second_neg_longitudes_deg_e[i] - first_neg_longitudes_deg_e[js]) ** 2
        )
        these_distances_deg = numpy.minimum(
            first_distances_deg, second_distances_deg
        )

        if numpy.min(these_distances_deg) > MAX_DISTANCE_DEG:
            second_pos_longitudes_deg_e[i] *= -1
            second_neg_longitudes_deg_e[i] *= -1

            first_distances_deg = numpy.sqrt(
                (second_latitudes_deg_n[i] - first_latitudes_deg_n[js]) ** 2 +
                (second_pos_longitudes_deg_e[i] - first_pos_longitudes_deg_e[js])
                ** 2
            )
            second_distances_deg = numpy.sqrt(
                (second_latitudes_deg_n[i] - first_latitudes_deg_n[js]) ** 2 +
                (second_neg_longitudes_deg_e[i] - first_neg_longitudes_deg_e[js])
                ** 2
            )
            these_distances_deg = numpy.minimum(
                first_distances_deg, second_distances_deg
            )

        if numpy.min(these_distances_deg) > MAX_DISTANCE_DEG:
            this_first_index = js[numpy.argmin(these_distances_deg)]

            warning_string = (
                'POTENTIAL ERROR: Cannot find distance match for CNN cyclone '
                '{0:s} at {1:s}.  CNN cyclone is at '
                '{2:.2f} deg N, {3:.2f} deg E; SHIPS cyclone is at '
                '{4:.2f} deg N, {5:.2f} deg E.'
            ).format(
                second_prediction_dict[prediction_io.CYCLONE_IDS_KEY][i],
                time_conversion.unix_sec_to_string(
                    second_init_times_unix_sec[i], '%Y-%m-%d-%H%M%S'
                ),
                second_latitudes_deg_n[i],
                second_pos_longitudes_deg_e[i],
                first_latitudes_deg_n[this_first_index],
                first_pos_longitudes_deg_e[this_first_index]
            )

            warnings.warn(warning_string)
            continue

        second_to_first_indices[i] = js[numpy.argmin(these_distances_deg)]

    second_indices = numpy.where(second_to_first_indices >= 0)[0]
    first_indices = second_to_first_indices[second_indices]

    print('Matched {0:d} of {1:d} second examples to a first example!'.format(
        len(second_indices), num_second_examples
    ))

    return first_indices, second_indices


def _compute_one_correlation(first_prediction_file_name,
                             second_prediction_file_name):
    """Computes correlation between predictions for two models.

    :param first_prediction_file_name: Path to predictions for first model.
        Will be read by `prediction_io.read_file`.
    :param second_prediction_file_name: Same but for second model.
    :return: correlation: Pearson correlation (scalar).
    """

    print('Reading data from: "{0:s}"...'.format(first_prediction_file_name))
    first_prediction_dict = prediction_io.read_file(first_prediction_file_name)

    print('Reading data from: "{0:s}"...'.format(second_prediction_file_name))
    second_prediction_dict = prediction_io.read_file(
        second_prediction_file_name
    )

    first_example_indices, second_example_indices = _match_examples(
        first_prediction_dict=first_prediction_dict,
        second_prediction_dict=second_prediction_dict
    )

    first_ri_probs = prediction_io.get_mean_predictions(
        first_prediction_dict
    )[first_example_indices, 0]

    second_ri_probs = prediction_io.get_mean_predictions(
        second_prediction_dict
    )[second_example_indices, 0]

    correlation = numpy.corrcoef(first_ri_probs, second_ri_probs)[0, 1]
    print('Correlation = {0:.4f}\n'.format(correlation))

    return correlation


def _run(top_nn_model_dir_names, nn_model_description_strings, output_dir_name):
    """Makes and plots prediction-correlation matrix.

    This is effectively the main method.

    :param top_nn_model_dir_names: See documentation at top of file.
    :param nn_model_description_strings: Same.
    :param output_dir_name: Same.
    """

    num_nn_models = len(top_nn_model_dir_names)
    assert len(nn_model_description_strings) == num_nn_models
    nn_model_description_strings = [
        s.replace('_', ' ') for s in nn_model_description_strings
    ]

    model_description_strings = (
        nn_model_description_strings + BASELINE_DESCRIPTION_STRINGS
    )
    model_description_strings_fancy = (
        nn_model_description_strings + BASELINE_DESCRIPTION_STRINGS_FANCY
    )
    top_model_dir_names = (
        top_nn_model_dir_names + [''] * len(BASELINE_DESCRIPTION_STRINGS)
    )
    num_models = len(top_model_dir_names)

    correlation_matrix = numpy.full((num_models, num_models), numpy.nan)

    for i in range(num_models):
        for j in range(num_models):
            if j >= i:
                continue
            if i == j:
                correlation_matrix[i, j] = 1.
                continue

            are_both_models_nn = not (
                model_description_strings[i] in BASELINE_DESCRIPTION_STRINGS or
                model_description_strings[j] in BASELINE_DESCRIPTION_STRINGS
            )
            are_both_models_baseline = (
                model_description_strings[i] in BASELINE_DESCRIPTION_STRINGS and
                model_description_strings[j] in BASELINE_DESCRIPTION_STRINGS
            )

            if are_both_models_nn:
                prediction_file_name_i = (
                    '{0:s}/real_time_testing/isotonic_regression/predictions.nc'
                ).format(top_model_dir_names[i])

                prediction_file_name_j = (
                    '{0:s}/real_time_testing/isotonic_regression/predictions.nc'
                ).format(top_model_dir_names[j])

            elif are_both_models_baseline:
                prediction_file_name_i = (
                    '{0:s}/real_time_testing_matched_with_ships/'
                    'isotonic_regression/ships_predictions_{1:s}.nc'
                ).format(
                    top_model_dir_names[0], model_description_strings[i]
                )

                prediction_file_name_j = (
                    '{0:s}/real_time_testing_matched_with_ships/'
                    'isotonic_regression/ships_predictions_{1:s}.nc'
                ).format(
                    top_model_dir_names[0], model_description_strings[j]
                )

            else:
                prediction_file_name_i = (
                    '{0:s}/real_time_testing_matched_with_ships/'
                    'isotonic_regression/cnn_predictions_cf_{1:s}.nc'
                ).format(
                    top_model_dir_names[i], model_description_strings[j]
                )

                prediction_file_name_j = (
                    '{0:s}/real_time_testing_matched_with_ships/'
                    'isotonic_regression/ships_predictions_{1:s}.nc'
                ).format(
                    top_model_dir_names[i], model_description_strings[j]
                )

            correlation_matrix[i, j] = _compute_one_correlation(
                first_prediction_file_name=prediction_file_name_i,
                second_prediction_file_name=prediction_file_name_j
            )
            correlation_matrix[j, i] = correlation_matrix[i, j] + 0.


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_nn_model_dir_names=getattr(
            INPUT_ARG_OBJECT, NN_MODEL_DIRS_ARG_NAME
        ),
        nn_model_description_strings=getattr(
            INPUT_ARG_OBJECT, NN_MODEL_DESCRIPTIONS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
