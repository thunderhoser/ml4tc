"""Helper methods for computing activation.

--- NOTATION ---

The following letters are used throughout this module.

T = number of input tensors to the model
E = number of examples (storm objects)
"""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

HIT_INDICES_KEY = 'hit_indices'
MISS_INDICES_KEY = 'miss_indices'
FALSE_ALARM_INDICES_KEY = 'false_alarm_indices'
CORRECT_NULL_INDICES_KEY = 'correct_null_indices'


def get_hilo_activation_examples(
        storm_activations, num_high_activation_examples,
        num_low_activation_examples, unique_storm_cells,
        full_storm_id_strings=None):
    """Finds examples (storm objects) with highest and lowest activations.

    E = number of examples

    :param storm_activations: length-E numpy array of model activations.
    :param num_high_activation_examples: Number of high-activation examples to
        return.
    :param num_low_activation_examples: Number of low-activation examples to
        return.
    :param unique_storm_cells: Boolean flag.  If True, each set will contain no
        more than one example per storm cell.  If False, each set may contain
        multiple examples from the same storm cell.
    :param full_storm_id_strings: [used only if `unique_storm_cells == True`]
        length-E list of full storm IDs.
    :return: low_indices: 1-D numpy array with indices of low-activation
        examples.
    :return: high_indices: 1-D numpy array with indices of high-activation
        examples.
    """

    error_checking.assert_is_numpy_array(storm_activations, num_dimensions=1)
    error_checking.assert_is_boolean(unique_storm_cells)
    num_examples = len(storm_activations)

    if unique_storm_cells:
        expected_dim = numpy.array([num_examples], dtype=int)

        error_checking.assert_is_string_list(full_storm_id_strings)
        error_checking.assert_is_numpy_array(
            numpy.array(full_storm_id_strings), exact_dimensions=expected_dim
        )

    error_checking.assert_is_integer(num_high_activation_examples)
    error_checking.assert_is_geq(num_high_activation_examples, 0)
    error_checking.assert_is_integer(num_low_activation_examples)
    error_checking.assert_is_geq(num_low_activation_examples, 0)
    error_checking.assert_is_greater(
        num_high_activation_examples + num_low_activation_examples, 0
    )

    num_low_activation_examples = min([
        num_low_activation_examples, num_examples
    ])
    num_high_activation_examples = min([
        num_high_activation_examples, num_examples
    ])

    low_indices = numpy.array([], dtype=int)
    high_indices = numpy.array([], dtype=int)

    if num_low_activation_examples > 0:
        sort_indices = numpy.argsort(storm_activations)

        if unique_storm_cells:
            these_id_strings = numpy.array(full_storm_id_strings)[sort_indices]
            _, these_unique_indices = numpy.unique(
                these_id_strings, return_index=True)

            these_unique_indices = numpy.sort(these_unique_indices)
            sort_indices = sort_indices[these_unique_indices]

        low_indices = sort_indices[:num_low_activation_examples]

    if num_high_activation_examples > 0:
        sort_indices = numpy.argsort(-1 * storm_activations)

        if unique_storm_cells:
            these_id_strings = numpy.array(full_storm_id_strings)[sort_indices]
            _, these_unique_indices = numpy.unique(
                these_id_strings, return_index=True)

            these_unique_indices = numpy.sort(these_unique_indices)
            sort_indices = sort_indices[these_unique_indices]

        high_indices = sort_indices[:num_high_activation_examples]

    return high_indices, low_indices


def get_contingency_table_extremes(
        storm_activations, storm_target_values, num_hits, num_misses,
        num_false_alarms, num_correct_nulls, unique_storm_cells,
        full_storm_id_strings=None):
    """Returns "contingency-table extremes".

    Specifically, this method returns the following:

    - best hits (positive examples with the highest activations)
    - worst misses (positive examples with the lowest activations)
    - worst false alarms (negative examples with the highest activations)
    - best correct nulls (negative examples with the lowest activations)

    DEFINITIONS

    One "example" is one storm object.
    A "negative example" is a storm object with target = 0.
    A "positive example" is a storm object with target = 1.
    The target variable must be binary.

    E = number of examples

    :param storm_activations: length-E numpy array of model activations.
    :param storm_target_values: length-E numpy array of target values.  These
        must be integers from 0...1.
    :param num_hits: Number of best hits.
    :param num_misses: Number of worst misses.
    :param num_false_alarms: Number of worst false alarms.
    :param num_correct_nulls: Number of best correct nulls.
    :param unique_storm_cells: See doc for `get_hilo_activation_examples`.
    :param full_storm_id_strings: Same.
    :return: ct_extreme_dict: Dictionary with the following keys.
    ct_extreme_dict['hit_indices']: 1-D numpy array with indices of best hits.
    ct_extreme_dict['miss_indices']: 1-D numpy array with indices of worst
        misses.
    ct_extreme_dict['false_alarm_indices']: 1-D numpy array with indices of
        worst false alarms.
    ct_extreme_dict['correct_null_indices']: 1-D numpy array with indices of
        best correct nulls.
    """

    error_checking.assert_is_numpy_array(storm_activations, num_dimensions=1)
    error_checking.assert_is_boolean(unique_storm_cells)

    num_examples = len(storm_activations)
    expected_dim = numpy.array([num_examples], dtype=int)

    if unique_storm_cells:
        error_checking.assert_is_string_list(full_storm_id_strings)
        error_checking.assert_is_numpy_array(
            numpy.array(full_storm_id_strings), exact_dimensions=expected_dim
        )

    error_checking.assert_is_integer_numpy_array(storm_target_values)
    error_checking.assert_is_geq_numpy_array(storm_target_values, 0)
    error_checking.assert_is_leq_numpy_array(storm_target_values, 1)
    error_checking.assert_is_numpy_array(
        storm_target_values, exact_dimensions=expected_dim)

    error_checking.assert_is_integer(num_hits)
    error_checking.assert_is_geq(num_hits, 0)
    error_checking.assert_is_integer(num_misses)
    error_checking.assert_is_geq(num_misses, 0)
    error_checking.assert_is_integer(num_false_alarms)
    error_checking.assert_is_geq(num_false_alarms, 0)
    error_checking.assert_is_integer(num_correct_nulls)
    error_checking.assert_is_geq(num_correct_nulls, 0)
    error_checking.assert_is_greater(
        num_hits + num_misses + num_false_alarms + num_correct_nulls, 0
    )

    positive_indices = numpy.where(storm_target_values == 1)[0]
    negative_indices = numpy.where(storm_target_values == 0)[0]

    num_hits = min([num_hits, len(positive_indices)])
    num_misses = min([num_misses, len(positive_indices)])
    num_false_alarms = min([num_false_alarms, len(negative_indices)])
    num_correct_nulls = min([num_correct_nulls, len(negative_indices)])

    hit_indices = numpy.array([], dtype=int)
    miss_indices = numpy.array([], dtype=int)
    false_alarm_indices = numpy.array([], dtype=int)
    correct_null_indices = numpy.array([], dtype=int)

    if num_hits > 0:
        these_indices = numpy.argsort(-1 * storm_activations[positive_indices])
        sort_indices = positive_indices[these_indices]

        if unique_storm_cells:
            these_id_strings = numpy.array(full_storm_id_strings)[sort_indices]
            _, these_unique_indices = numpy.unique(
                these_id_strings, return_index=True)

            these_unique_indices = numpy.sort(these_unique_indices)
            sort_indices = sort_indices[these_unique_indices]

        hit_indices = sort_indices[:num_hits]

    if num_misses > 0:
        these_indices = numpy.argsort(storm_activations[positive_indices])
        sort_indices = positive_indices[these_indices]

        if unique_storm_cells:
            these_id_strings = numpy.array(full_storm_id_strings)[sort_indices]
            _, these_unique_indices = numpy.unique(
                these_id_strings, return_index=True)

            these_unique_indices = numpy.sort(these_unique_indices)
            sort_indices = sort_indices[these_unique_indices]

        miss_indices = sort_indices[:num_misses]

    if num_false_alarms > 0:
        these_indices = numpy.argsort(-1 * storm_activations[negative_indices])
        sort_indices = negative_indices[these_indices]

        if unique_storm_cells:
            these_id_strings = numpy.array(full_storm_id_strings)[sort_indices]
            _, these_unique_indices = numpy.unique(
                these_id_strings, return_index=True)

            these_unique_indices = numpy.sort(these_unique_indices)
            sort_indices = sort_indices[these_unique_indices]

        false_alarm_indices = sort_indices[:num_false_alarms]

    if num_correct_nulls > 0:
        these_indices = numpy.argsort(storm_activations[negative_indices])
        sort_indices = negative_indices[these_indices]

        if unique_storm_cells:
            these_id_strings = numpy.array(full_storm_id_strings)[sort_indices]
            _, these_unique_indices = numpy.unique(
                these_id_strings, return_index=True)

            these_unique_indices = numpy.sort(these_unique_indices)
            sort_indices = sort_indices[these_unique_indices]

        correct_null_indices = sort_indices[:num_correct_nulls]

    return {
        HIT_INDICES_KEY: hit_indices,
        MISS_INDICES_KEY: miss_indices,
        FALSE_ALARM_INDICES_KEY: false_alarm_indices,
        CORRECT_NULL_INDICES_KEY: correct_null_indices
    }
