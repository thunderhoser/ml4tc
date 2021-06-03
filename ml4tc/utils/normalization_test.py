"""Unit tests for normalization.py."""

import unittest
import numpy
import scipy.stats
from ml4tc.utils import normalization

TOLERANCE = 1e-6

THESE_X_VALUES = numpy.linspace(200, 250, num=11, dtype=float)
THESE_Y_VALUES = numpy.linspace(260, 310, num=11, dtype=float)
TRAINING_VALUE_MATRIX = numpy.transpose(numpy.vstack((
    THESE_X_VALUES, THESE_Y_VALUES
)))

THESE_X_VALUES = numpy.array([231, 233, 218, numpy.nan, 203, 231, 233, 230])
THESE_Y_VALUES = numpy.array([311, 257, 314, 313, 289, 269, 280, numpy.nan])
ACTUAL_VALUE_MATRIX = numpy.transpose(numpy.vstack((
    THESE_X_VALUES, THESE_Y_VALUES
)))
ACTUAL_VALUE_MATRIX = numpy.expand_dims(ACTUAL_VALUE_MATRIX, axis=(1, 2))
ACTUAL_VALUE_MATRIX = numpy.repeat(ACTUAL_VALUE_MATRIX, repeats=11, axis=1)
ACTUAL_VALUE_MATRIX = numpy.repeat(ACTUAL_VALUE_MATRIX, repeats=6, axis=2)

THESE_X_VALUES = numpy.array([7, 7, 4, numpy.nan, 1, 7, 7, 6]) / 10
THESE_Y_VALUES = numpy.array([10, 0, 10, 10, 6, 2, 4, numpy.nan]) / 10
UNIFORM_VALUE_MATRIX = numpy.transpose(numpy.vstack((
    THESE_X_VALUES, THESE_Y_VALUES
)))
UNIFORM_VALUE_MATRIX = numpy.expand_dims(UNIFORM_VALUE_MATRIX, axis=(1, 2))
UNIFORM_VALUE_MATRIX = numpy.repeat(UNIFORM_VALUE_MATRIX, repeats=11, axis=1)
UNIFORM_VALUE_MATRIX = numpy.repeat(UNIFORM_VALUE_MATRIX, repeats=6, axis=2)

THESE_X_VALUES = numpy.array([235, 235, 220, numpy.nan, 205, 235, 235, 230])
THESE_Y_VALUES = numpy.array([310, 260, 310, 310, 290, 270, 280, numpy.nan])
DEUNIF_VALUE_MATRIX = numpy.transpose(numpy.vstack((
    THESE_X_VALUES, THESE_Y_VALUES
)))
DEUNIF_VALUE_MATRIX = numpy.expand_dims(DEUNIF_VALUE_MATRIX, axis=(1, 2))
DEUNIF_VALUE_MATRIX = numpy.repeat(DEUNIF_VALUE_MATRIX, repeats=11, axis=1)
DEUNIF_VALUE_MATRIX = numpy.repeat(DEUNIF_VALUE_MATRIX, repeats=6, axis=2)

NORMALIZED_VALUE_MATRIX = numpy.maximum(
    UNIFORM_VALUE_MATRIX, normalization.MIN_CUMULATIVE_DENSITY
)
NORMALIZED_VALUE_MATRIX = numpy.minimum(
    NORMALIZED_VALUE_MATRIX, normalization.MAX_CUMULATIVE_DENSITY
)
NORMALIZED_VALUE_MATRIX = scipy.stats.norm.ppf(
    NORMALIZED_VALUE_MATRIX, loc=0., scale=1.
)

THESE_X_VALUES = numpy.array([235, 235, 220, numpy.nan, 205, 235, 235, 230])
THESE_Y_VALUES = numpy.array([
    309.99995, 260.00005, 309.99995, 309.99995, 290, 270, 280, numpy.nan
])
DENORM_VALUE_MATRIX = numpy.transpose(numpy.vstack((
    THESE_X_VALUES, THESE_Y_VALUES
)))
DENORM_VALUE_MATRIX = numpy.expand_dims(DENORM_VALUE_MATRIX, axis=(1, 2))
DENORM_VALUE_MATRIX = numpy.repeat(DENORM_VALUE_MATRIX, repeats=11, axis=1)
DENORM_VALUE_MATRIX = numpy.repeat(DENORM_VALUE_MATRIX, repeats=6, axis=2)


class NormalizationTests(unittest.TestCase):
    """Each method is a unit test for normalization.py."""

    def test_actual_to_uniform_dist(self):
        """Ensures correct output from _actual_to_uniform_dist."""

        num_variables = ACTUAL_VALUE_MATRIX.shape[-1]
        this_value_matrix = numpy.full(ACTUAL_VALUE_MATRIX.shape, numpy.nan)

        for j in range(num_variables):
            this_value_matrix[..., j] = normalization._actual_to_uniform_dist(
                actual_values_new=ACTUAL_VALUE_MATRIX[..., j],
                actual_values_training=TRAINING_VALUE_MATRIX[..., j]
            )

        self.assertTrue(numpy.allclose(
            this_value_matrix, UNIFORM_VALUE_MATRIX, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_uniform_to_actual_dist(self):
        """Ensures correct output from _uniform_to_actual_dist."""

        num_variables = ACTUAL_VALUE_MATRIX.shape[-1]
        this_value_matrix = numpy.full(ACTUAL_VALUE_MATRIX.shape, numpy.nan)

        for j in range(num_variables):
            this_value_matrix[..., j] = normalization._uniform_to_actual_dist(
                uniform_values_new=UNIFORM_VALUE_MATRIX[..., j],
                actual_values_training=TRAINING_VALUE_MATRIX[..., j]
            )

        self.assertTrue(numpy.allclose(
            this_value_matrix, DEUNIF_VALUE_MATRIX, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_normalize_one_variable(self):
        """Ensures correct output from _normalize_one_variable."""

        num_variables = ACTUAL_VALUE_MATRIX.shape[-1]
        this_value_matrix = numpy.full(ACTUAL_VALUE_MATRIX.shape, numpy.nan)

        for j in range(num_variables):
            this_value_matrix[..., j] = normalization._normalize_one_variable(
                actual_values_new=ACTUAL_VALUE_MATRIX[..., j],
                actual_values_training=TRAINING_VALUE_MATRIX[..., j]
            )

        self.assertTrue(numpy.allclose(
            this_value_matrix, NORMALIZED_VALUE_MATRIX, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_denorm_one_variable(self):
        """Ensures correct output from _denorm_one_variable."""

        num_variables = ACTUAL_VALUE_MATRIX.shape[-1]
        this_value_matrix = numpy.full(ACTUAL_VALUE_MATRIX.shape, numpy.nan)

        for j in range(num_variables):
            this_value_matrix[..., j] = normalization._denorm_one_variable(
                normalized_values_new=NORMALIZED_VALUE_MATRIX[..., j],
                actual_values_training=TRAINING_VALUE_MATRIX[..., j]
            )

        self.assertTrue(numpy.allclose(
            this_value_matrix, DENORM_VALUE_MATRIX, atol=TOLERANCE,
            equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
