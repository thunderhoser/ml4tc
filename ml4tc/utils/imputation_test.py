"""Unit tests for imputation.py."""

import unittest
import numpy
from ml4tc.utils import imputation

TOLERANCE = 1e-6

# The following constants are used to test _interp_in_time.
INPUT_TIMES = numpy.array([0, 1, 2, 4, 6, 8, 9, 10, 11, 12], dtype=float)

FIRST_MIN_TEMPORAL_COVERAGE_FRACTION = 0.5
FIRST_MIN_NUM_TIMES = 4
FIRST_VALUES_BEFORE_INTERP = numpy.array(
    [3.2, -3.3, 5.2, 7.8, 7.5, -4.4, -2.8, 2.7, -4.0, -3.7], dtype=float
)
FIRST_VALUES_AFTER_INTERP = FIRST_VALUES_BEFORE_INTERP + 0.

SECOND_MIN_TEMPORAL_COVERAGE_FRACTION = 0.5
SECOND_MIN_NUM_TIMES = 4
SECOND_VALUES_BEFORE_INTERP = numpy.array([
    numpy.nan, -3.3, numpy.nan, 7.8, 7.5, -4.4, numpy.nan, 2.7, numpy.nan,
    numpy.nan
], dtype=float)
SECOND_VALUES_AFTER_INTERP = numpy.array(
    [-3.3, -3.3, 0.4, 7.8, 7.5, -4.4, -0.85, 2.7, 2.7, 2.7], dtype=float
)

THIRD_MIN_TEMPORAL_COVERAGE_FRACTION = 0.4
THIRD_MIN_NUM_TIMES = 4
THIRD_VALUES_BEFORE_INTERP = numpy.array([
    numpy.nan, numpy.nan, numpy.nan, 7.8, 7.5, -4.4, numpy.nan, 2.7, numpy.nan,
    numpy.nan
], dtype=float)
THIRD_VALUES_AFTER_INTERP = numpy.array(
    [7.8, 7.8, 7.8, 7.8, 7.5, -4.4, -0.85, 2.7, 2.7, 2.7], dtype=float
)

FOURTH_MIN_TEMPORAL_COVERAGE_FRACTION = 0.5
FOURTH_MIN_NUM_TIMES = 4
FOURTH_VALUES_BEFORE_INTERP = numpy.array([
    numpy.nan, numpy.nan, numpy.nan, 7.8, 7.5, -4.4, numpy.nan, 2.7, numpy.nan,
    numpy.nan
], dtype=float)
FOURTH_VALUES_AFTER_INTERP = FOURTH_VALUES_BEFORE_INTERP + 0.

# The following constants are used to test _interp_in_space.
FIRST_MIN_SPATIAL_COVERAGE_FRACTION = 0.5
FIRST_MIN_NUM_PIXELS = 3
FIRST_MATRIX_BEFORE_INTERP = numpy.array([1, 2, 3, numpy.nan])
FIRST_MATRIX_AFTER_INTERP = numpy.array([1, 2, 3, 3], dtype=float)

SECOND_MIN_SPATIAL_COVERAGE_FRACTION = 0.5
SECOND_MIN_NUM_PIXELS = 4
SECOND_MATRIX_BEFORE_INTERP = numpy.array([1, 2, 3, numpy.nan])
SECOND_MATRIX_AFTER_INTERP = SECOND_MATRIX_BEFORE_INTERP + 0.

THIRD_MIN_SPATIAL_COVERAGE_FRACTION = 0.5
THIRD_MIN_NUM_PIXELS = 4
THIRD_MATRIX_BEFORE_INTERP = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, numpy.nan, numpy.nan, 10],
    [numpy.nan, 12, numpy.nan, numpy.nan, 15]
])
THIRD_MATRIX_AFTER_INTERP = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, 7, 4, 10],
    [6, 12, 12, 15, 15]
], dtype=float)

FOURTH_MIN_SPATIAL_COVERAGE_FRACTION = 0.5
FOURTH_MIN_NUM_PIXELS = 11
FOURTH_MATRIX_BEFORE_INTERP = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, numpy.nan, numpy.nan, 10],
    [numpy.nan, 12, numpy.nan, numpy.nan, 15]
])
FOURTH_MATRIX_AFTER_INTERP = FOURTH_MATRIX_BEFORE_INTERP + 0.

FIFTH_MIN_SPATIAL_COVERAGE_FRACTION = 0.5
FIFTH_MIN_NUM_PIXELS = 4
FIFTH_MATRIX_BEFORE_INTERP = numpy.stack(
    (THIRD_MATRIX_BEFORE_INTERP, THIRD_MATRIX_BEFORE_INTERP), axis=0
)
FIFTH_MATRIX_AFTER_INTERP = numpy.stack(
    (THIRD_MATRIX_AFTER_INTERP, THIRD_MATRIX_AFTER_INTERP), axis=0
)

SIXTH_MIN_SPATIAL_COVERAGE_FRACTION = 0.5
SIXTH_MIN_NUM_PIXELS = 22
SIXTH_MATRIX_BEFORE_INTERP = numpy.stack(
    (FOURTH_MATRIX_BEFORE_INTERP, FOURTH_MATRIX_BEFORE_INTERP), axis=0
)
SIXTH_MATRIX_AFTER_INTERP = numpy.stack(
    (FOURTH_MATRIX_AFTER_INTERP, FOURTH_MATRIX_AFTER_INTERP), axis=0
)


class ImputationTests(unittest.TestCase):
    """Each method is a unit test for imputation.py."""

    def test_interp_in_time_first(self):
        """Ensures correct output from _interp_in_time.

        In this case, using first set of inputs.
        """

        these_values = imputation._interp_in_time(
            input_times=INPUT_TIMES,
            input_data_values=FIRST_VALUES_BEFORE_INTERP + 0.,
            min_coverage_fraction=FIRST_MIN_TEMPORAL_COVERAGE_FRACTION,
            min_num_times=FIRST_MIN_NUM_TIMES
        )

        self.assertTrue(numpy.allclose(
            these_values, FIRST_VALUES_AFTER_INTERP, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_interp_in_time_second(self):
        """Ensures correct output from _interp_in_time.

        In this case, using second set of inputs.
        """

        these_values = imputation._interp_in_time(
            input_times=INPUT_TIMES,
            input_data_values=SECOND_VALUES_BEFORE_INTERP + 0.,
            min_coverage_fraction=SECOND_MIN_TEMPORAL_COVERAGE_FRACTION,
            min_num_times=SECOND_MIN_NUM_TIMES
        )

        self.assertTrue(numpy.allclose(
            these_values, SECOND_VALUES_AFTER_INTERP, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_interp_in_time_third(self):
        """Ensures correct output from _interp_in_time.

        In this case, using third set of inputs.
        """

        these_values = imputation._interp_in_time(
            input_times=INPUT_TIMES,
            input_data_values=THIRD_VALUES_BEFORE_INTERP + 0.,
            min_coverage_fraction=THIRD_MIN_TEMPORAL_COVERAGE_FRACTION,
            min_num_times=THIRD_MIN_NUM_TIMES
        )

        self.assertTrue(numpy.allclose(
            these_values, THIRD_VALUES_AFTER_INTERP, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_interp_in_time_fourth(self):
        """Ensures correct output from _interp_in_time.

        In this case, using fourth set of inputs.
        """

        these_values = imputation._interp_in_time(
            input_times=INPUT_TIMES,
            input_data_values=FOURTH_VALUES_BEFORE_INTERP + 0.,
            min_coverage_fraction=FOURTH_MIN_TEMPORAL_COVERAGE_FRACTION,
            min_num_times=FOURTH_MIN_NUM_TIMES
        )

        self.assertTrue(numpy.allclose(
            these_values, FOURTH_VALUES_AFTER_INTERP, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_interp_in_space_first(self):
        """Ensures correct output from _interp_in_space.

        In this case, using first set of inputs.
        """

        this_matrix = imputation._interp_in_space(
            input_matrix=FIRST_MATRIX_BEFORE_INTERP + 0.,
            min_coverage_fraction=FIRST_MIN_SPATIAL_COVERAGE_FRACTION,
            min_num_pixels=FIRST_MIN_NUM_PIXELS
        )

        self.assertTrue(numpy.allclose(
            this_matrix, FIRST_MATRIX_AFTER_INTERP, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_interp_in_space_second(self):
        """Ensures correct output from _interp_in_space.

        In this case, using second set of inputs.
        """

        this_matrix = imputation._interp_in_space(
            input_matrix=SECOND_MATRIX_BEFORE_INTERP + 0.,
            min_coverage_fraction=SECOND_MIN_SPATIAL_COVERAGE_FRACTION,
            min_num_pixels=SECOND_MIN_NUM_PIXELS
        )

        self.assertTrue(numpy.allclose(
            this_matrix, SECOND_MATRIX_AFTER_INTERP, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_interp_in_space_third(self):
        """Ensures correct output from _interp_in_space.

        In this case, using third set of inputs.
        """

        this_matrix = imputation._interp_in_space(
            input_matrix=THIRD_MATRIX_BEFORE_INTERP + 0.,
            min_coverage_fraction=THIRD_MIN_SPATIAL_COVERAGE_FRACTION,
            min_num_pixels=THIRD_MIN_NUM_PIXELS
        )

        self.assertTrue(numpy.allclose(
            this_matrix, THIRD_MATRIX_AFTER_INTERP, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_interp_in_space_fourth(self):
        """Ensures correct output from _interp_in_space.

        In this case, using fourth set of inputs.
        """

        this_matrix = imputation._interp_in_space(
            input_matrix=FOURTH_MATRIX_BEFORE_INTERP + 0.,
            min_coverage_fraction=FOURTH_MIN_SPATIAL_COVERAGE_FRACTION,
            min_num_pixels=FOURTH_MIN_NUM_PIXELS
        )

        self.assertTrue(numpy.allclose(
            this_matrix, FOURTH_MATRIX_AFTER_INTERP, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_interp_in_space_fifth(self):
        """Ensures correct output from _interp_in_space.

        In this case, using fifth set of inputs.
        """

        this_matrix = imputation._interp_in_space(
            input_matrix=FIFTH_MATRIX_BEFORE_INTERP + 0.,
            min_coverage_fraction=FIFTH_MIN_SPATIAL_COVERAGE_FRACTION,
            min_num_pixels=FIFTH_MIN_NUM_PIXELS
        )

        self.assertTrue(numpy.allclose(
            this_matrix, FIFTH_MATRIX_AFTER_INTERP, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_interp_in_space_sixth(self):
        """Ensures correct output from _interp_in_space.

        In this case, using sixth set of inputs.
        """

        this_matrix = imputation._interp_in_space(
            input_matrix=SIXTH_MATRIX_BEFORE_INTERP + 0.,
            min_coverage_fraction=SIXTH_MIN_SPATIAL_COVERAGE_FRACTION,
            min_num_pixels=SIXTH_MIN_NUM_PIXELS
        )

        self.assertTrue(numpy.allclose(
            this_matrix, SIXTH_MATRIX_AFTER_INTERP, atol=TOLERANCE,
            equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
