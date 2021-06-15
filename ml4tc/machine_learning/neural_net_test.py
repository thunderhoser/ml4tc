"""Unit tests for neural_net.py."""

import unittest
import numpy
from ml4tc.machine_learning import neural_net

# The following constants are used to test _find_desired_times.
DESIRED_TIMES_UNIX_SEC = numpy.array(
    [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200], dtype=int
)
TOLERANCE_SEC = 1000

FIRST_ACTUAL_TIMES_UNIX_SEC = numpy.array(
    [2700, 5700, 1400, 100, 6300, 7100, 3600], dtype=int
)
FIRST_TIME_INDICES = numpy.array([3, 2, 2, 0, 6, 6, 1, 4, 5], dtype=int)

SECOND_ACTUAL_TIMES_UNIX_SEC = numpy.array(
    [7200, 4600, 5600, 3600, 1800, 0, 2600], dtype=int
)
SECOND_TIME_INDICES = numpy.array([5, 4, 4, 6, 3, 1, 2, 2, 0], dtype=int)

THIRD_ACTUAL_TIMES_UNIX_SEC = numpy.array(
    [3700, 6400, 3900, 1000, 200, 5700, 7100], dtype=int
)
THIRD_TIME_INDICES = numpy.array([4, 3, 3, 0, 0, 2, 5, 1, 6], dtype=int)

FOURTH_ACTUAL_TIMES_UNIX_SEC = numpy.array(
    [4300, 5400, 3400, 400, 6000, 2100, 2600], dtype=int
)
FOURTH_TIME_INDICES = None

# The following constants are used to test _discretize_intensity_change.
INTENSITY_CHANGE_M_S01 = 30.

FIRST_CLASS_CUTOFFS_M_S01 = numpy.array([30.])
FIRST_CLASS_FLAGS = numpy.array([0, 1], dtype=int)

SECOND_CLASS_CUTOFFS_M_S01 = numpy.array([31.])
SECOND_CLASS_FLAGS = numpy.array([1, 0], dtype=int)

THIRD_CLASS_CUTOFFS_M_S01 = numpy.array([-6, 1, 8, 15, 22, 29, 36], dtype=float)
THIRD_CLASS_FLAGS = numpy.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=int)

FOURTH_CLASS_CUTOFFS_M_S01 = numpy.array([-10, 0, 10, 20, 30, 40], dtype=float)
FOURTH_CLASS_FLAGS = numpy.array([0, 0, 0, 0, 0, 1, 0], dtype=int)


class NeuralNetTests(unittest.TestCase):
    """Each method is a unit test for neural_net.py."""

    def test_find_desired_times_first(self):
        """Ensures correct output from _find_desired_times.

        In this case, using first set of actual times.
        """

        these_indices = neural_net._find_desired_times(
            all_times_unix_sec=FIRST_ACTUAL_TIMES_UNIX_SEC,
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC,
            tolerance_sec=TOLERANCE_SEC
        )

        if FIRST_TIME_INDICES is None:
            self.assertTrue(these_indices is None)
        else:
            self.assertTrue(numpy.array_equal(
                these_indices, FIRST_TIME_INDICES
            ))

    def test_find_desired_times_second(self):
        """Ensures correct output from _find_desired_times.

        In this case, using second set of actual times.
        """

        these_indices = neural_net._find_desired_times(
            all_times_unix_sec=SECOND_ACTUAL_TIMES_UNIX_SEC,
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC,
            tolerance_sec=TOLERANCE_SEC
        )

        if SECOND_TIME_INDICES is None:
            self.assertTrue(these_indices is None)
        else:
            self.assertTrue(numpy.array_equal(
                these_indices, SECOND_TIME_INDICES
            ))

    def test_find_desired_times_third(self):
        """Ensures correct output from _find_desired_times.

        In this case, using third set of actual times.
        """

        these_indices = neural_net._find_desired_times(
            all_times_unix_sec=THIRD_ACTUAL_TIMES_UNIX_SEC,
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC,
            tolerance_sec=TOLERANCE_SEC
        )

        if THIRD_TIME_INDICES is None:
            self.assertTrue(these_indices is None)
        else:
            self.assertTrue(numpy.array_equal(
                these_indices, THIRD_TIME_INDICES
            ))

    def test_find_desired_times_fourth(self):
        """Ensures correct output from _find_desired_times.

        In this case, using fourth set of actual times.
        """

        these_indices = neural_net._find_desired_times(
            all_times_unix_sec=FOURTH_ACTUAL_TIMES_UNIX_SEC,
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC,
            tolerance_sec=TOLERANCE_SEC
        )

        if FOURTH_TIME_INDICES is None:
            self.assertTrue(these_indices is None)
        else:
            self.assertTrue(numpy.array_equal(
                these_indices, FOURTH_TIME_INDICES
            ))

    def test_discretize_intensity_change_first(self):
        """Ensures correct output from _discretize_intensity_change.

        In this case, using first set of cutoffs.
        """

        these_flags = neural_net._discretize_intensity_change(
            intensity_change_m_s01=INTENSITY_CHANGE_M_S01,
            class_cutoffs_m_s01=FIRST_CLASS_CUTOFFS_M_S01
        )
        self.assertTrue(numpy.array_equal(these_flags, FIRST_CLASS_FLAGS))

    def test_discretize_intensity_change_second(self):
        """Ensures correct output from _discretize_intensity_change.

        In this case, using second set of cutoffs.
        """

        these_flags = neural_net._discretize_intensity_change(
            intensity_change_m_s01=INTENSITY_CHANGE_M_S01,
            class_cutoffs_m_s01=SECOND_CLASS_CUTOFFS_M_S01
        )
        self.assertTrue(numpy.array_equal(these_flags, SECOND_CLASS_FLAGS))

    def test_discretize_intensity_change_third(self):
        """Ensures correct output from _discretize_intensity_change.

        In this case, using third set of cutoffs.
        """

        these_flags = neural_net._discretize_intensity_change(
            intensity_change_m_s01=INTENSITY_CHANGE_M_S01,
            class_cutoffs_m_s01=THIRD_CLASS_CUTOFFS_M_S01
        )
        self.assertTrue(numpy.array_equal(these_flags, THIRD_CLASS_FLAGS))

    def test_discretize_intensity_change_fourth(self):
        """Ensures correct output from _discretize_intensity_change.

        In this case, using fourth set of cutoffs.
        """

        these_flags = neural_net._discretize_intensity_change(
            intensity_change_m_s01=INTENSITY_CHANGE_M_S01,
            class_cutoffs_m_s01=FOURTH_CLASS_CUTOFFS_M_S01
        )
        self.assertTrue(numpy.array_equal(these_flags, FOURTH_CLASS_FLAGS))


if __name__ == '__main__':
    unittest.main()
