"""Unit tests for general_utils.py."""

import unittest
import numpy
from ml4tc.utils import general_utils

TOLERANCE = 1e-6

# The following constants are used to test speed_and_heading_to_uv.
STORM_SPEEDS_M_S01 = numpy.array(
    [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, numpy.nan], dtype=float
)
STORM_HEADINGS_DEG = numpy.array(
    [66, 77, 0, 45, 90, 135, 180, 225, 270, 315, 360, numpy.nan, 360],
    dtype=float
)

HALF_ROOT2 = numpy.sqrt(2) / 2

U_MOTIONS_M_S01 = numpy.array([
    0, 0, 0, 2 * HALF_ROOT2, 3, 4 * HALF_ROOT2, 0,
    -6 * HALF_ROOT2, -7, -8 * HALF_ROOT2, 0, numpy.nan, numpy.nan
])
V_MOTIONS_M_S01 = numpy.array([
    0, 0, 1, 2 * HALF_ROOT2, 0, -4 * HALF_ROOT2, -5,
    -6 * HALF_ROOT2, 0, 8 * HALF_ROOT2, 9, numpy.nan, numpy.nan
])

# The following constants are used to test fill_nans.
MATRIX_WITH_NANS_1D = numpy.array([1, 2, 3, numpy.nan])
MATRIX_WITHOUT_NANS_1D = numpy.array([1, 2, 3, 3], dtype=float)

MATRIX_WITH_NANS_2D = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, numpy.nan, numpy.nan, 10],
    [numpy.nan, 12, numpy.nan, numpy.nan, 15]
])
MATRIX_WITHOUT_NANS_2D = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, 7, 4, 10],
    [6, 12, 12, 15, 15]
])
MATRIX_WITH_NANS_3D = numpy.stack(
    (MATRIX_WITH_NANS_2D, MATRIX_WITH_NANS_2D), axis=0
)
MATRIX_WITHOUT_NANS_3D = numpy.stack(
    (MATRIX_WITHOUT_NANS_2D, MATRIX_WITHOUT_NANS_2D), axis=0
)


class GeneralUtilsTests(unittest.TestCase):
    """Each method is a unit test for general_utils.py."""

    def test_speed_and_heading_to_uv(self):
        """Ensures correct output from speed_and_heading_to_uv."""

        these_u_motions_m_s01, these_v_motions_m_s01 = (
            general_utils.speed_and_heading_to_uv(
                storm_speeds_m_s01=STORM_SPEEDS_M_S01,
                storm_headings_deg=STORM_HEADINGS_DEG
            )
        )

        self.assertTrue(numpy.allclose(
            these_u_motions_m_s01, U_MOTIONS_M_S01, atol=TOLERANCE,
            equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            these_v_motions_m_s01, V_MOTIONS_M_S01, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_fill_nans_1d(self):
        """Ensures correct output from fill_nans."""

        this_matrix_without_nans = general_utils.fill_nans(MATRIX_WITH_NANS_1D)
        self.assertTrue(numpy.allclose(
            this_matrix_without_nans, MATRIX_WITHOUT_NANS_1D, atol=TOLERANCE
        ))

    def test_fill_nans_2d(self):
        """Ensures correct output from fill_nans."""

        this_matrix_without_nans = general_utils.fill_nans(MATRIX_WITH_NANS_2D)
        self.assertTrue(numpy.allclose(
            this_matrix_without_nans, MATRIX_WITHOUT_NANS_2D, atol=TOLERANCE
        ))

    def test_fill_nans_3d(self):
        """Ensures correct output from fill_nans."""

        this_matrix_without_nans = general_utils.fill_nans(MATRIX_WITH_NANS_3D)
        self.assertTrue(numpy.allclose(
            this_matrix_without_nans, MATRIX_WITHOUT_NANS_3D, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
