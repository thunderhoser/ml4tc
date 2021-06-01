"""Unit tests for general_utils.py."""

import unittest
import numpy
from ml4tc.utils import general_utils

TOLERANCE = 1e-6

STORM_SPEEDS_M_S01 = numpy.array(
    [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float
)
STORM_HEADINGS_DEG = numpy.array(
    [66, 77, 0, 45, 90, 135, 180, 225, 270, 315, 360], dtype=float
)

HALF_ROOT2 = numpy.sqrt(2) / 2

U_MOTIONS_M_S01 = numpy.array([
    0, 0, 0, 2 * HALF_ROOT2, 3, 4 * HALF_ROOT2, 0,
    -6 * HALF_ROOT2, -7, -8 * HALF_ROOT2, 0
])
V_MOTIONS_M_S01 = numpy.array([
    0, 0, 1, 2 * HALF_ROOT2, 0, -4 * HALF_ROOT2, -5,
    -6 * HALF_ROOT2, 0, 8 * HALF_ROOT2, 9
])


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
            these_u_motions_m_s01, U_MOTIONS_M_S01, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_v_motions_m_s01, V_MOTIONS_M_S01, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
