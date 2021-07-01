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

# The following constants are used to test find_exact_times.
ACTUAL_TIMES_UNIX_SEC = numpy.array([7, 1, 12, 59, 15, 34], dtype=int)
FIRST_DESIRED_TIMES_UNIX_SEC = numpy.array([15, 1, 7, 59], dtype=int)
FIRST_DESIRED_INDICES = numpy.array([4, 1, 0, 3], dtype=int)
SECOND_DESIRED_TIMES_UNIX_SEC = numpy.array([15, 1, 7, 59, 2], dtype=int)
SECOND_DESIRED_INDICES = None

THIRD_START_TIME_UNIX_SEC = 5
THIRD_END_TIME_UNIX_SEC = 50
THIRD_DESIRED_INDICES = numpy.array([0, 2, 4, 5], dtype=int)

FOURTH_START_TIME_UNIX_SEC = 5
FOURTH_END_TIME_UNIX_SEC = 6
FOURTH_DESIRED_INDICES = None

# The following constants are used to test create_latlng_grid.
MIN_GRID_LATITUDE_DEG_N = 49.123
MAX_GRID_LATITUDE_DEG_N = 59.321
MIN_GRID_LONGITUDE_DEG_E = 240.567
MAX_GRID_LONGITUDE_DEG_E = -101.789
LATITUDE_SPACING_DEG = 1.
LONGITUDE_SPACING_DEG = 2.

GRID_POINT_LATITUDES_DEG = numpy.array(
    [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], dtype=float
)
GRID_POINT_LONGITUDES_DEG = numpy.array(
    [240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260], dtype=float
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

    def test_find_exact_times_first(self):
        """Ensures correct output from find_exact_times.

        In this case, using first set of input args.
        """

        these_indices = general_utils.find_exact_times(
            actual_times_unix_sec=ACTUAL_TIMES_UNIX_SEC,
            desired_times_unix_sec=FIRST_DESIRED_TIMES_UNIX_SEC
        )
        self.assertTrue(numpy.array_equal(these_indices, FIRST_DESIRED_INDICES))

    def test_find_exact_times_second(self):
        """Ensures correct output from find_exact_times.

        In this case, using second set of input args.
        """

        with self.assertRaises(IndexError):
            general_utils.find_exact_times(
                actual_times_unix_sec=ACTUAL_TIMES_UNIX_SEC,
                desired_times_unix_sec=SECOND_DESIRED_TIMES_UNIX_SEC
            )

    def test_find_exact_times_third(self):
        """Ensures correct output from find_exact_times.

        In this case, using third set of input args.
        """

        these_indices = general_utils.find_exact_times(
            actual_times_unix_sec=ACTUAL_TIMES_UNIX_SEC,
            first_desired_time_unix_sec=THIRD_START_TIME_UNIX_SEC,
            last_desired_time_unix_sec=THIRD_END_TIME_UNIX_SEC
        )
        self.assertTrue(numpy.array_equal(these_indices, THIRD_DESIRED_INDICES))

    def test_find_exact_times_fourth(self):
        """Ensures correct output from find_exact_times.

        In this case, using fourth set of input args.
        """

        with self.assertRaises(ValueError):
            general_utils.find_exact_times(
                actual_times_unix_sec=ACTUAL_TIMES_UNIX_SEC,
                first_desired_time_unix_sec=FOURTH_START_TIME_UNIX_SEC,
                last_desired_time_unix_sec=FOURTH_END_TIME_UNIX_SEC
            )

    def test_create_latlng_grid(self):
        """Ensures correct output from create_latlng_grid."""

        these_latitudes_deg, these_longitudes_deg = (
            general_utils.create_latlng_grid(
                min_latitude_deg_n=MIN_GRID_LATITUDE_DEG_N,
                max_latitude_deg_n=MAX_GRID_LATITUDE_DEG_N,
                latitude_spacing_deg=LATITUDE_SPACING_DEG,
                min_longitude_deg_e=MIN_GRID_LONGITUDE_DEG_E,
                max_longitude_deg_e=MAX_GRID_LONGITUDE_DEG_E,
                longitude_spacing_deg=LONGITUDE_SPACING_DEG
            )
        )

        self.assertTrue(numpy.allclose(
            these_latitudes_deg, GRID_POINT_LATITUDES_DEG, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg, GRID_POINT_LONGITUDES_DEG, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
