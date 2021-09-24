"""Unit tests for satellite_utils.py."""

import unittest
import numpy
from ml4tc.utils import satellite_utils

TOLERANCE = 1e-6

# The following constants are used to test _find_storm_center_px_space.
GRID_LATITUDES_DEG_N = numpy.array(
    [-10, -8, -6, -4, -2, 0, 3, 6, 9, 12, 20], dtype=float
)
GRID_LONGITUDES_DEG_E = numpy.array(
    [350, 355, 0, 5, 10, 15, 25, 35, 45, 55, 65, 75], dtype=float
)

FIRST_STORM_LATITUDE_DEG_N = 6.9
FIRST_STORM_LONGITUDE_DEG_E = 354.3
FIRST_STORM_ROW = 7.5
FIRST_STORM_COLUMN = 0.5

SECOND_STORM_LATITUDE_DEG_N = 16.7
SECOND_STORM_LONGITUDE_DEG_E = 26.3
SECOND_STORM_ROW = 9.5
SECOND_STORM_COLUMN = 6.5

THIRD_STORM_LATITUDE_DEG_N = 11.9
THIRD_STORM_LONGITUDE_DEG_E = 35.5
THIRD_STORM_ROW = 8.5
THIRD_STORM_COLUMN = 7.5

FOURTH_STORM_LATITUDE_DEG_N = -1.5
FOURTH_STORM_LONGITUDE_DEG_E = 51.4
FOURTH_STORM_ROW = 4.5
FOURTH_STORM_COLUMN = 8.5

FIFTH_STORM_LATITUDE_DEG_N = 15.1
FIFTH_STORM_LONGITUDE_DEG_E = 7.6
FIFTH_STORM_ROW = 9.5
FIFTH_STORM_COLUMN = 3.5

# The following constants are used to test _crop_image_around_storm_center.
UNCROPPED_DATA_MATRIX = numpy.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
    [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
    [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
    [97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
    [109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
    [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
], dtype=float)

NUM_CROPPED_ROWS = 4
NUM_CROPPED_COLUMNS = 6

FIRST_CROPPED_DATA_MATRIX = numpy.array([
    [73, 73, 73, 74, 75, 76],
    [85, 85, 85, 86, 87, 88],
    [97, 97, 97, 98, 99, 100],
    [109, 109, 109, 110, 111, 112]
], dtype=float)

FIRST_CROPPED_LATITUDES_DEG_N = numpy.array([3, 6, 9, 12], dtype=float)
FIRST_CROPPED_LONGITUDES_DEG_E = numpy.array(
    [340, 345, 350, 355, 0, 5], dtype=float
)

SECOND_CROPPED_DATA_MATRIX = numpy.array([
    [101, 102, 103, 104, 105, 106],
    [113, 114, 115, 116, 117, 118],
    [125, 126, 127, 128, 129, 130],
    [125, 126, 127, 128, 129, 130]
], dtype=float)

SECOND_CROPPED_LATITUDES_DEG_N = numpy.array([9, 12, 20, 28], dtype=float)
SECOND_CROPPED_LONGITUDES_DEG_E = numpy.array(
    [10, 15, 25, 35, 45, 55], dtype=float
)

THIRD_CROPPED_DATA_MATRIX = numpy.array([
    [90, 91, 92, 93, 94, 95],
    [102, 103, 104, 105, 106, 107],
    [114, 115, 116, 117, 118, 119],
    [126, 127, 128, 129, 130, 131]
], dtype=float)

THIRD_CROPPED_LATITUDES_DEG_N = numpy.array([6, 9, 12, 20], dtype=float)
THIRD_CROPPED_LONGITUDES_DEG_E = numpy.array(
    [15, 25, 35, 45, 55, 65], dtype=float
)

FOURTH_CROPPED_DATA_MATRIX = numpy.array([
    [43, 44, 45, 46, 47, 48],
    [55, 56, 57, 58, 59, 60],
    [67, 68, 69, 70, 71, 72],
    [79, 80, 81, 82, 83, 84]
], dtype=float)

FOURTH_CROPPED_LATITUDES_DEG_N = numpy.array([-4, -2, 0, 3], dtype=float)
FOURTH_CROPPED_LONGITUDES_DEG_E = numpy.array(
    [25, 35, 45, 55, 65, 75], dtype=float
)

FIFTH_CROPPED_DATA_MATRIX = numpy.array([
    [98, 99, 100, 101, 102, 103],
    [110, 111, 112, 113, 114, 115],
    [122, 123, 124, 125, 126, 127],
    [122, 123, 124, 125, 126, 127]
], dtype=float)

FIFTH_CROPPED_LATITUDES_DEG_N = numpy.array([9, 12, 20, 28], dtype=float)
FIFTH_CROPPED_LONGITUDES_DEG_E = numpy.array(
    [355, 0, 5, 10, 15, 25], dtype=float
)

# The following constants are used to test get_cyclone_id and parse_cyclone_id.
YEAR = 1998
BASIN_ID_STRING = 'AL'
CYCLONE_NUMBER = 5
CYCLONE_ID_STRING = '1998AL05'


class SatelliteUtilsTests(unittest.TestCase):
    """Each method is a unit test for satellite_utils.py."""

    def test_find_storm_center_px_space_first(self):
        """Ensures correct output from _find_storm_center_px_space.

        In this case, using first storm center.
        """

        this_row, this_column = satellite_utils._find_storm_center_px_space(
            storm_latitude_deg_n=FIRST_STORM_LATITUDE_DEG_N,
            storm_longitude_deg_e=FIRST_STORM_LONGITUDE_DEG_E,
            grid_latitudes_deg_n=GRID_LATITUDES_DEG_N + 0.,
            grid_longitudes_deg_e=GRID_LONGITUDES_DEG_E + 0.
        )

        self.assertTrue(this_row == FIRST_STORM_ROW)
        self.assertTrue(this_column == FIRST_STORM_COLUMN)

    def test_find_storm_center_px_space_second(self):
        """Ensures correct output from _find_storm_center_px_space.

        In this case, using second storm center.
        """

        this_row, this_column = satellite_utils._find_storm_center_px_space(
            storm_latitude_deg_n=SECOND_STORM_LATITUDE_DEG_N,
            storm_longitude_deg_e=SECOND_STORM_LONGITUDE_DEG_E,
            grid_latitudes_deg_n=GRID_LATITUDES_DEG_N + 0.,
            grid_longitudes_deg_e=GRID_LONGITUDES_DEG_E + 0.
        )

        self.assertTrue(this_row == SECOND_STORM_ROW)
        self.assertTrue(this_column == SECOND_STORM_COLUMN)

    def test_find_storm_center_px_space_third(self):
        """Ensures correct output from _find_storm_center_px_space.

        In this case, using third storm center.
        """

        this_row, this_column = satellite_utils._find_storm_center_px_space(
            storm_latitude_deg_n=THIRD_STORM_LATITUDE_DEG_N,
            storm_longitude_deg_e=THIRD_STORM_LONGITUDE_DEG_E,
            grid_latitudes_deg_n=GRID_LATITUDES_DEG_N + 0.,
            grid_longitudes_deg_e=GRID_LONGITUDES_DEG_E + 0.
        )

        self.assertTrue(this_row == THIRD_STORM_ROW)
        self.assertTrue(this_column == THIRD_STORM_COLUMN)

    def test_find_storm_center_px_space_fourth(self):
        """Ensures correct output from _find_storm_center_px_space.

        In this case, using fourth storm center.
        """

        this_row, this_column = satellite_utils._find_storm_center_px_space(
            storm_latitude_deg_n=FOURTH_STORM_LATITUDE_DEG_N,
            storm_longitude_deg_e=FOURTH_STORM_LONGITUDE_DEG_E,
            grid_latitudes_deg_n=GRID_LATITUDES_DEG_N + 0.,
            grid_longitudes_deg_e=GRID_LONGITUDES_DEG_E + 0.
        )

        self.assertTrue(this_row == FOURTH_STORM_ROW)
        self.assertTrue(this_column == FOURTH_STORM_COLUMN)

    def test_find_storm_center_px_space_fifth(self):
        """Ensures correct output from _find_storm_center_px_space.

        In this case, using fifth storm center.
        """

        this_row, this_column = satellite_utils._find_storm_center_px_space(
            storm_latitude_deg_n=FIFTH_STORM_LATITUDE_DEG_N,
            storm_longitude_deg_e=FIFTH_STORM_LONGITUDE_DEG_E,
            grid_latitudes_deg_n=GRID_LATITUDES_DEG_N + 0.,
            grid_longitudes_deg_e=GRID_LONGITUDES_DEG_E + 0.
        )

        self.assertTrue(this_row == FIFTH_STORM_ROW)
        self.assertTrue(this_column == FIFTH_STORM_COLUMN)

    def test_crop_image_around_storm_center_first(self):
        """Ensures correct output from _crop_image_around_storm_center.

        In this case, using first storm center.
        """

        (
            this_data_matrix, these_latitudes_deg_n, these_longitudes_deg_e
        ) = satellite_utils._crop_image_around_storm_center(
            data_matrix=UNCROPPED_DATA_MATRIX + 0.,
            grid_latitudes_deg_n=GRID_LATITUDES_DEG_N + 0.,
            grid_longitudes_deg_e=GRID_LONGITUDES_DEG_E + 0.,
            storm_row=FIRST_STORM_ROW, storm_column=FIRST_STORM_COLUMN,
            num_cropped_rows=NUM_CROPPED_ROWS,
            num_cropped_columns=NUM_CROPPED_COLUMNS
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, FIRST_CROPPED_DATA_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_latitudes_deg_n, FIRST_CROPPED_LATITUDES_DEG_N, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg_e, FIRST_CROPPED_LONGITUDES_DEG_E,
            atol=TOLERANCE
        ))

    def test_crop_image_around_storm_center_second(self):
        """Ensures correct output from _crop_image_around_storm_center.

        In this case, using second storm center.
        """

        (
            this_data_matrix, these_latitudes_deg_n, these_longitudes_deg_e
        ) = satellite_utils._crop_image_around_storm_center(
            data_matrix=UNCROPPED_DATA_MATRIX + 0.,
            grid_latitudes_deg_n=GRID_LATITUDES_DEG_N + 0.,
            grid_longitudes_deg_e=GRID_LONGITUDES_DEG_E + 0.,
            storm_row=SECOND_STORM_ROW, storm_column=SECOND_STORM_COLUMN,
            num_cropped_rows=NUM_CROPPED_ROWS,
            num_cropped_columns=NUM_CROPPED_COLUMNS
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, SECOND_CROPPED_DATA_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_latitudes_deg_n, SECOND_CROPPED_LATITUDES_DEG_N,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg_e, SECOND_CROPPED_LONGITUDES_DEG_E,
            atol=TOLERANCE
        ))

    def test_crop_image_around_storm_center_third(self):
        """Ensures correct output from _crop_image_around_storm_center.

        In this case, using third storm center.
        """

        (
            this_data_matrix, these_latitudes_deg_n, these_longitudes_deg_e
        ) = satellite_utils._crop_image_around_storm_center(
            data_matrix=UNCROPPED_DATA_MATRIX + 0.,
            grid_latitudes_deg_n=GRID_LATITUDES_DEG_N + 0.,
            grid_longitudes_deg_e=GRID_LONGITUDES_DEG_E + 0.,
            storm_row=THIRD_STORM_ROW, storm_column=THIRD_STORM_COLUMN,
            num_cropped_rows=NUM_CROPPED_ROWS,
            num_cropped_columns=NUM_CROPPED_COLUMNS
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, THIRD_CROPPED_DATA_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_latitudes_deg_n, THIRD_CROPPED_LATITUDES_DEG_N, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg_e, THIRD_CROPPED_LONGITUDES_DEG_E,
            atol=TOLERANCE
        ))

    def test_crop_image_around_storm_center_fourth(self):
        """Ensures correct output from _crop_image_around_storm_center.

        In this case, using fourth storm center.
        """

        (
            this_data_matrix, these_latitudes_deg_n, these_longitudes_deg_e
        ) = satellite_utils._crop_image_around_storm_center(
            data_matrix=UNCROPPED_DATA_MATRIX + 0.,
            grid_latitudes_deg_n=GRID_LATITUDES_DEG_N + 0.,
            grid_longitudes_deg_e=GRID_LONGITUDES_DEG_E + 0.,
            storm_row=FOURTH_STORM_ROW, storm_column=FOURTH_STORM_COLUMN,
            num_cropped_rows=NUM_CROPPED_ROWS,
            num_cropped_columns=NUM_CROPPED_COLUMNS
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, FOURTH_CROPPED_DATA_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_latitudes_deg_n, FOURTH_CROPPED_LATITUDES_DEG_N,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg_e, FOURTH_CROPPED_LONGITUDES_DEG_E,
            atol=TOLERANCE
        ))

    def test_crop_image_around_storm_center_fifth(self):
        """Ensures correct output from _crop_image_around_storm_center.

        In this case, using fifth storm center.
        """

        (
            this_data_matrix, these_latitudes_deg_n, these_longitudes_deg_e
        ) = satellite_utils._crop_image_around_storm_center(
            data_matrix=UNCROPPED_DATA_MATRIX + 0.,
            grid_latitudes_deg_n=GRID_LATITUDES_DEG_N + 0.,
            grid_longitudes_deg_e=GRID_LONGITUDES_DEG_E + 0.,
            storm_row=FIFTH_STORM_ROW, storm_column=FIFTH_STORM_COLUMN,
            num_cropped_rows=NUM_CROPPED_ROWS,
            num_cropped_columns=NUM_CROPPED_COLUMNS
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, FIFTH_CROPPED_DATA_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_latitudes_deg_n, FIFTH_CROPPED_LATITUDES_DEG_N, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg_e, FIFTH_CROPPED_LONGITUDES_DEG_E,
            atol=TOLERANCE
        ))

    def test_get_cyclone_id(self):
        """Ensures correct output from get_cyclone_id."""

        this_id_string = satellite_utils.get_cyclone_id(
            year=YEAR, basin_id_string=BASIN_ID_STRING,
            cyclone_number=CYCLONE_NUMBER
        )

        self.assertTrue(this_id_string == CYCLONE_ID_STRING)

    def test_parse_cyclone_id(self):
        """Ensures correct output from parse_cyclone_id."""

        this_year, this_basin_id_string, this_cyclone_number = (
            satellite_utils.parse_cyclone_id(CYCLONE_ID_STRING)
        )

        self.assertTrue(this_year == YEAR)
        self.assertTrue(this_basin_id_string == BASIN_ID_STRING)
        self.assertTrue(this_cyclone_number == CYCLONE_NUMBER)


if __name__ == '__main__':
    unittest.main()
