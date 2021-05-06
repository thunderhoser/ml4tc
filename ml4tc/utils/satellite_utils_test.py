"""Unit tests for satellite_utils.py."""

import unittest
from ml4tc.utils import satellite_utils

YEAR = 1998
BASIN_ID_STRING = 'AL'
CYCLONE_NUMBER = 5
CYCLONE_ID_STRING = '1998AL05'


class SatelliteUtilsTests(unittest.TestCase):
    """Each method is a unit test for satellite_utils.py."""

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
