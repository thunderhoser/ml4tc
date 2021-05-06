"""Unit tests for satellite_io.py."""

import unittest
from ml4tc.io import satellite_io

DIRECTORY_NAME = 'foo'
CYCLONE_ID_STRING = '1998AL05'
SATELLITE_FILE_NAME = 'foo/cira_satellite_1998AL05.nc'


class SatelliteIoTests(unittest.TestCase):
    """Each method is a unit test for satellite_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = satellite_io.find_file(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME)

    def test_file_name_to_cyclone_id(self):
        """Ensures correct output from file_name_to_cyclone_id."""

        this_id_string = satellite_io.file_name_to_cyclone_id(
            SATELLITE_FILE_NAME
        )

        self.assertTrue(this_id_string == CYCLONE_ID_STRING)


if __name__ == '__main__':
    unittest.main()
