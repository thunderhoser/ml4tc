"""Unit tests for cira_satellite_io.py."""

import unittest
from ml4tc.io import cira_satellite_io

TOLERANCE = 1e-6

TOP_DIRECTORY_NAME = 'foo'
CYCLONE_ID_STRING = '1998AL05'
VALID_TIME_UNIX_SEC = 907411500
SATELLITE_FILE_NAME = 'foo/1998/1998AL05/AL0598_19982761045M.nc'


class CiraSatelliteIoTests(unittest.TestCase):
    """Each method is a unit test for cira_satellite_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = cira_satellite_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME)

    def test_file_name_to_cyclone_id(self):
        """Ensures correct output from file_name_to_cyclone_id."""

        this_id_string = cira_satellite_io.file_name_to_cyclone_id(
            SATELLITE_FILE_NAME
        )

        self.assertTrue(this_id_string == CYCLONE_ID_STRING)


if __name__ == '__main__':
    unittest.main()
