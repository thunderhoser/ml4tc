"""Unit tests for new_cira_satellite_io.py."""

import unittest
import numpy
from ml4tc.io import new_cira_satellite_io

TOLERANCE = 1e-6

TOP_DIRECTORY_NAME = 'foo'
CYCLONE_ID_STRING = '2021EP12'

VALID_TIMES_UNIX_SEC = numpy.array([
    1628514000, 1628514300, 1628514600, 1628514900,
    1628515200, 1628515500, 1628515800, 1628516100,
    1628516400, 1628516700, 1628517000, 1628517300,
    1628517600
], dtype=int)

VALID_TIME_STRINGS = [
    '20210809130000', '20210809130005', '20210809130100', '20210809130105',
    '20210809130200', '20210809130205', '20210809130300', '20210809130305',
    '20210809130400', '20210809130405', '20210809130500', '20210809130505',
    '20210809140000'
]
SATELLITE_FILE_NAMES = [
    'foo/2021/2021EP12/TC-IRAR_v02r02_EP122021_s{0:s}_e{0:s}.nc'.format(t)
    for t in VALID_TIME_STRINGS
]


class NewCiraSatelliteIoTests(unittest.TestCase):
    """Each method is a unit test for new_cira_satellite_io.py."""

    def test_unix_time_to_file_name(self):
        """Ensures correct output from _unix_time_to_file_name."""

        for i in range(len(VALID_TIMES_UNIX_SEC)):
            this_time_string = new_cira_satellite_io._unix_time_to_file_name(
                VALID_TIMES_UNIX_SEC[i]
            )
            self.assertTrue(this_time_string == VALID_TIME_STRINGS[i])

    def test_file_name_time_to_unix(self):
        """Ensures correct output from _file_name_time_to_unix."""

        for i in range(len(VALID_TIME_STRINGS)):
            this_time_unix_sec = new_cira_satellite_io._file_name_time_to_unix(
                VALID_TIME_STRINGS[i]
            )
            self.assertTrue(this_time_unix_sec == VALID_TIMES_UNIX_SEC[i])

    def test_find_file(self):
        """Ensures correct output from find_file."""

        for i in range(len(VALID_TIMES_UNIX_SEC)):
            this_file_name = new_cira_satellite_io.find_file(
                top_directory_name=TOP_DIRECTORY_NAME,
                cyclone_id_string=CYCLONE_ID_STRING,
                valid_time_unix_sec=VALID_TIMES_UNIX_SEC[i],
                raise_error_if_missing=False
            )

            self.assertTrue(this_file_name == SATELLITE_FILE_NAMES[i])

    def test_file_name_to_cyclone_id(self):
        """Ensures correct output from file_name_to_cyclone_id."""

        for i in range(len(SATELLITE_FILE_NAMES)):
            this_id_string = new_cira_satellite_io.file_name_to_cyclone_id(
                SATELLITE_FILE_NAMES[i]
            )
            self.assertTrue(this_id_string == CYCLONE_ID_STRING)

    def test_file_name_to_time(self):
        """Ensures correct output from file_name_to_time."""

        for i in range(len(SATELLITE_FILE_NAMES)):
            this_time_unix_sec = new_cira_satellite_io.file_name_to_time(
                SATELLITE_FILE_NAMES[i]
            )
            self.assertTrue(this_time_unix_sec == VALID_TIMES_UNIX_SEC[i])


if __name__ == '__main__':
    unittest.main()
