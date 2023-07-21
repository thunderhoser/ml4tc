"""Unit tests for new_cira_satellite_io.py."""

import unittest
import numpy
from ml4tc.io import new_cira_satellite_io

TOLERANCE = 1e-6

TOP_DIRECTORY_NAME = 'foo'
CYCLONE_ID_STRING = '2021EP12'

VALID_TIMES_UNIX_SEC = numpy.array([
    1628514000, 1628514060, 1628514120, 1628514180, 1628514240,
    1628514300, 1628514360, 1628514420, 1628514480, 1628514540,
    1628514600, 1628514660, 1628514720, 1628514780, 1628514840,
    1628514900, 1628514960, 1628515020, 1628515080, 1628515140,
    1628515200, 1628515260, 1628515320, 1628515380, 1628515440,
    1628515500, 1628515560, 1628515620, 1628515680, 1628515740,
    1628515800, 1628515860, 1628515920, 1628515980, 1628516040,
    1628516100, 1628516160, 1628516220, 1628516280, 1628516340,
    1628516400, 1628516460, 1628516520, 1628516580, 1628516640,
    1628516700, 1628516760, 1628516820, 1628516880, 1628516940,
    1628517000, 1628517060, 1628517120, 1628517180, 1628517240,
    1628517300, 1628517360, 1628517420, 1628517480, 1628517540,
    1628517600
], dtype=int)

VALID_TIME_STRINGS = [
    '20210809130000', '20210809130001', '20210809130002', '20210809130003', '20210809130004',
    '20210809130005', '20210809130006', '20210809130007', '20210809130008', '20210809130009',
    '20210809130100', '20210809130101', '20210809130102', '20210809130103', '20210809130104',
    '20210809130105', '20210809130106', '20210809130107', '20210809130108', '20210809130109',
    '20210809130200', '20210809130201', '20210809130202', '20210809130203', '20210809130204',
    '20210809130205', '20210809130206', '20210809130207', '20210809130208', '20210809130209',
    '20210809130300', '20210809130301', '20210809130302', '20210809130303', '20210809130304',
    '20210809130305', '20210809130306', '20210809130307', '20210809130308', '20210809130309',
    '20210809130400', '20210809130401', '20210809130402', '20210809130403', '20210809130404',
    '20210809130405', '20210809130406', '20210809130407', '20210809130408', '20210809130409',
    '20210809130500', '20210809130501', '20210809130502', '20210809130503', '20210809130504',
    '20210809130505', '20210809130506', '20210809130507', '20210809130508', '20210809130509',
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
