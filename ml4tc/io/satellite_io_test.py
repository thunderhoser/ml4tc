"""Unit tests for satellite_io.py."""

import unittest
from ml4tc.io import satellite_io

DIRECTORY_NAME = 'foo'
CYCLONE_ID_STRING = '1998AL05'
SATELLITE_FILE_NAME_UNZIPPED = 'foo/cira_satellite_1998AL05.nc'
SATELLITE_FILE_NAME_ZIPPED = 'foo/cira_satellite_1998AL05.nc.gz'


class SatelliteIoTests(unittest.TestCase):
    """Each method is a unit test for satellite_io.py."""

    def test_find_file_zipped_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped file but will allow unzipped file.
        """

        this_file_name = satellite_io.find_file(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME_UNZIPPED)

    def test_find_file_zipped_no_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped file and will *not* allow unzipped
        file.
        """

        this_file_name = satellite_io.find_file(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            prefer_zipped=True, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME_ZIPPED)

    def test_find_file_unzipped_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped file but will allow zipped file.
        """

        this_file_name = satellite_io.find_file(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME_ZIPPED)

    def test_find_file_unzipped_no_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped file and will *not* allow zipped
        file.
        """

        this_file_name = satellite_io.find_file(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME_UNZIPPED)

    def test_file_name_to_cyclone_id_zipped(self):
        """Ensures correct output from file_name_to_cyclone_id.

        In this case, using name of zipped file.
        """

        this_id_string = satellite_io.file_name_to_cyclone_id(
            SATELLITE_FILE_NAME_ZIPPED
        )

        self.assertTrue(this_id_string == CYCLONE_ID_STRING)

    def test_file_name_to_cyclone_id_unzipped(self):
        """Ensures correct output from file_name_to_cyclone_id.

        In this case, using name of unzipped file.
        """

        this_id_string = satellite_io.file_name_to_cyclone_id(
            SATELLITE_FILE_NAME_UNZIPPED
        )

        self.assertTrue(this_id_string == CYCLONE_ID_STRING)


if __name__ == '__main__':
    unittest.main()
