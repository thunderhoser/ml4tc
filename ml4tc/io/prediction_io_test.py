"""Unit tests for prediction_io.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4tc.io import prediction_io

TOLERANCE = 1e-6

# The following constants are used to test subset*.
TARGET_CLASSES = numpy.array([0, 1, 2, 2, 1, 0, 0, 2, 1, 1, 0, 2], dtype=int)
FORECAST_PROB_MATRIX = numpy.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1. / 3, 1. / 3, 1. / 3],
    [0.5, 0.25, 0.25],
    [0.4, 0.4, 0.2],
    [0.6, 0.2, 0.2],
    [0.2, 0.1, 0.7],
    [0.8, 0.1, 0.1],
    [0.7, 0.15, 0.15],
    [0.2, 0.5, 0.3],
    [0.9, 0.1, 0]
])

MODEL_FILE_NAME = 'foo'
CYCLONE_ID_STRINGS = [
    '2005AL01', '2005WP01', '2005CP01', '2005EP01', '2005IO01', '2005SH01',
    '2005AL02', '2005WP02', '2005CP02', '2005EP02', '2005IO02', '2005SH02'
]
INIT_TIME_STRINGS = [
    '2005-01-01', '2005-02-02', '2005-03-03', '2005-04-04',
    '2005-05-05', '2005-06-06', '2005-07-07', '2005-08-08',
    '2005-09-09', '2005-10-10', '2005-11-11', '2005-12-12'
]
LATITUDES_DEG_N = numpy.full(12, 53.5)
LONGITUDES_DEG_E = numpy.full(12, 246.5)
INIT_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d') for t in INIT_TIME_STRINGS
], dtype=int)

FULL_PREDICTION_DICT = {
    prediction_io.TARGET_MATRIX_KEY: numpy.expand_dims(TARGET_CLASSES, axis=-1),
    prediction_io.PROBABILITY_MATRIX_KEY: FORECAST_PROB_MATRIX + 0.,
    prediction_io.CYCLONE_IDS_KEY: copy.deepcopy(CYCLONE_ID_STRINGS),
    prediction_io.STORM_LATITUDES_KEY: LATITUDES_DEG_N + 0.,
    prediction_io.STORM_LONGITUDES_KEY: LONGITUDES_DEG_E + 0.,
    prediction_io.INIT_TIMES_KEY: INIT_TIMES_UNIX_SEC + 0,
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME
}

DESIRED_MONTH = 11
THESE_INDICES = numpy.array([10], dtype=int)

PREDICTION_DICT_SUBSET_BY_MONTH = {
    prediction_io.TARGET_MATRIX_KEY:
        numpy.expand_dims(TARGET_CLASSES[THESE_INDICES], axis=-1),
    prediction_io.PROBABILITY_MATRIX_KEY:
        FORECAST_PROB_MATRIX[THESE_INDICES, ...],
    prediction_io.CYCLONE_IDS_KEY:
        [CYCLONE_ID_STRINGS[k] for k in THESE_INDICES],
    prediction_io.STORM_LATITUDES_KEY: LATITUDES_DEG_N[THESE_INDICES],
    prediction_io.STORM_LONGITUDES_KEY: LONGITUDES_DEG_E[THESE_INDICES],
    prediction_io.INIT_TIMES_KEY: INIT_TIMES_UNIX_SEC[THESE_INDICES],
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME
}

DESIRED_BASIN_ID_STRING = 'SL'
THESE_INDICES = numpy.array([], dtype=int)

PREDICTION_DICT_SUBSET_BY_BASIN = {
    prediction_io.TARGET_MATRIX_KEY:
        numpy.expand_dims(TARGET_CLASSES[THESE_INDICES], axis=-1),
    prediction_io.PROBABILITY_MATRIX_KEY:
        FORECAST_PROB_MATRIX[THESE_INDICES, ...],
    prediction_io.CYCLONE_IDS_KEY:
        [CYCLONE_ID_STRINGS[k] for k in THESE_INDICES],
    prediction_io.STORM_LATITUDES_KEY: LATITUDES_DEG_N[THESE_INDICES],
    prediction_io.STORM_LONGITUDES_KEY: LONGITUDES_DEG_E[THESE_INDICES],
    prediction_io.INIT_TIMES_KEY: INIT_TIMES_UNIX_SEC[THESE_INDICES],
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME
}

# The following constants are used to test find_file and file_name_to_metadata.
DIRECTORY_NAME = 'foo'
MONTH = 12
BASIN_ID_STRING = 'SH'
GRID_ROW = 0
GRID_COLUMN = 666

FILE_NAME_DEFAULT = 'foo/predictions.nc'
FILE_NAME_MONTHLY = 'foo/predictions_month=12.nc'
FILE_NAME_BASIN_SPECIFIC = 'foo/predictions_basin-id-string=SH.nc'
FILE_NAME_SPATIAL = (
    'foo/grid-row=000/predictions_grid-row=000_grid-column=666.nc'
)

METADATA_DICT_DEFAULT = {
    prediction_io.MONTH_KEY: None,
    prediction_io.BASIN_ID_KEY: None,
    prediction_io.GRID_ROW_KEY: None,
    prediction_io.GRID_COLUMN_KEY: None
}

METADATA_DICT_MONTHLY = {
    prediction_io.MONTH_KEY: 12,
    prediction_io.BASIN_ID_KEY: None,
    prediction_io.GRID_ROW_KEY: None,
    prediction_io.GRID_COLUMN_KEY: None
}

METADATA_DICT_BASIN_SPECIFIC = {
    prediction_io.MONTH_KEY: None,
    prediction_io.BASIN_ID_KEY: 'SH',
    prediction_io.GRID_ROW_KEY: None,
    prediction_io.GRID_COLUMN_KEY: None
}

METADATA_DICT_SPATIAL = {
    prediction_io.MONTH_KEY: None,
    prediction_io.BASIN_ID_KEY: None,
    prediction_io.GRID_ROW_KEY: 0,
    prediction_io.GRID_COLUMN_KEY: 666
}


def _compare_prediction_dicts(first_prediction_dict, second_prediction_dict):
    """Compares two dictionaries with predicted and actual target values.

    :param first_prediction_dict: See doc for `prediction_io.read_file`.
    :param second_prediction_dict: Same.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_prediction_dict.keys())
    second_keys = list(first_prediction_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    if not numpy.allclose(
            first_prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
            second_prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
            atol=TOLERANCE
    ):
        return False

    these_keys = [
        prediction_io.TARGET_MATRIX_KEY, prediction_io.INIT_TIMES_KEY
    ]
    for this_key in these_keys:
        if not numpy.array_equal(
                first_prediction_dict[this_key],
                second_prediction_dict[this_key]
        ):
            return False

    these_keys = [prediction_io.CYCLONE_IDS_KEY, prediction_io.MODEL_FILE_KEY]
    for this_key in these_keys:
        if (
                first_prediction_dict[this_key] !=
                second_prediction_dict[this_key]
        ):
            return False

    return True


class PredictionIoTests(unittest.TestCase):
    """Each method is a unit test for prediction_io.py."""

    def test_subset_by_month(self):
        """Ensures correct output from subset_by_month."""

        this_prediction_dict = prediction_io.subset_by_month(
            prediction_dict=copy.deepcopy(FULL_PREDICTION_DICT),
            desired_month=DESIRED_MONTH
        )
        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, PREDICTION_DICT_SUBSET_BY_MONTH
        ))

    def test_subset_by_basin(self):
        """Ensures correct output from subset_by_basin."""

        this_prediction_dict = prediction_io.subset_by_basin(
            prediction_dict=copy.deepcopy(FULL_PREDICTION_DICT),
            desired_basin_id_string=DESIRED_BASIN_ID_STRING
        )
        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, PREDICTION_DICT_SUBSET_BY_BASIN
        ))

    def test_find_file_default(self):
        """Ensures correct output from find_file.

        In this case, using default metadata (no splitting by time or space).
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME, raise_error_if_missing=False
        )
        self.assertTrue(this_file_name == FILE_NAME_DEFAULT)

    def test_find_file_monthly(self):
        """Ensures correct output from find_file.

        In this case, splitting by month.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME, month=MONTH,
            raise_error_if_missing=False
        )
        self.assertTrue(this_file_name == FILE_NAME_MONTHLY)

    def test_find_file_basin_specific(self):
        """Ensures correct output from find_file.

        In this case, splitting by basin.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME, basin_id_string=BASIN_ID_STRING,
            raise_error_if_missing=False
        )
        self.assertTrue(this_file_name == FILE_NAME_BASIN_SPECIFIC)

    def test_find_file_spatial(self):
        """Ensures correct output from find_file.

        In this case, splitting by space.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME, grid_row=GRID_ROW,
            grid_column=GRID_COLUMN, raise_error_if_missing=False
        )
        self.assertTrue(this_file_name == FILE_NAME_SPATIAL)

    def test_file_name_to_metadata_default(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, using default metadata (no splitting by time or space).
        """

        this_metadata_dict = prediction_io.file_name_to_metadata(
            FILE_NAME_DEFAULT
        )
        self.assertTrue(this_metadata_dict == METADATA_DICT_DEFAULT)

    def test_file_name_to_metadata_monthly(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, splitting by month.
        """

        this_metadata_dict = prediction_io.file_name_to_metadata(
            FILE_NAME_MONTHLY
        )
        self.assertTrue(this_metadata_dict == METADATA_DICT_MONTHLY)

    def test_file_name_to_metadata_basin_specific(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, splitting by basin.
        """

        this_metadata_dict = prediction_io.file_name_to_metadata(
            FILE_NAME_BASIN_SPECIFIC
        )
        self.assertTrue(this_metadata_dict == METADATA_DICT_BASIN_SPECIFIC)

    def test_file_name_to_metadata_spatial(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, splitting by space.
        """

        this_metadata_dict = prediction_io.file_name_to_metadata(
            FILE_NAME_SPATIAL
        )
        self.assertTrue(this_metadata_dict == METADATA_DICT_SPATIAL)


if __name__ == '__main__':
    unittest.main()
