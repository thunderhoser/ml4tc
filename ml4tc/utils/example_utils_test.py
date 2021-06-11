"""Unit tests for example_utils.py."""

import copy
import unittest
import numpy
import xarray
from ml4tc.utils import example_utils

TOLERANCE = 1e-6

VALID_TIMES_UNIX_SEC = numpy.array(
    [100, 100, 500, 1000, 1800, 2400, 2700, 3000, 3200, 4000, 4800, 5500], dtype=int
)
GRID_ROW_INDICES = numpy.array([0, 1, 2, 3, 4, 5], dtype=int)
GRID_COLUMN_INDICES = numpy.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
UNGRIDDED_PREDICTOR_NAMES = ['foo', 'bar', 'moo', 'hal']
GRIDDED_PREDICTOR_NAMES = ['brightness_temp_kelvins']

THIS_METADATA_DICT = {
    example_utils.SATELLITE_TIME_DIM: VALID_TIMES_UNIX_SEC,
    example_utils.SATELLITE_GRID_ROW_DIM: GRID_ROW_INDICES,
    example_utils.SATELLITE_GRID_COLUMN_DIM: GRID_COLUMN_INDICES,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM:
        numpy.array(UNGRIDDED_PREDICTOR_NAMES),
    example_utils.SATELLITE_PREDICTOR_GRIDDED_DIM:
        numpy.array(GRIDDED_PREDICTOR_NAMES)
}

PREDICTOR_MATRIX_UNGRIDDED = numpy.random.uniform(
    low=-5., high=10.,
    size=(len(VALID_TIMES_UNIX_SEC), len(UNGRIDDED_PREDICTOR_NAMES))
)
THESE_DIM = (
    example_utils.SATELLITE_TIME_DIM,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
)
THIS_DATA_DICT = {
    example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY:
        (THESE_DIM, PREDICTOR_MATRIX_UNGRIDDED)
}

THESE_DIM = (
    len(VALID_TIMES_UNIX_SEC), len(GRID_ROW_INDICES), len(GRID_COLUMN_INDICES),
    len(UNGRIDDED_PREDICTOR_NAMES)
)
PREDICTOR_MATRIX_GRIDDED = numpy.random.uniform(
    low=200., high=300., size=THESE_DIM
)
THESE_DIM = (
    example_utils.SATELLITE_TIME_DIM,
    example_utils.SATELLITE_GRID_ROW_DIM,
    example_utils.SATELLITE_GRID_COLUMN_DIM,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
)
THIS_DATA_DICT.update({
    example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY:
        (THESE_DIM, PREDICTOR_MATRIX_GRIDDED)
})

EXAMPLE_TABLE_XARRAY_ALL = xarray.Dataset(
    data_vars=THIS_METADATA_DICT, coords=THIS_DATA_DICT
)

FIRST_TIME_INTERVAL_SEC = 1200
THESE_INDICES = numpy.array([0, 3, 5, 8, 10, 11], dtype=int)

THIS_METADATA_DICT = {
    example_utils.SATELLITE_TIME_DIM: VALID_TIMES_UNIX_SEC[THESE_INDICES],
    example_utils.SATELLITE_GRID_ROW_DIM: GRID_ROW_INDICES,
    example_utils.SATELLITE_GRID_COLUMN_DIM: GRID_COLUMN_INDICES,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM:
        numpy.array(UNGRIDDED_PREDICTOR_NAMES),
    example_utils.SATELLITE_PREDICTOR_GRIDDED_DIM:
        numpy.array(GRIDDED_PREDICTOR_NAMES)
}

THIS_PREDICTOR_MATRIX_UNGRIDDED = PREDICTOR_MATRIX_UNGRIDDED[THESE_INDICES, :]
THESE_DIM = (
    example_utils.SATELLITE_TIME_DIM,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
)
THIS_DATA_DICT = {
    example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY:
        (THESE_DIM, THIS_PREDICTOR_MATRIX_UNGRIDDED)
}

THIS_PREDICTOR_MATRIX_GRIDDED = PREDICTOR_MATRIX_GRIDDED[THESE_INDICES, ...]
THESE_DIM = (
    example_utils.SATELLITE_TIME_DIM,
    example_utils.SATELLITE_GRID_ROW_DIM,
    example_utils.SATELLITE_GRID_COLUMN_DIM,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
)
THIS_DATA_DICT.update({
    example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY:
        (THESE_DIM, THIS_PREDICTOR_MATRIX_GRIDDED)
})

FIRST_EXAMPLE_TABLE_XARRAY = xarray.Dataset(
    data_vars=THIS_METADATA_DICT, coords=THIS_DATA_DICT
)

SECOND_TIME_INTERVAL_SEC = 600
THESE_INDICES = numpy.array([0, 2, 3, 4, 5, 7, 8, 9, 10, 11], dtype=int)

THIS_METADATA_DICT = {
    example_utils.SATELLITE_TIME_DIM: VALID_TIMES_UNIX_SEC[THESE_INDICES],
    example_utils.SATELLITE_GRID_ROW_DIM: GRID_ROW_INDICES,
    example_utils.SATELLITE_GRID_COLUMN_DIM: GRID_COLUMN_INDICES,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM:
        numpy.array(UNGRIDDED_PREDICTOR_NAMES),
    example_utils.SATELLITE_PREDICTOR_GRIDDED_DIM:
        numpy.array(GRIDDED_PREDICTOR_NAMES)
}

THIS_PREDICTOR_MATRIX_UNGRIDDED = PREDICTOR_MATRIX_UNGRIDDED[THESE_INDICES, :]
THESE_DIM = (
    example_utils.SATELLITE_TIME_DIM,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
)
THIS_DATA_DICT = {
    example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY:
        (THESE_DIM, THIS_PREDICTOR_MATRIX_UNGRIDDED)
}

THIS_PREDICTOR_MATRIX_GRIDDED = PREDICTOR_MATRIX_GRIDDED[THESE_INDICES, ...]
THESE_DIM = (
    example_utils.SATELLITE_TIME_DIM,
    example_utils.SATELLITE_GRID_ROW_DIM,
    example_utils.SATELLITE_GRID_COLUMN_DIM,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
)
THIS_DATA_DICT.update({
    example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY:
        (THESE_DIM, THIS_PREDICTOR_MATRIX_GRIDDED)
})

SECOND_EXAMPLE_TABLE_XARRAY = xarray.Dataset(
    data_vars=THIS_METADATA_DICT, coords=THIS_DATA_DICT
)

THIRD_TIME_INTERVAL_SEC = 1800
THESE_INDICES = numpy.array([0, 4, 8, 11], dtype=int)

THIS_METADATA_DICT = {
    example_utils.SATELLITE_TIME_DIM: VALID_TIMES_UNIX_SEC[THESE_INDICES],
    example_utils.SATELLITE_GRID_ROW_DIM: GRID_ROW_INDICES,
    example_utils.SATELLITE_GRID_COLUMN_DIM: GRID_COLUMN_INDICES,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM:
        numpy.array(UNGRIDDED_PREDICTOR_NAMES),
    example_utils.SATELLITE_PREDICTOR_GRIDDED_DIM:
        numpy.array(GRIDDED_PREDICTOR_NAMES)
}

THIS_PREDICTOR_MATRIX_UNGRIDDED = PREDICTOR_MATRIX_UNGRIDDED[THESE_INDICES, :]
THESE_DIM = (
    example_utils.SATELLITE_TIME_DIM,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
)
THIS_DATA_DICT = {
    example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY:
        (THESE_DIM, THIS_PREDICTOR_MATRIX_UNGRIDDED)
}

THIS_PREDICTOR_MATRIX_GRIDDED = PREDICTOR_MATRIX_GRIDDED[THESE_INDICES, ...]
THESE_DIM = (
    example_utils.SATELLITE_TIME_DIM,
    example_utils.SATELLITE_GRID_ROW_DIM,
    example_utils.SATELLITE_GRID_COLUMN_DIM,
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
)
THIS_DATA_DICT.update({
    example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY:
        (THESE_DIM, THIS_PREDICTOR_MATRIX_GRIDDED)
})

THIRD_EXAMPLE_TABLE_XARRAY = xarray.Dataset(
    data_vars=THIS_METADATA_DICT, coords=THIS_DATA_DICT
)


class ExampleUtilsTests(unittest.TestCase):
    """Each method is a unit test for example_utils.py."""

    def test_subset_satellite_times_first(self):
        """Ensures correct output from subset_satellite_times.

        In this case, subsetting to first time interval.
        """

        this_example_table_xarray = example_utils.subset_satellite_times(
            example_table_xarray=copy.deepcopy(EXAMPLE_TABLE_XARRAY_ALL),
            time_interval_sec=FIRST_TIME_INTERVAL_SEC
        )
        self.assertTrue(
            this_example_table_xarray.equals(FIRST_EXAMPLE_TABLE_XARRAY)
        )

    def test_subset_satellite_times_second(self):
        """Ensures correct output from subset_satellite_times.

        In this case, subsetting to second time interval.
        """

        this_example_table_xarray = example_utils.subset_satellite_times(
            example_table_xarray=copy.deepcopy(EXAMPLE_TABLE_XARRAY_ALL),
            time_interval_sec=SECOND_TIME_INTERVAL_SEC
        )
        self.assertTrue(
            this_example_table_xarray.equals(SECOND_EXAMPLE_TABLE_XARRAY)
        )

    def test_subset_satellite_times_third(self):
        """Ensures correct output from subset_satellite_times.

        In this case, subsetting to third time interval.
        """

        this_example_table_xarray = example_utils.subset_satellite_times(
            example_table_xarray=copy.deepcopy(EXAMPLE_TABLE_XARRAY_ALL),
            time_interval_sec=THIRD_TIME_INTERVAL_SEC
        )
        self.assertTrue(
            this_example_table_xarray.equals(THIRD_EXAMPLE_TABLE_XARRAY)
        )


if __name__ == '__main__':
    unittest.main()
