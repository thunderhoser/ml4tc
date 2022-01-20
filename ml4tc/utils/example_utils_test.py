"""Unit tests for example_utils.py."""

import copy
import unittest
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from ml4tc.io import ships_io
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

# The following constants are used to test _create_merged_sst_variable and
# _create_merged_ohc_variable.
THESE_TIME_STRINGS = [
    '2012-12-01', '2013-01-01', '2012-11-01', '2013-02-01'
]
THESE_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d')
    for t in THESE_TIME_STRINGS
], dtype=int)

THESE_FORECAST_HOURS = numpy.array([0, 6, 12, 18, 24, 30], dtype=int)
THESE_PREDICTOR_NAMES = [
    ships_io.NCODA_SST_KEY,
    ships_io.REYNOLDS_SST_DAILY_KEY,
    ships_io.REYNOLDS_SST_KEY,
    ships_io.CLIMO_SST_KEY,
    ships_io.NCODA_OHC_26C_KEY,
    ships_io.SATELLITE_OHC_KEY,
    ships_io.OHC_FROM_SST_AND_CLIMO_KEY,
    ships_io.NCODA_OHC_26C_CLIMO_KEY,
    ships_io.CLIMO_OHC_KEY
]

THIS_METADATA_DICT = {
    example_utils.SHIPS_VALID_TIME_DIM: THESE_TIMES_UNIX_SEC,
    example_utils.SHIPS_FORECAST_HOUR_DIM: THESE_FORECAST_HOURS,
    example_utils.SHIPS_PREDICTOR_FORECAST_DIM:
        numpy.array(THESE_PREDICTOR_NAMES)
}

NAN = numpy.nan

FIRST_SST_MATRIX_KELVINS = numpy.array([
    [275, NAN, 275.2, 275.3, NAN, NAN],
    [290, NAN, 290, 291, 290, 291],
    [300, 300.5, NAN, 301.5, 302, 302.5],
    [273.15, 273.16, 273.17, 273.18, 273.19, 273.2]
])

SECOND_SST_MATRIX_KELVINS = 1. + numpy.array([
    [275, NAN, NAN, 275.3, 275.5, 275.7],
    [NAN, 291, 290, 291, 290, NAN],
    [300, 300.5, 301, 301.5, 302, 302.5],
    [273.15, 273.16, NAN, 273.18, 273.19, 273.2]
])

THIRD_SST_MATRIX_KELVINS = -1. + numpy.array([
    [275, 275.1, 275.2, 275.3, 275.5, 275.7],
    [290, 291, 290, 291, 290, 291],
    [NAN, 300.5, 301, 301.5, 302, 302.5],
    [273.15, NAN, 273.17, 273.18, NAN, 273.2]
])

FOURTH_SST_MATRIX_KELVINS = 0.5 + numpy.array([
    [NAN, 275.1, NAN, 275.3, 275.5, 275.7],
    [290, 291, 290, 291, 290, 291],
    [300, 300.5, 301, 301.5, 302, 302.5],
    [NAN, 273.16, NAN, 273.18, 273.19, 273.2]
])

THIS_SST_MATRIX_KELVINS = numpy.stack((
    FIRST_SST_MATRIX_KELVINS, SECOND_SST_MATRIX_KELVINS,
    THIRD_SST_MATRIX_KELVINS, FOURTH_SST_MATRIX_KELVINS
), axis=-1)

MERGED_SST_MATRIX_KELVINS = numpy.array([
    [276, 274.1, 274.2, 276.3, 276.5, 276.7],
    [290, 292, 290, 291, 290, 291],
    [301, 301.5, 302, 302.5, 303, 303.5],
    [273.15, 273.16, 273.17, 273.18, 273.19, 273.2]
])

FIRST_OHC_MATRIX_J_M02 = numpy.array([
    [NAN, 105, 110, 115, 120, 125],
    [250, 250, NAN, NAN, NAN, 250],
    [NAN, 390, 380, 370, NAN, 350],
    [NAN, NAN, NAN, 350, NAN, 300]
], dtype=float)

SECOND_OHC_MATRIX_J_M02 = 10. + numpy.array([
    [100, 105, 110, NAN, 120, NAN],
    [250, 250, 250, 250, 250, 250],
    [400, 390, 380, 370, 360, NAN],
    [300, NAN, 400, 350, NAN, 300]
], dtype=float)

THIRD_OHC_MATRIX_J_M02 = -10. + numpy.array([
    [NAN, 105, 110, 115, NAN, 125],
    [250, 250, 250, 250, NAN, NAN],
    [400, NAN, 380, 370, 360, 350],
    [300, 350, 400, 350, 300, NAN]
], dtype=float)

FOURTH_OHC_MATRIX_J_M02 = 50. + numpy.array([
    [100, NAN, NAN, 115, NAN, 125],
    [NAN, NAN, NAN, NAN, NAN, 250],
    [400, NAN, 380, NAN, 360, 350],
    [NAN, 350, 400, NAN, 300, 300]
], dtype=float)

FIFTH_OHC_MATRIX_J_M02 = -50. + numpy.array([
    [NAN, 105, 110, 115, 120, 125],
    [250, 250, NAN, 250, 250, 250],
    [NAN, 390, 380, 370, NAN, 350],
    [300, 350, 400, NAN, 300, 300]
], dtype=float)

THIS_OHC_MATRIX_J_M02 = 1e7 * numpy.stack((
    FIRST_OHC_MATRIX_J_M02, SECOND_OHC_MATRIX_J_M02, THIRD_OHC_MATRIX_J_M02,
    FOURTH_OHC_MATRIX_J_M02, FIFTH_OHC_MATRIX_J_M02
), axis=-1)

MERGED_OHC_MATRIX_J_M02 = 1e7 * numpy.array([
    [110, 105, 110, 115, 120, 125],
    [250, 250, 260, 260, 260, 250],
    [410, 390, 380, 370, 370, 350],
    [310, 340, 410, 350, 290, 300]
], dtype=float)

THIS_PREDICTOR_MATRIX = numpy.concatenate(
    (THIS_SST_MATRIX_KELVINS, THIS_OHC_MATRIX_J_M02), axis=-1
)

THESE_DIM = (
    example_utils.SHIPS_VALID_TIME_DIM,
    example_utils.SHIPS_FORECAST_HOUR_DIM,
    example_utils.SHIPS_PREDICTOR_FORECAST_DIM
)
THIS_DATA_DICT = {
    example_utils.SHIPS_PREDICTORS_FORECAST_KEY:
        (THESE_DIM, THIS_PREDICTOR_MATRIX)
}
ORIG_EXAMPLE_TABLE_XARRAY = xarray.Dataset(
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

    def test_create_merged_sst_variable(self):
        """Ensures correct output from _create_merged_sst_variable."""

        this_example_table_xarray = example_utils._create_merged_sst_variable(
            copy.deepcopy(ORIG_EXAMPLE_TABLE_XARRAY)
        )

        all_predictor_names = this_example_table_xarray.coords[
            example_utils.SHIPS_PREDICTOR_FORECAST_DIM
        ].values.tolist()

        merged_sst_index = all_predictor_names.index(ships_io.MERGED_SST_KEY)

        this_merged_sst_matrix_kelvins = this_example_table_xarray[
            example_utils.SHIPS_PREDICTORS_FORECAST_KEY
        ].values[..., merged_sst_index]

        self.assertTrue(numpy.allclose(
            this_merged_sst_matrix_kelvins, MERGED_SST_MATRIX_KELVINS,
            atol=TOLERANCE
        ))

    def test_create_merged_ohc_variable(self):
        """Ensures correct output from _create_merged_ohc_variable."""

        this_example_table_xarray = example_utils._create_merged_ohc_variable(
            copy.deepcopy(ORIG_EXAMPLE_TABLE_XARRAY)
        )

        all_predictor_names = this_example_table_xarray.coords[
            example_utils.SHIPS_PREDICTOR_FORECAST_DIM
        ].values.tolist()

        merged_ohc_index = all_predictor_names.index(ships_io.MERGED_OHC_KEY)

        this_merged_ohc_matrix_j_m02 = this_example_table_xarray[
            example_utils.SHIPS_PREDICTORS_FORECAST_KEY
        ].values[..., merged_ohc_index]

        self.assertTrue(numpy.allclose(
            this_merged_ohc_matrix_j_m02, MERGED_OHC_MATRIX_J_M02,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
