"""Unit tests for neural_net.py."""

import unittest
import numpy
import xarray
from ml4tc.utils import example_utils
from ml4tc.machine_learning import neural_net

TOLERANCE = 1e-6

# The following constants are used to test _find_desired_times.
DESIRED_TIMES_UNIX_SEC = numpy.array(
    [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200], dtype=int
)
TOLERANCE_SEC = 1000

FIRST_ACTUAL_TIMES_UNIX_SEC = numpy.array(
    [2700, 5700, 1400, 100, 6300, 7100, 3600], dtype=int
)
FIRST_TIME_INDICES = numpy.array([3, 2, 2, 0, 6, 6, 1, 4, 5], dtype=int)

SECOND_ACTUAL_TIMES_UNIX_SEC = numpy.array(
    [7200, 4600, 5600, 3600, 1800, 0, 2600], dtype=int
)
SECOND_TIME_INDICES = numpy.array([5, 4, 4, 6, 3, 1, 2, 2, 0], dtype=int)

THIRD_ACTUAL_TIMES_UNIX_SEC = numpy.array(
    [3700, 6400, 3900, 1000, 200, 5700, 7100], dtype=int
)
THIRD_TIME_INDICES = numpy.array([4, 3, 3, 0, 0, 2, 5, 1, 6], dtype=int)

FOURTH_ACTUAL_TIMES_UNIX_SEC = numpy.array(
    [4300, 5400, 3400, 400, 6000, 2100, 2600], dtype=int
)
FOURTH_TIME_INDICES = numpy.array(
    [3, 3, 5, 6, 2, 0, 1, 4, neural_net.MISSING_INDEX], dtype=int
)

# The following constants are used to test _interp_missing_times.
NAN = numpy.nan
TIMES_FOR_INTERP_SEC = numpy.array([0, 1, 2, 3, 4, 5], dtype=float)

THIS_MATRIX = numpy.array([
    [NAN, 0, 0, 0],
    [1, -5, 2, NAN],
    [2, NAN, 4, NAN],
    [3, NAN, 6, NAN],
    [NAN, -50, 10, NAN],
    [NAN, -100, 100, NAN]
], dtype=float)

DATA_MATRIX_NON_SPATIAL_BEFORE_INTERP = numpy.stack(
    (THIS_MATRIX, 2 * THIS_MATRIX), axis=0
)

THIS_MATRIX = numpy.array([
    [1, 0, 0, 0],
    [1, -5, 2, 0],
    [2, -20, 4, 0],
    [3, -35, 6, 0],
    [3, -50, 10, 0],
    [3, -100, 100, 0]
], dtype=float)

DATA_MATRIX_NON_SPATIAL_AFTER_INTERP = numpy.stack(
    (THIS_MATRIX, 2 * THIS_MATRIX), axis=0
)

DATA_MATRIX_SPATIAL_BEFORE_INTERP = numpy.expand_dims(
    DATA_MATRIX_NON_SPATIAL_BEFORE_INTERP, axis=1
)
DATA_MATRIX_SPATIAL_BEFORE_INTERP = numpy.expand_dims(
    DATA_MATRIX_SPATIAL_BEFORE_INTERP, axis=1
)
DATA_MATRIX_SPATIAL_BEFORE_INTERP = numpy.repeat(
    DATA_MATRIX_SPATIAL_BEFORE_INTERP, axis=1, repeats=480
)
DATA_MATRIX_SPATIAL_BEFORE_INTERP = numpy.repeat(
    DATA_MATRIX_SPATIAL_BEFORE_INTERP, axis=2, repeats=640
)

DATA_MATRIX_SPATIAL_AFTER_INTERP = numpy.expand_dims(
    DATA_MATRIX_NON_SPATIAL_AFTER_INTERP, axis=1
)
DATA_MATRIX_SPATIAL_AFTER_INTERP = numpy.expand_dims(
    DATA_MATRIX_SPATIAL_AFTER_INTERP, axis=1
)
DATA_MATRIX_SPATIAL_AFTER_INTERP = numpy.repeat(
    DATA_MATRIX_SPATIAL_AFTER_INTERP, axis=1, repeats=480
)
DATA_MATRIX_SPATIAL_AFTER_INTERP = numpy.repeat(
    DATA_MATRIX_SPATIAL_AFTER_INTERP, axis=2, repeats=640
)

# The following constants are used to test _discretize_intensity_change.
INTENSITY_CHANGE_M_S01 = 30.

FIRST_CLASS_CUTOFFS_M_S01 = numpy.array([30.])
FIRST_CLASS_FLAGS = numpy.array([0, 1], dtype=int)

SECOND_CLASS_CUTOFFS_M_S01 = numpy.array([31.])
SECOND_CLASS_FLAGS = numpy.array([1, 0], dtype=int)

THIRD_CLASS_CUTOFFS_M_S01 = numpy.array([-6, 1, 8, 15, 22, 29, 36], dtype=float)
THIRD_CLASS_FLAGS = numpy.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=int)

FOURTH_CLASS_CUTOFFS_M_S01 = numpy.array([-10, 0, 10, 20, 30, 40], dtype=float)
FOURTH_CLASS_FLAGS = numpy.array([0, 0, 0, 0, 0, 1, 0], dtype=int)

# The following constants are used to test _ships_goes_xarray_to_keras and
# _ships_forecasts_xarray_to_keras.
GOES_PREDICTORS_EXAMPLE1_LAG1 = numpy.array([0, 1, 2, 3, 4, 5], dtype=float)
GOES_PREDICTORS_EXAMPLE1_LAG2 = numpy.array(
    [0, -1, -2, -3, -4, -5], dtype=float
)

GOES_PRED_MATRIX_EXAMPLE1 = numpy.stack(
    (GOES_PREDICTORS_EXAMPLE1_LAG1, GOES_PREDICTORS_EXAMPLE1_LAG2), axis=0
)
GOES_PRED_MATRIX_EXAMPLE2 = 2 * GOES_PRED_MATRIX_EXAMPLE1
GOES_PRED_MATRIX_EXAMPLE3 = 3 * GOES_PRED_MATRIX_EXAMPLE1

GOES_PRED_MATRIX_STANDARD = numpy.stack((
    GOES_PRED_MATRIX_EXAMPLE1, GOES_PRED_MATRIX_EXAMPLE2,
    GOES_PRED_MATRIX_EXAMPLE3
), axis=0)

FORECAST_PREDICTORS_EXAMPLE1_HOUR1 = numpy.array([0, 0, 0, 0, 0], dtype=float)
FORECAST_PREDICTORS_EXAMPLE1_HOUR2 = numpy.array([2, 4, 6, 8, 10], dtype=float)
FORECAST_PREDICTORS_EXAMPLE1_HOUR3 = numpy.array([5, 4, 3, 2, 1], dtype=float)
FORECAST_PREDICTORS_EXAMPLE1_HOUR4 = numpy.array(
    [-100, -10, 0, 10, 100], dtype=float
)

FORECAST_PRED_MATRIX_EXAMPLE1 = numpy.stack((
    FORECAST_PREDICTORS_EXAMPLE1_HOUR1, FORECAST_PREDICTORS_EXAMPLE1_HOUR2,
    FORECAST_PREDICTORS_EXAMPLE1_HOUR3, FORECAST_PREDICTORS_EXAMPLE1_HOUR4
), axis=0)

FORECAST_PRED_MATRIX_EXAMPLE2 = 5 * FORECAST_PRED_MATRIX_EXAMPLE1
FORECAST_PRED_MATRIX_EXAMPLE3 = 10 * FORECAST_PRED_MATRIX_EXAMPLE1

FORECAST_PRED_MATRIX_STANDARD = numpy.stack((
    FORECAST_PRED_MATRIX_EXAMPLE1, FORECAST_PRED_MATRIX_EXAMPLE2,
    FORECAST_PRED_MATRIX_EXAMPLE3
), axis=0)

FORECAST_PRED_MATRIX_STANDARD_FIRST12H = (
    FORECAST_PRED_MATRIX_STANDARD[:, 1:3, :]
)

METADATA_DICT = {
    example_utils.SHIPS_LAG_TIME_DIM: numpy.array([3, numpy.nan]),
    example_utils.SHIPS_PREDICTOR_LAGGED_DIM: ['a', 'b', 'c', 'd', 'e', 'f'],
    example_utils.SHIPS_VALID_TIME_DIM:
        numpy.array([0, 21600, 43200], dtype=int),
    example_utils.SHIPS_FORECAST_HOUR_DIM:
        numpy.array([-12, 0, 12, 24], dtype=int),
    example_utils.SHIPS_PREDICTOR_FORECAST_DIM: ['A', 'B', 'C', 'D', 'E']
}

GOES_DIMENSIONS = (
    example_utils.SHIPS_VALID_TIME_DIM, example_utils.SHIPS_LAG_TIME_DIM,
    example_utils.SHIPS_PREDICTOR_LAGGED_DIM
)
FORECAST_DIMENSIONS = (
    example_utils.SHIPS_VALID_TIME_DIM, example_utils.SHIPS_FORECAST_HOUR_DIM,
    example_utils.SHIPS_PREDICTOR_FORECAST_DIM
)
MAIN_DATA_DICT = {
    example_utils.SHIPS_PREDICTORS_LAGGED_KEY:
        (GOES_DIMENSIONS, GOES_PRED_MATRIX_STANDARD),
    example_utils.SHIPS_PREDICTORS_FORECAST_KEY:
        (FORECAST_DIMENSIONS, FORECAST_PRED_MATRIX_STANDARD)
}

EXAMPLE_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=METADATA_DICT
)

GOES_PREDICTOR_INDICES = numpy.linspace(
    0, len(GOES_PREDICTORS_EXAMPLE1_LAG1) - 1,
    num=len(GOES_PREDICTORS_EXAMPLE1_LAG1), dtype=int
)
FORECAST_PREDICTOR_INDICES = numpy.linspace(
    0, len(FORECAST_PREDICTORS_EXAMPLE1_HOUR1) - 1,
    num=len(FORECAST_PREDICTORS_EXAMPLE1_HOUR1), dtype=int
)

MAX_FORECAST_HOUR = 12

GOES_PREDICTORS_EXAMPLE1_DESIRED = numpy.array(
    [0, -1, -2, -3, -4, -5], dtype=float
)
GOES_PREDICTORS_EXAMPLE2_DESIRED = numpy.array(
    [0, -2, -4, -6, -8, -10], dtype=float
)
GOES_PREDICTORS_EXAMPLE3_DESIRED = numpy.array(
    [0, -3, -6, -9, -12, -15], dtype=float
)

FORECAST_PRED_MATRIX_EXAMPLE1_DESIRED = numpy.array([
    [2, 4, 6, 8, 10],
    [5, 4, 3, 2, 1]
], dtype=float)

FORECAST_PRED_MATRIX_EXAMPLE2_DESIRED = numpy.array([
    [10, 20, 30, 40, 50],
    [25, 20, 15, 10, 5]
], dtype=float)

FORECAST_PRED_MATRIX_EXAMPLE3_DESIRED = numpy.array([
    [20, 40, 60, 80, 100],
    [50, 40, 30, 20, 10]
], dtype=float)

# The following constants are used to test combine_ships_predictors and
# separate_ships_predictors.
SHIPS_GOES_PREDICTOR_MATRIX_3D = numpy.stack((
    GOES_PREDICTORS_EXAMPLE1_DESIRED, GOES_PREDICTORS_EXAMPLE2_DESIRED,
    GOES_PREDICTORS_EXAMPLE3_DESIRED
), axis=0)

SHIPS_GOES_PREDICTOR_MATRIX_3D = numpy.expand_dims(
    SHIPS_GOES_PREDICTOR_MATRIX_3D, axis=-2
)
SHIPS_GOES_PREDICTOR_MATRIX_3D = numpy.repeat(
    SHIPS_GOES_PREDICTOR_MATRIX_3D, axis=-2, repeats=2
)
SHIPS_GOES_PREDICTOR_MATRIX_3D[:, 1, :] = (
    10 * SHIPS_GOES_PREDICTOR_MATRIX_3D[:, 0, :]
)

SHIPS_FORECAST_PREDICTOR_MATRIX_3D = numpy.stack((
    FORECAST_PRED_MATRIX_EXAMPLE1_DESIRED,
    FORECAST_PRED_MATRIX_EXAMPLE2_DESIRED,
    FORECAST_PRED_MATRIX_EXAMPLE3_DESIRED
), axis=0)

SHIPS_PREDICTOR_MATRIX_2D = numpy.array([
    [0, -1, -2, -3, -4, -5, 0, -10, -20, -30, -40, -50,
     2, 4, 6, 8, 10, 5, 4, 3, 2, 1],
    [0, -2, -4, -6, -8, -10, 0, -20, -40, -60, -80, -100,
     10, 20, 30, 40, 50, 25, 20, 15, 10, 5],
    [0, -3, -6, -9, -12, -15, 0, -30, -60, -90, -120, -150,
     20, 40, 60, 80, 100, 50, 40, 30, 20, 10]
], dtype=float)

SHIPS_PREDICTOR_MATRIX_2D_GOES_ONLY = numpy.array([
    [0, -1, -2, -3, -4, -5, 0, -10, -20, -30, -40, -50],
    [0, -2, -4, -6, -8, -10, 0, -20, -40, -60, -80, -100],
    [0, -3, -6, -9, -12, -15, 0, -30, -60, -90, -120, -150]
], dtype=float)

SHIPS_PREDICTOR_MATRIX_2D_FCST_ONLY = numpy.array([
    [2, 4, 6, 8, 10, 5, 4, 3, 2, 1],
    [10, 20, 30, 40, 50, 25, 20, 15, 10, 5],
    [20, 40, 60, 80, 100, 50, 40, 30, 20, 10]
], dtype=float)


class NeuralNetTests(unittest.TestCase):
    """Each method is a unit test for neural_net.py."""

    def test_find_desired_times_first(self):
        """Ensures correct output from _find_desired_times.

        In this case, using first set of actual times.
        """

        these_indices = neural_net._find_desired_times(
            all_times_unix_sec=FIRST_ACTUAL_TIMES_UNIX_SEC,
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC,
            tolerance_sec=TOLERANCE_SEC, max_num_missing_times=0
        )

        self.assertTrue(numpy.array_equal(
            these_indices, FIRST_TIME_INDICES
        ))

    def test_find_desired_times_second(self):
        """Ensures correct output from _find_desired_times.

        In this case, using second set of actual times.
        """

        these_indices = neural_net._find_desired_times(
            all_times_unix_sec=SECOND_ACTUAL_TIMES_UNIX_SEC,
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC,
            tolerance_sec=TOLERANCE_SEC, max_num_missing_times=0
        )

        self.assertTrue(numpy.array_equal(
            these_indices, SECOND_TIME_INDICES
        ))

    def test_find_desired_times_third(self):
        """Ensures correct output from _find_desired_times.

        In this case, using third set of actual times.
        """

        these_indices = neural_net._find_desired_times(
            all_times_unix_sec=THIRD_ACTUAL_TIMES_UNIX_SEC,
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC,
            tolerance_sec=TOLERANCE_SEC, max_num_missing_times=0
        )

        self.assertTrue(numpy.array_equal(
            these_indices, THIRD_TIME_INDICES
        ))

    def test_find_desired_times_fourth_allow_missing(self):
        """Ensures correct output from _find_desired_times.

        In this case, using fourth set of actual times and will allow missing
        times.
        """

        these_indices = neural_net._find_desired_times(
            all_times_unix_sec=FOURTH_ACTUAL_TIMES_UNIX_SEC,
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC,
            tolerance_sec=TOLERANCE_SEC, max_num_missing_times=1
        )

        self.assertTrue(numpy.array_equal(
            these_indices, FOURTH_TIME_INDICES
        ))

    def test_find_desired_times_fourth_no_allow_missing(self):
        """Ensures correct output from _find_desired_times.

        In this case, using fourth set of actual times and will *not* allow
        missing times.
        """

        these_indices = neural_net._find_desired_times(
            all_times_unix_sec=FOURTH_ACTUAL_TIMES_UNIX_SEC,
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC,
            tolerance_sec=TOLERANCE_SEC, max_num_missing_times=0
        )

        self.assertTrue(these_indices is None)

    def test_interp_missing_times_non_spatial(self):
        """Ensures correct output from _interp_missing_times.

        In this case, data are non-spatial.
        """

        this_data_matrix = neural_net._interp_missing_times(
            data_matrix=DATA_MATRIX_NON_SPATIAL_BEFORE_INTERP + 0.,
            times_sec=TIMES_FOR_INTERP_SEC
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, DATA_MATRIX_NON_SPATIAL_AFTER_INTERP,
            atol=TOLERANCE
        ))

    def test_interp_missing_times_spatial(self):
        """Ensures correct output from _interp_missing_times.

        In this case, data are spatial.
        """

        this_data_matrix = neural_net._interp_missing_times(
            data_matrix=DATA_MATRIX_SPATIAL_BEFORE_INTERP + 0.,
            times_sec=TIMES_FOR_INTERP_SEC
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, DATA_MATRIX_SPATIAL_AFTER_INTERP,
            atol=TOLERANCE
        ))

    def test_discretize_intensity_change_first(self):
        """Ensures correct output from _discretize_intensity_change.

        In this case, using first set of cutoffs.
        """

        these_flags = neural_net._discretize_intensity_change(
            intensity_change_m_s01=INTENSITY_CHANGE_M_S01,
            class_cutoffs_m_s01=FIRST_CLASS_CUTOFFS_M_S01
        )
        self.assertTrue(numpy.array_equal(these_flags, FIRST_CLASS_FLAGS))

    def test_discretize_intensity_change_second(self):
        """Ensures correct output from _discretize_intensity_change.

        In this case, using second set of cutoffs.
        """

        these_flags = neural_net._discretize_intensity_change(
            intensity_change_m_s01=INTENSITY_CHANGE_M_S01,
            class_cutoffs_m_s01=SECOND_CLASS_CUTOFFS_M_S01
        )
        self.assertTrue(numpy.array_equal(these_flags, SECOND_CLASS_FLAGS))

    def test_discretize_intensity_change_third(self):
        """Ensures correct output from _discretize_intensity_change.

        In this case, using third set of cutoffs.
        """

        these_flags = neural_net._discretize_intensity_change(
            intensity_change_m_s01=INTENSITY_CHANGE_M_S01,
            class_cutoffs_m_s01=THIRD_CLASS_CUTOFFS_M_S01
        )
        self.assertTrue(numpy.array_equal(these_flags, THIRD_CLASS_FLAGS))

    def test_discretize_intensity_change_fourth(self):
        """Ensures correct output from _discretize_intensity_change.

        In this case, using fourth set of cutoffs.
        """

        these_flags = neural_net._discretize_intensity_change(
            intensity_change_m_s01=INTENSITY_CHANGE_M_S01,
            class_cutoffs_m_s01=FOURTH_CLASS_CUTOFFS_M_S01
        )
        self.assertTrue(numpy.array_equal(these_flags, FOURTH_CLASS_FLAGS))

    def test_ships_goes_xarray_to_keras_example1(self):
        """Ensures correct output from _ships_goes_xarray_to_keras.

        In this case, extracting values from first example (init time).
        """

        these_predictor_values = neural_net._ships_goes_xarray_to_keras(
            example_table_xarray=EXAMPLE_TABLE_XARRAY,
            init_time_index=0, predictor_indices=GOES_PREDICTOR_INDICES
        )
        self.assertTrue(numpy.allclose(
            these_predictor_values, GOES_PREDICTORS_EXAMPLE1_DESIRED,
            atol=TOLERANCE
        ))

    def test_ships_goes_xarray_to_keras_example2(self):
        """Ensures correct output from _ships_goes_xarray_to_keras.

        In this case, extracting values from second example (init time).
        """

        these_predictor_values = neural_net._ships_goes_xarray_to_keras(
            example_table_xarray=EXAMPLE_TABLE_XARRAY,
            init_time_index=1, predictor_indices=GOES_PREDICTOR_INDICES
        )
        self.assertTrue(numpy.allclose(
            these_predictor_values, GOES_PREDICTORS_EXAMPLE2_DESIRED,
            atol=TOLERANCE
        ))

    def test_ships_goes_xarray_to_keras_example3(self):
        """Ensures correct output from _ships_goes_xarray_to_keras.

        In this case, extracting values from third example (init time).
        """

        these_predictor_values = neural_net._ships_goes_xarray_to_keras(
            example_table_xarray=EXAMPLE_TABLE_XARRAY,
            init_time_index=2, predictor_indices=GOES_PREDICTOR_INDICES
        )
        self.assertTrue(numpy.allclose(
            these_predictor_values, GOES_PREDICTORS_EXAMPLE3_DESIRED,
            atol=TOLERANCE
        ))

    def test_ships_forecasts_xarray_to_keras_example1(self):
        """Ensures correct output from _ships_forecasts_xarray_to_keras.

        In this case, extracting values from first example (init time).
        """

        this_predictor_matrix = neural_net._ships_forecasts_xarray_to_keras(
            example_table_xarray=EXAMPLE_TABLE_XARRAY,
            init_time_index=0, predictor_indices=FORECAST_PREDICTOR_INDICES,
            max_forecast_hour=MAX_FORECAST_HOUR, test_mode=True
        )
        self.assertTrue(numpy.allclose(
            this_predictor_matrix, FORECAST_PRED_MATRIX_EXAMPLE1_DESIRED,
            atol=TOLERANCE
        ))

    def test_ships_forecasts_xarray_to_keras_example2(self):
        """Ensures correct output from _ships_forecasts_xarray_to_keras.

        In this case, extracting values from second example (init time).
        """

        this_predictor_matrix = neural_net._ships_forecasts_xarray_to_keras(
            example_table_xarray=EXAMPLE_TABLE_XARRAY,
            init_time_index=1, predictor_indices=FORECAST_PREDICTOR_INDICES,
            max_forecast_hour=MAX_FORECAST_HOUR, test_mode=True
        )
        self.assertTrue(numpy.allclose(
            this_predictor_matrix, FORECAST_PRED_MATRIX_EXAMPLE2_DESIRED,
            atol=TOLERANCE
        ))

    def test_ships_forecasts_xarray_to_keras_example3(self):
        """Ensures correct output from _ships_forecasts_xarray_to_keras.

        In this case, extracting values from third example (init time).
        """

        this_predictor_matrix = neural_net._ships_forecasts_xarray_to_keras(
            example_table_xarray=EXAMPLE_TABLE_XARRAY,
            init_time_index=2, predictor_indices=FORECAST_PREDICTOR_INDICES,
            max_forecast_hour=MAX_FORECAST_HOUR, test_mode=True
        )
        self.assertTrue(numpy.allclose(
            this_predictor_matrix, FORECAST_PRED_MATRIX_EXAMPLE3_DESIRED,
            atol=TOLERANCE
        ))

    def test_combine_ships_predictors_both(self):
        """Ensures correct output from combine_ships_predictors.

        In this case, using both GOES and forecast predictors.
        """

        this_predictor_matrix = neural_net.combine_ships_predictors(
            ships_goes_predictor_matrix_3d=SHIPS_GOES_PREDICTOR_MATRIX_3D,
            ships_forecast_predictor_matrix_3d=
            SHIPS_FORECAST_PREDICTOR_MATRIX_3D
        )
        self.assertTrue(numpy.allclose(
            this_predictor_matrix, SHIPS_PREDICTOR_MATRIX_2D, atol=TOLERANCE
        ))

    def test_separate_ships_predictors_both(self):
        """Ensures correct output from separate_ships_predictors.

        In this case, using both GOES and forecast predictors.
        """

        (
            this_goes_predictor_matrix, this_forecast_predictor_matrix
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=SHIPS_PREDICTOR_MATRIX_2D,
            num_goes_predictors=SHIPS_GOES_PREDICTOR_MATRIX_3D.shape[2],
            num_forecast_predictors=SHIPS_FORECAST_PREDICTOR_MATRIX_3D.shape[2],
            num_forecast_hours=SHIPS_FORECAST_PREDICTOR_MATRIX_3D.shape[1]
        )

        self.assertTrue(numpy.allclose(
            this_goes_predictor_matrix, SHIPS_GOES_PREDICTOR_MATRIX_3D,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_forecast_predictor_matrix, SHIPS_FORECAST_PREDICTOR_MATRIX_3D,
            atol=TOLERANCE
        ))

    def test_combine_ships_predictors_goes_only(self):
        """Ensures correct output from combine_ships_predictors.

        In this case, using only GOES predictors.
        """

        this_predictor_matrix = neural_net.combine_ships_predictors(
            ships_goes_predictor_matrix_3d=SHIPS_GOES_PREDICTOR_MATRIX_3D,
            ships_forecast_predictor_matrix_3d=
            numpy.full((SHIPS_GOES_PREDICTOR_MATRIX_3D.shape[0], 0, 0), 0.)
        )
        self.assertTrue(numpy.allclose(
            this_predictor_matrix, SHIPS_PREDICTOR_MATRIX_2D_GOES_ONLY,
            atol=TOLERANCE
        ))

    def test_separate_ships_predictors_goes_only(self):
        """Ensures correct output from separate_ships_predictors.

        In this case, using only GOES predictors.
        """

        (
            this_goes_predictor_matrix, this_forecast_predictor_matrix
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=SHIPS_PREDICTOR_MATRIX_2D_GOES_ONLY,
            num_goes_predictors=SHIPS_GOES_PREDICTOR_MATRIX_3D.shape[2],
            num_forecast_predictors=0,
            num_forecast_hours=0
        )

        self.assertTrue(numpy.allclose(
            this_goes_predictor_matrix, SHIPS_GOES_PREDICTOR_MATRIX_3D,
            atol=TOLERANCE
        ))
        self.assertTrue(this_forecast_predictor_matrix.size == 0)

    def test_combine_ships_predictors_forecast_only(self):
        """Ensures correct output from combine_ships_predictors.

        In this case, using only forecast predictors.
        """

        this_predictor_matrix = neural_net.combine_ships_predictors(
            ships_goes_predictor_matrix_3d=
            numpy.full((SHIPS_FORECAST_PREDICTOR_MATRIX_3D.shape[0], 0, 0), 0.),
            ships_forecast_predictor_matrix_3d=
            SHIPS_FORECAST_PREDICTOR_MATRIX_3D
        )
        self.assertTrue(numpy.allclose(
            this_predictor_matrix, SHIPS_PREDICTOR_MATRIX_2D_FCST_ONLY,
            atol=TOLERANCE
        ))

    def test_separate_ships_predictors_forecast_only(self):
        """Ensures correct output from separate_ships_predictors.

        In this case, using only forecast predictors.
        """

        (
            this_goes_predictor_matrix, this_forecast_predictor_matrix
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=SHIPS_PREDICTOR_MATRIX_2D_FCST_ONLY,
            num_goes_predictors=0,
            num_forecast_predictors=SHIPS_FORECAST_PREDICTOR_MATRIX_3D.shape[2],
            num_forecast_hours=SHIPS_FORECAST_PREDICTOR_MATRIX_3D.shape[1]
        )

        self.assertTrue(this_goes_predictor_matrix.size == 0)
        self.assertTrue(numpy.allclose(
            this_forecast_predictor_matrix, SHIPS_FORECAST_PREDICTOR_MATRIX_3D,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
