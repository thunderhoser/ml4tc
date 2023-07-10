"""Unit tests for permutation.py."""

import unittest
import numpy
from ml4tc.utils import satellite_utils
from ml4tc.machine_learning import neural_net
from ml4tc.machine_learning import permutation

TOLERANCE = 1e-6

NUM_GOES_PREDICTORS = 17
NUM_GOES_LAG_TIMES = 3
NUM_FORECAST_PREDICTORS = 125
NUM_FORECAST_HOURS = 2
NUM_UNGRIDDED_SAT_PREDICTORS = 16

NUM_SCALAR_SHIPS_PREDICTORS = (
    NUM_GOES_PREDICTORS * NUM_GOES_LAG_TIMES +
    NUM_FORECAST_PREDICTORS * NUM_FORECAST_HOURS
)

# The following constants are used to test _permute_values and
# _depermute_values.
PREDICTOR_MATRIX_GRIDDED_SAT = numpy.random.normal(
    loc=0., scale=1., size=(10, 640, 480, 4, 1)
)
PREDICTOR_MATRIX_UNGRIDDED_SAT = numpy.random.normal(
    loc=0., scale=1., size=(10, 4, NUM_UNGRIDDED_SAT_PREDICTORS)
)
PREDICTOR_MATRIX_SHIPS = numpy.random.normal(
    loc=0., scale=1., size=(10, NUM_SCALAR_SHIPS_PREDICTORS)
)

TRAINING_OPTION_DICT = {
    neural_net.SATELLITE_LAG_TIMES_KEY:
        numpy.array([1800, 1200, 600, 0], dtype=int),
    neural_net.SHIPS_GOES_PREDICTORS_KEY: ['a'] * NUM_GOES_PREDICTORS,
    neural_net.SHIPS_FORECAST_PREDICTORS_KEY: ['b'] * NUM_FORECAST_PREDICTORS,
    neural_net.SATELLITE_PREDICTORS_KEY: ['c'] * NUM_UNGRIDDED_SAT_PREDICTORS,
    neural_net.SHIPS_MAX_FORECAST_HOUR_KEY: 6
}
MODEL_METADATA_DICT = {
    neural_net.TRAINING_OPTIONS_KEY: TRAINING_OPTION_DICT
}

# The following constants are used to test _predictor_indices_to_metadata.
FIRST_RESULT_DICT = {
    permutation.PERMUTED_MATRICES_KEY: numpy.array([0, 1, 2], dtype=int),
    permutation.PERMUTED_VARIABLES_KEY: numpy.array([0, 10, 15], dtype=int)
}
FIRST_PREDICTOR_NAMES = [satellite_utils.BRIGHTNESS_TEMPERATURE_KEY, 'c', 'a']

SECOND_RESULT_DICT = {
    permutation.PERMUTED_MATRICES_KEY: numpy.array([0, 1, 2], dtype=int),
    permutation.PERMUTED_VARIABLES_KEY: numpy.array([0, 10, 100], dtype=int)
}
SECOND_PREDICTOR_NAMES = [satellite_utils.BRIGHTNESS_TEMPERATURE_KEY, 'c', 'b']


class PermutationTests(unittest.TestCase):
    """Each method is a unit test for permutation.py."""

    def test_permute_values_gridded_sat_all_lags(self):
        """Ensures correct output from _permute_values.

        In this case, the predictor matrix contains gridded satellite data and
        permutation is over all model lag times.
        """

        new_predictor_matrix, permuted_value_matrix = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX_GRIDDED_SAT + 0.,
                predictor_type_enum=0,
                variable_index=0, model_lag_time_index=None,
                permuted_value_matrix=None
            )
        )

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_GRIDDED_SAT, atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=new_predictor_matrix + 0.,
            predictor_type_enum=0,
            variable_index=0, model_lag_time_index=None,
            permuted_value_matrix=permuted_value_matrix
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_permute_values_gridded_sat_one_lag(self):
        """Ensures correct output from _permute_values.

        In this case, the predictor matrix contains gridded satellite data and
        permutation is over one model lag time.
        """

        new_predictor_matrix, permuted_value_matrix = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX_GRIDDED_SAT + 0.,
                predictor_type_enum=0,
                variable_index=0, model_lag_time_index=1,
                permuted_value_matrix=None
            )
        )

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_GRIDDED_SAT, atol=TOLERANCE
        ))

        num_lag_times = new_predictor_matrix.shape[-2]
        indices_to_compare = numpy.arange(num_lag_times) != 1

        self.assertTrue(numpy.allclose(
            new_predictor_matrix[..., indices_to_compare, :],
            PREDICTOR_MATRIX_GRIDDED_SAT[..., indices_to_compare, :],
            atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=new_predictor_matrix + 0.,
            predictor_type_enum=0,
            variable_index=0, model_lag_time_index=1,
            permuted_value_matrix=permuted_value_matrix
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_permute_values_ungridded_sat_all_lags(self):
        """Ensures correct output from _permute_values.

        In this case, the predictor matrix contains ungridded satellite data and
        permutation is over all model lag times.
        """

        new_predictor_matrix, permuted_value_matrix = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX_UNGRIDDED_SAT + 0.,
                predictor_type_enum=1,
                variable_index=0, model_lag_time_index=None,
                permuted_value_matrix=None
            )
        )

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_UNGRIDDED_SAT, atol=TOLERANCE
        ))

        num_variables = new_predictor_matrix.shape[-1]
        indices_to_compare = numpy.arange(num_variables) != 0

        self.assertTrue(numpy.allclose(
            new_predictor_matrix[..., indices_to_compare],
            PREDICTOR_MATRIX_UNGRIDDED_SAT[..., indices_to_compare],
            atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=new_predictor_matrix + 0.,
            predictor_type_enum=1,
            variable_index=0, model_lag_time_index=None,
            permuted_value_matrix=permuted_value_matrix
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_permute_values_ungridded_sat_one_lag(self):
        """Ensures correct output from _permute_values.

        In this case, the predictor matrix contains ungridded satellite data and
        permutation is over one model lag time.
        """

        new_predictor_matrix, permuted_value_matrix = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX_UNGRIDDED_SAT + 0.,
                predictor_type_enum=1,
                variable_index=0, model_lag_time_index=1,
                permuted_value_matrix=None
            )
        )

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_UNGRIDDED_SAT, atol=TOLERANCE
        ))

        num_variables = new_predictor_matrix.shape[-1]
        second_indices = numpy.arange(num_variables) != 0

        num_lag_times = new_predictor_matrix.shape[-2]
        first_indices = numpy.arange(num_lag_times) != 1

        self.assertTrue(numpy.allclose(
            new_predictor_matrix[..., second_indices][..., first_indices, :],
            PREDICTOR_MATRIX_UNGRIDDED_SAT[
                ..., second_indices
            ][..., first_indices, :],
            atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=new_predictor_matrix + 0.,
            predictor_type_enum=1,
            variable_index=0, model_lag_time_index=1,
            permuted_value_matrix=permuted_value_matrix
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_permute_values_ships_all_lags_goes(self):
        """Ensures correct output from _permute_values.

        In this case, the predictor matrix contains SHIPS data; permutation is
        over all model lag times; and a GOES predictor is permuted.
        """

        new_predictor_matrix, permuted_value_matrix = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX_SHIPS + 0.,
                predictor_type_enum=2,
                variable_index=0, model_metadata_dict=MODEL_METADATA_DICT,
                model_lag_time_index=None, permuted_value_matrix=None
            )
        )

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_SHIPS, atol=TOLERANCE
        ))

        (
            new_goes_matrix_3d, new_forecast_matrix_3d
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=new_predictor_matrix,
            num_goes_predictors=NUM_GOES_PREDICTORS,
            num_forecast_predictors=NUM_FORECAST_PREDICTORS,
            num_forecast_hours=NUM_FORECAST_HOURS
        )

        (
            orig_goes_matrix_3d, orig_forecast_matrix_3d
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=PREDICTOR_MATRIX_SHIPS,
            num_goes_predictors=NUM_GOES_PREDICTORS,
            num_forecast_predictors=NUM_FORECAST_PREDICTORS,
            num_forecast_hours=NUM_FORECAST_HOURS
        )

        self.assertTrue(numpy.allclose(
            new_forecast_matrix_3d, orig_forecast_matrix_3d, atol=TOLERANCE
        ))

        num_variables = new_goes_matrix_3d.shape[-1]
        indices_to_compare = numpy.arange(num_variables) != 0

        self.assertTrue(numpy.allclose(
            new_goes_matrix_3d[..., indices_to_compare],
            orig_goes_matrix_3d[..., indices_to_compare],
            atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=new_predictor_matrix + 0.,
            predictor_type_enum=2,
            variable_index=0, model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=None,
            permuted_value_matrix=permuted_value_matrix
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_permute_values_ships_one_lag_goes(self):
        """Ensures correct output from _permute_values.

        In this case, the predictor matrix contains SHIPS data; permutation is
        over one model lag time; and a GOES predictor is permuted.
        """

        new_predictor_matrix, permuted_value_matrix = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX_SHIPS + 0.,
                predictor_type_enum=2,
                variable_index=0, model_metadata_dict=MODEL_METADATA_DICT,
                model_lag_time_index=1, permuted_value_matrix=None
            )
        )

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_SHIPS, atol=TOLERANCE
        ))

        (
            new_goes_matrix_3d, new_forecast_matrix_3d
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=new_predictor_matrix,
            num_goes_predictors=NUM_GOES_PREDICTORS,
            num_forecast_predictors=NUM_FORECAST_PREDICTORS,
            num_forecast_hours=NUM_FORECAST_HOURS
        )

        (
            orig_goes_matrix_3d, orig_forecast_matrix_3d
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=PREDICTOR_MATRIX_SHIPS,
            num_goes_predictors=NUM_GOES_PREDICTORS,
            num_forecast_predictors=NUM_FORECAST_PREDICTORS,
            num_forecast_hours=NUM_FORECAST_HOURS
        )

        self.assertTrue(numpy.allclose(
            new_forecast_matrix_3d, orig_forecast_matrix_3d, atol=TOLERANCE
        ))

        num_variables = new_goes_matrix_3d.shape[-1]
        second_indices = numpy.arange(num_variables) != 0

        num_lag_times = new_goes_matrix_3d.shape[-2]
        first_indices = numpy.arange(num_lag_times) != 1

        self.assertTrue(numpy.allclose(
            new_goes_matrix_3d[..., second_indices][..., first_indices, :],
            orig_goes_matrix_3d[..., second_indices][..., first_indices, :],
            atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=new_predictor_matrix + 0.,
            predictor_type_enum=2,
            variable_index=0, model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=1,
            permuted_value_matrix=permuted_value_matrix
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_permute_values_ships_all_lags_forecast(self):
        """Ensures correct output from _permute_values.

        In this case, the predictor matrix contains SHIPS data; permutation is
        over all model lag times; and a forecast predictor is permuted.
        """

        new_predictor_matrix, permuted_value_matrix = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX_SHIPS + 0.,
                predictor_type_enum=2,
                variable_index=NUM_GOES_PREDICTORS,
                model_metadata_dict=MODEL_METADATA_DICT,
                model_lag_time_index=None, permuted_value_matrix=None
            )
        )

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_SHIPS, atol=TOLERANCE
        ))

        (
            new_goes_matrix_3d, new_forecast_matrix_3d
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=new_predictor_matrix,
            num_goes_predictors=NUM_GOES_PREDICTORS,
            num_forecast_predictors=NUM_FORECAST_PREDICTORS,
            num_forecast_hours=NUM_FORECAST_HOURS
        )

        (
            orig_goes_matrix_3d, orig_forecast_matrix_3d
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=PREDICTOR_MATRIX_SHIPS,
            num_goes_predictors=NUM_GOES_PREDICTORS,
            num_forecast_predictors=NUM_FORECAST_PREDICTORS,
            num_forecast_hours=NUM_FORECAST_HOURS
        )

        self.assertTrue(numpy.allclose(
            new_goes_matrix_3d, orig_goes_matrix_3d, atol=TOLERANCE
        ))

        num_variables = new_forecast_matrix_3d.shape[-1]
        indices_to_compare = numpy.arange(num_variables) != 0

        self.assertTrue(numpy.allclose(
            new_forecast_matrix_3d[..., indices_to_compare],
            orig_forecast_matrix_3d[..., indices_to_compare],
            atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=new_predictor_matrix + 0.,
            predictor_type_enum=2,
            variable_index=NUM_GOES_PREDICTORS,
            model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=None,
            permuted_value_matrix=permuted_value_matrix
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_permute_values_ships_one_lag_forecast(self):
        """Ensures correct output from _permute_values.

        In this case, the predictor matrix contains SHIPS data; permutation is
        over one model lag time; and a forecast predictor is permuted.
        """

        new_predictor_matrix, permuted_value_matrix = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX_SHIPS + 0.,
                predictor_type_enum=2,
                variable_index=NUM_GOES_PREDICTORS,
                model_metadata_dict=MODEL_METADATA_DICT,
                model_lag_time_index=1, permuted_value_matrix=None
            )
        )

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_SHIPS, atol=TOLERANCE
        ))

        (
            new_goes_matrix_3d, new_forecast_matrix_3d
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=new_predictor_matrix,
            num_goes_predictors=NUM_GOES_PREDICTORS,
            num_forecast_predictors=NUM_FORECAST_PREDICTORS,
            num_forecast_hours=NUM_FORECAST_HOURS
        )

        (
            orig_goes_matrix_3d, orig_forecast_matrix_3d
        ) = neural_net.separate_ships_predictors(
            ships_predictor_matrix_2d=PREDICTOR_MATRIX_SHIPS,
            num_goes_predictors=NUM_GOES_PREDICTORS,
            num_forecast_predictors=NUM_FORECAST_PREDICTORS,
            num_forecast_hours=NUM_FORECAST_HOURS
        )

        self.assertTrue(numpy.allclose(
            new_goes_matrix_3d, orig_goes_matrix_3d, atol=TOLERANCE
        ))

        num_variables = new_forecast_matrix_3d.shape[-1]
        second_indices = numpy.arange(num_variables) != 0

        num_lag_times = new_forecast_matrix_3d.shape[-2]
        first_indices = numpy.arange(num_lag_times) != 1

        self.assertTrue(numpy.allclose(
            new_forecast_matrix_3d[..., second_indices][..., first_indices, :],
            orig_forecast_matrix_3d[..., second_indices][..., first_indices, :],
            atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=new_predictor_matrix + 0.,
            predictor_type_enum=2,
            variable_index=NUM_GOES_PREDICTORS,
            model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=1,
            permuted_value_matrix=permuted_value_matrix
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_depermute_values_gridded_sat_all_lags(self):
        """Ensures correct output from _depermute_values.

        In this case, the predictor matrix contains gridded satellite data and
        permutation is over all model lag times.
        """

        new_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_GRIDDED_SAT + 0.,
            predictor_type_enum=0,
            variable_index=0, model_lag_time_index=None,
            permuted_value_matrix=None
        )[0]

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            clean_predictor_matrix=PREDICTOR_MATRIX_GRIDDED_SAT,
            predictor_type_enum=0,
            variable_index=0, model_lag_time_index=None
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_GRIDDED_SAT, atol=TOLERANCE
        ))

    def test_depermute_values_gridded_sat_one_lag(self):
        """Ensures correct output from _depermute_values.

        In this case, the predictor matrix contains gridded satellite data and
        permutation is over one model lag time.
        """

        new_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_GRIDDED_SAT + 0.,
            predictor_type_enum=0,
            variable_index=0, model_lag_time_index=1,
            permuted_value_matrix=None
        )[0]

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            clean_predictor_matrix=PREDICTOR_MATRIX_GRIDDED_SAT,
            predictor_type_enum=0,
            variable_index=0, model_lag_time_index=1
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_GRIDDED_SAT, atol=TOLERANCE
        ))

    def test_depermute_values_ungridded_sat_all_lags(self):
        """Ensures correct output from _depermute_values.

        In this case, the predictor matrix contains ungridded satellite data and
        permutation is over all model lag times.
        """

        new_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_UNGRIDDED_SAT + 0.,
            predictor_type_enum=1,
            variable_index=0, model_lag_time_index=None,
            permuted_value_matrix=None
        )[0]

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            clean_predictor_matrix=PREDICTOR_MATRIX_UNGRIDDED_SAT,
            predictor_type_enum=1,
            variable_index=0, model_lag_time_index=None
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_UNGRIDDED_SAT, atol=TOLERANCE
        ))

    def test_depermute_values_ungridded_sat_one_lag(self):
        """Ensures correct output from _depermute_values.

        In this case, the predictor matrix contains ungridded satellite data and
        permutation is over one model lag time.
        """

        new_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_UNGRIDDED_SAT + 0.,
            predictor_type_enum=1,
            variable_index=0, model_lag_time_index=1,
            permuted_value_matrix=None
        )[0]

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            clean_predictor_matrix=PREDICTOR_MATRIX_UNGRIDDED_SAT,
            predictor_type_enum=1,
            variable_index=0, model_lag_time_index=1
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_UNGRIDDED_SAT, atol=TOLERANCE
        ))

    def test_depermute_values_ships_all_lags_goes(self):
        """Ensures correct output from _depermute_values.

        In this case, the predictor matrix contains SHIPS data; permutation is
        over all model lag times; and a GOES predictor is permuted.
        """

        new_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_SHIPS + 0.,
            predictor_type_enum=2,
            variable_index=0, model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=None, permuted_value_matrix=None
        )[0]

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            clean_predictor_matrix=PREDICTOR_MATRIX_SHIPS,
            predictor_type_enum=2,
            variable_index=0, model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=None
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_SHIPS, atol=TOLERANCE
        ))

    def test_depermute_values_ships_one_lag_goes(self):
        """Ensures correct output from _depermute_values.

        In this case, the predictor matrix contains SHIPS data; permutation is
        over one model lag time; and a GOES predictor is permuted.
        """

        new_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_SHIPS + 0.,
            predictor_type_enum=2,
            variable_index=0, model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=1, permuted_value_matrix=None
        )[0]

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            clean_predictor_matrix=PREDICTOR_MATRIX_SHIPS,
            predictor_type_enum=2,
            variable_index=0, model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=1
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_SHIPS, atol=TOLERANCE
        ))

    def test_depermute_values_ships_all_lags_forecast(self):
        """Ensures correct output from _depermute_values.

        In this case, the predictor matrix contains SHIPS data; permutation is
        over all model lag times; and a forecast predictor is permuted.
        """

        new_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_SHIPS + 0.,
            predictor_type_enum=2,
            variable_index=NUM_GOES_PREDICTORS,
            model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=None, permuted_value_matrix=None
        )[0]

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            clean_predictor_matrix=PREDICTOR_MATRIX_SHIPS,
            predictor_type_enum=2,
            variable_index=NUM_GOES_PREDICTORS,
            model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=None
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_SHIPS, atol=TOLERANCE
        ))

    def test_depermute_values_ships_one_lag_forecast(self):
        """Ensures correct output from _depermute_values.

        In this case, the predictor matrix contains SHIPS data; permutation is
        over one model lag time; and a forecast predictor is permuted.
        """

        new_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_SHIPS + 0.,
            predictor_type_enum=2,
            variable_index=NUM_GOES_PREDICTORS,
            model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=1, permuted_value_matrix=None
        )[0]

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            clean_predictor_matrix=PREDICTOR_MATRIX_SHIPS,
            predictor_type_enum=2,
            variable_index=NUM_GOES_PREDICTORS,
            model_metadata_dict=MODEL_METADATA_DICT,
            model_lag_time_index=1
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_SHIPS, atol=TOLERANCE
        ))

    def test_predictor_indices_to_metadata_first(self):
        """Ensures correct output from _predictor_indices_to_metadata.

        In this case, using first set of results.
        """

        these_predictor_names = permutation._predictor_indices_to_metadata(
            model_metadata_dict=MODEL_METADATA_DICT,
            one_step_result_dict=FIRST_RESULT_DICT
        )
        self.assertTrue(these_predictor_names == FIRST_PREDICTOR_NAMES)

    def test_predictor_indices_to_metadata_second(self):
        """Ensures correct output from _predictor_indices_to_metadata.

        In this case, using first set of results.
        """

        these_predictor_names = permutation._predictor_indices_to_metadata(
            model_metadata_dict=MODEL_METADATA_DICT,
            one_step_result_dict=SECOND_RESULT_DICT
        )
        self.assertTrue(these_predictor_names == SECOND_PREDICTOR_NAMES)


if __name__ == '__main__':
    unittest.main()
