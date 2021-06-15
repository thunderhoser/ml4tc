"""Methods for training and applying neural nets."""

import os
import sys
import copy
import random
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import example_io
import example_utils
import satellite_utils

# TODO(thunderhoser): Need unit tests for private methods.

MINUTES_TO_SECONDS = 60
HOURS_TO_SECONDS = 3600
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

DEFAULT_CLASS_CUTOFFS_KT = numpy.array(
    [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25], dtype=float
)
DEFAULT_CLASS_CUTOFFS_M_S01 = DEFAULT_CLASS_CUTOFFS_KT * KT_TO_METRES_PER_SECOND


def _find_desired_times(
        all_times_unix_sec, desired_times_unix_sec, tolerance_sec):
    """Finds desired times.

    L = number of desired times

    :param all_times_unix_sec: 1-D numpy array with all times available.
    :param desired_times_unix_sec: length-L numpy array of desired times.
    :param tolerance_sec: Tolerance.
    :return: desired_indices: length-L numpy array of indices into
        `all_times_unix_sec`.  If not all desired times were found, the output
        will be None, instead.
    """

    desired_indices = []

    for t in desired_times_unix_sec:
        differences_sec = numpy.absolute(all_times_unix_sec - t)
        min_index = numpy.argmin(differences_sec)

        if differences_sec[min_index] > tolerance_sec:
            return None

        desired_indices.append(min_index)

    return numpy.array(desired_indices, dtype=int)


def _discretize_intensity_change(intensity_change_m_s01, class_cutoffs_m_s01):
    """Discretizes intensity change into class.

    K = number of classes

    :param intensity_change_m_s01: Intensity change (metres per second).
    :param class_cutoffs_m_s01: numpy array (length K - 1) of class cutoffs.
    :return: class_flags: length-K numpy array of flags.  One value will be 1,
        and the others will be 0.
    """

    class_index = numpy.searchsorted(
        class_cutoffs_m_s01, intensity_change_m_s01, side='right'
    )
    class_flags = numpy.full(len(class_cutoffs_m_s01) + 1, 0, dtype=int)
    class_flags[class_index] = 1

    return class_flags


def _read_one_example_file(
        example_file_name, num_examples_desired, lead_time_hours,
        satellite_lag_times_minutes, ships_lag_times_hours,
        satellite_predictor_names, ships_predictor_names_lagged,
        ships_predictor_names_forecast, class_cutoffs_m_s01):
    """Reads one example file for generator.

    :param example_file_name: Path to input file.  Will be read by
        `example_io.read_file`.
    :param num_examples_desired: Number of examples desired.
    :param lead_time_hours: See doc for `input_generator`.
    :param satellite_lag_times_minutes: Same.
    :param ships_lag_times_hours: Same.
    :param satellite_predictor_names: Same.
    :param ships_predictor_names_lagged: Same.
    :param ships_predictor_names_forecast: Same.
    :param class_cutoffs_m_s01: Same.
    :return: predictor_matrices: Same.
    :return: target_matrix: Same.
    """

    print('Reading data from: "{0:s}"...'.format(example_file_name))
    xt = example_io.read_file(example_file_name)

    init_times_unix_sec = xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values
    numpy.random.shuffle(init_times_unix_sec)

    satellite_lag_times_sec = satellite_lag_times_minutes * MINUTES_TO_SECONDS
    ships_lag_times_sec = ships_lag_times_hours * HOURS_TO_SECONDS
    lead_time_sec = lead_time_hours * HOURS_TO_SECONDS

    satellite_time_indices_by_example = []
    ships_time_indices_by_example = []
    target_time_indices_by_example = []

    for t in init_times_unix_sec:
        these_satellite_indices = _find_desired_times(
            all_times_unix_sec=
            xt.coords[example_utils.SATELLITE_TIME_DIM].values,
            desired_times_unix_sec=t - satellite_lag_times_sec,
            tolerance_sec=600
        )
        if these_satellite_indices is None:
            continue

        these_ships_indices = _find_desired_times(
            all_times_unix_sec=
            xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values,
            desired_times_unix_sec=t - ships_lag_times_sec,
            tolerance_sec=0
        )
        if these_ships_indices is None:
            continue

        these_target_indices = _find_desired_times(
            all_times_unix_sec=
            xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values,
            desired_times_unix_sec=
            numpy.array([t, t + lead_time_sec], dtype=int),
            tolerance_sec=0
        )
        if these_target_indices is None:
            continue

        satellite_time_indices_by_example.append(
            these_satellite_indices
        )
        ships_time_indices_by_example.append(these_ships_indices)
        target_time_indices_by_example.append(these_target_indices)

        if len(ships_time_indices_by_example) == num_examples_desired:
            break

    num_examples = len(ships_time_indices_by_example)
    num_grid_rows = (
        xt[example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY].values.shape[1]
    )
    num_grid_columns = (
        xt[example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY].values.shape[2]
    )
    these_dim = (
        num_examples, num_grid_rows, num_grid_columns,
        len(satellite_lag_times_sec), 1
    )
    brightness_temp_matrix = numpy.full(these_dim, numpy.nan)

    these_dim = (
        num_examples, len(satellite_lag_times_sec),
        len(satellite_predictor_names)
    )
    satellite_predictor_matrix = numpy.full(these_dim, numpy.nan)

    num_ships_lag_times = len(
        xt.coords[example_utils.SHIPS_LAG_TIME_DIM].values
    )
    num_ships_forecast_hours = len(
        xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values
    )
    num_ships_channels_lagged = (
        num_ships_lag_times * len(ships_predictor_names_lagged)
    )
    num_ships_channels = (
        num_ships_channels_lagged +
        num_ships_forecast_hours * len(ships_predictor_names_forecast)
    )

    these_dim = (num_examples, len(ships_lag_times_sec), num_ships_channels)
    ships_predictor_matrix = numpy.full(these_dim, numpy.nan)
    target_matrix = numpy.full(
        (num_examples, len(DEFAULT_CLASS_CUTOFFS_M_S01)), -1, dtype=int
    )

    all_satellite_predictor_names = (
        xt.coords[
            example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
        ].values.tolist()
    )
    satellite_predictor_indices = numpy.array([
        all_satellite_predictor_names.index(n)
        for n in satellite_predictor_names
    ], dtype=int)

    all_ships_predictor_names_lagged = (
        xt.coords[example_utils.SHIPS_PREDICTOR_LAGGED_DIM].values.tolist()
    )
    ships_predictor_indices_lagged = numpy.array([
        all_ships_predictor_names_lagged.index(n)
        for n in ships_predictor_names_lagged
    ], dtype=int)

    all_ships_predictor_names_forecast = (
        xt.coords[example_utils.SHIPS_PREDICTOR_FORECAST_DIM].values.tolist()
    )
    ships_predictor_indices_forecast = numpy.array([
        all_ships_predictor_names_forecast.index(n)
        for n in ships_predictor_names_forecast
    ], dtype=int)

    for i in range(num_examples):
        for j in range(len(satellite_lag_times_sec)):
            k = satellite_time_indices_by_example[i][j]

            brightness_temp_matrix[i, ..., j, 0] = xt[
                example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY
            ].values[k, ..., 0]

            satellite_predictor_matrix[i, j, :] = xt[
                example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY
            ].values[k, satellite_predictor_indices]

        for j in range(len(ships_lag_times_sec)):
            k = ships_time_indices_by_example[i][j]

            lagged_values = numpy.ravel(
                xt[
                    example_utils.SHIPS_PREDICTORS_LAGGED_KEY
                ].values[k, :, ships_predictor_indices_lagged]
            )
            ships_predictor_matrix[i, j, :len(lagged_values)] = lagged_values

            forecast_values = numpy.ravel(
                xt[
                    example_utils.SHIPS_PREDICTORS_FORECAST_KEY
                ].values[k, :, ships_predictor_indices_forecast]
            )
            ships_predictor_matrix[i, j, len(lagged_values):] = forecast_values

        intensity_change_m_s01 = numpy.diff(
            xt[example_utils.STORM_INTENSITY_KEY].values[
                target_time_indices_by_example[i]
            ]
        )[0]

        target_matrix[i, :] = _discretize_intensity_change(
            intensity_change_m_s01=intensity_change_m_s01,
            class_cutoffs_m_s01=class_cutoffs_m_s01
        )

    predictor_matrices = [
        brightness_temp_matrix, satellite_predictor_matrix,
        ships_predictor_matrix
    ]

    return predictor_matrices, target_matrix


def input_generator(
        example_dir_name, years, lead_time_hours, satellite_lag_times_minutes,
        ships_lag_times_hours, satellite_predictor_names,
        ships_predictor_names_lagged, ships_predictor_names_forecast,
        num_examples_per_batch, max_examples_per_cyclone_in_batch,
        class_cutoffs_m_s01):
    """Generates training data for neural net.

    E = number of examples per batch
    M = number of rows in satellite grid
    N = number of columns in satellite grid
    T_sat = number of lag times for satellite-based predictors
    T_ships = number of lag times for SHIPS predictors
    C_sat = number of channels for ungridded satellite-based predictors
    C_ships = number of channels for SHIPS predictors
    K = number of classes

    :param example_dir_name: Name of directory with example files.  Files
        therein will be found by `example_io.find_file` and read by
        `example_io.read_file`.
    :param years: 1-D numpy array of training years.
    :param lead_time_hours: Lead time for predicting storm intensity.
    :param satellite_lag_times_minutes: 1-D numpy array of lag times for
        satellite-based predictors.
    :param ships_lag_times_hours: 1-D numpy array of lag times for SHIPS
        predictors.
    :param satellite_predictor_names: 1-D list with names of satellite-based
        predictors to use.
    :param ships_predictor_names_lagged: 1-D list with names of lagged SHIPS
        predictors to use.
    :param ships_predictor_names_forecast: 1-D list with names of forecast SHIPS
        predictors to use.
    :param num_examples_per_batch: Number of examples per batch.
    :param max_examples_per_cyclone_in_batch: Max number of examples (time
        steps) from one cyclone in a batch.
    :param class_cutoffs_m_s01: numpy array (length K - 1) of class cutoffs.
    :return: predictor_matrices: 1-D list with the following elements.

        brightness_temp_matrix: numpy array (E x M x N x T_sat x 1) of
        brightness temperatures.

        satellite_predictor_matrix: numpy array (E x T_sat x C_sat) of
        satellite-based predictors.

        ships_predictor_matrix: numpy array (E x T_ships x C_ships) of
        SHIPS predictors.

    :return: target_matrix: E-by-K numpy array of integers (0 or 1), indicating
        true classes.
    """

    # TODO(thunderhoser): Deal with binary classification.

    # TODO(thunderhoser): For SHIPS predictors, all lag times and forecast times
    # are currently flattened along the channel axis.  I might change this in
    # the future to make better use of time series.

    # Check input args.
    error_checking.assert_is_integer_numpy_array(years)
    error_checking.assert_is_numpy_array(years, num_dimensions=1)
    error_checking.assert_is_integer(lead_time_hours)
    assert numpy.mod(lead_time_hours, 6) == 0

    error_checking.assert_is_integer_numpy_array(satellite_lag_times_minutes)
    error_checking.assert_is_geq_numpy_array(satellite_lag_times_minutes, 0)
    error_checking.assert_is_numpy_array(
        satellite_lag_times_minutes, num_dimensions=1
    )

    error_checking.assert_is_integer_numpy_array(ships_lag_times_hours)
    error_checking.assert_is_geq_numpy_array(ships_lag_times_hours, 0)
    assert numpy.all(numpy.mod(ships_lag_times_hours, 6) == 0)
    error_checking.assert_is_numpy_array(
        ships_lag_times_hours, num_dimensions=1
    )

    # TODO(thunderhoser): Allow either list to be empty.
    error_checking.assert_is_string_list(satellite_predictor_names)
    error_checking.assert_is_string_list(ships_predictor_names_lagged)
    error_checking.assert_is_string_list(ships_predictor_names_forecast)

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(max_examples_per_cyclone_in_batch)
    error_checking.assert_is_geq(max_examples_per_cyclone_in_batch, 5)
    error_checking.assert_is_greater(
        num_examples_per_batch, max_examples_per_cyclone_in_batch
    )

    error_checking.assert_is_numpy_array(class_cutoffs_m_s01, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(class_cutoffs_m_s01), 0.
    )
    assert numpy.all(numpy.isfinite(class_cutoffs_m_s01))

    # Do actual stuff.
    cyclone_id_strings = example_io.find_cyclones(
        directory_name=example_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [satellite_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )
    good_flags = numpy.array([y in years for y in cyclone_years], dtype=bool)
    good_indices = numpy.where(good_flags)[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    random.shuffle(cyclone_id_strings)

    example_file_names = [
        example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=True
        )
        for c in cyclone_id_strings
    ]

    file_index = 0

    while True:
        predictor_matrices = None
        target_matrix = None
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if file_index == len(example_file_names):
                file_index = 0

            num_examples_to_read = min([
                max_examples_per_cyclone_in_batch,
                num_examples_per_batch - num_examples_in_memory
            ])

            these_predictor_matrices, this_target_matrix = (
                _read_one_example_file(
                    example_file_name=example_file_names[file_index],
                    num_examples_desired=num_examples_to_read,
                    lead_time_hours=lead_time_hours,
                    satellite_lag_times_minutes=satellite_lag_times_minutes,
                    ships_lag_times_hours=ships_lag_times_hours,
                    satellite_predictor_names=satellite_predictor_names,
                    ships_predictor_names_lagged=ships_predictor_names_lagged,
                    ships_predictor_names_forecast=
                    ships_predictor_names_forecast,
                    class_cutoffs_m_s01=class_cutoffs_m_s01
                )
            )

            if predictor_matrices is None:
                predictor_matrices = copy.deepcopy(these_predictor_matrices)
                target_matrix = this_target_matrix + 0
            else:
                for j in range(len(predictor_matrices)):
                    predictor_matrices[j] = numpy.concatenate(
                        (predictor_matrices[j], these_predictor_matrices[j]),
                        axis=0
                    )

                target_matrix = numpy.concatenate(
                    (target_matrix, this_target_matrix), axis=0
                )

            num_examples_in_memory = target_matrix.shape[0]

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        target_matrix = target_matrix.astype('float16')

        yield predictor_matrices, target_matrix
