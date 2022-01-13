"""Plotting methods for predictor variables."""

import numpy
import xarray
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import ships_io
from ml4tc.utils import example_utils
from ml4tc.utils import satellite_utils
from ml4tc.utils import normalization
from ml4tc.machine_learning import neural_net
from ml4tc.plotting import ships_plotting
from ml4tc.plotting import scalar_satellite_plotting
from ml4tc.scripts import plot_satellite

MINUTES_TO_SECONDS = 60
HOURS_TO_SECONDS = 3600


def plot_scalar_satellite_one_example(
        predictor_matrices_one_example, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec):
    """Plots scalar (ungridded) satellite-based predictors for one example.

    "For one example" means for each lag time and one forecast-initialization
    time.  Explainable-ML heat maps may eventually be plotted on top of these
    colour maps.

    T = number of input tensors to model

    :param predictor_matrices_one_example: length-3 list, where each element is
        either None or a numpy array formatted in the same way as the training
        data.  The first axis (i.e., the example axis) of each numpy array
        should have length 1.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `satellite_utils.parse_cyclone_id`).
    :param init_time_unix_sec: Forecast-initialization time.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :return: pathless_output_file_name: Suggested pathless name for image file.
    """

    # Check input args.
    error_checking.assert_is_list(predictor_matrices_one_example)

    for this_predictor_matrix in predictor_matrices_one_example:
        if this_predictor_matrix is None:
            continue

        error_checking.assert_is_numpy_array_without_nan(this_predictor_matrix)

    satellite_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_integer(init_time_unix_sec)

    # Housekeeping.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    lag_times_sec = (
        MINUTES_TO_SECONDS *
        validation_option_dict[neural_net.SATELLITE_LAG_TIMES_KEY]
    )

    num_predictors = predictor_matrices_one_example[1].shape[-1]
    predictor_indices = numpy.linspace(
        0, num_predictors - 1, num=num_predictors, dtype=int
    )

    valid_times_unix_sec = init_time_unix_sec - lag_times_sec
    num_valid_times = len(valid_times_unix_sec)
    valid_time_indices = numpy.linspace(
        0, num_valid_times - 1, num=num_valid_times, dtype=int
    )

    # Create xarray table.
    metadata_dict = {
        example_utils.SATELLITE_TIME_DIM: init_time_unix_sec - lag_times_sec,
        example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM:
            validation_option_dict[neural_net.SATELLITE_PREDICTORS_KEY]
    }

    dimensions = (
        example_utils.SATELLITE_TIME_DIM,
        example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
    )
    main_data_dict = {
        example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY: (
            dimensions,
            predictor_matrices_one_example[1][0, ...]
        ),
        satellite_utils.CYCLONE_ID_KEY: (
            (example_utils.SATELLITE_TIME_DIM,),
            [cyclone_id_string] * len(lag_times_sec)
        )
    }

    example_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    # Do plotting.
    return scalar_satellite_plotting.plot_colour_map_multi_times(
        example_table_xarray=example_table_xarray,
        time_indices=valid_time_indices, predictor_indices=predictor_indices
    )


def plot_brightness_temp_one_example(
        predictor_matrices_one_example, model_metadata_dict,
        cyclone_id_string, init_time_unix_sec, normalization_table_xarray,
        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e):
    """Plots brightness-temperature maps for one example.

    "For one example" means for each lag time and one forecast-initialization
    time.  Explainable-ML heat maps may eventually be plotted on top of these
    colour maps.

    M = number of rows in grid
    N = number of columns in grid
    L = number of model lag times
    P = number of points in border set

    :param predictor_matrices_one_example: See doc for
        `plot_scalar_satellite_one_example`.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param normalization_table_xarray: xarray table returned by
        `normalization.read_file`.
    :param grid_latitude_matrix_deg_n: numpy array of latitudes (deg north).  If
        regular grids, this should have dimensions M x L.  If irregular grids,
        should have dimensions M x N x L.
    :param grid_longitude_matrix_deg_e: numpy array of longitudes (deg east).
        If regular grids, this should have dimensions N x L.  If irregular
        grids, should have dimensions M x N x L.
    :param border_latitudes_deg_n: length-P numpy array of latitudes
        (deg north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes
        (deg east).
    :return: figure_objects: length-L list of figure handles (instances of
        `matplotlib.figure.Figure`).
    :return: axes_objects: length-L list of axes handles (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    :return: pathless_output_file_names: length-L list of suggested pathless
        names for image files.
    """

    # Check input args.
    error_checking.assert_is_list(predictor_matrices_one_example)

    for this_predictor_matrix in predictor_matrices_one_example:
        if this_predictor_matrix is None:
            continue

        error_checking.assert_is_numpy_array_without_nan(this_predictor_matrix)

    satellite_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_integer(init_time_unix_sec)

    error_checking.assert_is_numpy_array(grid_latitude_matrix_deg_n)
    regular_grids = len(grid_latitude_matrix_deg_n.shape) == 2

    if regular_grids:
        error_checking.assert_is_numpy_array(
            grid_latitude_matrix_deg_n, num_dimensions=2
        )

        num_model_lag_times = grid_latitude_matrix_deg_n.shape[1]
        expected_dim = numpy.array(
            [grid_longitude_matrix_deg_e.shape[0], num_model_lag_times],
            dtype=int
        )
        error_checking.assert_is_numpy_array(
            grid_longitude_matrix_deg_e, exact_dimensions=expected_dim
        )
    else:
        error_checking.assert_is_numpy_array(
            grid_latitude_matrix_deg_n, num_dimensions=3
        )
        error_checking.assert_is_numpy_array(
            grid_longitude_matrix_deg_e,
            exact_dimensions=
            numpy.array(grid_latitude_matrix_deg_n.shape, dtype=int)
        )

    # Denormalize brightness temperatures.
    nt = normalization_table_xarray
    predictor_names_norm = list(
        nt.coords[normalization.SATELLITE_PREDICTOR_GRIDDED_DIM].values
    )

    k = predictor_names_norm.index(satellite_utils.BRIGHTNESS_TEMPERATURE_KEY)
    training_values = (
        nt[normalization.SATELLITE_PREDICTORS_GRIDDED_KEY].values[:, k]
    )
    training_values = training_values[numpy.isfinite(training_values)]

    brightness_temp_matrix_kelvins = normalization._denorm_one_variable(
        normalized_values_new=predictor_matrices_one_example[0],
        actual_values_training=training_values
    )[..., 0]

    # Housekeeping.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    model_lag_times_sec = (
        MINUTES_TO_SECONDS *
        validation_option_dict[neural_net.SATELLITE_LAG_TIMES_KEY]
    )
    num_model_lag_times = len(model_lag_times_sec)

    num_grid_rows = brightness_temp_matrix_kelvins.shape[1]
    num_grid_columns = brightness_temp_matrix_kelvins.shape[2]
    grid_row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=int
    )
    grid_column_indices = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=int
    )

    # For each model lag time:
    figure_objects = [None] * num_model_lag_times
    axes_objects = [None] * num_model_lag_times
    pathless_output_file_names = [''] * num_model_lag_times

    for j in range(num_model_lag_times):
        valid_time_unix_sec = init_time_unix_sec - model_lag_times_sec[j]

        metadata_dict = {
            satellite_utils.GRID_ROW_DIM: grid_row_indices,
            satellite_utils.GRID_COLUMN_DIM: grid_column_indices,
            satellite_utils.TIME_DIM:
                numpy.array([valid_time_unix_sec], dtype=int)
        }

        dimensions = (
            satellite_utils.TIME_DIM,
            satellite_utils.GRID_ROW_DIM, satellite_utils.GRID_COLUMN_DIM
        )
        main_data_dict = {
            satellite_utils.CYCLONE_ID_KEY: (
                (satellite_utils.TIME_DIM,),
                [cyclone_id_string]
            ),
            satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
                dimensions,
                brightness_temp_matrix_kelvins[[0], ..., j]
            )
        }

        if regular_grids:
            main_data_dict.update({
                satellite_utils.GRID_LATITUDE_KEY: (
                    (satellite_utils.TIME_DIM, satellite_utils.GRID_ROW_DIM),
                    numpy.transpose(grid_latitude_matrix_deg_n[:, [j]])
                ),
                satellite_utils.GRID_LONGITUDE_KEY: (
                    (satellite_utils.TIME_DIM, satellite_utils.GRID_COLUMN_DIM),
                    numpy.transpose(grid_longitude_matrix_deg_e[:, [j]])
                )
            })
        else:
            dimensions = (
                satellite_utils.TIME_DIM, satellite_utils.GRID_ROW_DIM,
                satellite_utils.GRID_COLUMN_DIM
            )
            this_latitude_matrix_deg_n = numpy.expand_dims(
                grid_latitude_matrix_deg_n[..., j], axis=0
            )
            this_longitude_matrix_deg_e = numpy.expand_dims(
                grid_longitude_matrix_deg_e[..., j], axis=0
            )

            main_data_dict.update({
                satellite_utils.GRID_LATITUDE_KEY: (
                    dimensions, this_latitude_matrix_deg_n
                ),
                satellite_utils.GRID_LONGITUDE_KEY: (
                    dimensions, this_longitude_matrix_deg_e
                )
            })

        example_table_xarray = xarray.Dataset(
            data_vars=main_data_dict, coords=metadata_dict
        )
        figure_objects[j], axes_objects[j], pathless_output_file_names[j] = (
            plot_satellite.plot_one_satellite_image(
                satellite_table_xarray=example_table_xarray, time_index=0,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                cbar_orientation_string=None, output_dir_name=None
            )
        )

    return figure_objects, axes_objects, pathless_output_file_names


def plot_lagged_ships_one_example(
        predictor_matrices_one_example, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec, builtin_lag_times_hours, forecast_hours):
    """Plots lagged SHIPS predictors for one example.

    "For one example" means for each lag time and one forecast-initialization
    time.  Explainable-ML heat maps may eventually be plotted on top of these
    colour maps.

    :param predictor_matrices_one_example: See doc for
        `plot_scalar_satellite_one_example`.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param builtin_lag_times_hours: 1-D numpy array of built-in lag times for
        lagged SHIPS predictors.
    :param forecast_hours: 1-D numpy array of forecast hours for forecast SHIPS
        predictors.
    :return: figure_objects: See doc for `plot_brightness_temp_one_example`.
    :return: axes_objects: Same.
    :return: pathless_output_file_names: Same.
    """

    # Check input args.
    error_checking.assert_is_list(predictor_matrices_one_example)

    for this_predictor_matrix in predictor_matrices_one_example:
        if this_predictor_matrix is None:
            continue

        error_checking.assert_is_numpy_array_without_nan(this_predictor_matrix)

    satellite_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_integer(init_time_unix_sec)
    error_checking.assert_is_numpy_array(
        builtin_lag_times_hours, num_dimensions=1
    )
    error_checking.assert_is_numpy_array(forecast_hours, num_dimensions=1)
    error_checking.assert_is_numpy_array_without_nan(forecast_hours)

    # Housekeeping.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    model_lag_times_sec = (
        HOURS_TO_SECONDS *
        validation_option_dict[neural_net.SHIPS_LAG_TIMES_KEY]
    )

    lagged_predictor_names = (
        validation_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
    )
    num_lagged_predictors = len(lagged_predictor_names)
    lagged_predictor_indices = numpy.linspace(
        0, num_lagged_predictors - 1, num=num_lagged_predictors, dtype=int
    )

    forecast_predictor_names = (
        validation_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
    )
    num_forecast_predictors = (
        0 if forecast_predictor_names is None
        else len(forecast_predictor_names)
    )

    num_forecast_hours = len(forecast_hours)
    num_builtin_lag_times = len(builtin_lag_times_hours)
    num_model_lag_times = len(model_lag_times_sec)

    # For each model lag time:
    figure_objects = [None] * num_model_lag_times
    axes_objects = [None] * num_model_lag_times
    pathless_output_file_names = [''] * num_model_lag_times

    for j in range(num_model_lag_times):

        # Create xarray table.
        valid_time_unix_sec = init_time_unix_sec - model_lag_times_sec[j]

        metadata_dict = {
            example_utils.SHIPS_LAG_TIME_DIM: builtin_lag_times_hours,
            example_utils.SHIPS_VALID_TIME_DIM:
                numpy.array([valid_time_unix_sec], dtype=int),
            example_utils.SHIPS_PREDICTOR_LAGGED_DIM: lagged_predictor_names
        }

        predictor_matrix = numpy.expand_dims(
            predictor_matrices_one_example[2][0, j, :], axis=0
        )
        predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)
        predictor_matrix = neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=predictor_matrix,
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=num_builtin_lag_times,
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=num_forecast_hours
        )[0][:, 0, ...]

        dimensions = (
            example_utils.SHIPS_VALID_TIME_DIM,
            example_utils.SHIPS_LAG_TIME_DIM,
            example_utils.SHIPS_PREDICTOR_LAGGED_DIM
        )
        main_data_dict = {
            example_utils.SHIPS_PREDICTORS_LAGGED_KEY: (
                dimensions, predictor_matrix
            ),
            ships_io.CYCLONE_ID_KEY: (
                (example_utils.SHIPS_VALID_TIME_DIM,),
                [cyclone_id_string]
            )
        }

        this_table_xarray = xarray.Dataset(
            data_vars=main_data_dict, coords=metadata_dict
        )

        # Do plotting.
        figure_objects[j], axes_objects[j], pathless_output_file_names[j] = (
            ships_plotting.plot_lagged_predictors_one_init_time(
                example_table_xarray=this_table_xarray, init_time_index=0,
                predictor_indices=lagged_predictor_indices,
            )
        )

    return figure_objects, axes_objects, pathless_output_file_names


def plot_forecast_ships_one_example(
        predictor_matrices_one_example, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec, builtin_lag_times_hours, forecast_hours):
    """Plots forecast SHIPS predictors for one example.

    "For one example" means for each lag time and one forecast-initialization
    time.  Explainable-ML heat maps may eventually be plotted on top of these
    colour maps.

    :param predictor_matrices_one_example: See doc for
        `plot_scalar_satellite_one_example`.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param builtin_lag_times_hours: See doc for `plot_lagged_ships_one_example`.
    :param forecast_hours: Same.
    :return: figure_objects: Same.
    :return: axes_objects: Same.
    :return: pathless_output_file_names: Same.
    """

    # Check input args.
    error_checking.assert_is_list(predictor_matrices_one_example)

    for this_predictor_matrix in predictor_matrices_one_example:
        if this_predictor_matrix is None:
            continue

        error_checking.assert_is_numpy_array_without_nan(this_predictor_matrix)

    satellite_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_integer(init_time_unix_sec)
    error_checking.assert_is_numpy_array(
        builtin_lag_times_hours, num_dimensions=1
    )
    error_checking.assert_is_numpy_array(forecast_hours, num_dimensions=1)
    error_checking.assert_is_numpy_array_without_nan(forecast_hours)

    # Housekeeping.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    model_lag_times_sec = (
        HOURS_TO_SECONDS *
        validation_option_dict[neural_net.SHIPS_LAG_TIMES_KEY]
    )

    forecast_predictor_names = (
        validation_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
    )
    num_forecast_predictors = len(forecast_predictor_names)
    forecast_predictor_indices = numpy.linspace(
        0, num_forecast_predictors - 1, num=num_forecast_predictors, dtype=int
    )

    lagged_predictor_names = (
        validation_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
    )
    num_lagged_predictors = (
        0 if lagged_predictor_names is None
        else len(lagged_predictor_names)
    )

    num_forecast_hours = len(forecast_hours)
    num_builtin_lag_times = len(builtin_lag_times_hours)
    num_model_lag_times = len(model_lag_times_sec)

    # For each model lag time:
    figure_objects = [None] * num_model_lag_times
    axes_objects = [None] * num_model_lag_times
    pathless_output_file_names = [''] * num_model_lag_times

    for j in range(num_model_lag_times):

        # Create xarray table.
        valid_time_unix_sec = init_time_unix_sec - model_lag_times_sec[j]

        metadata_dict = {
            example_utils.SHIPS_FORECAST_HOUR_DIM: forecast_hours,
            example_utils.SHIPS_VALID_TIME_DIM:
                numpy.array([valid_time_unix_sec], dtype=int),
            example_utils.SHIPS_PREDICTOR_FORECAST_DIM:
                forecast_predictor_names
        }

        predictor_matrix = numpy.expand_dims(
            predictor_matrices_one_example[2][0, j, :], axis=0
        )
        predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)
        predictor_matrix = neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=predictor_matrix,
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=num_builtin_lag_times,
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=num_forecast_hours
        )[1][:, 0, ...]

        dimensions = (
            example_utils.SHIPS_VALID_TIME_DIM,
            example_utils.SHIPS_FORECAST_HOUR_DIM,
            example_utils.SHIPS_PREDICTOR_FORECAST_DIM
        )
        main_data_dict = {
            example_utils.SHIPS_PREDICTORS_FORECAST_KEY: (
                dimensions, predictor_matrix
            ),
            ships_io.CYCLONE_ID_KEY: (
                (example_utils.SHIPS_VALID_TIME_DIM,),
                [cyclone_id_string]
            )
        }

        this_table_xarray = xarray.Dataset(
            data_vars=main_data_dict, coords=metadata_dict
        )

        # Do plotting.
        figure_objects[j], axes_objects[j], pathless_output_file_names[j] = (
            ships_plotting.plot_fcst_predictors_one_init_time(
                example_table_xarray=this_table_xarray, init_time_index=0,
                predictor_indices=forecast_predictor_indices
            )
        )

    return figure_objects, axes_objects, pathless_output_file_names
