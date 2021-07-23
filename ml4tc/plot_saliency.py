"""Plots saliency maps."""

import os
import sys
import copy
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_general_utils
import time_conversion
import file_system_utils
import error_checking
import imagemagick_utils
import ships_io
import example_io
import border_io
import example_utils
import satellite_utils
import normalization
import saliency
import neural_net
import plotting_utils
import ships_plotting
import satellite_plotting
import scalar_satellite_plotting
import plot_satellite

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MINUTES_TO_SECONDS = 60
HOURS_TO_SECONDS = 3600
MAX_COLOUR_PERCENTILE = 99.

SHIPS_FORECAST_HOURS = numpy.linspace(-12, 120, num=23, dtype=int)
SHIPS_BUILTIN_LAG_TIMES_HOURS = numpy.array([numpy.nan, 0, 1.5, 3])

DEFAULT_FONT_SIZE = 20
COLOUR_BAR_FONT_SIZE = 12

BRIGHTNESS_TEMP_CMAP_OBJECT = pyplot.get_cmap('BuGn')
SCALAR_SATELLITE_FONT_SIZE = 20
SCALAR_SATELLITE_CMAP_OBJECT = pyplot.get_cmap('binary')
LAGGED_SHIPS_FONT_SIZE = 20
LAGGED_SHIPS_CMAP_OBJECT = pyplot.get_cmap('binary')
FORECAST_SHIPS_FONT_SIZE = 10
FORECAST_SHIPS_CMAP_OBJECT = pyplot.get_cmap('binary')

FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file.  Will be read by `saliency.read_file`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with input examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be used to denormalize '
    'brightness-temperature maps before plotting).  Will be read by '
    '`normalization.read_file`.'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Images will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_FILE_ARG_NAME, type=str, required=True,
    help=SALIENCY_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def plot_brightness_temp(
        predictor_matrices_one_example, model_metadata_dict,
        cyclone_id_string, init_time_unix_sec, normalization_table_xarray,
        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e):
    """Plots brightness-temperature map for each lag time at one init time.

    M = number of rows in grid
    N = number of columns in grid
    L = number of model lag times
    P = number of points in border set

    :param predictor_matrices_one_example: See doc for
        `plot_lagged_ships_predictors`.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param normalization_table_xarray: xarray table returned by
        `normalization.read_file`.
    :param grid_latitude_matrix_deg_n: M-by-L numpy array of grid-point
        latitudes (deg north).
    :param grid_longitude_matrix_deg_e: N-by-L numpy array of grid-point
        longitudes (deg east).
    :param border_latitudes_deg_n: length-P numpy array of latitudes
        (deg north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes
        (deg east).
    :return: figure_objects: See doc for
        `plot_lagged_ships_predictors`.
    :return: axes_objects: Same.
    :return: pathless_output_file_names: Same.
    """

    # Check input args.
    error_checking.assert_is_list(predictor_matrices_one_example)
    for this_predictor_matrix in predictor_matrices_one_example:
        error_checking.assert_is_numpy_array_without_nan(this_predictor_matrix)

    satellite_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_integer(init_time_unix_sec)

    error_checking.assert_is_numpy_array(
        grid_latitude_matrix_deg_n, num_dimensions=2
    )

    num_model_lag_times = grid_latitude_matrix_deg_n.shape[1]
    expected_dim = numpy.array(
        [grid_longitude_matrix_deg_e.shape[0], num_model_lag_times], dtype=int
    )
    error_checking.assert_is_numpy_array(
        grid_longitude_matrix_deg_e, exact_dimensions=expected_dim
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
            ),
            satellite_utils.GRID_LATITUDE_KEY: (
                (satellite_utils.TIME_DIM, satellite_utils.GRID_ROW_DIM),
                numpy.transpose(grid_latitude_matrix_deg_n[:, [j]])
            ),
            satellite_utils.GRID_LONGITUDE_KEY: (
                (satellite_utils.TIME_DIM, satellite_utils.GRID_COLUMN_DIM),
                numpy.transpose(grid_longitude_matrix_deg_e[:, [j]])
            )
        }

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


def plot_scalar_satellite_predictors(
        predictor_matrices_one_example, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec):
    """Plots scalar satellite predictors for each lag time at one init time.

    T = number of input tensors to model

    :param predictor_matrices_one_example: length-T list of numpy arrays,
        formatted in the same way as the training data.  The first axis (i.e.,
        the example axis) of each numpy array should have length 1.
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
            [cyclone_id_string]
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


def plot_lagged_ships_predictors(
        predictor_matrices_one_example, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec, builtin_lag_times_hours, forecast_hours):
    """Plots lagged SHIPS predictors for each lag time at one init time.

    L = number of model lag times (not built-in lag times)

    :param predictor_matrices_one_example: See doc for
        `plot_scalar_satellite_predictors`.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param builtin_lag_times_hours: 1-D numpy array of built-in lag times for
        lagged SHIPS predictors.
    :param forecast_hours: 1-D numpy array of forecast hours for forecast SHIPS
        predictors.
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

    num_forecast_predictors = len(
        validation_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
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


def plot_forecast_ships_predictors(
        predictor_matrices_one_example, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec, builtin_lag_times_hours, forecast_hours):
    """Plots lagged SHIPS predictors for each lag time at one init time.

    :param predictor_matrices_one_example: See doc for
        `plot_scalar_satellite_predictors`.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param builtin_lag_times_hours: See doc for `plot_lagged_ships_predictors`.
    :param forecast_hours: Same.
    :return: figure_objects: Same.
    :return: axes_objects: Same.
    :return: pathless_output_file_names: Same.
    """

    # Check input args.
    error_checking.assert_is_list(predictor_matrices_one_example)
    for this_predictor_matrix in predictor_matrices_one_example:
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

    num_lagged_predictors = len(
        validation_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
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


def _plot_scalar_satellite_saliency(
        data_dict, saliency_dict, model_metadata_dict,
        cyclone_id_string, init_time_unix_sec, output_dir_name):
    """Plots saliency for scalar satellite for each lag time at one init time.

    :param data_dict: Dictionary returned by `neural_net.create_inputs`.
    :param saliency_dict: Dictionary returned by `saliency.read_file`.
    :param model_metadata_dict: See doc for `plot_scalar_satellite_predictors`.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    predictor_example_index = numpy.where(
        data_dict[neural_net.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]
    saliency_example_index = numpy.where(
        saliency_dict[saliency.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]

    predictor_matrices_one_example = [
        p[[predictor_example_index], ...]
        for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]
    saliency_matrices_one_example = [
        s[[saliency_example_index], ...]
        for s in saliency_dict[saliency.SALIENCY_KEY]
    ]

    figure_object, axes_object, pathless_output_file_name = (
        plot_scalar_satellite_predictors(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            init_time_unix_sec=init_time_unix_sec
        )
    )

    all_saliency_values = numpy.concatenate([
        numpy.ravel(s) for s in saliency_matrices_one_example
    ])
    max_absolute_colour_value = numpy.percentile(
        numpy.absolute(all_saliency_values), MAX_COLOUR_PERCENTILE
    )
    scalar_satellite_plotting.plot_pm_signs_multi_times(
        data_matrix=saliency_matrices_one_example[1][0, ...],
        axes_object=axes_object,
        font_size=SCALAR_SATELLITE_FONT_SIZE,
        colour_map_object=SCALAR_SATELLITE_CMAP_OBJECT,
        max_absolute_colour_value=max_absolute_colour_value
    )

    output_file_name = '{0:s}/{1:s}'.format(
        output_dir_name, pathless_output_file_name
    )
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    temporary_cbar_file_name = '{0:s}/{1:s}_cbar.jpg'.format(
        output_dir_name,
        '.'.join(pathless_output_file_name.split('.')[:-1])
    )
    colour_norm_object = pyplot.Normalize(
        vmin=scalar_satellite_plotting.MIN_NORMALIZED_VALUE,
        vmax=scalar_satellite_plotting.MAX_NORMALIZED_VALUE
    )
    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        temporary_cbar_file_name=temporary_cbar_file_name,
        colour_map_object=scalar_satellite_plotting.COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Predictor'
    )

    colour_norm_object = pyplot.Normalize(
        vmin=0., vmax=max_absolute_colour_value
    )
    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        temporary_cbar_file_name=temporary_cbar_file_name,
        colour_map_object=SCALAR_SATELLITE_CMAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Absolute saliency'
    )


def _plot_brightness_temp_saliency(
        data_dict, saliency_dict, model_metadata_dict,
        cyclone_id_string, init_time_unix_sec, normalization_table_xarray,
        border_latitudes_deg_n, border_longitudes_deg_e, output_dir_name):
    """Plots saliency for brightness temp for each lag time at one init time.

    P = number of points in border set

    :param data_dict: See doc for `_plot_scalar_satellite_saliency`.
    :param saliency_dict: Same.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param normalization_table_xarray: xarray table returned by
        `normalization.read_file`.
    :param border_latitudes_deg_n: length-P numpy array of latitudes
        (deg north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes
        (deg east).
    :param output_dir_name: Same.
    """

    predictor_example_index = numpy.where(
        data_dict[neural_net.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]
    saliency_example_index = numpy.where(
        saliency_dict[saliency.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]

    predictor_matrices_one_example = [
        p[[predictor_example_index], ...]
        for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]
    saliency_matrices_one_example = [
        s[[saliency_example_index], ...]
        for s in saliency_dict[saliency.SALIENCY_KEY]
    ]
    grid_latitude_matrix_deg_n = data_dict[
        neural_net.GRID_LATITUDE_MATRIX_KEY
    ][predictor_example_index, ...]

    grid_longitude_matrix_deg_e = data_dict[
        neural_net.GRID_LONGITUDE_MATRIX_KEY
    ][predictor_example_index, ...]

    figure_objects, axes_objects, pathless_output_file_names = (
        plot_brightness_temp(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            init_time_unix_sec=init_time_unix_sec,
            grid_latitude_matrix_deg_n=grid_latitude_matrix_deg_n,
            grid_longitude_matrix_deg_e=grid_longitude_matrix_deg_e,
            normalization_table_xarray=normalization_table_xarray,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e
        )
    )

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    num_model_lag_times = len(
        validation_option_dict[neural_net.SHIPS_LAG_TIMES_KEY]
    )

    all_saliency_values = numpy.concatenate([
        numpy.ravel(s) for s in saliency_matrices_one_example
    ])
    max_abs_contour_value = numpy.percentile(
        numpy.absolute(all_saliency_values), MAX_COLOUR_PERCENTILE
    )

    panel_file_names = [''] * num_model_lag_times

    for k in range(num_model_lag_times):
        this_saliency_matrix = (
            gg_general_utils.apply_gaussian_filter(
                input_matrix=saliency_matrices_one_example[0][0, ..., k, 0],
                e_folding_radius_grid_cells=2.
            )
        )

        # TODO(thunderhoser): Do 0.001 thing everywhere.
        satellite_plotting.plot_saliency(
            saliency_matrix=this_saliency_matrix, axes_object=axes_objects[k],
            latitudes_deg_n=grid_latitude_matrix_deg_n[:, k],
            longitudes_deg_e=grid_longitude_matrix_deg_e[:, k],
            min_abs_contour_value=0.001,
            max_abs_contour_value=max_abs_contour_value,
            half_num_contours=10,
            colour_map_object=BRIGHTNESS_TEMP_CMAP_OBJECT
        )

        panel_file_names[k] = '{0:s}/{1:s}'.format(
            output_dir_name, pathless_output_file_names[k]
        )
        print('Saving figure to file: "{0:s}"...'.format(
            panel_file_names[k]
        ))
        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[k],
            output_file_name=panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    concat_figure_file_name = '{0:s}/{1:s}_{2:s}_brightness_temp.jpg'.format(
        output_dir_name, cyclone_id_string, init_time_string
    )
    plotting_utils.concat_panels(
        panel_file_names=panel_file_names,
        concat_figure_file_name=concat_figure_file_name
    )

    temporary_cbar_file_name = (
        '{0:s}/{1:s}_{2:s}_brightness_temp_cbar.jpg'
    ).format(
        output_dir_name, cyclone_id_string, init_time_string
    )
    colour_map_object, colour_norm_object = (
        satellite_plotting.get_colour_scheme()
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        temporary_cbar_file_name=temporary_cbar_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Brightness temp (K)'
    )

    colour_norm_object = pyplot.Normalize(
        vmin=0.001, vmax=max_abs_contour_value
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        temporary_cbar_file_name=temporary_cbar_file_name,
        colour_map_object=BRIGHTNESS_TEMP_CMAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Absolute saliency'
    )


def _plot_lagged_ships_saliency(
        data_dict, saliency_dict, model_metadata_dict,
        cyclone_id_string, init_time_unix_sec, output_dir_name):
    """Plots saliency for lagged SHIPS for each lag time at one init time.

    :param data_dict: See doc for `_plot_scalar_satellite_saliency`.
    :param saliency_dict: Same.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param output_dir_name: Same.
    """

    predictor_example_index = numpy.where(
        data_dict[neural_net.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]
    saliency_example_index = numpy.where(
        saliency_dict[saliency.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]

    predictor_matrices_one_example = [
        p[[predictor_example_index], ...]
        for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]
    saliency_matrices_one_example = [
        s[[saliency_example_index], ...]
        for s in saliency_dict[saliency.SALIENCY_KEY]
    ]

    figure_objects, axes_objects, pathless_output_file_names = (
        plot_lagged_ships_predictors(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            builtin_lag_times_hours=SHIPS_BUILTIN_LAG_TIMES_HOURS,
            forecast_hours=SHIPS_FORECAST_HOURS,
            init_time_unix_sec=init_time_unix_sec
        )
    )

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    num_lagged_predictors = len(
        validation_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
    )
    num_forecast_predictors = len(
        validation_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
    )
    num_model_lag_times = len(
        validation_option_dict[neural_net.SHIPS_LAG_TIMES_KEY]
    )

    saliency_matrix = neural_net.ships_predictors_3d_to_4d(
        predictor_matrix_3d=saliency_matrices_one_example[2][[0], ...],
        num_lagged_predictors=num_lagged_predictors,
        num_builtin_lag_times=len(SHIPS_BUILTIN_LAG_TIMES_HOURS),
        num_forecast_predictors=num_forecast_predictors,
        num_forecast_hours=len(SHIPS_FORECAST_HOURS)
    )[0][0, ...]

    all_saliency_values = numpy.concatenate([
        numpy.ravel(s) for s in saliency_matrices_one_example
    ])
    max_absolute_colour_value = numpy.percentile(
        numpy.absolute(all_saliency_values), MAX_COLOUR_PERCENTILE
    )

    panel_file_names = [''] * num_model_lag_times

    for k in range(num_model_lag_times):
        ships_plotting.plot_pm_signs_one_init_time(
            data_matrix=saliency_matrix[k, ...],
            axes_object=axes_objects[k],
            font_size=LAGGED_SHIPS_FONT_SIZE,
            colour_map_object=LAGGED_SHIPS_CMAP_OBJECT,
            max_absolute_colour_value=max_absolute_colour_value
        )

        panel_file_names[k] = '{0:s}/{1:s}'.format(
            output_dir_name, pathless_output_file_names[k]
        )
        print('Saving figure to file: "{0:s}"...'.format(
            panel_file_names[k]
        ))
        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[k],
            output_file_name=panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/{1:s}_{2:s}_ships_lagged.jpg'.format(
        output_dir_name, cyclone_id_string,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
    )
    plotting_utils.concat_panels(
        panel_file_names=panel_file_names,
        concat_figure_file_name=concat_figure_file_name
    )

    temporary_cbar_file_name = '{0:s}/{1:s}_{2:s}_ships_lagged_cbar.jpg'.format(
        output_dir_name, cyclone_id_string,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
    )
    colour_norm_object = pyplot.Normalize(
        vmin=ships_plotting.MIN_NORMALIZED_VALUE,
        vmax=ships_plotting.MAX_NORMALIZED_VALUE
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        temporary_cbar_file_name=temporary_cbar_file_name,
        colour_map_object=ships_plotting.COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Predictor'
    )

    colour_norm_object = pyplot.Normalize(
        vmin=0., vmax=max_absolute_colour_value
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        temporary_cbar_file_name=temporary_cbar_file_name,
        colour_map_object=LAGGED_SHIPS_CMAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Absolute saliency'
    )


def _plot_forecast_ships_saliency(
        data_dict, saliency_dict, model_metadata_dict,
        cyclone_id_string, init_time_unix_sec, output_dir_name):
    """Plots saliency for forecast SHIPS for each lag time at one init time.

    :param data_dict: See doc for `_plot_scalar_satellite_saliency`.
    :param saliency_dict: Same.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param output_dir_name: Same.
    """

    predictor_example_index = numpy.where(
        data_dict[neural_net.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]
    saliency_example_index = numpy.where(
        saliency_dict[saliency.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]

    predictor_matrices_one_example = [
        p[[predictor_example_index], ...]
        for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]
    saliency_matrices_one_example = [
        s[[saliency_example_index], ...]
        for s in saliency_dict[saliency.SALIENCY_KEY]
    ]

    figure_objects, axes_objects, pathless_output_file_names = (
        plot_forecast_ships_predictors(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            builtin_lag_times_hours=SHIPS_BUILTIN_LAG_TIMES_HOURS,
            forecast_hours=SHIPS_FORECAST_HOURS,
            init_time_unix_sec=init_time_unix_sec
        )
    )

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    num_lagged_predictors = len(
        validation_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
    )
    num_forecast_predictors = len(
        validation_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
    )
    num_model_lag_times = len(
        validation_option_dict[neural_net.SHIPS_LAG_TIMES_KEY]
    )

    saliency_matrix = neural_net.ships_predictors_3d_to_4d(
        predictor_matrix_3d=saliency_matrices_one_example[2][[0], ...],
        num_lagged_predictors=num_lagged_predictors,
        num_builtin_lag_times=len(SHIPS_BUILTIN_LAG_TIMES_HOURS),
        num_forecast_predictors=num_forecast_predictors,
        num_forecast_hours=len(SHIPS_FORECAST_HOURS)
    )[1][0, ...]

    all_saliency_values = numpy.concatenate([
        numpy.ravel(s) for s in saliency_matrices_one_example
    ])
    max_absolute_colour_value = numpy.percentile(
        numpy.absolute(all_saliency_values), MAX_COLOUR_PERCENTILE
    )

    panel_file_names = [''] * num_model_lag_times

    for k in range(num_model_lag_times):
        ships_plotting.plot_pm_signs_one_init_time(
            data_matrix=saliency_matrix[k, ...],
            axes_object=axes_objects[k],
            font_size=FORECAST_SHIPS_FONT_SIZE,
            colour_map_object=FORECAST_SHIPS_CMAP_OBJECT,
            max_absolute_colour_value=max_absolute_colour_value
        )

        panel_file_names[k] = '{0:s}/{1:s}'.format(
            output_dir_name, pathless_output_file_names[k]
        )
        print('Saving figure to file: "{0:s}"...'.format(
            panel_file_names[k]
        ))
        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[k],
            output_file_name=panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/{1:s}_{2:s}_ships_forecast.jpg'.format(
        output_dir_name, cyclone_id_string,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
    )
    plotting_utils.concat_panels(
        panel_file_names=panel_file_names,
        concat_figure_file_name=concat_figure_file_name
    )

    temporary_cbar_file_name = (
        '{0:s}/{1:s}_{2:s}_ships_forecast_cbar.jpg'
    ).format(
        output_dir_name, cyclone_id_string,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
    )
    colour_norm_object = pyplot.Normalize(
        vmin=ships_plotting.MIN_NORMALIZED_VALUE,
        vmax=ships_plotting.MAX_NORMALIZED_VALUE
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        temporary_cbar_file_name=temporary_cbar_file_name,
        colour_map_object=ships_plotting.COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Predictor'
    )

    colour_norm_object = pyplot.Normalize(
        vmin=0., vmax=max_absolute_colour_value
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        temporary_cbar_file_name=temporary_cbar_file_name,
        colour_map_object=FORECAST_SHIPS_CMAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Absolute saliency'
    )


def _run(saliency_file_name, example_dir_name, normalization_file_name,
         output_dir_name):
    """Plots saliency maps.

    This is effectively the main method.

    :param saliency_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param normalization_file_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read files.
    print('Reading data from: "{0:s}"...'.format(saliency_file_name))
    saliency_dict = saliency.read_file(saliency_file_name)

    model_file_name = saliency_dict[saliency.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    base_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    # Find example files.
    unique_cyclone_id_strings = numpy.unique(
        numpy.array(saliency_dict[saliency.CYCLONE_IDS_KEY])
    )
    num_cyclones = len(unique_cyclone_id_strings)

    unique_example_file_names = [
        example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        for c in unique_cyclone_id_strings
    ]

    for i in range(num_cyclones):
        option_dict = copy.deepcopy(base_option_dict)
        option_dict[neural_net.EXAMPLE_FILE_KEY] = unique_example_file_names[i]

        print(SEPARATOR_STRING)
        data_dict = neural_net.create_inputs(option_dict)
        print(SEPARATOR_STRING)

        example_indices = numpy.where(
            numpy.array(saliency_dict[saliency.CYCLONE_IDS_KEY]) ==
            unique_cyclone_id_strings[i]
        )[0]

        for j in example_indices:
            _plot_brightness_temp_saliency(
                data_dict=data_dict, saliency_dict=saliency_dict,
                model_metadata_dict=model_metadata_dict,
                cyclone_id_string=unique_cyclone_id_strings[i],
                init_time_unix_sec=saliency_dict[saliency.INIT_TIMES_KEY][j],
                normalization_table_xarray=normalization_table_xarray,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                output_dir_name=output_dir_name
            )

            _plot_scalar_satellite_saliency(
                data_dict=data_dict, saliency_dict=saliency_dict,
                model_metadata_dict=model_metadata_dict,
                cyclone_id_string=unique_cyclone_id_strings[i],
                init_time_unix_sec=saliency_dict[saliency.INIT_TIMES_KEY][j],
                output_dir_name=output_dir_name
            )

            _plot_lagged_ships_saliency(
                data_dict=data_dict, saliency_dict=saliency_dict,
                model_metadata_dict=model_metadata_dict,
                cyclone_id_string=unique_cyclone_id_strings[i],
                init_time_unix_sec=saliency_dict[saliency.INIT_TIMES_KEY][j],
                output_dir_name=output_dir_name
            )

            _plot_forecast_ships_saliency(
                data_dict=data_dict, saliency_dict=saliency_dict,
                model_metadata_dict=model_metadata_dict,
                cyclone_id_string=unique_cyclone_id_strings[i],
                init_time_unix_sec=saliency_dict[saliency.INIT_TIMES_KEY][j],
                output_dir_name=output_dir_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_name=getattr(INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
