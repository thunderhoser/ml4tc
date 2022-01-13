"""Plots SHIPS data."""

import os
import sys
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import time_conversion
import error_checking
import ships_io
import example_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

DEFAULT_FONT_SIZE = 20
FORECAST_HOUR_FONT_SIZE = 7
LAG_TIME_FONT_SIZE = 20
PREDICTOR_FONT_SIZE = 7

MIN_NORMALIZED_VALUE = -3.
MAX_NORMALIZED_VALUE = 3.
COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)


def plot_lagged_predictors_one_init_time(
        example_table_xarray, init_time_index, predictor_indices,
        info_string=None, figure_object=None, axes_object=None
):
    """Plots lagged predictors for one initialization time.

    :param example_table_xarray: xarray table in format returned by
        `example_io.read_file`.
    :param init_time_index: Index of initial time to plot.
    :param predictor_indices: 1-D numpy array with indices of predictors to
        plot.
    :param info_string: Info string (to be appended to title).
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).  If None, will create new figure.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If None, will create new
        axes.
    :return: figure_object: See input doc.
    :return: axes_object: See input doc.
    :return: pathless_output_file_name: Pathless name for output file.
    """

    xt = example_table_xarray

    lag_times_hours = xt.coords[example_utils.SHIPS_LAG_TIME_DIM].values
    predictor_matrix = (
        xt[example_utils.SHIPS_PREDICTORS_LAGGED_KEY].values[
            init_time_index, ...
        ][:, predictor_indices]
    )
    predictor_matrix = predictor_matrix.astype(float)

    if figure_object is None or axes_object is None:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    colour_norm_object = pyplot.Normalize(
        vmin=MIN_NORMALIZED_VALUE, vmax=MAX_NORMALIZED_VALUE
    )
    axes_object.imshow(
        predictor_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
        norm=colour_norm_object
    )

    y_tick_values = numpy.linspace(
        0, predictor_matrix.shape[0] - 1, num=predictor_matrix.shape[0],
        dtype=float
    )
    y_tick_labels = ['{0:.1f}'.format(t) for t in lag_times_hours]
    pyplot.yticks(y_tick_values, y_tick_labels, fontsize=LAG_TIME_FONT_SIZE)
    axes_object.set_ylabel('Lag time (hours)')

    x_tick_values = numpy.linspace(
        0, predictor_matrix.shape[1] - 1, num=predictor_matrix.shape[1],
        dtype=float
    )
    x_tick_labels = (
        xt.coords[example_utils.SHIPS_PREDICTOR_LAGGED_DIM].values[
            predictor_indices
        ].tolist()
    )
    pyplot.xticks(
        x_tick_values, x_tick_labels, rotation=90., fontsize=PREDICTOR_FONT_SIZE
    )

    init_time_unix_sec = (
        xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values[init_time_index]
    )
    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    cyclone_id_string = xt[ships_io.CYCLONE_ID_KEY].values[init_time_index]
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    title_string = 'SHIPS for {0:s} at {1:s}'.format(
        cyclone_id_string, init_time_string
    )
    if info_string is not None:
        title_string += '; {0:s}'.format(info_string)

    axes_object.set_title(title_string)

    pathless_output_file_name = '{0:s}_{1:s}_ships_lagged.jpg'.format(
        cyclone_id_string, init_time_string
    )

    return figure_object, axes_object, pathless_output_file_name


def plot_fcst_predictors_one_init_time(
        example_table_xarray, init_time_index, predictor_indices,
        info_string=None, figure_object=None, axes_object=None
):
    """Plots forecast predictors for one initialization time.

    :param example_table_xarray: See doc for
        `plot_lagged_predictors_one_init_time`.
    :param init_time_index: Same.
    :param predictor_indices: Same.
    :param info_string: Same.
    :param figure_object: Same.
    :param axes_object: Same.
    :return: figure_object: Same.
    :return: axes_object: Same.
    :return: pathless_output_file_name: Same.
    """

    xt = example_table_xarray

    forecast_times_hours = numpy.round(
        xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values
    ).astype(int)

    predictor_matrix = (
        xt[example_utils.SHIPS_PREDICTORS_FORECAST_KEY].values[
            init_time_index, ...
        ][:, predictor_indices]
    )
    predictor_matrix = predictor_matrix.astype(float)

    if figure_object is None or axes_object is None:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    colour_norm_object = pyplot.Normalize(
        vmin=MIN_NORMALIZED_VALUE, vmax=MAX_NORMALIZED_VALUE
    )
    axes_object.imshow(
        predictor_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
        norm=colour_norm_object
    )

    y_tick_values = numpy.linspace(
        0, predictor_matrix.shape[0] - 1, num=predictor_matrix.shape[0],
        dtype=float
    )
    y_tick_labels = ['{0:d}'.format(t) for t in forecast_times_hours]
    pyplot.yticks(
        y_tick_values, y_tick_labels, fontsize=FORECAST_HOUR_FONT_SIZE
    )
    axes_object.set_ylabel('Forecast time (hours)')

    x_tick_values = numpy.linspace(
        0, predictor_matrix.shape[1] - 1, num=predictor_matrix.shape[1],
        dtype=float
    )
    x_tick_labels = (
        xt.coords[example_utils.SHIPS_PREDICTOR_FORECAST_DIM].values[
            predictor_indices
        ].tolist()
    )
    pyplot.xticks(
        x_tick_values, x_tick_labels, rotation=90., fontsize=PREDICTOR_FONT_SIZE
    )

    init_time_unix_sec = (
        xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values[init_time_index]
    )
    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    cyclone_id_string = xt[ships_io.CYCLONE_ID_KEY].values[init_time_index]
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    title_string = 'SHIPS for {0:s} at {1:s}'.format(
        cyclone_id_string, init_time_string
    )
    if info_string is not None:
        title_string += '; {0:s}'.format(info_string)

    axes_object.set_title(title_string)

    pathless_output_file_name = '{0:s}_{1:s}_ships_forecast.jpg'.format(
        cyclone_id_string, init_time_string
    )

    return figure_object, axes_object, pathless_output_file_name


def plot_pm_signs_one_init_time(
        data_matrix, axes_object, font_size, colour_map_object,
        max_absolute_colour_value):
    """Plots data at one initialization time with plus-minus signs.

    Positive values will be plotted as plus signs; negative values will be
    plotted as minus signs; and the colour map will be applied to absolute
    values only.

    This method is good for plotting saliency, input * gradient, or any other
    explainable-ML result that can be either positive or negative.

    V = number of SHIPS variables
    T = number of lag times or forecast hours

    :param data_matrix: T-by-V numpy array of data values to plot.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param font_size: Font size for plus and minus signs.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param max_absolute_colour_value: Max absolute value in colour scheme.
    """

    error_checking.assert_is_numpy_array_without_nan(data_matrix)
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
    error_checking.assert_is_geq(max_absolute_colour_value, 0.)
    max_absolute_colour_value = max([max_absolute_colour_value, 0.001])

    num_grid_rows = data_matrix.shape[0]
    num_grid_columns = data_matrix.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = grids.get_xy_grid_points(
        x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
        x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    colour_norm_object = pyplot.Normalize(
        vmin=0., vmax=max_absolute_colour_value
    )
    rgb_matrix = colour_map_object(colour_norm_object(
        numpy.absolute(data_matrix)
    ))[..., :-1]

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if data_matrix[i, j] >= 0:
                axes_object.text(
                    x_coords[j], y_coords[i], '+', fontsize=font_size,
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='center', transform=axes_object.transAxes
                )
            else:
                axes_object.text(
                    x_coords[j], y_coords[i], '_', fontsize=font_size,
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='bottom', transform=axes_object.transAxes
                )


def plot_raw_numbers_one_init_time(
        data_matrix, axes_object, font_size, colour_map_object,
        min_colour_value, max_colour_value, number_format_string):
    """Plots data at one initialization time with raw numbers (text).

    This method is good for plotting saliency, input * gradient, or any other
    explainable-ML result that can have only one sign (positive or negative).

    V = number of SHIPS variables
    T = number of lag times or forecast hours

    :param data_matrix: T-by-V numpy array of data values to plot.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param font_size: Font size.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param number_format_string: Format for printing numbers.  Valid examples
        are ".2f", "d", etc.
    """

    error_checking.assert_is_numpy_array_without_nan(data_matrix)
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
    error_checking.assert_is_greater(max_colour_value, min_colour_value)
    error_checking.assert_is_string(number_format_string)

    number_format_string = '{0:' + number_format_string + '}'

    num_grid_rows = data_matrix.shape[0]
    num_grid_columns = data_matrix.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = grids.get_xy_grid_points(
        x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
        x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    rgb_matrix = colour_map_object(colour_norm_object(
        numpy.absolute(data_matrix)
    ))[..., :-1]

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            axes_object.text(
                x_coords[j], y_coords[i],
                number_format_string.format(data_matrix[i, j]),
                fontsize=font_size, color=rgb_matrix[i, j, ...],
                horizontalalignment='center', verticalalignment='center',
                transform=axes_object.transAxes
            )


# def plot_colour_map_one_init_time(
#         data_matrix, colour_map_object, colour_norm_object,
#         figure_object=None, axes_object=None):
#     """Plots data at one initialization time with colour map.
#
#     This method is good for plotting the heat map from any explainable-ML
#     method.
#
#     :param data_matrix: See doc for `plot_pm_signs_one_init_time`.
#     :param colour_map_object: Colour scheme (instance of
#         `matplotlib.pyplot.cm`).
#     :param colour_norm_object: Colour normalization (maps from data space to
#         colour-bar space, which goes from 0...1).  This is an instance of
#         `matplotlib.colors.Normalize`.
#     :param figure_object: See doc for `plot_lagged_predictors_one_init_time`.
#     :param axes_object: Same.
#     :return: figure_object: Same.
#     :return: axes_object: Same.
#     """
#
#     error_checking.assert_is_numpy_array_without_nan(data_matrix)
#     error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
#
#     num_grid_rows = data_matrix.shape[0]
#     num_grid_columns = data_matrix.shape[1]
#     x_coord_spacing = num_grid_columns ** -1
#     y_coord_spacing = num_grid_rows ** -1
#
#     x_coords, y_coords = grids.get_xy_grid_points(
#         x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
#         x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
#         num_rows=num_grid_rows, num_columns=num_grid_columns
#     )
#
#     if figure_object is None or axes_object is None:
#         figure_object, axes_object = pyplot.subplots(
#             1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
#         )
