"""Plots scalar (ungridded) satellite data."""

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
import example_utils
import satellite_utils

TIME_FORMAT_SECONDS = '%Y-%m-%d-%H%M%S'
TIME_FORMAT_MINUTES = '%Y-%m-%d-%H%M'

MIN_NORMALIZED_VALUE = -3.
MAX_NORMALIZED_VALUE = 3.
COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

BAR_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
BAR_EDGE_COLOUR = numpy.full(3, 0.)
BAR_FONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
BAR_EDGE_WIDTH = 2.
BAR_FONT_SIZE = 20

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

DEFAULT_FONT_SIZE = 30
DEFAULT_TIME_TICK_FONT_SIZE = 30
DEFAULT_PREDICTOR_TICK_FONT_SIZE = 20

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)


def plot_bar_graph_one_time(
        example_table_xarray, time_index, predictor_indices, info_string=None,
        figure_object=None, axes_object=None):
    """Plots predictors at one time as bar graph.

    :param example_table_xarray: xarray table in format returned by
        `example_io.read_file`.
    :param time_index: Index of valid time to plot.
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

    error_checking.assert_is_integer(time_index)
    error_checking.assert_is_geq(time_index, 0)
    error_checking.assert_is_integer_numpy_array(predictor_indices)
    error_checking.assert_is_geq_numpy_array(predictor_indices, 0)
    if info_string is not None:
        error_checking.assert_is_string(info_string)

    xt = example_table_xarray
    predictor_values = (
        xt[example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY].values[
            time_index, predictor_indices
        ]
    )

    num_predictors = len(predictor_values)
    y_coords = numpy.linspace(
        0, num_predictors - 1, num=num_predictors, dtype=float
    )

    if figure_object is None or axes_object is None:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    axes_object.barh(
        y_coords, predictor_values, color=BAR_FACE_COLOUR,
        edgecolor=BAR_EDGE_COLOUR, linewidth=BAR_EDGE_WIDTH
    )

    pyplot.yticks([], [])
    axes_object.set_xlim(MIN_NORMALIZED_VALUE, MAX_NORMALIZED_VALUE)

    predictor_names = xt.coords[
        example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
    ].values[predictor_indices].tolist()

    for j in range(num_predictors):
        axes_object.text(
            0, y_coords[j], predictor_names[j], color=BAR_FONT_COLOUR,
            horizontalalignment='center', verticalalignment='center',
            fontsize=BAR_FONT_SIZE, fontweight='bold'
        )

    valid_time_unix_sec = (
        xt.coords[example_utils.SATELLITE_TIME_DIM].values[time_index]
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT_SECONDS
    )
    cyclone_id_string = xt[satellite_utils.CYCLONE_ID_KEY].values[time_index]
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    title_string = 'Satellite for {0:s} at {1:s}'.format(
        cyclone_id_string, valid_time_string
    )
    if info_string is not None:
        title_string += '; {0:s}'.format(info_string)

    axes_object.set_title(title_string)

    pathless_output_file_name = '{0:s}_{1:s}_scalar_satellite.jpg'.format(
        cyclone_id_string, valid_time_string
    )

    return figure_object, axes_object, pathless_output_file_name


def plot_colour_map_multi_times(
        example_table_xarray, time_indices, predictor_indices,
        time_tick_font_size=DEFAULT_TIME_TICK_FONT_SIZE,
        predictor_tick_font_size=DEFAULT_PREDICTOR_TICK_FONT_SIZE,
        info_string=None, figure_object=None, axes_object=None):
    """Plots predictors at many times as colour map.

    :param example_table_xarray: See doc for `plot_bar_graph_one_time`.
    :param time_indices: 1-D numpy array with indices of valid times to plot.
    :param predictor_indices: See doc for `plot_bar_graph_one_time`.
    :param time_tick_font_size: Font size for tick labels on y-axis.
    :param predictor_tick_font_size: Font size for tick labels on x-axis.
    :param info_string: Same.
    :param figure_object: Same.
    :param axes_object: Same.
    :return: figure_object: Same.
    :return: axes_object: Same.
    :return: pathless_output_file_name: Same.
    """

    error_checking.assert_is_integer_numpy_array(time_indices)
    error_checking.assert_is_geq_numpy_array(time_indices, 0)
    error_checking.assert_is_integer_numpy_array(predictor_indices)
    error_checking.assert_is_geq_numpy_array(predictor_indices, 0)
    if info_string is not None:
        error_checking.assert_is_string(info_string)

    xt = example_table_xarray
    predictor_matrix = (
        xt[example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY].values[
            time_indices, :
        ][:, predictor_indices]
    )
    predictor_matrix = predictor_matrix.astype(float)

    valid_times_unix_sec = (
        xt.coords[example_utils.SATELLITE_TIME_DIM].values[time_indices]
    )
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_MINUTES)
        for t in valid_times_unix_sec
    ]
    cyclone_id_strings = xt[satellite_utils.CYCLONE_ID_KEY].values[time_indices]
    if not isinstance(cyclone_id_strings[0], str):
        cyclone_id_strings = [s.decode('utf-8') for s in cyclone_id_strings]

    cyclone_id_strings = numpy.unique(numpy.array(cyclone_id_strings))
    assert len(cyclone_id_strings) == 1
    cyclone_id_string = cyclone_id_strings[0]

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
    y_tick_labels = ['{0:s}'.format(t) for t in valid_time_strings]
    pyplot.yticks(y_tick_values, y_tick_labels, fontsize=time_tick_font_size)
    axes_object.set_ylabel('Valid time')

    x_tick_values = numpy.linspace(
        0, predictor_matrix.shape[1] - 1, num=predictor_matrix.shape[1],
        dtype=float
    )
    x_tick_labels = xt.coords[
        example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
    ].values[predictor_indices].tolist()

    pyplot.xticks(
        x_tick_values, x_tick_labels, rotation=90.,
        fontsize=predictor_tick_font_size
    )

    title_string = 'Satellite for {0:s}'.format(cyclone_id_string)
    if info_string is not None:
        title_string += '; {0:s}'.format(info_string)

    axes_object.set_title(title_string)

    pathless_output_file_name = '{0:s}_{1:s}-{2:s}_scalar_satellite.jpg'.format(
        cyclone_id_string,
        valid_time_strings[numpy.argmin(valid_times_unix_sec)].replace('-', ''),
        valid_time_strings[numpy.argmax(valid_times_unix_sec)].replace('-', '')
    )

    return figure_object, axes_object, pathless_output_file_name


def plot_pm_signs_multi_times(
        data_matrix, axes_object, font_size, colour_map_object,
        max_absolute_colour_value):
    """Plots data at many times with plus-minus signs.

    Positive values will be plotted as plus signs; negative values will be
    plotted as minus signs; and the colour map will be applied to absolute
    values only.

    This method is good for plotting saliency, input * gradient, or any other
    explainable-ML result that can be either positive or negative.

    V = number of satellite variables
    T = number of times

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


def plot_raw_numbers_multi_times(
        data_matrix, axes_object, font_size, colour_map_object,
        min_colour_value, max_colour_value, number_format_string,
        plot_in_log_space=False):
    """Plots data at many times with raw numbers (text).

    This method is good for plotting saliency, input * gradient, or any other
    explainable-ML result that can have only one sign (positive or negative).

    V = number of satellite variables
    T = number of times

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
    :param plot_in_log_space: Boolean flag.  If True (False), colours will be
        scaled logarithmically (linearly).
    """

    error_checking.assert_is_numpy_array_without_nan(data_matrix)
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
    error_checking.assert_is_greater(max_colour_value, min_colour_value)
    error_checking.assert_is_string(number_format_string)
    error_checking.assert_is_boolean(plot_in_log_space)

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

    if plot_in_log_space:
        rgb_matrix = colour_map_object(colour_norm_object(
            numpy.log10(1 + numpy.absolute(data_matrix))
        ))[..., :-1]
    else:
        rgb_matrix = colour_map_object(colour_norm_object(
            numpy.absolute(data_matrix)
        ))[..., :-1]

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            axes_object.text(
                x_coords[j], y_coords[i],
                number_format_string.format(data_matrix[i, j]),
                fontsize=font_size,
                fontstyle='italic' if data_matrix[i, j] < 0 else 'normal',
                fontweight='bold' if data_matrix[i, j] > 0 else 'normal',
                color=rgb_matrix[i, j, ...],
                horizontalalignment='center', verticalalignment='center',
                transform=axes_object.transAxes
            )
