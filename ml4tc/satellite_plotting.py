"""Plotting methods for satellite data."""

import os
import sys
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import longitude_conversion as lng_conversion
import error_checking
import gg_plotting_utils

DEFAULT_MIN_TEMP_KELVINS = 190.
DEFAULT_MAX_TEMP_KELVINS = 310.
DEFAULT_CUTOFF_TEMP_KELVINS = 240.


def _grid_points_to_edges(grid_point_coords):
    """Converts grid-point coordinates to grid-cell-edge coordinates.

    P = number of grid points

    :param grid_point_coords: length-P numpy array of grid-point coordinates, in
        increasing order.
    :return: grid_cell_edge_coords: length-(P + 1) numpy array of grid-cell-edge
        coordinates, also in increasing order.
    """

    grid_cell_edge_coords = (grid_point_coords[:-1] + grid_point_coords[1:]) / 2
    first_edge_coords = (
        grid_point_coords[0] - numpy.diff(grid_point_coords[:2]) / 2
    )
    last_edge_coords = (
        grid_point_coords[-1] + numpy.diff(grid_point_coords[-2:]) / 2
    )

    return numpy.concatenate((
        first_edge_coords, grid_cell_edge_coords, last_edge_coords
    ))


def get_colour_scheme(
        min_temp_kelvins=DEFAULT_MIN_TEMP_KELVINS,
        max_temp_kelvins=DEFAULT_MAX_TEMP_KELVINS,
        cutoff_temp_kelvins=DEFAULT_CUTOFF_TEMP_KELVINS):
    """Returns colour scheme for brightness temperature.

    :param min_temp_kelvins: Minimum temperature in colour scheme.
    :param max_temp_kelvins: Max temperature in colour scheme.
    :param cutoff_temp_kelvins: Cutoff between grey and non-grey colours.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    error_checking.assert_is_greater(max_temp_kelvins, cutoff_temp_kelvins)
    error_checking.assert_is_greater(cutoff_temp_kelvins, min_temp_kelvins)

    normalized_values = numpy.linspace(0, 1, num=1001, dtype=float)

    grey_colour_map_object = pyplot.get_cmap('Greys')
    grey_temps_kelvins = numpy.linspace(
        cutoff_temp_kelvins, max_temp_kelvins, num=1001, dtype=float
    )
    grey_rgb_matrix = grey_colour_map_object(normalized_values)[:, :-1]

    plasma_colour_map_object = pyplot.get_cmap('plasma')
    plasma_temps_kelvins = numpy.linspace(
        min_temp_kelvins, cutoff_temp_kelvins, num=1001, dtype=float
    )
    plasma_rgb_matrix = plasma_colour_map_object(normalized_values)[:, :-1]

    boundary_temps_kelvins = numpy.concatenate(
        (plasma_temps_kelvins, grey_temps_kelvins), axis=0
    )
    rgb_matrix = numpy.concatenate(
        (plasma_rgb_matrix, grey_rgb_matrix), axis=0
    )

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        boundary_temps_kelvins, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def add_colour_bar(
        brightness_temp_matrix_kelvins, axes_object, colour_map_object,
        colour_norm_object, orientation_string, font_size):
    """Adds colour bar to plot.

    :param brightness_temp_matrix_kelvins: See doc for `plot_2d_grid_regular`.
    :param axes_object: Same.
    :param colour_map_object: See doc for `get_colour_scheme`.
    :param colour_norm_object: Same.
    :param orientation_string: Orientation ("vertical" or "horizontal").
    :param font_size: Font size for labels on colour bar.
    :return: colour_bar_object: See doc for `plot_2d_grid_regular`.
    """

    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, num_dimensions=2
    )
    error_checking.assert_is_geq_numpy_array(
        brightness_temp_matrix_kelvins, 0, allow_nan=True
    )
    error_checking.assert_is_string(orientation_string)
    error_checking.assert_is_greater(font_size, 0)

    if orientation_string == 'horizontal' and font_size > 30:
        padding = 0.15
    else:
        padding = None

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=brightness_temp_matrix_kelvins,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=orientation_string,
        extend_min=True, extend_max=True, font_size=font_size, padding=padding
    )

    num_tick_values = 1 + int(numpy.round(
        (DEFAULT_MAX_TEMP_KELVINS - DEFAULT_MIN_TEMP_KELVINS) / 10
    ))
    tick_values = numpy.linspace(
        DEFAULT_MIN_TEMP_KELVINS, DEFAULT_MAX_TEMP_KELVINS, num=num_tick_values,
        dtype=int
    )

    tick_strings = ['{0:d}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return colour_bar_object


def plot_2d_grid_regular(
        brightness_temp_matrix_kelvins, axes_object, latitudes_deg_n,
        longitudes_deg_e, cbar_orientation_string='vertical', font_size=30.):
    """Plots brightness temperature on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param brightness_temp_matrix_kelvins: M-by-N numpy array of brightness
        temperatures.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param latitudes_deg_n: length-M numpy array of grid-point latitudes (deg
        north).
    :param longitudes_deg_e: length-N numpy array of grid-point longitudes (deg
        east).
    :param cbar_orientation_string: Colour-bar orientation.  May be
        "horizontal", "vertical", or None.
    :param font_size: Font size.
    :return: colour_bar_object: Colour-bar handle (instance of
        `matplotlib.pyplot.colorbar`).
    """

    error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(latitudes_deg_n), 0.
    )

    error_checking.assert_is_numpy_array(longitudes_deg_e, num_dimensions=1)
    longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        longitudes_deg_e
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(longitudes_deg_e), 0.
    )

    num_rows = len(latitudes_deg_n)
    num_columns = len(longitudes_deg_e)
    expected_dim = numpy.array([num_rows, num_columns], dtype=int)

    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, exact_dimensions=expected_dim
    )
    error_checking.assert_is_greater_numpy_array(
        brightness_temp_matrix_kelvins, 0., allow_nan=True
    )

    if cbar_orientation_string is not None:
        error_checking.assert_is_string(cbar_orientation_string)

    edge_latitudes_deg_n = _grid_points_to_edges(latitudes_deg_n)
    edge_longitudes_deg_e = _grid_points_to_edges(longitudes_deg_e)
    edge_temp_matrix_kelvins = grids.latlng_field_grid_points_to_edges(
        field_matrix=brightness_temp_matrix_kelvins,
        min_latitude_deg=1., min_longitude_deg=1.,
        lat_spacing_deg=1e-6, lng_spacing_deg=1e-6
    )[0]

    edge_temp_matrix_kelvins = numpy.ma.masked_where(
        numpy.isnan(edge_temp_matrix_kelvins), edge_temp_matrix_kelvins
    )
    colour_map_object, colour_norm_object = get_colour_scheme()

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    axes_object.pcolormesh(
        edge_longitudes_deg_e, edge_latitudes_deg_n, edge_temp_matrix_kelvins,
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', zorder=-1e11
    )

    if cbar_orientation_string is None:
        return None

    return add_colour_bar(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object, colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=cbar_orientation_string, font_size=font_size
    )