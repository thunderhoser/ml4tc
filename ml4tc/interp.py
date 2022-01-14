"""Interpolation methods."""

import os
import sys
import numpy
import scipy.interpolate

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_general_utils
import error_checking

NEAREST_NEIGHBOUR_METHOD_STRING = 'nearest'
SPLINE_METHOD_STRING = 'spline'
SPATIAL_INTERP_METHOD_STRINGS = [
    NEAREST_NEIGHBOUR_METHOD_STRING, SPLINE_METHOD_STRING
]

DEFAULT_SPLINE_DEGREE = 3
SMOOTHING_FACTOR_FOR_SPATIAL_INTERP = 0


def _nn_interp_from_xy_grid_to_points(
        input_matrix, sorted_grid_point_x_metres, sorted_grid_point_y_metres,
        query_x_coords_metres, query_y_coords_metres):
    """Nearest-neighbour interpolation from x-y grid to scattered points.

    :param input_matrix: See doc for `interp_from_xy_grid_to_points`.
    :param sorted_grid_point_x_metres: Same.
    :param sorted_grid_point_y_metres: Same.
    :param query_x_coords_metres: Same.
    :param query_y_coords_metres: Same.
    :return: interp_values: Same.
    """

    num_query_points = len(query_x_coords_metres)
    interp_values = numpy.full(num_query_points, numpy.nan)

    for i in range(num_query_points):
        _, this_row = gg_general_utils.find_nearest_value(
            sorted_grid_point_y_metres, query_y_coords_metres[i])
        _, this_column = gg_general_utils.find_nearest_value(
            sorted_grid_point_x_metres, query_x_coords_metres[i])
        interp_values[i] = input_matrix[this_row, this_column]

    return interp_values


def check_spatial_interp_method(interp_method_string):
    """Ensures that spatial-interpolation method is valid.

    :param interp_method_string: Interp method.
    :raises: ValueError: if `interp_method_string not in
        SPATIAL_INTERP_METHOD_STRINGS`.
    """

    error_checking.assert_is_string(interp_method_string)
    if interp_method_string not in SPATIAL_INTERP_METHOD_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid spatial-interp methods (listed above) do not '
            'include "{1:s}".'
        ).format(str(SPATIAL_INTERP_METHOD_STRINGS), interp_method_string)

        raise ValueError(error_string)


def interp_from_xy_grid_to_points(
        input_matrix, sorted_grid_point_x_metres, sorted_grid_point_y_metres,
        query_x_coords_metres, query_y_coords_metres,
        method_string=NEAREST_NEIGHBOUR_METHOD_STRING,
        spline_degree=DEFAULT_SPLINE_DEGREE, extrapolate=False):
    """Interpolation from x-y grid to scattered points.

    M = number of rows (unique y-coordinates at grid points)
    N = number of columns (unique x-coordinates at grid points)
    Q = number of query points

    :param input_matrix: M-by-N numpy array of gridded data.
    :param sorted_grid_point_x_metres: length-N numpy array with x-coordinates
        of grid points.  Must be sorted in ascending order.  Also,
        sorted_grid_point_x_metres[j] must match input_matrix[:, j].
    :param sorted_grid_point_y_metres: length-M numpy array with y-coordinates
        of grid points.  Must be sorted in ascending order.  Also,
        sorted_grid_point_y_metres[i] must match input_matrix[i, :].
    :param query_x_coords_metres: length-Q numpy array with x-coordinates of
        query points.
    :param query_y_coords_metres: length-Q numpy array with y-coordinates of
        query points.
    :param method_string: Interpolation method (must be accepted by
        `check_spatial_interp_method`).
    :param spline_degree: [used only if method_string = "spline"]
        Polynomial degree for spline interpolation (1 for linear, 2 for
        quadratic, 3 for cubic).
    :param extrapolate: Boolean flag.  If True, will extrapolate to points
        outside the domain (specified by `sorted_grid_point_x_metres` and
        `sorted_grid_point_y_metres`).  If False, will throw an error if there
        are query points outside the domain.
    :return: interp_values: length-Q numpy array of interpolated values.
    """

    error_checking.assert_is_numpy_array_without_nan(sorted_grid_point_x_metres)
    error_checking.assert_is_numpy_array(
        sorted_grid_point_x_metres, num_dimensions=1)
    num_grid_columns = len(sorted_grid_point_x_metres)

    error_checking.assert_is_numpy_array_without_nan(sorted_grid_point_y_metres)
    error_checking.assert_is_numpy_array(
        sorted_grid_point_y_metres, num_dimensions=1)
    num_grid_rows = len(sorted_grid_point_y_metres)

    error_checking.assert_is_real_numpy_array(input_matrix)
    error_checking.assert_is_numpy_array(
        input_matrix, exact_dimensions=numpy.array(
            [num_grid_rows, num_grid_columns]))

    error_checking.assert_is_numpy_array_without_nan(query_x_coords_metres)
    error_checking.assert_is_numpy_array(
        query_x_coords_metres, num_dimensions=1)
    num_query_points = len(query_x_coords_metres)

    error_checking.assert_is_numpy_array_without_nan(query_y_coords_metres)
    error_checking.assert_is_numpy_array(
        query_y_coords_metres, exact_dimensions=numpy.array([num_query_points]))

    error_checking.assert_is_boolean(extrapolate)
    if not extrapolate:
        error_checking.assert_is_geq_numpy_array(
            query_x_coords_metres, numpy.min(sorted_grid_point_x_metres))
        error_checking.assert_is_leq_numpy_array(
            query_x_coords_metres, numpy.max(sorted_grid_point_x_metres))
        error_checking.assert_is_geq_numpy_array(
            query_y_coords_metres, numpy.min(sorted_grid_point_y_metres))
        error_checking.assert_is_leq_numpy_array(
            query_y_coords_metres, numpy.max(sorted_grid_point_y_metres))

    check_spatial_interp_method(method_string)
    if method_string == NEAREST_NEIGHBOUR_METHOD_STRING:
        return _nn_interp_from_xy_grid_to_points(
            input_matrix=input_matrix,
            sorted_grid_point_x_metres=sorted_grid_point_x_metres,
            sorted_grid_point_y_metres=sorted_grid_point_y_metres,
            query_x_coords_metres=query_x_coords_metres,
            query_y_coords_metres=query_y_coords_metres)

    interp_object = scipy.interpolate.RectBivariateSpline(
        sorted_grid_point_y_metres, sorted_grid_point_x_metres, input_matrix,
        kx=spline_degree, ky=spline_degree,
        s=SMOOTHING_FACTOR_FOR_SPATIAL_INTERP)

    return interp_object(
        query_y_coords_metres, query_x_coords_metres, grid=False)
