"""Imputation of missing data."""

import os
import sys
import numpy
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import general_utils
import example_utils
import normalization

LARGE_NEGATIVE_NUMBER = -1e10


def _interp_in_time(input_times, input_data_values, min_coverage_fraction,
                    min_num_times):
    """Interpolates in time to fill missing values (NaN).

    T = number of times

    :param input_times: length-T numpy array of times (any units).
    :param input_data_values: length-T numpy array of data values.
    :param min_coverage_fraction: Minimum coverage fraction.  If fewer times
        have actual values (not NaN), then interpolation will not be done.
    :param min_num_times: Minimum number of times.  If fewer times have actual
        values (not NaN), then interpolation will not be done.
    :return: output_data_values: length-T numpy array of data values after
        interpolation.
    """

    missing_flags = numpy.isnan(input_data_values)
    if not numpy.any(missing_flags):
        return input_data_values

    real_indices = numpy.where(numpy.invert(missing_flags))[0]
    coverage_fraction = float(len(real_indices)) / len(input_data_values)

    if len(real_indices) < min_num_times:
        return input_data_values
    if coverage_fraction < min_coverage_fraction:
        return input_data_values

    fill_values = (
        input_data_values[real_indices][0], input_data_values[real_indices][-1]
    )

    interp_object = interp1d(
        input_times[real_indices], input_data_values[real_indices],
        kind='linear', bounds_error=False, assume_sorted=True,
        fill_value=fill_values
    )

    return interp_object(input_times)


def _interp_in_space(input_matrix, min_coverage_fraction, min_num_pixels):
    """Interpolates in space to fill missing values (NaN).

    This method assumes that the grid is equidistant.

    M = number of rows in grid
    N = number of columns in grid

    :param input_matrix: M-by-N numpy array of data values.
    :param min_coverage_fraction: Minimum coverage fraction.  If fewer pixels
        have actual values (not NaN), then interpolation will not be done.
    :param min_num_pixels: Minimum number of pixels.  If fewer pixels have
        actual values (not NaN), then interpolation will not be done.
    :return: output_matrix: M-by-N numpy array of data values after
        interpolation.
    """

    missing_flag_matrix = numpy.isnan(input_matrix)
    if not numpy.any(missing_flag_matrix):
        return input_matrix

    num_real_pixels = numpy.sum(numpy.invert(missing_flag_matrix))
    coverage_fraction = float(num_real_pixels) / input_matrix.size

    if num_real_pixels < min_num_pixels:
        return input_matrix
    if coverage_fraction < min_coverage_fraction:
        return input_matrix

    return general_utils.fill_nans(input_matrix)


def impute_examples(
        example_table_xarray_unnorm, normalization_table_xarray,
        min_temporal_coverage_fraction, min_num_times,
        min_spatial_coverage_fraction, min_num_pixels,
        fill_value_for_isotherm_stuff=LARGE_NEGATIVE_NUMBER):
    """Imputes data in learning examples.

    :param example_table_xarray_unnorm: xarray table with unnormalized learning
        examples, in format returned by `example_io.read_file`.
    :param normalization_table_xarray: xarray table with normalization
        parameters, in format returned by `normalization.read_file`.
    :param min_temporal_coverage_fraction: See doc for `_interp_in_time`.
    :param min_num_times: Same.
    :param min_spatial_coverage_fraction: See doc for `_interp_in_space`.
    :param min_num_pixels: Same.
    :param fill_value_for_isotherm_stuff: Fill value for isotherm-related
        variables.  If None, will use climatology as fill value.  I suggest
        setting a large negative fill value, because missing values for
        isotherm-related variables usually mean that the given isotherm does not
        exist in the ocean column -- not that the value is just unknown.
    :return: example_table_xarray_unnorm: Same as input but with missing values
        imputed.
    """

    error_checking.assert_is_geq(min_temporal_coverage_fraction, 0.25)
    error_checking.assert_is_less_than(min_temporal_coverage_fraction, 1.)
    error_checking.assert_is_integer(min_num_times)
    error_checking.assert_is_geq(min_num_times, 4)
    error_checking.assert_is_geq(min_spatial_coverage_fraction, 0.25)
    error_checking.assert_is_less_than(min_spatial_coverage_fraction, 1.)
    error_checking.assert_is_integer(min_num_pixels)
    error_checking.assert_is_geq(min_num_pixels, 10000)

    if fill_value_for_isotherm_stuff is not None:
        error_checking.assert_is_real_number(fill_value_for_isotherm_stuff)

    xt = example_table_xarray_unnorm
    nt = normalization_table_xarray

    example_predictor_names = (
        xt.coords[example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM].values
    ).tolist()
    example_predictor_matrix = (
        xt[example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY].values
    )
    norm_predictor_names = (
        nt.coords[normalization.SATELLITE_PREDICTOR_UNGRIDDED_DIM].values
    ).tolist()
    valid_times_unix_sec = (
        xt.coords[example_utils.SATELLITE_TIME_DIM].values
    ).astype(float)

    for j in range(len(example_predictor_names)):
        print('Imputing values of {0:s}...'.format(example_predictor_names[j]))
        orig_nan_fraction = numpy.mean(numpy.isnan(
            example_predictor_matrix[:, j]
        ))

        example_predictor_matrix[:, j] = _interp_in_time(
            input_times=valid_times_unix_sec,
            input_data_values=example_predictor_matrix[:, j],
            min_coverage_fraction=min_temporal_coverage_fraction,
            min_num_times=min_num_times
        )

        new_nan_fraction = numpy.mean(numpy.isnan(
            example_predictor_matrix[:, j]
        ))
        print((
            'Original and new NaN fractions after interp = {0:.4f}, {1:.4f}'
        ).format(
            orig_nan_fraction, new_nan_fraction
        ))

        k = norm_predictor_names.index(example_predictor_names[j])
        climo_value = numpy.nanmedian(
            nt[normalization.SATELLITE_PREDICTORS_UNGRIDDED_KEY].values[:, k]
        )
        if numpy.isnan(climo_value):
            climo_value = 0.

        example_predictor_matrix[:, j][
            numpy.isnan(example_predictor_matrix[:, j])
        ] = climo_value

        assert not numpy.any(numpy.isnan(example_predictor_matrix[:, j]))

    xt[example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY].values = (
        example_predictor_matrix
    )
    print('\n')

    example_predictor_names = (
        xt.coords[example_utils.SATELLITE_PREDICTOR_GRIDDED_DIM].values
    ).tolist()
    example_predictor_matrix = (
        xt[example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY].values
    )
    num_times = example_predictor_matrix.shape[0]
    norm_predictor_names = (
        nt.coords[normalization.SATELLITE_PREDICTOR_GRIDDED_DIM].values
    ).tolist()

    for j in range(len(example_predictor_names)):
        print('Imputing values of {0:s}...'.format(example_predictor_names[j]))
        orig_nan_fraction = numpy.mean(numpy.isnan(
            example_predictor_matrix[..., j]
        ))

        for i in range(num_times):
            example_predictor_matrix[i, ..., j] = _interp_in_space(
                input_matrix=example_predictor_matrix[i, ..., j],
                min_coverage_fraction=min_spatial_coverage_fraction,
                min_num_pixels=min_num_pixels
            )

        new_nan_fraction = numpy.mean(numpy.isnan(
            example_predictor_matrix[..., j]
        ))
        print((
            'Original and new NaN fractions after interp = {0:.4f}, {1:.4f}'
        ).format(
            orig_nan_fraction, new_nan_fraction
        ))

        k = norm_predictor_names.index(example_predictor_names[j])
        climo_value = numpy.nanmedian(
            nt[normalization.SATELLITE_PREDICTORS_GRIDDED_KEY].values[:, k]
        )
        if numpy.isnan(climo_value):
            climo_value = 0.

        example_predictor_matrix[..., j][
            numpy.isnan(example_predictor_matrix[..., j])
        ] = climo_value

        assert not numpy.any(numpy.isnan(example_predictor_matrix[..., j]))

    xt[example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY].values = (
        example_predictor_matrix
    )
    print('\n')

    example_predictor_names = (
        xt.coords[example_utils.SHIPS_PREDICTOR_LAGGED_DIM].values
    ).tolist()
    example_predictor_matrix = (
        xt[example_utils.SHIPS_PREDICTORS_LAGGED_KEY].values
    )
    num_lag_times = example_predictor_matrix.shape[1]
    norm_predictor_names = (
        nt.coords[normalization.SHIPS_PREDICTOR_LAGGED_DIM].values
    ).tolist()
    valid_times_unix_sec = (
        xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values
    ).astype(float)

    for j in range(len(example_predictor_names)):
        print('Imputing values of {0:s}...'.format(example_predictor_names[j]))
        orig_nan_fraction = numpy.mean(numpy.isnan(
            example_predictor_matrix[..., j]
        ))

        for i in range(num_lag_times):
            example_predictor_matrix[:, i, j] = _interp_in_time(
                input_times=valid_times_unix_sec,
                input_data_values=example_predictor_matrix[:, i, j],
                min_coverage_fraction=min_temporal_coverage_fraction,
                min_num_times=min_num_times
            )

        new_nan_fraction = numpy.mean(numpy.isnan(
            example_predictor_matrix[..., j]
        ))
        print((
            'Original and new NaN fractions after interp = {0:.4f}, {1:.4f}'
        ).format(
            orig_nan_fraction, new_nan_fraction
        ))

        k = norm_predictor_names.index(example_predictor_names[j])
        climo_value = numpy.nanmedian(
            nt[normalization.SHIPS_PREDICTORS_LAGGED_KEY].values[:, k]
        )
        if numpy.isnan(climo_value):
            climo_value = 0.

        if (
                '_isotherm_' in example_predictor_names[j] and
                fill_value_for_isotherm_stuff is not None
        ):
            climo_value = fill_value_for_isotherm_stuff + 0.

        example_predictor_matrix[..., j][
            numpy.isnan(example_predictor_matrix[..., j])
        ] = climo_value

        assert not numpy.any(numpy.isnan(example_predictor_matrix[..., j]))

    xt[example_utils.SHIPS_PREDICTORS_LAGGED_KEY].values = (
        example_predictor_matrix
    )
    print('\n')

    example_predictor_names = (
        xt.coords[example_utils.SHIPS_PREDICTOR_FORECAST_DIM].values
    ).tolist()
    example_predictor_matrix = (
        xt[example_utils.SHIPS_PREDICTORS_FORECAST_KEY].values
    )
    num_times = example_predictor_matrix.shape[0]
    norm_predictor_names = (
        nt.coords[normalization.SHIPS_PREDICTOR_FORECAST_DIM].values
    ).tolist()
    forecast_hours = (
        xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values
    ).astype(float)

    for j in range(len(example_predictor_names)):
        print('Imputing values of {0:s}...'.format(example_predictor_names[j]))
        orig_nan_fraction = numpy.mean(numpy.isnan(
            example_predictor_matrix[..., j]
        ))

        for i in range(num_times):
            example_predictor_matrix[i, :, j] = _interp_in_time(
                input_times=forecast_hours,
                input_data_values=example_predictor_matrix[i, :, j],
                min_coverage_fraction=min_temporal_coverage_fraction,
                min_num_times=min_num_times
            )

        new_nan_fraction = numpy.mean(numpy.isnan(
            example_predictor_matrix[..., j]
        ))
        print((
            'Original and new NaN fractions after interp = {0:.4f}, {1:.4f}'
        ).format(
            orig_nan_fraction, new_nan_fraction
        ))

        k = norm_predictor_names.index(example_predictor_names[j])
        climo_value = numpy.nanmedian(
            nt[normalization.SHIPS_PREDICTORS_FORECAST_KEY].values[:, k]
        )
        if numpy.isnan(climo_value):
            climo_value = 0.

        if (
                '_isotherm_' in example_predictor_names[j] and
                fill_value_for_isotherm_stuff is not None
        ):
            climo_value = fill_value_for_isotherm_stuff + 0.

        example_predictor_matrix[..., j][
            numpy.isnan(example_predictor_matrix[..., j])
        ] = climo_value

        assert not numpy.any(numpy.isnan(example_predictor_matrix[..., j]))

    return xt
