"""Methods for normalizing predictor variables."""

import os
import sys
import numpy
import xarray
import scipy.stats

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import example_io
import example_utils

MIN_CUMULATIVE_DENSITY = 1e-6
MAX_CUMULATIVE_DENSITY = 1. - 1e-6

GRIDDED_INDEX_DIM = 'gridded_index'
UNGRIDDED_INDEX_DIM = 'ungridded_index'
SATELLITE_PREDICTOR_UNGRIDDED_DIM = (
    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
)
SATELLITE_PREDICTOR_GRIDDED_DIM = example_utils.SATELLITE_PREDICTOR_GRIDDED_DIM
SHIPS_PREDICTOR_LAGGED_DIM = example_utils.SHIPS_PREDICTOR_LAGGED_DIM
SHIPS_PREDICTOR_FORECAST_DIM = example_utils.SHIPS_PREDICTOR_FORECAST_DIM

SATELLITE_PREDICTORS_UNGRIDDED_KEY = (
    example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY
)
SATELLITE_PREDICTORS_GRIDDED_KEY = (
    example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY
)
SHIPS_PREDICTORS_LAGGED_KEY = example_utils.SHIPS_PREDICTORS_LAGGED_KEY
SHIPS_PREDICTORS_FORECAST_KEY = example_utils.SHIPS_PREDICTORS_FORECAST_KEY


def _actual_to_uniform_dist(actual_values_new, actual_values_training):
    """Converts values from actual to uniform distribution.

    :param actual_values_new: numpy array of actual (physical) values to
        convert.
    :param actual_values_training: numpy array of actual (physical) values in
        training data.
    :return: uniform_values_new: numpy array (same shape as `actual_values_new`)
        with rescaled values from 0...1.
    """

    error_checking.assert_is_numpy_array_without_nan(actual_values_training)

    actual_values_new_1d = numpy.ravel(actual_values_new)
    real_indices = numpy.where(
        numpy.invert(numpy.isnan(actual_values_new_1d))
    )[0]

    if len(real_indices) == 0:
        return actual_values_new

    search_indices = numpy.searchsorted(
        numpy.sort(numpy.ravel(actual_values_training)),
        actual_values_new_1d[real_indices],
        side='left'
    ).astype(float)

    uniform_values_new_1d = actual_values_new_1d + 0.
    num_values = actual_values_training.size
    uniform_values_new_1d[real_indices] = search_indices / (num_values - 1)
    uniform_values_new_1d[uniform_values_new_1d > 1.] = 1.

    return numpy.reshape(uniform_values_new_1d, actual_values_new.shape)


def _uniform_to_actual_dist(uniform_values_new, actual_values_training):
    """Converts values from uniform to actual distribution.

    This method is the inverse of `_actual_to_uniform_dist`.

    :param uniform_values_new: See doc for `_actual_to_uniform_dist`.
    :param actual_values_training: Same.
    :return: actual_values_new: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(actual_values_training)

    uniform_values_new_1d = numpy.ravel(uniform_values_new)
    real_indices = numpy.where(
        numpy.invert(numpy.isnan(uniform_values_new_1d))
    )[0]

    if len(real_indices) == 0:
        return uniform_values_new

    actual_values_new_1d = uniform_values_new_1d + 0.
    actual_values_new_1d[real_indices] = numpy.percentile(
        numpy.ravel(actual_values_training),
        100 * uniform_values_new_1d[real_indices],
        interpolation='linear'
    )

    return numpy.reshape(actual_values_new_1d, uniform_values_new.shape)


def _normalize_one_variable(actual_values_new, actual_values_training):
    """Normalizes one variable.

    :param actual_values_new: See doc for `_actual_to_uniform_dist`.
    :param actual_values_training: Same.
    :return: normalized_values_new: numpy array (same shape as
        `actual_values_new`) with normalized values (z-scores).
    """

    uniform_values_new = _actual_to_uniform_dist(
        actual_values_new=actual_values_new,
        actual_values_training=actual_values_training
    )

    uniform_values_new_1d = numpy.ravel(uniform_values_new)
    real_indices = numpy.where(
        numpy.invert(numpy.isnan(uniform_values_new_1d))
    )[0]

    uniform_values_new_1d[real_indices] = numpy.maximum(
        uniform_values_new_1d[real_indices], MIN_CUMULATIVE_DENSITY
    )
    uniform_values_new_1d[real_indices] = numpy.minimum(
        uniform_values_new_1d[real_indices], MAX_CUMULATIVE_DENSITY
    )
    uniform_values_new_1d[real_indices] = scipy.stats.norm.ppf(
        uniform_values_new_1d[real_indices], loc=0., scale=1.
    )

    return numpy.reshape(uniform_values_new_1d, uniform_values_new.shape)


def _denorm_one_variable(normalized_values_new, actual_values_training):
    """Denormalizes one variable.

    This method is the inverse of `_normalize_one_variable`.

    :param normalized_values_new: See doc for `_normalize_one_variable`.
    :param actual_values_training: Same.
    :return: actual_values_new: Same.
    """

    normalized_values_new_1d = numpy.ravel(normalized_values_new)
    real_indices = numpy.where(
        numpy.invert(numpy.isnan(normalized_values_new_1d))
    )[0]

    uniform_values_new_1d = normalized_values_new_1d + 0.
    uniform_values_new_1d[real_indices] = scipy.stats.norm.cdf(
        normalized_values_new_1d[real_indices], loc=0., scale=1.
    )
    uniform_values_new = numpy.reshape(
        uniform_values_new_1d, normalized_values_new.shape
    )

    return _uniform_to_actual_dist(
        uniform_values_new=uniform_values_new,
        actual_values_training=actual_values_training
    )


def get_normalization_params(example_file_names, num_values_per_ungridded,
                             num_values_per_gridded):
    """Computes normalizn params (set of reference values) for each predictor.

    :param example_file_names: 1-D list of paths to example files.  Each will be
        read by `example_io.read_file`.
    :param num_values_per_ungridded: Number of reference values to save for each
        ungridded predictor.
    :param num_values_per_gridded: Number of reference values to save for each
        gridded predictor.
    :return: normalization_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    error_checking.assert_is_string_list(example_file_names)
    error_checking.assert_is_integer(num_values_per_ungridded)
    error_checking.assert_is_geq(num_values_per_ungridded, 1000)
    error_checking.assert_is_integer(num_values_per_gridded)
    error_checking.assert_is_geq(num_values_per_gridded, 10000)

    print('Reading data from: "{0:s}"...'.format(example_file_names[0]))
    first_table_xarray = example_io.read_file(example_file_names[0])

    gridded_indices = numpy.linspace(
        0, num_values_per_gridded - 1, num=num_values_per_gridded, dtype=int
    )
    ungridded_indices = numpy.linspace(
        0, num_values_per_ungridded - 1, num=num_values_per_ungridded, dtype=int
    )
    normalization_metadata_dict = {
        GRIDDED_INDEX_DIM: gridded_indices,
        UNGRIDDED_INDEX_DIM: ungridded_indices,
        SATELLITE_PREDICTOR_UNGRIDDED_DIM:
            first_table_xarray.coords[SATELLITE_PREDICTOR_UNGRIDDED_DIM].values,
        SATELLITE_PREDICTOR_GRIDDED_DIM:
            first_table_xarray.coords[SATELLITE_PREDICTOR_GRIDDED_DIM].values,
        SHIPS_PREDICTOR_LAGGED_DIM:
            first_table_xarray.coords[SHIPS_PREDICTOR_LAGGED_DIM].values,
        SHIPS_PREDICTOR_FORECAST_DIM:
            first_table_xarray.coords[SHIPS_PREDICTOR_FORECAST_DIM].values
    }

    num_satellite_predictors_ungridded = len(
        normalization_metadata_dict[SATELLITE_PREDICTOR_UNGRIDDED_DIM]
    )
    num_satellite_predictors_gridded = len(
        normalization_metadata_dict[SATELLITE_PREDICTOR_GRIDDED_DIM]
    )
    num_ships_predictors_lagged = len(
        normalization_metadata_dict[SHIPS_PREDICTOR_LAGGED_DIM]
    )
    num_ships_predictors_forecast = len(
        normalization_metadata_dict[SHIPS_PREDICTOR_FORECAST_DIM]
    )

    these_dim = (UNGRIDDED_INDEX_DIM, SATELLITE_PREDICTOR_UNGRIDDED_DIM)
    this_array = numpy.full(
        (num_values_per_ungridded, num_satellite_predictors_ungridded),
        numpy.nan
    )
    normalization_data_dict = {
        SATELLITE_PREDICTORS_UNGRIDDED_KEY: [these_dim, this_array]
    }

    these_dim = (GRIDDED_INDEX_DIM, SATELLITE_PREDICTOR_GRIDDED_DIM)
    this_array = numpy.full(
        (num_values_per_gridded, num_satellite_predictors_gridded), numpy.nan
    )
    normalization_data_dict[SATELLITE_PREDICTORS_GRIDDED_KEY] = [
        these_dim, this_array
    ]

    these_dim = (UNGRIDDED_INDEX_DIM, SHIPS_PREDICTOR_LAGGED_DIM)
    this_array = numpy.full(
        (num_values_per_ungridded, num_ships_predictors_lagged), numpy.nan
    )
    normalization_data_dict[SHIPS_PREDICTORS_LAGGED_KEY] = [
        these_dim, this_array
    ]

    these_dim = (UNGRIDDED_INDEX_DIM, SHIPS_PREDICTOR_FORECAST_DIM)
    this_array = numpy.full(
        (num_values_per_ungridded, num_ships_predictors_forecast), numpy.nan
    )
    normalization_data_dict[SHIPS_PREDICTORS_FORECAST_KEY] = [
        these_dim, this_array
    ]

    num_files = len(example_file_names)
    num_values_per_gridded_per_file = float(num_values_per_gridded) / num_files
    num_values_per_ungridded_per_file = (
        float(num_values_per_ungridded) / num_files
    )

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(example_file_names[i]))
        example_table_xarray = example_io.read_file(example_file_names[i])

        for this_key in normalization_data_dict:
            predictor_matrix = example_table_xarray[this_key].values

            for j in range(predictor_matrix.shape[-1]):
                predictor_values = predictor_matrix[..., j]
                predictor_values = predictor_values[
                    numpy.isfinite(predictor_values)
                ]

                if len(predictor_values) == 0:
                    continue

                if this_key == SATELLITE_PREDICTORS_GRIDDED_KEY:
                    num_values_expected = int(numpy.round(
                        (i + 1) * num_values_per_gridded_per_file
                    ))
                else:
                    num_values_expected = int(numpy.round(
                        (i + 1) * num_values_per_ungridded_per_file
                    ))

                first_nan_index = numpy.where(
                    numpy.isnan(normalization_data_dict[this_key][1][:, j])
                )[0][0]

                num_values_needed = num_values_expected - first_nan_index

                if len(predictor_values) > num_values_needed:
                    predictor_values = numpy.random.choice(
                        predictor_values, size=num_values_needed, replace=False
                    )

                normalization_data_dict[this_key][1][
                    first_nan_index:(first_nan_index + len(predictor_values)), j
                ] = predictor_values

    for this_key in normalization_data_dict:
        normalization_data_dict[this_key] = tuple(
            normalization_data_dict[this_key]
        )

    return xarray.Dataset(
        data_vars=normalization_data_dict, coords=normalization_metadata_dict
    )


def normalize_data(example_table_xarray, normalization_table_xarray):
    """Normalizes all predictor variables.

    :param example_table_xarray: xarray table returned by
        `example_io.read_file`.
    :param normalization_table_xarray: See doc for `get_normalization_params`.
    :return: example_table_xarray: Normalized version of input.
    """

    xt = example_table_xarray
    nt = normalization_table_xarray

    predictor_names = list(
        xt.coords[SATELLITE_PREDICTOR_UNGRIDDED_DIM].values
    )
    predictor_names_norm = list(
        nt.coords[SATELLITE_PREDICTOR_UNGRIDDED_DIM].values
    )

    for j in range(len(predictor_names)):
        k = predictor_names_norm.index(predictor_names[j])
        training_values = nt[SATELLITE_PREDICTORS_UNGRIDDED_KEY].values[:, k]
        training_values = training_values[numpy.isfinite(training_values)]

        xt[SATELLITE_PREDICTORS_UNGRIDDED_KEY].values[..., j] = (
            _normalize_one_variable(
                actual_values_new=
                xt[SATELLITE_PREDICTORS_UNGRIDDED_KEY].values[..., j],
                actual_values_training=training_values
            )
        )

    predictor_names = list(
        xt.coords[SATELLITE_PREDICTOR_GRIDDED_DIM].values
    )
    predictor_names_norm = list(
        nt.coords[SATELLITE_PREDICTOR_GRIDDED_DIM].values
    )

    for j in range(len(predictor_names)):
        k = predictor_names_norm.index(predictor_names[j])
        training_values = nt[SATELLITE_PREDICTORS_GRIDDED_KEY].values[:, k]
        training_values = training_values[numpy.isfinite(training_values)]

        xt[SATELLITE_PREDICTORS_GRIDDED_KEY].values[..., j] = (
            _normalize_one_variable(
                actual_values_new=
                xt[SATELLITE_PREDICTORS_GRIDDED_KEY].values[..., j],
                actual_values_training=training_values
            )
        )

    predictor_names = list(xt.coords[SHIPS_PREDICTOR_LAGGED_DIM].values)
    predictor_names_norm = list(nt.coords[SHIPS_PREDICTOR_LAGGED_DIM].values)

    for j in range(len(predictor_names)):
        k = predictor_names_norm.index(predictor_names[j])
        training_values = nt[SHIPS_PREDICTORS_LAGGED_KEY].values[:, k]
        training_values = training_values[numpy.isfinite(training_values)]

        xt[SHIPS_PREDICTORS_LAGGED_KEY].values[..., j] = (
            _normalize_one_variable(
                actual_values_new=
                xt[SHIPS_PREDICTORS_LAGGED_KEY].values[..., j],
                actual_values_training=training_values
            )
        )

    predictor_names = list(xt.coords[SHIPS_PREDICTOR_FORECAST_DIM].values)
    predictor_names_norm = list(nt.coords[SHIPS_PREDICTOR_FORECAST_DIM].values)

    for j in range(len(predictor_names)):
        k = predictor_names_norm.index(predictor_names[j])
        training_values = nt[SHIPS_PREDICTORS_FORECAST_KEY].values[:, k]
        training_values = training_values[numpy.isfinite(training_values)]

        xt[SHIPS_PREDICTORS_FORECAST_KEY].values[..., j] = (
            _normalize_one_variable(
                actual_values_new=
                xt[SHIPS_PREDICTORS_FORECAST_KEY].values[..., j],
                actual_values_training=training_values
            )
        )

    return xt


def denormalize_data(example_table_xarray, normalization_table_xarray):
    """Denormalizes all predictor variables.

    :param example_table_xarray: xarray table returned by
        `example_io.read_file`.
    :param normalization_table_xarray: See doc for `get_normalization_params`.
    :return: example_table_xarray: Denormalized version of input.
    """

    xt = example_table_xarray
    nt = normalization_table_xarray

    predictor_names = list(
        xt.coords[SATELLITE_PREDICTOR_UNGRIDDED_DIM].values
    )
    predictor_names_norm = list(
        nt.coords[SATELLITE_PREDICTOR_UNGRIDDED_DIM].values
    )

    for j in range(len(predictor_names)):
        k = predictor_names_norm.index(predictor_names[j])
        training_values = nt[SATELLITE_PREDICTORS_UNGRIDDED_KEY].values[:, k]
        training_values = training_values[numpy.isfinite(training_values)]

        xt[SATELLITE_PREDICTORS_UNGRIDDED_KEY].values[..., j] = (
            _denorm_one_variable(
                normalized_values_new=
                xt[SATELLITE_PREDICTORS_UNGRIDDED_KEY].values[..., j],
                actual_values_training=training_values
            )
        )

    predictor_names = list(
        xt.coords[SATELLITE_PREDICTOR_GRIDDED_DIM].values
    )
    predictor_names_norm = list(
        nt.coords[SATELLITE_PREDICTOR_GRIDDED_DIM].values
    )

    for j in range(len(predictor_names)):
        k = predictor_names_norm.index(predictor_names[j])
        training_values = nt[SATELLITE_PREDICTORS_GRIDDED_KEY].values[:, k]
        training_values = training_values[numpy.isfinite(training_values)]

        xt[SATELLITE_PREDICTORS_GRIDDED_KEY].values[..., j] = (
            _denorm_one_variable(
                normalized_values_new=
                xt[SATELLITE_PREDICTORS_GRIDDED_KEY].values[..., j],
                actual_values_training=training_values
            )
        )

    predictor_names = list(xt.coords[SHIPS_PREDICTOR_LAGGED_DIM].values)
    predictor_names_norm = list(nt.coords[SHIPS_PREDICTOR_LAGGED_DIM].values)

    for j in range(len(predictor_names)):
        k = predictor_names_norm.index(predictor_names[j])
        training_values = nt[SHIPS_PREDICTORS_LAGGED_KEY].values[:, k]
        training_values = training_values[numpy.isfinite(training_values)]

        xt[SHIPS_PREDICTORS_LAGGED_KEY].values[..., j] = (
            _denorm_one_variable(
                normalized_values_new=
                xt[SHIPS_PREDICTORS_LAGGED_KEY].values[..., j],
                actual_values_training=training_values
            )
        )

    predictor_names = list(xt.coords[SHIPS_PREDICTOR_FORECAST_DIM].values)
    predictor_names_norm = list(nt.coords[SHIPS_PREDICTOR_FORECAST_DIM].values)

    for j in range(len(predictor_names)):
        k = predictor_names_norm.index(predictor_names[j])
        training_values = nt[SHIPS_PREDICTORS_FORECAST_KEY].values[:, k]
        training_values = training_values[numpy.isfinite(training_values)]

        xt[SHIPS_PREDICTORS_FORECAST_KEY].values[..., j] = (
            _denorm_one_variable(
                normalized_values_new=
                xt[SHIPS_PREDICTORS_FORECAST_KEY].values[..., j],
                actual_values_training=training_values
            )
        )

    return xt


def write_file(normalization_table_xarray, netcdf_file_name):
    """Writes normalization params to NetCDF file.

    :param normalization_table_xarray: xarray table in format returned by
        `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    normalization_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_file(netcdf_file_name):
    """Reads normalization params from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: normalization_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)
