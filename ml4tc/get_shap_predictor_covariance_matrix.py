"""Computes covariance matrix between Shapley values and predictor values.

There is one covariance for each pair of grid points, so the matrix can get
huge.
"""

import os
import sys
import shutil
import argparse
from multiprocessing import Pool
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import saliency

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LAG_TIME_INDEX = -1
CHANNEL_INDEX = 0
NUM_SLICES_PER_DIMENSION = 4

SHAPLEY_PIXEL_DIM = 'shapley_pixel'
PREDICTOR_PIXEL_DIM = 'predictor_pixel'
COVARIANCE_KEY = 'covariance'

INPUT_FILES_ARG_NAME = 'input_shapley_file_names'
# NUM_EXAMPLES_ARG_NAME = 'num_examples_to_keep'
COARSENING_FACTOR_ARG_NAME = 'spatial_coarsening_factor'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files, each containing Shapley values for a '
    'different set of examples (one example = one TC at one time).  These '
    'files will be read by `saliency.read_file`.'
)
# NUM_EXAMPLES_HELP_STRING = (
#     'Number of examples to keep, i.e., to use in computing the covariances.'
# )
COARSENING_FACTOR_HELP_STRING = (
    'Will use every [K]th grid point (in both the row and column dimensions), '
    'where K = {0:s}.'
).format(COARSENING_FACTOR_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (zarr format).  The covariance matrix will be written '
    'here by `_write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
# INPUT_ARG_PARSER.add_argument(
#     '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
#     help=NUM_EXAMPLES_HELP_STRING
# )
INPUT_ARG_PARSER.add_argument(
    '--' + COARSENING_FACTOR_ARG_NAME, type=int, required=True,
    help=COARSENING_FACTOR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _get_covariance_matrix(shapley_matrix, predictor_matrix):
    """Computes covariance matrix between Shapley values and predictor values.

    E = number of examples
    S = number of grid points for Shapley values
    P = number of grid points for predictor values

    :param shapley_matrix: E-by-S numpy array of Shapley values.
    :param predictor_matrix: E-by-P numpy array of predictor values.
    :return: covariance_matrix: S-by-P numpy array of covariances.
    """

    assert not numpy.any(numpy.isnan(shapley_matrix))
    assert not numpy.any(numpy.isnan(predictor_matrix))

    mean_norm_shapley_value_by_pixel = numpy.mean(shapley_matrix, axis=0)
    mean_norm_predictor_by_pixel = numpy.mean(predictor_matrix, axis=0)

    num_examples = shapley_matrix.shape[0]
    num_shapley_pixels = shapley_matrix.shape[1]
    num_predictor_pixels = predictor_matrix.shape[1]
    covariance_matrix = numpy.full(
        (num_shapley_pixels, num_predictor_pixels), numpy.nan,
        dtype=numpy.float32
    )

    for i in range(num_shapley_pixels):
        print('Have computed {0:d} of {1:d} covariances...'.format(
            i * num_predictor_pixels,
            num_shapley_pixels * num_predictor_pixels
        ))

        for j in range(num_predictor_pixels):
            covariance_matrix[i, j] = numpy.sum(
                (shapley_matrix[:, i] - mean_norm_shapley_value_by_pixel[i]) *
                (predictor_matrix[:, j] - mean_norm_predictor_by_pixel[j])
            )

    print('Have computed all {0:d} covariances!'.format(
        num_shapley_pixels * num_predictor_pixels
    ))
    covariance_matrix = covariance_matrix / (num_examples - 1)

    print('Number of NaN covariances = {0:d} of {1:d}'.format(
        numpy.sum(numpy.isnan(covariance_matrix)),
        covariance_matrix.size
    ))
    covariance_matrix[numpy.isnan(covariance_matrix)] = 0.

    return covariance_matrix


def _write_results(zarr_file_name, covariance_matrix):
    """Writes covariance matrix to zarr file.

    P = number of pixels

    :param zarr_file_name: Path to output file.
    :param covariance_matrix: P-by-P numpy array of covariances, where the
        [i, j] element is the covariance between normalized Shapley value at the
        [i]th pixel and normalized predictor value at the [j]th pixel.
    """

    error_checking.assert_is_string(zarr_file_name)
    if os.path.isdir(zarr_file_name):
        shutil.rmtree(zarr_file_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=zarr_file_name
    )

    num_pixels = covariance_matrix.shape[0]
    pixel_indices = numpy.linspace(0, num_pixels - 1, num=num_pixels, dtype=int)

    metadata_dict = {
        SHAPLEY_PIXEL_DIM: pixel_indices,
        PREDICTOR_PIXEL_DIM: pixel_indices
    }
    main_data_dict = {
        COVARIANCE_KEY: (
            (SHAPLEY_PIXEL_DIM, PREDICTOR_PIXEL_DIM),
            covariance_matrix
        )
    }
    covariance_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    covariance_table_xarray.to_zarr(
        store=zarr_file_name, mode='w',
        encoding={COVARIANCE_KEY: {'dtype': 'float32'}}
    )


def _run(shapley_file_names, spatial_coarsening_factor, output_file_name):
    """Computes covariance matrix between Shapley values and predictor values.

    This is effectively the main method.

    :param shapley_file_names: See documentation at top of file.
    :param spatial_coarsening_factor: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(spatial_coarsening_factor, 1)
    if spatial_coarsening_factor == 1:
        spatial_coarsening_factor = None

    shapley_matrix = None
    norm_predictor_matrix = None

    # TODO(thunderhoser): Ensure matching saliency metadata for input files.
    for this_file_name in shapley_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_saliency_dict = saliency.read_file(this_file_name)

        this_dummy_input_grad_matrix = this_saliency_dict[
            saliency.THREE_INPUT_GRAD_KEY
        ][0][..., LAG_TIME_INDEX, CHANNEL_INDEX]

        this_dummy_saliency_matrix = this_saliency_dict[
            saliency.THREE_SALIENCY_KEY
        ][0][..., LAG_TIME_INDEX, CHANNEL_INDEX]

        if spatial_coarsening_factor is not None:
            this_dummy_input_grad_matrix = this_dummy_input_grad_matrix[
                :, ::spatial_coarsening_factor, ::spatial_coarsening_factor
            ]
            this_dummy_saliency_matrix = this_dummy_saliency_matrix[
                :, ::spatial_coarsening_factor, ::spatial_coarsening_factor
            ]

        this_predictor_matrix = numpy.divide(
            this_dummy_input_grad_matrix, this_dummy_saliency_matrix
        )
        this_predictor_matrix[
            numpy.invert(numpy.isfinite(this_predictor_matrix))
        ] = 0.

        this_shapley_matrix = this_dummy_input_grad_matrix

        if shapley_matrix is None:
            shapley_matrix = this_shapley_matrix + 0.
            norm_predictor_matrix = this_predictor_matrix + 0.
        else:
            shapley_matrix = numpy.concatenate(
                (shapley_matrix, this_shapley_matrix), axis=0
            )
            norm_predictor_matrix = numpy.concatenate(
                (norm_predictor_matrix, this_predictor_matrix), axis=0
            )

    print(SEPARATOR_STRING)

    mean_shapley_value = numpy.mean(shapley_matrix)
    stdev_shapley_value = numpy.std(shapley_matrix, ddof=1)
    norm_shapley_matrix = (
        (shapley_matrix - mean_shapley_value) / stdev_shapley_value
    )
    del shapley_matrix

    mean_predictor_value = numpy.mean(norm_predictor_matrix)
    stdev_predictor_value = numpy.std(norm_predictor_matrix, ddof=1)
    double_norm_predictor_matrix = (
        (norm_predictor_matrix - mean_predictor_value) / stdev_predictor_value
    )
    del norm_predictor_matrix

    num_examples = norm_shapley_matrix.shape[0]
    num_grid_rows = norm_shapley_matrix.shape[1]
    num_grid_columns = norm_shapley_matrix.shape[2]
    num_pixels = num_grid_rows * num_grid_columns
    these_dim = (num_examples, num_pixels)

    norm_shapley_matrix = numpy.reshape(norm_shapley_matrix, these_dim)
    double_norm_predictor_matrix = numpy.reshape(
        double_norm_predictor_matrix, these_dim
    )

    slice_indices_normalized = numpy.linspace(
        0, 1, num=NUM_SLICES_PER_DIMENSION + 1, dtype=float
    )
    start_pixels = numpy.round(
        num_pixels * slice_indices_normalized[:-1]
    ).astype(int)
    end_pixels = numpy.round(
        num_pixels * slice_indices_normalized[1:]
    ).astype(int)

    start_pixels_for_shapley_values, start_pixels_for_predictors = (
        numpy.meshgrid(start_pixels, start_pixels)
    )
    start_pixels_for_shapley_values = numpy.ravel(
        start_pixels_for_shapley_values
    )
    start_pixels_for_predictors = numpy.ravel(start_pixels_for_predictors)

    end_pixels_for_shapley_values, end_pixels_for_predictors = (
        numpy.meshgrid(end_pixels, end_pixels)
    )
    end_pixels_for_shapley_values = numpy.ravel(
        end_pixels_for_shapley_values
    )
    end_pixels_for_predictors = numpy.ravel(end_pixels_for_predictors)

    argument_list = []

    for s_start, s_end, p_start, p_end in zip(
            start_pixels_for_shapley_values, end_pixels_for_shapley_values,
            start_pixels_for_predictors, end_pixels_for_predictors
    ):
        argument_list.append((
            norm_shapley_matrix[:, s_start:s_end],
            double_norm_predictor_matrix[:, p_start:p_end]
        ))

    covariance_matrix = numpy.full(
        (num_pixels, num_pixels), numpy.nan, dtype=numpy.float32
    )

    with Pool() as pool_object:
        covariance_submatrices = pool_object.starmap(
            _get_covariance_matrix, argument_list
        )

        for k in range(len(argument_list)):
            s_start = start_pixels_for_shapley_values[k]
            s_end = end_pixels_for_shapley_values[k]
            p_start = start_pixels_for_predictors[k]
            p_end = end_pixels_for_predictors[k]

            covariance_matrix[s_start:s_end, p_start:p_end] = (
                covariance_submatrices[k]
            )

    print(SEPARATOR_STRING)
    assert not numpy.any(numpy.isnan(covariance_matrix))

    print('Root mean squared covariance (RMSC) = {0:.4f}'.format(
        numpy.sqrt(numpy.mean(covariance_matrix ** 2))
    ))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    _write_results(
        zarr_file_name=output_file_name,
        covariance_matrix=covariance_matrix
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        shapley_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        spatial_coarsening_factor=getattr(
            INPUT_ARG_OBJECT, COARSENING_FACTOR_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
