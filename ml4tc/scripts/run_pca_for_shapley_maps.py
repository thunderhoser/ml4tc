"""Runs PCA (principal-component analysis) for maps of Shapley values."""

import os
import shutil
import argparse
import numpy
import xarray
from sklearn.decomposition import IncrementalPCA
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tc.machine_learning import saliency

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LAG_TIME_INDEX = -1
CHANNEL_INDEX = 0

EXAMPLE_DIM = 'example'
PRINCIPAL_COMPONENT_DIM = 'principal_component'
FEATURE_DIM = 'feature'
GRID_ROW_DIM = 'grid_row'
GRID_COLUMN_DIM = 'grid_column'

EOF_KEY = 'empirical_orthogonal_function'
EIGENVALUE_KEY = 'eigenvalue'
STANDARDIZED_PC_KEY = 'standardized_principal_component'
REGRESSED_SHAPLEY_VALUE_KEY = 'regressed_shapley_value'
REGRESSED_PREDICTOR_KEY = 'regressed_predictor'

INPUT_FILES_ARG_NAME = 'input_shapley_file_names'
NUM_EXAMPLES_ARG_NAME = 'num_examples_to_keep'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files, each containing Shapley values for a '
    'different set of examples (one example = one TC at one time).  These '
    'files will be read by `saliency.read_file`.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to keep, i.e., to use in fitting the PCA.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (zarr format).  Parameters of the fitted PCA will be '
    'written here by `_write_pca_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _write_pca_results(
        zarr_file_name, eof_matrix, eigenvalues, standardized_pc_matrix,
        regressed_shapley_matrix, regressed_predictor_matrix):
    """Writes PCA results to zarr file.

    E = number of data examples
    P = number of principal components
    M = number of rows in grid
    N = number of columns in grid
    F = number of features = 2 * M * N

    :param zarr_file_name: Path to output file.
    :param eof_matrix: F-by-P numpy array, where each column is an empirical
        orthogonal function.
    :param eigenvalues: length-P numpy array of eigenvalues.
    :param standardized_pc_matrix: E-by-P numpy array, where each column is a
        standardized (mean = 0 and stdev = 1) series of principal-component
        loadings.  (I think "loadings" is the correct terminology??)
    :param regressed_shapley_matrix: P-by-M-by-N numpy array of Shapley values
        regressed onto EOFs.
    :param regressed_predictor_matrix: P-by-M-by-N numpy array of predictor
        values regressed onto EOFs.
    """

    error_checking.assert_is_string(zarr_file_name)
    if os.path.isdir(zarr_file_name):
        shutil.rmtree(zarr_file_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=zarr_file_name
    )

    num_examples = standardized_pc_matrix.shape[0]
    num_principal_components = standardized_pc_matrix.shape[1]
    num_features = eof_matrix.shape[0]
    num_grid_rows = regressed_shapley_matrix.shape[1]
    num_grid_columns = regressed_shapley_matrix.shape[2]

    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )
    pc_indices = numpy.linspace(
        0, num_principal_components - 1, num=num_principal_components, dtype=int
    )
    feature_indices = numpy.linspace(
        0, num_features - 1, num=num_features, dtype=int
    )
    row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=int
    )
    column_indices = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=int
    )

    metadata_dict = {
        EXAMPLE_DIM: example_indices,
        PRINCIPAL_COMPONENT_DIM: pc_indices,
        FEATURE_DIM: feature_indices,
        GRID_ROW_DIM: row_indices,
        GRID_COLUMN_DIM: column_indices
    }

    main_data_dict = {
        EOF_KEY: (
            (FEATURE_DIM, PRINCIPAL_COMPONENT_DIM),
            eof_matrix
        ),
        EIGENVALUE_KEY: (
            (PRINCIPAL_COMPONENT_DIM,),
            eigenvalues
        ),
        STANDARDIZED_PC_KEY: (
            (EXAMPLE_DIM, PRINCIPAL_COMPONENT_DIM),
            standardized_pc_matrix
        ),
        REGRESSED_SHAPLEY_VALUE_KEY: (
            (PRINCIPAL_COMPONENT_DIM, GRID_ROW_DIM, GRID_COLUMN_DIM),
            regressed_shapley_matrix
        ),
        REGRESSED_PREDICTOR_KEY: (
            (PRINCIPAL_COMPONENT_DIM, GRID_ROW_DIM, GRID_COLUMN_DIM),
            regressed_predictor_matrix
        )
    }

    pca_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    encoding_dict = {
        EOF_KEY: {'dtype': 'float32'},
        EIGENVALUE_KEY: {'dtype': 'float32'},
        STANDARDIZED_PC_KEY: {'dtype': 'float32'},
        REGRESSED_SHAPLEY_VALUE_KEY: {'dtype': 'float32'},
        REGRESSED_PREDICTOR_KEY: {'dtype': 'float32'}
    }
    pca_table_xarray.to_zarr(
        store=zarr_file_name, mode='w', encoding=encoding_dict
    )


def _run(shapley_file_names, num_examples_to_keep, output_file_name):
    """Runs PCA (principal-component analysis) for maps of Shapley values.

    This is effectively the main method.

    :param shapley_file_names: See documentation at top of file.
    :param num_examples_to_keep: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(num_examples_to_keep, 100)

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

    num_examples = shapley_matrix.shape[0]

    if num_examples > num_examples_to_keep:
        all_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        good_indices = numpy.random.choice(
            all_indices, size=num_examples_to_keep, replace=False
        )

        shapley_matrix = shapley_matrix[good_indices, ...]
        norm_predictor_matrix = norm_predictor_matrix[good_indices, ...]

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
    these_dim = (num_examples, num_grid_rows * num_grid_columns)

    norm_feature_matrix = numpy.concatenate((
        numpy.reshape(norm_shapley_matrix, these_dim),
        numpy.reshape(double_norm_predictor_matrix, these_dim)
    ), axis=1)

    del norm_shapley_matrix
    del double_norm_predictor_matrix

    print('Running PCA...')
    pca_object = IncrementalPCA(n_components=num_examples, whiten=False)
    pca_object.fit(norm_feature_matrix)

    eof_matrix = numpy.transpose(pca_object.components_)  # EOFs are columns.
    eigenvalues = pca_object.singular_values_ ** 2

    print('Computing principal-component series...')
    first_matrix = numpy.dot(norm_feature_matrix, eof_matrix)
    second_matrix = numpy.linalg.inv(numpy.diag(numpy.sqrt(eigenvalues)))
    principal_component_matrix = numpy.dot(first_matrix, second_matrix)

    print('Standardizing principal-component series...')
    principal_component_means = numpy.mean(
        principal_component_matrix, axis=0, keepdims=True
    )
    principal_component_stdevs = numpy.std(
        principal_component_matrix, ddof=1, axis=0, keepdims=True
    )
    standardized_pc_matrix = (
        (principal_component_matrix - principal_component_means) /
        principal_component_stdevs
    )

    print('Regressing features onto each EOF...')
    regressed_feature_matrix = numpy.full(
        (num_examples, 2 * num_grid_rows * num_grid_columns), numpy.nan
    )

    for i in range(num_examples):
        this_matrix = numpy.dot(
            numpy.transpose(norm_feature_matrix),
            standardized_pc_matrix[:, [i]]
        )
        regressed_feature_matrix[i, :] = (
            numpy.squeeze(this_matrix) / num_examples
        )

    num_grid_points = num_grid_rows * num_grid_columns
    regressed_shapley_matrix = numpy.reshape(
        regressed_feature_matrix[:, :num_grid_points],
        (num_examples, num_grid_rows, num_grid_columns)
    )
    regressed_predictor_matrix = numpy.reshape(
        regressed_feature_matrix[:, num_grid_points:],
        (num_examples, num_grid_rows, num_grid_columns)
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    _write_pca_results(
        zarr_file_name=output_file_name,
        eof_matrix=eof_matrix,
        eigenvalues=eigenvalues,
        standardized_pc_matrix=standardized_pc_matrix,
        regressed_shapley_matrix=regressed_shapley_matrix,
        regressed_predictor_matrix=regressed_predictor_matrix
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        shapley_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        num_examples_to_keep=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
