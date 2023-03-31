"""Runs MCA (maximum-covariance analysis) for maps of Shapley values."""

import argparse
import numpy
import xarray
import netCDF4
from sklearn.decomposition import IncrementalPCA
from gewittergefahr.gg_utils import file_system_utils
from ml4tc.machine_learning import saliency
from ml4tc.scripts import \
    get_shap_predictor_covariance_matrix as get_covar_matrix

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LAG_TIME_INDEX = -1
CHANNEL_INDEX = 0

PRINCIPAL_COMPONENT_DIM = 'principal_component'
GRID_ROW_DIM = 'grid_row'
GRID_COLUMN_DIM = 'grid_column'
PIXEL_DIM = 'pixel'

SHAPLEY_SINGULAR_VALUE_KEY = 'shapley_singular_value'
PREDICTOR_SINGULAR_VALUE_KEY = 'predictor_singular_value'
SHAPLEY_EXPANSION_COEFF_KEY = 'shapley_expansion_coefficient'
PREDICTOR_EXPANSION_COEFF_KEY = 'predictor_expansion_coefficient'
EIGENVALUE_KEY = 'eigenvalue'
REGRESSED_SHAPLEY_VALUE_KEY = 'regressed_shapley_value'
REGRESSED_PREDICTOR_KEY = 'regressed_predictor'

SHAPLEY_FILES_ARG_NAME = 'input_shapley_file_names'
COVARIANCE_FILE_ARG_NAME = 'input_covariance_file_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SHAPLEY_FILES_HELP_STRING = (
    'List of paths to Shapley files, each containing Shapley values for a '
    'different set of examples (one example = one TC at one time).  These '
    'files will be read by `saliency.read_file`.'
)
COVARIANCE_FILE_HELP_STRING = (
    'Path to covariance file.  This should contain the P-by-P covariance '
    'matrix (where P = num pixels) between the Shapley and predictor values, '
    'created by the script get_shap_predictor_covariance_matrix.py, using the '
    'exact same Shapley files.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Parameters of the fitted MCA will be written here '
    'by `_write_mca_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=SHAPLEY_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COVARIANCE_FILE_ARG_NAME, type=str, required=True,
    help=COVARIANCE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _write_mca_results(
        netcdf_file_name,
        shapley_singular_value_matrix, predictor_singular_value_matrix,
        shapley_expansion_coeff_matrix, predictor_expansion_coeff_matrix,
        eigenvalues, regressed_shapley_matrix, regressed_predictor_matrix):
    """Writes MCA results to NetCDF file.

    P = number of principal components
    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param shapley_singular_value_matrix: MN-by-P numpy array, where each column
        is a singular vector for the Shapley values.
    :param predictor_singular_value_matrix: MN-by-P numpy array, where each
        column is a singular vector for the predictor values.
    :param shapley_expansion_coeff_matrix: MN-by-P numpy array, where each
        column is a vector of expansion coefficients for the Shapley values.
    :param predictor_expansion_coeff_matrix: MN-by-P numpy array, where each
        column is a vector of expansion coefficients for the predictor values.
    :param eigenvalues: length-P numpy array of eigenvalues.
    :param regressed_shapley_matrix: P-by-M-by-N numpy array of Shapley values
        regressed onto singular vectors.
    :param regressed_predictor_matrix: P-by-M-by-N numpy array of predictor
        values regressed onto singular vectors.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(netcdf_file_name, 'w', format='NETCDF4')

    num_principal_components = shapley_singular_value_matrix.shape[1]
    num_grid_rows = regressed_shapley_matrix.shape[1]
    num_grid_columns = regressed_shapley_matrix.shape[2]
    num_pixels = num_grid_rows * num_grid_columns

    dataset_object.createDimension(
        PRINCIPAL_COMPONENT_DIM, num_principal_components
    )
    dataset_object.createDimension(GRID_ROW_DIM, num_grid_rows)
    dataset_object.createDimension(GRID_COLUMN_DIM, num_grid_columns)
    dataset_object.createDimension(PIXEL_DIM, num_pixels)

    dataset_object.createVariable(
        SHAPLEY_SINGULAR_VALUE_KEY, datatype=numpy.float32,
        dimensions=(PIXEL_DIM, PRINCIPAL_COMPONENT_DIM)
    )
    dataset_object.variables[SHAPLEY_SINGULAR_VALUE_KEY][:] = (
        shapley_singular_value_matrix
    )

    dataset_object.createVariable(
        PREDICTOR_SINGULAR_VALUE_KEY, datatype=numpy.float32,
        dimensions=(PIXEL_DIM, PRINCIPAL_COMPONENT_DIM)
    )
    dataset_object.variables[PREDICTOR_SINGULAR_VALUE_KEY][:] = (
        predictor_singular_value_matrix
    )

    dataset_object.createVariable(
        SHAPLEY_EXPANSION_COEFF_KEY, datatype=numpy.float32,
        dimensions=(PIXEL_DIM, PRINCIPAL_COMPONENT_DIM)
    )
    dataset_object.variables[SHAPLEY_EXPANSION_COEFF_KEY][:] = (
        shapley_expansion_coeff_matrix
    )

    dataset_object.createVariable(
        PREDICTOR_EXPANSION_COEFF_KEY, datatype=numpy.float32,
        dimensions=(PIXEL_DIM, PRINCIPAL_COMPONENT_DIM)
    )
    dataset_object.variables[PREDICTOR_EXPANSION_COEFF_KEY][:] = (
        predictor_expansion_coeff_matrix
    )

    dataset_object.createVariable(
        EIGENVALUE_KEY, datatype=numpy.float32,
        dimensions=PRINCIPAL_COMPONENT_DIM
    )
    dataset_object.variables[EIGENVALUE_KEY][:] = eigenvalues

    dataset_object.createVariable(
        REGRESSED_SHAPLEY_VALUE_KEY, datatype=numpy.float32,
        dimensions=(PRINCIPAL_COMPONENT_DIM, GRID_ROW_DIM, GRID_COLUMN_DIM)
    )
    dataset_object.variables[REGRESSED_SHAPLEY_VALUE_KEY][:] = (
        regressed_shapley_matrix
    )

    dataset_object.createVariable(
        REGRESSED_PREDICTOR_KEY, datatype=numpy.float32,
        dimensions=(PRINCIPAL_COMPONENT_DIM, GRID_ROW_DIM, GRID_COLUMN_DIM)
    )
    dataset_object.variables[REGRESSED_PREDICTOR_KEY][:] = (
        regressed_predictor_matrix
    )

    dataset_object.close()


def _run(shapley_file_names, covariance_file_name, output_file_name):
    """Runs MCA (maximum-covariance analysis) for maps of Shapley values.

    This is effectively the same method.

    :param shapley_file_names: See documentation at top of file.
    :param covariance_file_name: Same.
    :param output_file_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(covariance_file_name))
    covariance_table_xarray = xarray.open_dataset(covariance_file_name)
    covariance_matrix = covariance_table_xarray[get_covar_matrix.COVARIANCE_KEY]

    num_covariance_pixels = covariance_matrix.shape[0]

    shapley_matrix = None
    norm_predictor_matrix = None
    spatial_coarsening_factor = None

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

        if spatial_coarsening_factor is None:
            num_orig_pixels = (
                this_dummy_saliency_matrix.shape[1] *
                this_dummy_saliency_matrix.shape[2]
            )
            spatial_coarsening_factor_float = (
                float(num_orig_pixels) / num_covariance_pixels
            )
            spatial_coarsening_factor = int(numpy.round(
                spatial_coarsening_factor_float
            ))

            assert numpy.isclose(
                spatial_coarsening_factor, spatial_coarsening_factor_float,
                rtol=0.01
            )

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

    print('Running PCA...')
    pca_object = IncrementalPCA(n_components=num_examples, whiten=False)
    pca_object.fit(covariance_matrix)

    predictor_singular_value_matrix = numpy.transpose(pca_object.components_)
    eigenvalues = pca_object.singular_values_ ** 2

    print('Computing left singular vectors (for Shapley values)...')
    first_matrix = numpy.dot(
        covariance_matrix, predictor_singular_value_matrix
    )
    second_matrix = numpy.linalg.inv(numpy.diag(numpy.sqrt(eigenvalues)))
    shapley_singular_value_matrix = numpy.dot(first_matrix, second_matrix)

    del covariance_matrix

    print('Computing expansion coefficients...')
    shapley_expansion_coeff_matrix = numpy.dot(
        norm_shapley_matrix, shapley_singular_value_matrix
    )
    predictor_expansion_coeff_matrix = numpy.dot(
        double_norm_predictor_matrix, predictor_singular_value_matrix
    )

    print('Standardizing expansion coefficients...')
    these_means = numpy.mean(
        shapley_expansion_coeff_matrix, axis=0, keepdims=True
    )
    these_stdevs = numpy.std(
        shapley_expansion_coeff_matrix, ddof=1, axis=0, keepdims=True
    )
    shapley_expansion_coeff_matrix = (
        (shapley_expansion_coeff_matrix - these_means) / these_stdevs
    )

    these_means = numpy.mean(
        predictor_expansion_coeff_matrix, axis=0, keepdims=True
    )
    these_stdevs = numpy.std(
        predictor_expansion_coeff_matrix, ddof=1, axis=0, keepdims=True
    )
    predictor_expansion_coeff_matrix = (
        (predictor_expansion_coeff_matrix - these_means) / these_stdevs
    )

    print('Regressing Shapley values onto each left singular vector...')
    regressed_shapley_matrix = numpy.full((num_examples, num_pixels), numpy.nan)

    for i in range(num_examples):
        this_matrix = numpy.dot(
            numpy.transpose(norm_shapley_matrix),
            shapley_expansion_coeff_matrix[:, [i]]
        )
        regressed_shapley_matrix[i, :] = (
            numpy.squeeze(this_matrix) / num_examples
        )

    print('Regressing predictor values onto each right singular vector...')
    regressed_predictor_matrix = numpy.full(
        (num_examples, num_pixels), numpy.nan
    )

    for i in range(num_examples):
        this_matrix = numpy.dot(
            numpy.transpose(double_norm_predictor_matrix),
            predictor_expansion_coeff_matrix[:, [i]]
        )
        regressed_predictor_matrix[i, :] = (
            numpy.squeeze(this_matrix) / num_examples
        )

    regressed_shapley_matrix = numpy.reshape(
        regressed_shapley_matrix,
        (num_examples, num_grid_rows, num_grid_columns)
    )
    regressed_predictor_matrix = numpy.reshape(
        regressed_predictor_matrix,
        (num_examples, num_grid_rows, num_grid_columns)
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    _write_mca_results(
        netcdf_file_name=output_file_name,
        shapley_singular_value_matrix=shapley_singular_value_matrix,
        predictor_singular_value_matrix=predictor_singular_value_matrix,
        shapley_expansion_coeff_matrix=shapley_expansion_coeff_matrix,
        predictor_expansion_coeff_matrix=predictor_expansion_coeff_matrix,
        eigenvalues=eigenvalues,
        regressed_shapley_matrix=regressed_shapley_matrix,
        regressed_predictor_matrix=regressed_predictor_matrix
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        shapley_file_names=getattr(INPUT_ARG_OBJECT, SHAPLEY_FILES_ARG_NAME),
        covariance_file_name=getattr(
            INPUT_ARG_OBJECT, COVARIANCE_FILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
