"""Runs PCA (principal-component analysis) for maps of Shapley values."""

import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4tc.machine_learning import saliency

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
    'Path to output file.  Parameters of the fitted PCA will be written here '
    'by `_write_pca_results`.'
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


def _run(shapley_file_names, num_examples_to_keep, output_file_name):
    """Runs PCA (principal-component analysis) for maps of Shapley values.

    This is effectively the main method.

    :param shapley_file_names: See documentation at top of file.
    :param num_examples_to_keep: Same.
    :param output_file_name: Same.
    """

    # (1) Read data.
    # (2) Filter data if necessary.
    # (3) Run PCA.
    # (4) Develop format for output file.
    # (5) Write plotting script.

    error_checking.assert_is_geq(num_examples_to_keep, 100)

    shapley_matrix = None
    predictor_matrix = None

    # TODO(thunderhoser): Ensure matching saliency metadata for input files.
    for this_file_name in shapley_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_saliency_dict = saliency.read_file(this_file_name)

        this_dummy_input_grad_matrix = (
            this_saliency_dict[saliency.THREE_INPUT_GRAD_KEY][0]
        )
        this_dummy_saliency_matrix = (
            this_saliency_dict[saliency.THREE_SALIENCY_KEY][0]
        )
        this_predictor_matrix = numpy.divide(
            this_dummy_input_grad_matrix, this_dummy_saliency_matrix
        )

        this_predictor_matrix[
            numpy.invert(numpy.isfinite(this_predictor_matrix))
        ] = 0.
        this_shapley_matrix = this_dummy_input_grad_matrix

        if shapley_matrix is None:
            shapley_matrix = this_shapley_matrix + 0.
            predictor_matrix = this_predictor_matrix + 0.
        else:
            shapley_matrix = numpy.concatenate(
                (shapley_matrix, this_shapley_matrix), axis=0
            )
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=0
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
        predictor_matrix = predictor_matrix[good_indices, ...]

    num_examples = shapley_matrix.shape[0]

    mean_shapley_value = numpy.mean(shapley_matrix)
    stdev_shapley_value = numpy.std(shapley_matrix, ddof=1)
    shapley_matrix = (shapley_matrix - mean_shapley_value) / stdev_shapley_value

    mean_shapley_value = numpy.mean(shapley_matrix)
    stdev_shapley_value = numpy.std(shapley_matrix, ddof=1)
    shapley_matrix = (shapley_matrix - mean_shapley_value) / stdev_shapley_value




if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        shapley_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        num_examples_to_keep=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
