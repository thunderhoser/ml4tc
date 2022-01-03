"""Composites (averages) many saliency maps."""

import argparse
import numpy
from gewittergefahr.gg_utils import prob_matched_means as pmm
from ml4tc.machine_learning import saliency

NUM_INPUT_MATRICES = 3

INPUT_FILES_ARG_NAME = 'input_saliency_file_names'
USE_PMM_ARG_NAME = 'use_pmm'
MAX_PERCENTILE_ARG_NAME = 'pmm_max_percentile_level'
OUTPUT_FILE_ARG_NAME = 'output_saliency_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files, containing saliency maps to composite.  Each'
    ' file will be read by `saliency.read_file`.'
)
USE_PMM_HELP_STRING = (
    'Boolean flag.  If 1, will use probability-matched means (PMM) to do '
    'compositing.  If 0, will use simple average.'
)
MAX_PERCENTILE_HELP_STRING = (
    'Max percentile level for PMM, ranging from 0...100.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Composite saliency map will be written here by '
    '`saliency.write_composite_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_PMM_ARG_NAME, type=int, required=True, help=USE_PMM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.5,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_names, use_pmm, pmm_max_percentile_level, output_file_name):
    """Composites (averages) many saliency maps.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param use_pmm: Same.
    :param pmm_max_percentile_level: Same.
    :param output_file_name: Same.
    """

    saliency_matrices = [None] * NUM_INPUT_MATRICES
    input_times_grad_matrices = [None] * NUM_INPUT_MATRICES
    model_file_name = None

    for this_file_name in input_file_names:
        # TODO(thunderhoser): Ensure matching saliency metadata for input files.

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_saliency_dict = saliency.read_file(this_file_name)
        model_file_name = this_saliency_dict[saliency.MODEL_FILE_KEY]

        new_saliency_matrices = this_saliency_dict[saliency.THREE_SALIENCY_KEY]
        new_input_grad_matrices = (
            this_saliency_dict[saliency.THREE_INPUT_GRAD_KEY]
        )

        for j in range(NUM_INPUT_MATRICES):
            if new_saliency_matrices[j] is None:
                continue

            if saliency_matrices[j] is None:
                saliency_matrices[j] = new_saliency_matrices[j] + 0.
                input_times_grad_matrices[j] = new_input_grad_matrices[j] + 0.
            else:
                saliency_matrices[j] = numpy.concatenate(
                    (saliency_matrices[j], new_saliency_matrices[j]),
                    axis=0
                )
                input_times_grad_matrices[j] = numpy.concatenate(
                    (input_times_grad_matrices[j], new_input_grad_matrices[j]),
                    axis=0
                )

    predictor_matrices = [
        None if a is None else numpy.divide(a, b)
        for a, b in zip(input_times_grad_matrices, saliency_matrices)
    ]

    mean_saliency_matrices = [None] * NUM_INPUT_MATRICES
    mean_input_grad_matrices = [None] * NUM_INPUT_MATRICES
    mean_predictor_matrices = [None] * NUM_INPUT_MATRICES

    for j in range(len(saliency_matrices)):
        if saliency_matrices[j] is None:
            continue

        predictor_matrices[j][
            numpy.invert(numpy.isfinite(predictor_matrices[j]))
        ] = 0.

        if use_pmm:
            mean_saliency_matrices[j] = pmm.run_pmm_many_variables(
                input_matrix=saliency_matrices[j],
                max_percentile_level=pmm_max_percentile_level
            )
            mean_input_grad_matrices[j] = pmm.run_pmm_many_variables(
                input_matrix=input_times_grad_matrices[j],
                max_percentile_level=pmm_max_percentile_level
            )
            mean_predictor_matrices[j] = pmm.run_pmm_many_variables(
                input_matrix=predictor_matrices[j],
                max_percentile_level=pmm_max_percentile_level
            )
        else:
            mean_saliency_matrices[j] = numpy.mean(saliency_matrices[j], axis=0)
            mean_input_grad_matrices[j] = numpy.mean(
                input_times_grad_matrices[j], axis=0
            )
            mean_predictor_matrices[j] = numpy.mean(
                predictor_matrices[j], axis=0
            )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    saliency.write_composite_file(
        netcdf_file_name=output_file_name,
        three_saliency_matrices=mean_saliency_matrices,
        three_input_grad_matrices=mean_input_grad_matrices,
        three_predictor_matrices=mean_predictor_matrices,
        model_file_name=model_file_name, use_pmm=use_pmm,
        pmm_max_percentile_level=pmm_max_percentile_level
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        use_pmm=bool(getattr(INPUT_ARG_OBJECT, USE_PMM_ARG_NAME)),
        pmm_max_percentile_level=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
