"""Average saliency maps created with SmoothGrad."""

import os
import sys
import copy
import glob
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import saliency

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_PATTERN_ARG_NAME = 'input_saliency_file_pattern'
NUM_SAMPLES_ARG_NAME = 'num_smoothgrad_samples'
OUTPUT_FILE_ARG_NAME = 'output_saliency_file_name'

INPUT_PATTERN_HELP_STRING = (
    'Glob pattern for input files, containing saliency maps to be averaged.  '
    'Each file will be read by `saliency.read_file`.'
)
NUM_SAMPLES_HELP_STRING = (
    'Number of SmoothGrad samples expected.  This script will raise an error '
    'if the number of files found (i.e., matching the glob pattern) is '
    'smaller.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  The average saliency map will be written here by '
    '`saliency.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_SAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_pattern, num_smoothgrad_samples, output_file_name):
    """Average saliency maps created with SmoothGrad.

    This is effectively the main method.

    :param input_file_pattern: See documentation at top of file.
    :param num_smoothgrad_samples: Same.
    :param output_file_name: Same.
    :raises: ValueError: if number of files found < number of expected
        SmoothGrad samples.
    """

    error_checking.assert_is_greater(num_smoothgrad_samples, 1)

    input_file_names = glob.glob(input_file_pattern)
    input_file_names.sort()

    if len(input_file_names) < num_smoothgrad_samples:
        error_string = (
            'Expected at least {0:d} SmoothGrad samples.  Instead, found {1:d} '
            'files with pattern: "{2:s}"'
        ).format(
            num_smoothgrad_samples, len(input_file_names), input_file_pattern
        )

        raise ValueError(error_string)

    if num_smoothgrad_samples > len(input_file_names):
        file_indices = numpy.linspace(
            0, len(input_file_names) - 1, num=len(input_file_names), dtype=int
        )
        file_indices = numpy.random.choice(
            file_indices, size=num_smoothgrad_samples, replace=False
        )
        input_file_names = [input_file_names[k] for k in file_indices]

    three_saliency_matrices = [None] * 3
    three_input_grad_matrices = [None] * 3
    saliency_dict = dict()

    for i in range(num_smoothgrad_samples):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        new_saliency_dict = saliency.read_file(input_file_names[i])

        sort_indices = numpy.argsort(new_saliency_dict[saliency.INIT_TIMES_KEY])
        new_saliency_dict[saliency.INIT_TIMES_KEY] = (
            new_saliency_dict[saliency.INIT_TIMES_KEY][sort_indices]
        )
        new_saliency_dict[saliency.CYCLONE_IDS_KEY] = [
            new_saliency_dict[saliency.CYCLONE_IDS_KEY][k] for k in sort_indices
        ]

        for j in range(len(new_saliency_dict[saliency.THREE_SALIENCY_KEY])):
            if new_saliency_dict[saliency.THREE_SALIENCY_KEY][j] is None:
                continue

            new_saliency_dict[saliency.THREE_SALIENCY_KEY][j] = (
                new_saliency_dict[saliency.THREE_SALIENCY_KEY][j][
                    sort_indices, ...
                ]
            )
            new_saliency_dict[saliency.THREE_INPUT_GRAD_KEY][j] = (
                new_saliency_dict[saliency.THREE_INPUT_GRAD_KEY][j][
                    sort_indices, ...
                ]
            )

        if not bool(saliency_dict):
            saliency_dict = copy.deepcopy(new_saliency_dict)

        assert (
            saliency_dict[saliency.CYCLONE_IDS_KEY] ==
            new_saliency_dict[saliency.CYCLONE_IDS_KEY]
        )
        assert numpy.array_equal(
            saliency_dict[saliency.INIT_TIMES_KEY],
            new_saliency_dict[saliency.INIT_TIMES_KEY]
        )
        assert (
            saliency_dict[saliency.MODEL_FILE_KEY] ==
            new_saliency_dict[saliency.MODEL_FILE_KEY]
        )
        assert (
            saliency_dict[saliency.LAYER_NAME_KEY] ==
            new_saliency_dict[saliency.LAYER_NAME_KEY]
        )
        assert numpy.allclose(
            saliency_dict[saliency.NEURON_INDICES_KEY],
            new_saliency_dict[saliency.NEURON_INDICES_KEY],
            atol=1e-6, equal_nan=True
        )
        assert numpy.isclose(
            saliency_dict[saliency.IDEAL_ACTIVATION_KEY],
            new_saliency_dict[saliency.IDEAL_ACTIVATION_KEY],
            atol=1e-6
        )

        new_saliency_matrices = new_saliency_dict[saliency.THREE_SALIENCY_KEY]
        new_input_grad_matrices = new_saliency_dict[
            saliency.THREE_INPUT_GRAD_KEY
        ]

        if all([s is None for s in three_saliency_matrices]):
            for j in range(len(three_saliency_matrices)):
                if new_saliency_matrices[j] is None:
                    continue

                three_saliency_matrices[j] = new_saliency_matrices[j] + 0.
                three_input_grad_matrices[j] = (
                    new_input_grad_matrices[j] + 0.
                )

            continue

        for j in range(len(three_saliency_matrices)):
            if new_saliency_matrices[j] is None:
                continue

            concat_saliency_matrix = numpy.stack(
                (new_saliency_matrices[j], three_saliency_matrices[j]),
                axis=-1
            )
            three_saliency_matrices[j] = numpy.average(
                concat_saliency_matrix, axis=-1,
                weights=numpy.array([j, 1], dtype=float)
            )

            concat_input_grad_matrix = numpy.stack(
                (new_input_grad_matrices[j], three_input_grad_matrices[j]),
                axis=-1
            )
            three_input_grad_matrices[j] = numpy.average(
                concat_input_grad_matrix, axis=-1,
                weights=numpy.array([j, 1], dtype=float)
            )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    saliency.write_file(
        netcdf_file_name=output_file_name,
        three_saliency_matrices=three_saliency_matrices,
        three_input_grad_matrices=three_input_grad_matrices,
        cyclone_id_strings=saliency_dict[saliency.CYCLONE_IDS_KEY],
        init_times_unix_sec=saliency_dict[saliency.INIT_TIMES_KEY],
        model_file_name=saliency_dict[saliency.MODEL_FILE_KEY],
        layer_name=saliency_dict[saliency.LAYER_NAME_KEY],
        neuron_indices=saliency_dict[saliency.NEURON_INDICES_KEY],
        ideal_activation=saliency_dict[saliency.IDEAL_ACTIVATION_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_pattern=getattr(INPUT_ARG_OBJECT, INPUT_PATTERN_ARG_NAME),
        num_smoothgrad_samples=getattr(INPUT_ARG_OBJECT, NUM_SAMPLES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
