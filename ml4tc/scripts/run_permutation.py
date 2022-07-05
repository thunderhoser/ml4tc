"""Runs permutation-based importance test."""

import copy
import argparse
import numpy
from ml4tc.io import example_io
from ml4tc.utils import satellite_utils
from ml4tc.machine_learning import neural_net
from ml4tc.machine_learning import permutation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
YEARS_ARG_NAME = 'years'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
DO_BACKWARDS_ARG_NAME = 'do_backwards_test'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
NUM_STEPS_ARG_NAME = 'num_steps'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with input examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
YEARS_HELP_STRING = (
    'Model will be applied to tropical cyclones in these years.  If you want to'
    ' use specific cyclones instead, leave this argument alone.'
)
CYCLONE_IDS_HELP_STRING = (
    'Model will be applied to these tropical cyclones.  If you want to use full'
    ' years instead, leave this argument alone.'
)
DO_BACKWARDS_HELP_STRING = (
    'Boolean flag.  If 1, will run backwards permutation test.  If 0, will run '
    'forward permutation test.'
)
NUM_BOOTSTRAP_HELP_STRING = (
    'Number of bootstrap replicates used to estimate cost function.'
)
NUM_STEPS_HELP_STRING = (
    'Number of steps to carry out.  Will keep going until N predictors are '
    'permanently (de)permuted, where N = `num_steps`.  If negative, will keep '
    'going until all predictors are (de)permuted.'
)
OUTPUT_FILE_HELP_STRING = (
    'Name of output file.  Results will be saved here by '
    '`permutation.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEARS_ARG_NAME, type=int, nargs='+', required=False, default=[-1],
    help=YEARS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_IDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=CYCLONE_IDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DO_BACKWARDS_ARG_NAME, type=int, required=False, default=0,
    help=DO_BACKWARDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=False, default=1000,
    help=NUM_BOOTSTRAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_STEPS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_STEPS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_dir_name, years, cyclone_id_strings,
         do_backwards_test, num_bootstrap_reps, num_steps, output_file_name):
    """Runs permutation-based importance test.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param years: Same.
    :param cyclone_id_strings: Same.
    :param do_backwards_test: Same.
    :param num_bootstrap_reps: Same.
    :param num_steps: Same.
    :param output_file_name: Same.
    """

    # Process input args.
    if len(years) == 1 and years[0] < 0:
        years = None
    if num_steps < 0:
        num_steps = None

    # Read model.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )
    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    # Find example files.
    if years is not None:
        cyclone_id_strings = example_io.find_cyclones(
            directory_name=example_dir_name, raise_error_if_all_missing=True
        )

        cyclone_years = numpy.array([
            satellite_utils.parse_cyclone_id(c)[0]
            for c in cyclone_id_strings
        ], dtype=int)

        good_flags = numpy.array(
            [c in years for c in cyclone_years], dtype=float
        )
        good_indices = numpy.where(good_flags)[0]

        cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
        cyclone_id_strings.sort()

    example_file_names = [
        example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        for c in cyclone_id_strings
    ]

    print(SEPARATOR_STRING)

    # Read examples.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    three_predictor_matrices = [None]
    target_class_matrix = None

    for i in range(len(example_file_names)):
        this_option_dict = copy.deepcopy(validation_option_dict)
        this_option_dict[neural_net.EXAMPLE_FILE_KEY] = example_file_names[i]
        this_data_dict = neural_net.create_inputs(this_option_dict)

        new_predictor_matrices = (
            this_data_dict[neural_net.PREDICTOR_MATRICES_KEY]
        )
        this_target_class_matrix = this_data_dict[neural_net.TARGET_MATRIX_KEY]

        if (
                this_target_class_matrix is None or
                this_target_class_matrix.size == 0
        ):
            continue

        if target_class_matrix is None:
            three_predictor_matrices = copy.deepcopy(new_predictor_matrices)
            target_class_matrix = this_target_class_matrix + 0
        else:
            for j in range(len(three_predictor_matrices)):
                if three_predictor_matrices[j] is None:
                    continue

                three_predictor_matrices[j] = numpy.concatenate((
                    three_predictor_matrices[j], new_predictor_matrices[j]
                ), axis=0)

            target_class_matrix = numpy.concatenate(
                (target_class_matrix, this_target_class_matrix), axis=0
            )

    print(SEPARATOR_STRING)

    if do_backwards_test:
        result_dict = permutation.run_backwards_test(
            three_predictor_matrices=three_predictor_matrices,
            target_class_matrix=target_class_matrix,
            model_object=model_object, model_metadata_dict=model_metadata_dict,
            cost_function=permutation.make_auc_cost_function(),
            num_bootstrap_reps=num_bootstrap_reps, num_steps=num_steps
        )
    else:
        result_dict = permutation.run_forward_test(
            three_predictor_matrices=three_predictor_matrices,
            target_class_matrix=target_class_matrix,
            model_object=model_object, model_metadata_dict=model_metadata_dict,
            cost_function=permutation.make_auc_cost_function(),
            num_bootstrap_reps=num_bootstrap_reps, num_steps=num_steps
        )

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    permutation.write_file(
        result_dict=result_dict, netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int),
        cyclone_id_strings=getattr(INPUT_ARG_OBJECT, CYCLONE_IDS_ARG_NAME),
        do_backwards_test=bool(getattr(
            INPUT_ARG_OBJECT, DO_BACKWARDS_ARG_NAME
        )),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        num_steps=getattr(INPUT_ARG_OBJECT, NUM_STEPS_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
