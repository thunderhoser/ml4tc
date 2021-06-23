"""Applies trained neural net in inference mode."""

import copy
import pickle
import argparse
import numpy
from sklearn.metrics import roc_auc_score
from gewittergefahr.gg_utils import file_system_utils
from ml4tc.io import example_io
from ml4tc.utils import satellite_utils
from ml4tc.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_EXAMPLES_PER_BATCH = 32
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
YEARS_ARG_NAME = 'years'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of input directory, containing examples to predict.  Files therein '
    'will be found by `example_io.find_file` and read by '
    '`example_io.read_file`.'
)
YEARS_HELP_STRING = 'Model will be applied to tropical cyclones in these years.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output (Pickle) file.  Predictions and target values will be saved'
    ' here.'
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
    '--' + YEARS_ARG_NAME, type=int, nargs='+', required=True,
    help=YEARS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_dir_name, years, output_file_name):
    """Applies trained neural net in inference mode.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param years: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)
    training_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    cyclone_id_strings = example_io.find_cyclones(
        directory_name=example_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [satellite_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )

    good_flags = numpy.array([c in years for c in cyclone_years], dtype=float)
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

    target_array = None
    forecast_prob_array = None

    for this_example_file_name in example_file_names:
        this_option_dict = copy.deepcopy(training_option_dict)
        this_option_dict[neural_net.EXAMPLE_FILE_KEY] = this_example_file_name

        these_predictor_matrices, this_target_array = (
            neural_net.create_inputs(this_option_dict)
        )
        if this_target_array.size == 0:
            continue

        this_prob_array = neural_net.apply_model(
            model_object=model_object,
            predictor_matrices=these_predictor_matrices,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=True
        )

        if target_array is None:
            target_array = this_target_array + 0
            forecast_prob_array = this_prob_array + 0.
        else:
            target_array = numpy.concatenate(
                (target_array, this_target_array), axis=0
            )
            forecast_prob_array = numpy.concatenate(
                (forecast_prob_array, this_prob_array), axis=0
            )

        print(SEPARATOR_STRING)

    print('Writing predictions and target values to: "{0:s}"...'.format(
        output_file_name
    ))

    pickle_file_handle = open(output_file_name, 'wb')
    pickle.dump(target_array, pickle_file_handle)
    pickle.dump(forecast_prob_array, pickle_file_handle)
    pickle_file_handle.close()

    # TODO(thunderhoser): HACK.
    if len(target_array.shape) == 1:
        auc = roc_auc_score(target_array, forecast_prob_array)
        print(auc)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
