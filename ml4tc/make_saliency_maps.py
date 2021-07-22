"""Creates saliency maps."""

import os
import sys
import copy
import argparse
import numpy
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_io
import satellite_utils
import neural_net
import saliency

tensorflow.compat.v1.disable_eager_execution()

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
YEARS_ARG_NAME = 'years'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
LAYER_NAME_ARG_NAME = 'layer_name'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with input examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
YEARS_HELP_STRING = (
    'Will create saliency maps for tropical cyclones in these years.  If you '
    'want to use specific cyclones instead, leave this argument alone.'
)
CYCLONE_IDS_HELP_STRING = (
    'Will create saliency maps for these tropical cyclones.  If you want to use'
    ' full years instead, leave this argument alone.'
)
LAYER_NAME_HELP_STRING = 'Name of layer with relevant neuron.'
NEURON_INDICES_HELP_STRING = (
    '1-D numpy array with indices of relevant neuron.  Must have length D - 1, '
    'where D = number of dimensions in layer output.  The first dimension is '
    'the batch dimension, which always has length `None` in Keras.'
)
IDEAL_ACTIVATION_HELP_STRING = (
    'Ideal neuron activation, used to define loss function.  The loss function '
    'will be (neuron_activation - ideal_activation)**2.'
)
OUTPUT_FILE_HELP_STRING = (
    'Name of output file.  Results will be saved here by '
    '`saliency.write_file`.'
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
    '--' + LAYER_NAME_ARG_NAME, type=str, required=True,
    help=LAYER_NAME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=True,
    help=NEURON_INDICES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=True,
    help=IDEAL_ACTIVATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_dir_name, years, unique_cyclone_id_strings,
         layer_name, neuron_indices, ideal_activation, output_file_name):
    """Creates saliency maps.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param years: Same.
    :param unique_cyclone_id_strings: Same.
    :param layer_name: Same.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param output_file_name: Same.
    """

    # Process input args.
    if len(years) == 1 and years[0] < 0:
        years = None

    # Read model.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )
    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    # Find example files.
    if years is None:
        unique_cyclone_id_strings = numpy.unique(
            numpy.array(unique_cyclone_id_strings)
        )
    else:
        unique_cyclone_id_strings = example_io.find_cyclones(
            directory_name=example_dir_name, raise_error_if_all_missing=True
        )

        cyclone_years = numpy.array([
            satellite_utils.parse_cyclone_id(c)[0]
            for c in unique_cyclone_id_strings
        ], dtype=int)

        good_flags = numpy.array(
            [c in years for c in cyclone_years], dtype=float
        )
        good_indices = numpy.where(good_flags)[0]

        unique_cyclone_id_strings = [
            unique_cyclone_id_strings[k] for k in good_indices
        ]
        unique_cyclone_id_strings.sort()

    example_file_names = [
        example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        for c in unique_cyclone_id_strings
    ]

    print(SEPARATOR_STRING)

    # Read examples.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    saliency_matrices = [None]
    cyclone_id_strings = []
    init_times_unix_sec = numpy.array([], dtype=int)

    for i in range(len(example_file_names)):
        this_option_dict = copy.deepcopy(validation_option_dict)
        this_option_dict[neural_net.EXAMPLE_FILE_KEY] = example_file_names[i]
        this_data_dict = neural_net.create_inputs(this_option_dict)
        print(SEPARATOR_STRING)

        these_predictor_matrices = (
            this_data_dict[neural_net.PREDICTOR_MATRICES_KEY]
        )
        this_target_array = this_data_dict[neural_net.TARGET_ARRAY_KEY]

        if this_target_array.size == 0:
            continue

        this_num_examples = these_predictor_matrices[0].shape[0]
        cyclone_id_strings += [unique_cyclone_id_strings[i]] * this_num_examples
        init_times_unix_sec = numpy.concatenate((
            init_times_unix_sec,
            this_data_dict[neural_net.INIT_TIMES_KEY]
        ))

        these_saliency_matrices = saliency.get_saliency_one_neuron(
            model_object=model_object,
            predictor_matrices=these_predictor_matrices,
            layer_name=layer_name, neuron_indices=neuron_indices,
            ideal_activation=ideal_activation
        )
        print(len(these_saliency_matrices))
        print(SEPARATOR_STRING)

        if saliency_matrices[0] is None:
            saliency_matrices = copy.deepcopy(these_saliency_matrices)
            print(len(saliency_matrices))
        else:
            for j in range(len(saliency_matrices)):
                saliency_matrices[j] = numpy.concatenate(
                    (saliency_matrices[j], these_saliency_matrices[j]), axis=0
                )

            print(len(saliency_matrices))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    saliency.write_file(
        netcdf_file_name=output_file_name, saliency_matrices=saliency_matrices,
        cyclone_id_strings=cyclone_id_strings,
        init_times_unix_sec=init_times_unix_sec,
        model_file_name=model_file_name,
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int),
        unique_cyclone_id_strings=getattr(
            INPUT_ARG_OBJECT, CYCLONE_IDS_ARG_NAME
        ),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        neuron_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int
        ),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
