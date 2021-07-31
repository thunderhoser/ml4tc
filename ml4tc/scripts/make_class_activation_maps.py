"""Creates class-activation maps (CAM)."""

import copy
import argparse
import numpy
import tensorflow
from ml4tc.io import example_io
from ml4tc.utils import satellite_utils
from ml4tc.machine_learning import gradcam
from ml4tc.machine_learning import neural_net

tensorflow.compat.v1.disable_eager_execution()

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
NUM_EXAMPLES_PER_CHUNK = 100

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
YEARS_ARG_NAME = 'years'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
TARGET_CLASS_ARG_NAME = 'target_class'
TARGET_LAYER_ARG_NAME = 'target_layer_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with input examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
YEARS_HELP_STRING = (
    'Will create class-activation maps for tropical cyclones in these years.  '
    'If you want to use specific cyclones instead, leave this argument alone.'
)
CYCLONE_IDS_HELP_STRING = (
    'Will create class-activation maps for these tropical cyclones.  If you '
    'want to use full years instead, leave this argument alone.'
)
TARGET_CLASS_HELP_STRING = (
    'Class-activation maps will be created for this class.  Must be an integer '
    'in 0...(K - 1), where K = number of classes.'
)
TARGET_LAYER_HELP_STRING = (
    'Name of target layer.  Neuron-importance weights will be based on '
    'activations in this layer.  Layer must have spatial outputs.'
)
OUTPUT_FILE_HELP_STRING = (
    'Name of output file.  Results will be saved here by `gradcam.write_file`.'
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
    '--' + TARGET_CLASS_ARG_NAME, type=int, required=True,
    help=TARGET_CLASS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_LAYER_ARG_NAME, type=str, required=True,
    help=TARGET_LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_dir_name, years, unique_cyclone_id_strings,
         target_class, target_layer_name, output_file_name):
    """Creates class-activation maps (CAM).

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param years: Same.
    :param unique_cyclone_id_strings: Same.
    :param target_class: Same.
    :param target_layer_name: Same.
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

    # Create CAMs for one example (i.e., one tropical cyclone) at a time.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    class_activation_matrix = None
    cyclone_id_strings = []
    init_times_unix_sec = numpy.array([], dtype=int)

    num_examples_done = 0

    for i in range(len(example_file_names)):
        this_option_dict = copy.deepcopy(validation_option_dict)
        this_option_dict[neural_net.EXAMPLE_FILE_KEY] = example_file_names[i]

        print(SEPARATOR_STRING)
        this_data_dict = neural_net.create_inputs(this_option_dict)
        print(SEPARATOR_STRING)

        these_predictor_matrices = [
            m for m in this_data_dict[neural_net.PREDICTOR_MATRICES_KEY]
            if m is not None
        ]
        this_target_array = this_data_dict[neural_net.TARGET_ARRAY_KEY]

        if this_target_array.size == 0:
            continue

        this_num_examples = these_predictor_matrices[0].shape[0]
        cyclone_id_strings += [unique_cyclone_id_strings[i]] * this_num_examples
        init_times_unix_sec = numpy.concatenate((
            init_times_unix_sec,
            this_data_dict[neural_net.INIT_TIMES_KEY]
        ))

        for j in range(this_num_examples):
            if numpy.mod(j, 5) == 0:
                print((
                    'Have created CAM for {0:d} of {1:d} time steps in tropical'
                    ' cyclone...'
                ).format(
                    j, this_num_examples
                ))

            this_cam_matrix = gradcam.run_gradcam(
                model_object=model_object,
                predictor_matrices_one_example=
                [p[[j], ...] for p in these_predictor_matrices],
                target_class=target_class, target_layer_name=target_layer_name
            )

            if class_activation_matrix is None:
                dimensions = (NUM_EXAMPLES_PER_CHUNK,) + this_cam_matrix.shape
                class_activation_matrix = numpy.full(dimensions, numpy.nan)

            if num_examples_done >= class_activation_matrix.shape[0]:
                dimensions = (NUM_EXAMPLES_PER_CHUNK,) + this_cam_matrix.shape
                class_activation_matrix = numpy.concatenate((
                    class_activation_matrix,
                    numpy.full(dimensions, numpy.nan)
                ))

            class_activation_matrix[num_examples_done, ...] = this_cam_matrix
            num_examples_done += 1

        print((
            'Have created CAM for all {0:d} time steps in tropical cyclone!'
        ).format(
            this_num_examples
        ))

    print(SEPARATOR_STRING)
    class_activation_matrix = class_activation_matrix[:num_examples_done, ...]

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    gradcam.write_file(
        netcdf_file_name=output_file_name,
        class_activation_matrix=class_activation_matrix,
        cyclone_id_strings=cyclone_id_strings,
        init_times_unix_sec=init_times_unix_sec,
        model_file_name=model_file_name,
        target_class=target_class, target_layer_name=target_layer_name
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
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        target_layer_name=getattr(INPUT_ARG_OBJECT, TARGET_LAYER_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
