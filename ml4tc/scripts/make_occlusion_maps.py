"""Creates occlusion maps."""

import copy
import argparse
import numpy
from ml4tc.io import example_io
from ml4tc.utils import satellite_utils
from ml4tc.machine_learning import neural_net
from ml4tc.machine_learning import occlusion

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
YEARS_ARG_NAME = 'years'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
TARGET_CLASS_ARG_NAME = 'target_class'
HALF_WINDOW_SIZE_ARG_NAME = 'half_window_size_px'
STRIDE_LENGTH_ARG_NAME = 'stride_length_px'
FILL_VALUE_ARG_NAME = 'fill_value'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with input examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
YEARS_HELP_STRING = (
    'Will create occlusion maps for tropical cyclones in these years.  If you '
    'want to use specific cyclones instead, leave this argument alone.'
)
CYCLONE_IDS_HELP_STRING = (
    'Will create occlusion maps for these tropical cyclones.  If you want to '
    'use full years instead, leave this argument alone.'
)
TARGET_CLASS_HELP_STRING = (
    'Occlusion maps will be created for this class.  Must be an integer in '
    '0...(K - 1), where K = number of classes.'
)
HALF_WINDOW_SIZE_HELP_STRING = (
    'Half-size of occlusion window (pixels).  If half-size is P, the full '
    'window will (2 * P + 1) rows by (2 * P + 1) columns.'
)
STRIDE_LENGTH_HELP_STRING = 'Stride length for occlusion window (pixels).'
FILL_VALUE_HELP_STRING = (
    'Fill value.  Inside the occlusion window, all brightness temperatures will'
    ' be assigned this value, to simulate missing data.'
)
OUTPUT_FILE_HELP_STRING = (
    'Name of output file.  Results will be saved here by '
    '`occlusion.write_file`.'
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
    '--' + HALF_WINDOW_SIZE_ARG_NAME, type=int, required=True,
    help=HALF_WINDOW_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + STRIDE_LENGTH_ARG_NAME, type=int, required=True,
    help=STRIDE_LENGTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FILL_VALUE_ARG_NAME, type=float, required=False, default=0,
    help=FILL_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_dir_name, years, unique_cyclone_id_strings,
         target_class, half_window_size_px, stride_length_px, fill_value,
         output_file_name):
    """Creates occlusion maps.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param years: Same.
    :param unique_cyclone_id_strings: Same.
    :param target_class: Same.
    :param half_window_size_px: Same.
    :param stride_length_px: Same.
    :param fill_value: Same.
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

    # Create occlusion maps.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    three_occlusion_prob_matrices = [None]
    three_norm_occlusion_matrices = [None]
    cyclone_id_strings = []
    init_times_unix_sec = numpy.array([], dtype=int)

    for i in range(len(example_file_names)):
        this_option_dict = copy.deepcopy(validation_option_dict)
        this_option_dict[neural_net.EXAMPLE_FILE_KEY] = example_file_names[i]
        this_data_dict = neural_net.create_inputs(this_option_dict)
        print(SEPARATOR_STRING)

        if this_data_dict[neural_net.INIT_TIMES_KEY].size == 0:
            continue

        these_predictor_matrices = (
            this_data_dict[neural_net.PREDICTOR_MATRICES_KEY]
        )

        this_num_examples = these_predictor_matrices[0].shape[0]
        cyclone_id_strings += [unique_cyclone_id_strings[i]] * this_num_examples
        init_times_unix_sec = numpy.concatenate((
            init_times_unix_sec,
            this_data_dict[neural_net.INIT_TIMES_KEY]
        ))

        new_occlusion_prob_matrices, new_original_probs = (
            occlusion.get_occlusion_maps(
                model_object=model_object,
                model_metadata_dict=model_metadata_dict,
                predictor_matrices=these_predictor_matrices,
                target_class=target_class,
                half_window_size_px=half_window_size_px,
                stride_length_px=stride_length_px, fill_value=fill_value
            )
        )
        new_norm_occlusion_matrices = occlusion.normalize_occlusion_maps(
            occlusion_prob_matrices=new_occlusion_prob_matrices,
            original_probs=new_original_probs
        )

        if all([m is None for m in three_occlusion_prob_matrices]):
            three_occlusion_prob_matrices = copy.deepcopy(
                new_occlusion_prob_matrices
            )
            three_norm_occlusion_matrices = copy.deepcopy(
                new_norm_occlusion_matrices
            )
        else:
            for j in range(len(three_occlusion_prob_matrices)):
                if three_occlusion_prob_matrices[j] is None:
                    continue

                three_occlusion_prob_matrices[j] = numpy.concatenate((
                    three_occlusion_prob_matrices[j],
                    new_occlusion_prob_matrices[j]
                ), axis=0)

                three_norm_occlusion_matrices[j] = numpy.concatenate((
                    three_norm_occlusion_matrices[j],
                    new_norm_occlusion_matrices[j]
                ), axis=0)

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    occlusion.write_file(
        netcdf_file_name=output_file_name,
        three_occlusion_prob_matrices=three_occlusion_prob_matrices,
        three_norm_occlusion_matrices=three_norm_occlusion_matrices,
        cyclone_id_strings=cyclone_id_strings,
        init_times_unix_sec=init_times_unix_sec,
        model_file_name=model_file_name,
        target_class=target_class, half_window_size_px=half_window_size_px,
        stride_length_px=stride_length_px, fill_value=fill_value
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
        half_window_size_px=getattr(
            INPUT_ARG_OBJECT, HALF_WINDOW_SIZE_ARG_NAME
        ),
        stride_length_px=getattr(INPUT_ARG_OBJECT, STRIDE_LENGTH_ARG_NAME),
        fill_value=getattr(INPUT_ARG_OBJECT, FILL_VALUE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
