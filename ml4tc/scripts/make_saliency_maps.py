"""Creates saliency maps."""

import copy
import argparse
import numpy
import tensorflow
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import example_io
from ml4tc.utils import satellite_utils
from ml4tc.machine_learning import neural_net
from ml4tc.machine_learning import saliency

tensorflow.compat.v1.disable_eager_execution()

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
YEARS_ARG_NAME = 'years'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
LAYER_NAME_ARG_NAME = 'layer_name'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
NUM_SMOOTHGRAD_SAMPLES_ARG_NAME = 'num_smoothgrad_samples'
SMOOTHGRAD_STDEV_ARG_NAME = 'smoothgrad_noise_stdev'
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
    'This is a weird one.  See doc for `saliency.get_saliency_one_neuron`.'
)
IDEAL_ACTIVATION_HELP_STRING = (
    'Ideal neuron activation, used to define loss function.  The loss function '
    'will be (neuron_activation - ideal_activation)**2.'
)
NUM_SMOOTHGRAD_SAMPLES_HELP_STRING = (
    'Number of samples for SmoothGrad.  If you do not want to use SmoothGrad, '
    'make this argument <= 0.'
)
SMOOTHGRAD_STDEV_HELP_STRING = (
    'Standard deviation of Gaussian noise for SmoothGrad.  If you do not want '
    'to use SmoothGrad, leave this as the default.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results will be saved here by '
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
    '--' + NEURON_INDICES_ARG_NAME, type=float, nargs='+', required=True,
    help=NEURON_INDICES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=True,
    help=IDEAL_ACTIVATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SMOOTHGRAD_SAMPLES_ARG_NAME, type=int, required=False, default=0,
    help=NUM_SMOOTHGRAD_SAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHGRAD_STDEV_ARG_NAME, type=float, required=False, default=-1,
    help=SMOOTHGRAD_STDEV_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_dir_name, years, unique_cyclone_id_strings,
         layer_name, neuron_indices, ideal_activation, num_smoothgrad_samples,
         smoothgrad_noise_stdev, output_file_name):
    """Creates saliency maps.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param years: Same.
    :param unique_cyclone_id_strings: Same.
    :param layer_name: Same.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param num_smoothgrad_samples: Same.
    :param smoothgrad_noise_stdev: Same.
    :param output_file_name: Same.
    """

    # Process input args.
    if len(years) == 1 and years[0] < 0:
        years = None

    use_smoothgrad = num_smoothgrad_samples > 0
    if use_smoothgrad:
        error_checking.assert_is_greater(smoothgrad_noise_stdev, 0.)

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

    # Create saliency maps.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    three_saliency_matrices = [None]
    three_input_grad_matrices = [None]
    cyclone_id_strings = []
    init_times_unix_sec = numpy.array([], dtype=int)

    for i in range(len(example_file_names)):
        this_option_dict = copy.deepcopy(validation_option_dict)
        this_option_dict[neural_net.EXAMPLE_FILE_KEY] = example_file_names[i]
        this_data_dict = neural_net.create_inputs(this_option_dict)
        print(SEPARATOR_STRING)

        if this_data_dict[neural_net.INIT_TIMES_KEY].size == 0:
            continue

        new_predictor_matrices = (
            this_data_dict[neural_net.PREDICTOR_MATRICES_KEY]
        )

        this_num_examples = new_predictor_matrices[0].shape[0]
        cyclone_id_strings += [unique_cyclone_id_strings[i]] * this_num_examples
        init_times_unix_sec = numpy.concatenate((
            init_times_unix_sec,
            this_data_dict[neural_net.INIT_TIMES_KEY]
        ))

        if use_smoothgrad:
            new_saliency_matrices = [None]

            for k in num_smoothgrad_samples:
                these_predictor_matrices = [
                    None if p is None
                    else p + numpy.random.normal(
                        loc=0., scale=smoothgrad_noise_stdev, size=p.shape
                    )
                    for p in new_predictor_matrices
                ]

                these_saliency_matrices = saliency.get_saliency_one_neuron(
                    model_object=model_object,
                    three_predictor_matrices=these_predictor_matrices,
                    layer_name=layer_name, neuron_indices=neuron_indices,
                    ideal_activation=ideal_activation
                )

                if all([s is None for s in new_saliency_matrices]):
                    for j in range(len(these_saliency_matrices)):
                        if these_saliency_matrices[j] is None:
                            continue

                        new_saliency_matrices[j] = numpy.full(
                            these_saliency_matrices[j].shape +
                            (num_smoothgrad_samples,),
                            numpy.nan
                        )

                for j in range(len(these_saliency_matrices)):
                    if these_saliency_matrices[j] is None:
                        continue

                    new_saliency_matrices[j][..., k] = (
                        these_saliency_matrices[j]
                    )

            new_saliency_matrices = [
                None if s is None else numpy.mean(s, axis=-1)
                for s in new_saliency_matrices
            ]
        else:
            new_saliency_matrices = saliency.get_saliency_one_neuron(
                model_object=model_object,
                three_predictor_matrices=new_predictor_matrices,
                layer_name=layer_name, neuron_indices=neuron_indices,
                ideal_activation=ideal_activation
            )

        new_input_grad_matrices = [
            None if p is None
            else p * s
            for p, s in zip(new_predictor_matrices, new_saliency_matrices)
        ]
        print(SEPARATOR_STRING)

        if all([m is None for m in three_saliency_matrices]):
            three_saliency_matrices = copy.deepcopy(new_saliency_matrices)
            three_input_grad_matrices = copy.deepcopy(new_input_grad_matrices)
        else:
            for j in range(len(three_saliency_matrices)):
                if three_saliency_matrices[j] is None:
                    continue

                three_saliency_matrices[j] = numpy.concatenate((
                    three_saliency_matrices[j], new_saliency_matrices[j]
                ), axis=0)

                three_input_grad_matrices[j] = numpy.concatenate((
                    three_input_grad_matrices[j],
                    new_input_grad_matrices[j]
                ), axis=0)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    saliency.write_file(
        netcdf_file_name=output_file_name,
        three_saliency_matrices=three_saliency_matrices,
        three_input_grad_matrices=three_input_grad_matrices,
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
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=float
        ),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        num_smoothgrad_samples=getattr(
            INPUT_ARG_OBJECT, NUM_SMOOTHGRAD_SAMPLES_ARG_NAME
        ),
        smoothgrad_noise_stdev=getattr(
            INPUT_ARG_OBJECT, SMOOTHGRAD_STDEV_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
