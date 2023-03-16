"""Creates maps of Shapley values."""

import os
import sys
import copy
import argparse
import numpy
import tensorflow
import keras.layers
import keras.models
import shap
import shap.explainers

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import example_io
import neural_net
import saliency

# TODO(thunderhoser): This script assumes that the model outputs an ensemble
# (e.g., CRPS-constrained) of predictions, and it adds a global-average-pooling
# layer to the end of the model.

tensorflow.compat.v1.disable_v2_behavior()
# tensorflow.compat.v1.disable_eager_execution()
# tensorflow.config.threading.set_inter_op_parallelism_threads(1)
# tensorflow.config.threading.set_intra_op_parallelism_threads(1)

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
BASELINE_CYCLONE_IDS_ARG_NAME = 'baseline_cyclone_id_strings'
MAX_BASELINE_EXAMPLES_ARG_NAME = 'max_num_baseline_examples'
NEW_CYCLONE_IDS_ARG_NAME = 'new_cyclone_id_strings'
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
BASELINE_CYCLONE_IDS_HELP_STRING = (
    'List of IDs in format "yyyyBBNN".  Will use these cyclones to create the '
    'baseline.'
)
MAX_BASELINE_EXAMPLES_HELP_STRING = (
    'Max number of baseline examples.  If this script finds more baseline '
    'examples, it will randomly subset.  Running SHAP with too many baseline '
    'examples is very computationally expensive.'
)
NEW_CYCLONE_IDS_HELP_STRING = (
    'List of IDs in format "yyyyBBNN".  Will compute Shapley values for these '
    'cyclones.'
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
    '--' + BASELINE_CYCLONE_IDS_ARG_NAME, type=str, nargs='+', required=True,
    help=BASELINE_CYCLONE_IDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_BASELINE_EXAMPLES_ARG_NAME, type=int, required=True,
    help=MAX_BASELINE_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_CYCLONE_IDS_ARG_NAME, type=str, nargs='+', required=True,
    help=NEW_CYCLONE_IDS_HELP_STRING
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


def _predictor_matrices_to_num_examples(predictor_matrices):
    """Determines number of examples from list of predictor matrices.

    :param predictor_matrices: 1-D list of numpy arrays.  The first axis of
        each numpy array is the example axis.
    :return: num_examples: Integer.
    """

    for this_matrix in predictor_matrices:
        if this_matrix is None:
            continue

        return this_matrix.shape[0]

    return None


def _run(model_file_name, example_dir_name, baseline_cyclone_id_strings,
         max_num_baseline_examples, new_cyclone_id_strings,
         num_smoothgrad_samples, smoothgrad_noise_stdev,
         output_file_name):
    """Creates maps of Shapley values.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param baseline_cyclone_id_strings: Same.
    :param max_num_baseline_examples: Same.
    :param new_cyclone_id_strings: Same.
    :param num_smoothgrad_samples: Same.
    :param smoothgrad_noise_stdev: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    use_smoothgrad = num_smoothgrad_samples > 0
    if use_smoothgrad:
        error_checking.assert_is_greater(smoothgrad_noise_stdev, 0.)

    error_checking.assert_is_geq(max_num_baseline_examples, 20)
    error_checking.assert_is_leq(max_num_baseline_examples, 200)

    # Append global-average-pooling layer to model.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    orig_model_object = neural_net.read_model(model_file_name)

    output_layer_object = orig_model_object.output
    output_layer_object = keras.layers.GlobalAveragePooling1D(
        data_format='channels_first'
    )(output_layer_object)

    model_object = keras.models.Model(
        inputs=orig_model_object.input, outputs=output_layer_object
    )

    # Find satellite data for baseline and new cyclones.
    baseline_cyclone_id_strings = list(set(baseline_cyclone_id_strings))
    baseline_example_file_names = [
        example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        for c in baseline_cyclone_id_strings
    ]

    max_num_baseline_examples_per_file = int(numpy.ceil(
        float(max_num_baseline_examples) / len(baseline_example_file_names)
    ))

    new_cyclone_id_strings = list(set(new_cyclone_id_strings))
    new_example_file_names = [
        example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        for c in new_cyclone_id_strings
    ]

    # Read satellite data for baseline.
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )

    three_baseline_predictor_matrices = [None] * 3
    print(SEPARATOR_STRING)

    for this_file_name in baseline_example_file_names:
        validation_option_dict[neural_net.EXAMPLE_FILE_KEY] = this_file_name
        data_dict = neural_net.create_inputs(validation_option_dict)

        this_num_examples = _predictor_matrices_to_num_examples(
            data_dict[neural_net.PREDICTOR_MATRICES_KEY]
        )
        if this_num_examples == 0:
            continue

        good_indices = numpy.linspace(
            0, this_num_examples - 1, num=this_num_examples, dtype=int
        )
        if this_num_examples > max_num_baseline_examples_per_file:
            good_indices = numpy.random.choice(
                good_indices, size=max_num_baseline_examples_per_file,
                replace=True
            )

        for j in range(len(three_baseline_predictor_matrices)):
            if data_dict[neural_net.PREDICTOR_MATRICES_KEY][j] is None:
                continue

            if three_baseline_predictor_matrices[j] is None:
                three_baseline_predictor_matrices[j] = data_dict[
                    neural_net.PREDICTOR_MATRICES_KEY
                ][j][good_indices, ...]
            else:
                three_baseline_predictor_matrices[j] = numpy.concatenate((
                    three_baseline_predictor_matrices[j],
                    data_dict[neural_net.PREDICTOR_MATRICES_KEY][j][
                        good_indices, ...
                    ]
                ), axis=0)

    print(SEPARATOR_STRING)

    num_baseline_examples = _predictor_matrices_to_num_examples(
        three_baseline_predictor_matrices
    )

    if num_baseline_examples > max_num_baseline_examples:
        example_indices = numpy.linspace(
            0, num_baseline_examples - 1, num=num_baseline_examples, dtype=int
        )
        example_indices = numpy.random.choice(
            example_indices, size=max_num_baseline_examples, replace=False
        )
        three_baseline_predictor_matrices = [
            None if p is None else p[example_indices, ...]
            for p in three_baseline_predictor_matrices
        ]

    # Do actual stuff.
    explainer_object = shap.DeepExplainer(
        model=model_object,
        data=[p for p in three_baseline_predictor_matrices if p is not None]
    )

    three_shapley_value_matrices = [None] * 3
    cyclone_id_strings = []
    init_times_unix_sec = numpy.array([], dtype=int)

    for i in range(len(new_example_file_names)):
        validation_option_dict[neural_net.EXAMPLE_FILE_KEY] = (
            new_example_file_names[i]
        )
        data_dict = neural_net.create_inputs(validation_option_dict)

        if data_dict[neural_net.INIT_TIMES_KEY].size == 0:
            print(SEPARATOR_STRING)
            continue

        new_predictor_matrices = (
            data_dict[neural_net.PREDICTOR_MATRICES_KEY]
        )
        num_examples_new = new_predictor_matrices[0].shape[0]

        cyclone_id_strings += [new_cyclone_id_strings[i]] * num_examples_new
        init_times_unix_sec = numpy.concatenate((
            init_times_unix_sec,
            data_dict[neural_net.INIT_TIMES_KEY]
        ))

        if use_smoothgrad:
            new_shapley_matrices = [None] * 3

            for k in range(num_smoothgrad_samples):
                these_predictor_matrices = [
                    None if p is None
                    else p + numpy.random.normal(
                        loc=0., scale=smoothgrad_noise_stdev, size=p.shape
                    )
                    for p in new_predictor_matrices
                ]

                these_shapley_matrices = explainer_object.shap_values(
                    X=[p for p in these_predictor_matrices if p is not None],
                    check_additivity=False
                )
                if isinstance(these_shapley_matrices[0], list):
                    these_shapley_matrices = these_shapley_matrices[0]

                if all([s is None for s in new_shapley_matrices]):
                    j_minor = -1

                    for j in range(len(new_shapley_matrices)):
                        if these_predictor_matrices[j] is None:
                            continue

                        j_minor += 1

                        new_shapley_matrices[j] = numpy.full(
                            these_shapley_matrices[j_minor].shape +
                            (num_smoothgrad_samples,),
                            numpy.nan
                        )

                j_minor = -1

                for j in range(len(new_shapley_matrices)):
                    if these_predictor_matrices[j] is None:
                        continue

                    j_minor += 1

                    new_shapley_matrices[j][..., k] = (
                        these_shapley_matrices[j_minor]
                    )

            new_shapley_matrices = [
                None if s is None else numpy.mean(s, axis=-1)
                for s in new_shapley_matrices
            ]
        else:
            these_shapley_matrices = explainer_object.shap_values(
                X=[p for p in new_predictor_matrices if p is not None],
                check_additivity=False
            )
            if isinstance(these_shapley_matrices[0], list):
                these_shapley_matrices = these_shapley_matrices[0]

            new_shapley_matrices = [None] * 3
            j_minor = -1

            for j in range(len(new_shapley_matrices)):
                if new_predictor_matrices[j] is None:
                    continue

                j_minor += 1
                new_shapley_matrices[j] = these_shapley_matrices[j_minor] + 0.

        print(SEPARATOR_STRING)

        if all([m is None for m in three_shapley_value_matrices]):
            three_shapley_value_matrices = copy.deepcopy(new_shapley_matrices)
        else:
            for j in range(len(three_shapley_value_matrices)):
                if three_shapley_value_matrices[j] is None:
                    continue

                three_shapley_value_matrices[j] = numpy.concatenate((
                    three_shapley_value_matrices[j], new_shapley_matrices[j]
                ), axis=0)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    saliency.write_file(
        netcdf_file_name=output_file_name,
        three_saliency_matrices=three_shapley_value_matrices,
        three_input_grad_matrices=three_shapley_value_matrices,
        cyclone_id_strings=cyclone_id_strings,
        init_times_unix_sec=init_times_unix_sec,
        model_file_name=model_file_name,
        layer_name='mean_output', neuron_indices=numpy.array([0, numpy.nan]),
        ideal_activation=1.
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        baseline_cyclone_id_strings=getattr(
            INPUT_ARG_OBJECT, BASELINE_CYCLONE_IDS_ARG_NAME
        ),
        max_num_baseline_examples=getattr(
            INPUT_ARG_OBJECT, MAX_BASELINE_EXAMPLES_ARG_NAME
        ),
        new_cyclone_id_strings=getattr(
            INPUT_ARG_OBJECT, NEW_CYCLONE_IDS_ARG_NAME
        ),
        num_smoothgrad_samples=getattr(
            INPUT_ARG_OBJECT, NUM_SMOOTHGRAD_SAMPLES_ARG_NAME
        ),
        smoothgrad_noise_stdev=getattr(
            INPUT_ARG_OBJECT, SMOOTHGRAD_STDEV_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
