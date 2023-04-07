"""Creates maps of Shapley values."""

import argparse
import numpy
import tensorflow
import keras.layers
import keras.models
import shap
import shap.explainers
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import example_io
from ml4tc.io import prediction_io
from ml4tc.machine_learning import neural_net
from ml4tc.machine_learning import saliency

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
NEW_CYCLONES_PREDICTION_FILE_ARG_NAME = 'new_cyclones_prediction_file_name'
NUM_SMOOTHGRAD_SAMPLES_ARG_NAME = 'num_smoothgrad_samples'
SMOOTHGRAD_STDEV_ARG_NAME = 'smoothgrad_noise_stdev'
OUTPUT_FILE_ARG_NAME = 'output_file_name_sans_sample_num'

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
    'cyclones.  If you would rather get new cyclone IDs from a prediction '
    'file, leave this argument alone.'
)
NEW_CYCLONES_PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file, to be read by `prediction_io.read_file`.  Will '
    'compute Shapley values for all examples (i.e., all cyclone/time pairs) in '
    'this file.  If you would rather specify new cyclone IDs as a list, leave '
    'this argument alone.'
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
    'Path to output file, not including sample number.  If {0:s} <= 1, this '
    'should the actual file path.  However, if {0:s} > 1, this should be just '
    'the beginning of the file path.  For example, suppose that {1:s} is '
    '"foo/shapley_maps" and {0:s} = 5.  Then the output files will be '
    '"foo/shapley_maps_sample000001.zarr", '
    '"foo/shapley_maps_sample000002.zarr", '
    '"foo/shapley_maps_sample000003.zarr", '
    '"foo/shapley_maps_sample000004.zarr", '
    'and "foo/shapley_maps_sample000005.zarr".  Either way, the output file(s) '
    'will be written by `saliency.write_file`.'
).format(
    NUM_SMOOTHGRAD_SAMPLES_ARG_NAME, OUTPUT_FILE_ARG_NAME
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
    '--' + NEW_CYCLONE_IDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=NEW_CYCLONE_IDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_CYCLONES_PREDICTION_FILE_ARG_NAME, type=str, required=False,
    default='', help=NEW_CYCLONES_PREDICTION_FILE_HELP_STRING
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


def _apply_shap_sans_smoothgrad(
        explainer_object, desired_cyclone_id_strings,
        desired_cyclone_init_times_unix_sec, example_dir_name,
        model_file_name, model_metadata_dict, output_file_name):
    """Applies SHAP with SmoothGrad.

    :param explainer_object: See doc for `_apply_shap_with_smoothgrad`.
    :param desired_cyclone_id_strings: Same.
    :param desired_cyclone_init_times_unix_sec: Same.
    :param example_dir_name: Same.
    :param model_file_name: Same.
    :param model_metadata_dict: Same.
    :param output_file_name: Same.
    """

    validation_option_dict = model_metadata_dict[
        neural_net.VALIDATION_OPTIONS_KEY
    ]

    three_predictor_matrices = [None] * 3
    three_shapley_matrices = [None] * 3
    cyclone_id_strings = []
    init_times_unix_sec = numpy.array([], dtype=int)

    for this_cyclone_id_string in list(set(desired_cyclone_id_strings)):
        this_example_file_name = example_io.find_file(
            directory_name=example_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )

        validation_option_dict[neural_net.EXAMPLE_FILE_KEY] = (
            this_example_file_name
        )
        data_dict = neural_net.create_inputs(validation_option_dict)
        print(SEPARATOR_STRING)

        these_desired_indices = numpy.where(
            numpy.array(desired_cyclone_id_strings) == this_cyclone_id_string
        )[0]

        if desired_cyclone_init_times_unix_sec is None:
            num_init_times = len(data_dict[neural_net.INIT_TIMES_KEY])
            these_dict_indices = numpy.linspace(
                0, num_init_times - 1, num=num_init_times, dtype=int
            )
        else:
            these_dict_indices = numpy.array([
                numpy.where(data_dict[neural_net.INIT_TIMES_KEY] == t)[0][0]
                for t in
                desired_cyclone_init_times_unix_sec[these_desired_indices]
            ], dtype=int)

        num_examples_new = len(these_dict_indices)
        cyclone_id_strings += [this_cyclone_id_string] * num_examples_new
        init_times_unix_sec = numpy.concatenate((
            init_times_unix_sec,
            data_dict[neural_net.INIT_TIMES_KEY][these_dict_indices]
        ))

        new_predictor_matrices = [
            None if p is None
            else p[these_dict_indices, ...]
            for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
        ]

        for j in range(len(new_predictor_matrices)):
            if new_predictor_matrices[j] is None:
                continue

            if three_predictor_matrices[j] is None:
                three_predictor_matrices[j] = new_predictor_matrices[j] + 0.
            else:
                three_predictor_matrices[j] = numpy.concatenate(
                    (three_predictor_matrices[j], new_predictor_matrices[j]),
                    axis=0
                )

        new_shapley_matrices = explainer_object.shap_values(
            X=[p for p in new_predictor_matrices if p is not None],
            check_additivity=False
        )
        if isinstance(new_shapley_matrices[0], list):
            new_shapley_matrices = new_shapley_matrices[0]

        j_minor = -1
        for j in range(len(three_shapley_matrices)):
            if new_predictor_matrices[j] is None:
                continue

            j_minor += 1

            if three_shapley_matrices[j] is None:
                three_shapley_matrices[j] = new_shapley_matrices[j_minor] + 0.
            else:
                three_shapley_matrices[j] = numpy.concatenate(
                    (three_shapley_matrices[j], new_shapley_matrices[j_minor]),
                    axis=0
                )

    dummy_input_times_grad_matrices = three_shapley_matrices
    dummy_saliency_matrices = [
        None if ig is None else numpy.divide(ig, p)
        for ig, p in
        zip(dummy_input_times_grad_matrices, three_predictor_matrices)
    ]

    for j in range(len(dummy_saliency_matrices)):
        if dummy_saliency_matrices[j] is None:
            continue

        dummy_saliency_matrices[j][
            numpy.isnan(dummy_saliency_matrices[j])
        ] = numpy.inf

    # TODO(thunderhoser): If this method runs into memory issues, I could write
    # one output file per cyclone, instead of one for all cyclones.
    print('Writing results to: "{0:s}"...'.format(output_file_name))
    saliency.write_file(
        zarr_file_name=output_file_name,
        three_saliency_matrices=dummy_saliency_matrices,
        three_input_grad_matrices=dummy_input_times_grad_matrices,
        cyclone_id_strings=cyclone_id_strings,
        init_times_unix_sec=init_times_unix_sec,
        model_file_name=model_file_name,
        layer_name='mean_output', neuron_indices=numpy.array([0, numpy.nan]),
        ideal_activation=1.
    )


def _apply_shap_with_smoothgrad(
        explainer_object, desired_cyclone_id_strings, example_dir_name,
        num_smoothgrad_samples, smoothgrad_noise_stdev,
        model_file_name, model_metadata_dict,
        output_file_name_sans_sample_num):
    """Applies SHAP with SmoothGrad.

    N = number of desired cyclones

    :param explainer_object: Instance of `shap.DeepExplainer`.
    :param desired_cyclone_id_strings: length-N list of desired cyclone IDs.
    :param example_dir_name: See documentation at top of file.
    :param num_smoothgrad_samples: Number of SmoothGrad samples.
    :param smoothgrad_noise_stdev: Noise level for SmoothGrad.
    :param model_file_name: Path to trained model.
    :param model_metadata_dict: Dictionary with metadata for trained model,
        in format returned by `neural_net.read_metafile`.
    :param output_file_name_sans_sample_num: See documentation at top of file.
    """

    validation_option_dict = model_metadata_dict[
        neural_net.VALIDATION_OPTIONS_KEY
    ]

    three_predictor_matrices = [None] * 3
    cyclone_id_strings = []
    init_times_unix_sec = numpy.array([], dtype=int)

    for this_cyclone_id_string in desired_cyclone_id_strings:
        this_example_file_name = example_io.find_file(
            directory_name=example_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )

        validation_option_dict[neural_net.EXAMPLE_FILE_KEY] = (
            this_example_file_name
        )
        data_dict = neural_net.create_inputs(validation_option_dict)
        print(SEPARATOR_STRING)

        num_examples_new = len(data_dict[neural_net.INIT_TIMES_KEY])
        if num_examples_new == 0:
            continue

        cyclone_id_strings += [this_cyclone_id_string] * num_examples_new
        init_times_unix_sec = numpy.concatenate((
            init_times_unix_sec,
            data_dict[neural_net.INIT_TIMES_KEY]
        ))

        new_predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]

        for j in range(len(new_predictor_matrices)):
            if new_predictor_matrices[j] is None:
                continue

            if three_predictor_matrices[j] is None:
                three_predictor_matrices[j] = new_predictor_matrices[j] + 0.
            else:
                three_predictor_matrices[j] = numpy.concatenate(
                    (three_predictor_matrices[j], new_predictor_matrices[j]),
                    axis=0
                )

    for k in range(num_smoothgrad_samples):
        predictor_matrices_k = [
            None if p is None
            else p + numpy.random.normal(
                loc=0., scale=smoothgrad_noise_stdev, size=p.shape
            )
            for p in three_predictor_matrices
        ]

        shapley_matrices_k = explainer_object.shap_values(
            X=[p for p in predictor_matrices_k if p is not None],
            check_additivity=False
        )
        if isinstance(shapley_matrices_k[0], list):
            shapley_matrices_k = shapley_matrices_k[0]

        three_shapley_matrices = [None] * 3
        j_minor = -1

        for j in range(len(three_shapley_matrices)):
            if three_predictor_matrices[j] is None:
                continue

            j_minor += 1
            three_shapley_matrices[j] = shapley_matrices_k[j_minor]

        output_file_name = '{0:s}_sample{1:06d}.zarr'.format(
            output_file_name_sans_sample_num, k
        )

        dummy_input_times_grad_matrices = three_shapley_matrices
        dummy_saliency_matrices = [
            None if ig is None else numpy.divide(ig, p)
            for ig, p in
            zip(dummy_input_times_grad_matrices, predictor_matrices_k)
        ]

        for j in range(len(dummy_saliency_matrices)):
            if dummy_saliency_matrices[j] is None:
                continue

            # NaN occurs when dividing 0 by 0.  This ensures that, if I try to
            # reconstruct the predictors from dummy saliency and
            # input-times-grad values (as predictor = I*G / saliency), the
            # reconstruct predictor will again be 0.
            dummy_saliency_matrices[j][
                numpy.isnan(dummy_saliency_matrices[j])
            ] = numpy.inf

        print('Writing results to: "{0:s}"...'.format(output_file_name))
        saliency.write_file(
            zarr_file_name=output_file_name,
            three_saliency_matrices=dummy_saliency_matrices,
            three_input_grad_matrices=dummy_input_times_grad_matrices,
            cyclone_id_strings=cyclone_id_strings,
            init_times_unix_sec=init_times_unix_sec,
            model_file_name=model_file_name,
            layer_name='mean_output',
            neuron_indices=numpy.array([0, numpy.nan]),
            ideal_activation=1.
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
         new_cyclones_prediction_file_name, num_smoothgrad_samples,
         smoothgrad_noise_stdev, output_file_name_sans_sample_num):
    """Creates maps of Shapley values.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param baseline_cyclone_id_strings: Same.
    :param max_num_baseline_examples: Same.
    :param new_cyclone_id_strings: Same.
    :param new_cyclones_prediction_file_name: Same.
    :param num_smoothgrad_samples: Same.
    :param smoothgrad_noise_stdev: Same.
    :param output_file_name_sans_sample_num: Same.
    """

    # Check input args.
    use_smoothgrad = num_smoothgrad_samples > 0
    if use_smoothgrad:
        error_checking.assert_is_greater(smoothgrad_noise_stdev, 0.)

    if use_smoothgrad and output_file_name_sans_sample_num.endswith('.nc'):
        output_file_name_sans_sample_num = (
            output_file_name_sans_sample_num[:-3]
        )

    if use_smoothgrad and output_file_name_sans_sample_num.endswith('.zarr'):
        output_file_name_sans_sample_num = (
            output_file_name_sans_sample_num[:-5]
        )

    error_checking.assert_is_geq(max_num_baseline_examples, 20)
    error_checking.assert_is_leq(max_num_baseline_examples, 200)

    if len(new_cyclone_id_strings) == 1 and new_cyclone_id_strings[0] == '':
        new_cyclone_id_strings = None

    if new_cyclone_id_strings is None:
        print('Reading cyclone IDs and times from: "{0:s}"...'.format(
            new_cyclones_prediction_file_name
        ))
        new_cyclones_prediction_dict = prediction_io.read_file(
            new_cyclones_prediction_file_name
        )

        new_cyclone_id_strings = new_cyclones_prediction_dict[
            prediction_io.CYCLONE_IDS_KEY
        ]
        new_cyclone_init_times_unix_sec = new_cyclones_prediction_dict[
            prediction_io.INIT_TIMES_KEY
        ]
    else:
        new_cyclone_init_times_unix_sec = None

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

    del data_dict
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
    del three_baseline_predictor_matrices

    # The if-else statement below is designed for memory-efficiency.
    if num_smoothgrad_samples > 1:
        # This method reads all the new cyclones into memory at once, then
        # applies SHAP multiple times (once per SmoothGrad iteration).
        _apply_shap_with_smoothgrad(
            explainer_object=explainer_object,
            desired_cyclone_id_strings=list(set(new_cyclone_id_strings)),
            example_dir_name=example_dir_name,
            num_smoothgrad_samples=num_smoothgrad_samples,
            smoothgrad_noise_stdev=smoothgrad_noise_stdev,
            model_file_name=model_file_name,
            model_metadata_dict=model_metadata_dict,
            output_file_name_sans_sample_num=output_file_name_sans_sample_num
        )
    else:
        # For each new cyclone C, this method reads the data for C into memory,
        # applies SHAP to C, then purges the data for C from memory.
        _apply_shap_sans_smoothgrad(
            explainer_object=explainer_object,
            desired_cyclone_id_strings=new_cyclone_id_strings,
            desired_cyclone_init_times_unix_sec=new_cyclone_init_times_unix_sec,
            example_dir_name=example_dir_name,
            model_file_name=model_file_name,
            model_metadata_dict=model_metadata_dict,
            output_file_name=output_file_name_sans_sample_num
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
        new_cyclones_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, NEW_CYCLONES_PREDICTION_FILE_ARG_NAME
        ),
        num_smoothgrad_samples=getattr(
            INPUT_ARG_OBJECT, NUM_SMOOTHGRAD_SAMPLES_ARG_NAME
        ),
        smoothgrad_noise_stdev=getattr(
            INPUT_ARG_OBJECT, SMOOTHGRAD_STDEV_ARG_NAME
        ),
        output_file_name_sans_sample_num=getattr(
            INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME
        )
    )
