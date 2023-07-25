"""Applies trained neural net in inference mode."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import example_io
import prediction_io
import satellite_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_EXAMPLES_PER_BATCH = 32
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
YEARS_ARG_NAME = 'years'
NUM_DROPOUT_ITERS_ARG_NAME = 'num_dropout_iterations'
USE_QUANTILES_ARG_NAME = 'use_quantiles'
MAX_ENSEMBLE_SIZE_ARG_NAME = 'max_ensemble_size'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of input directory, containing examples to predict.  Files therein '
    'will be found by `example_io.find_file` and read by '
    '`example_io.read_file`.'
)
YEARS_HELP_STRING = 'Model will be applied to tropical cyclones in these years.'
NUM_DROPOUT_ITERS_HELP_STRING = (
    'Number of iterations for Monte Carlo dropout.  If you do not want to use '
    'MC dropout, make this argument <= 0.'
)
USE_QUANTILES_HELP_STRING = (
    '[used only if NN does quantile regression] Boolean flag.  If 1, will save '
    'predictions for every quantile.  If 0, will save only mean predictions.'
)
MAX_ENSEMBLE_SIZE_HELP_STRING = (
    '[used only if NN does uncertainty quantification with something other '
    'than quantile regression] Max ensemble size.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Predictions and targets will be written here by'
    ' `prediction_io.write_file`, to an exact location determined by '
    '`prediction_io.find_file`.'
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
    '--' + NUM_DROPOUT_ITERS_ARG_NAME, type=int, required=False, default=0,
    help=NUM_DROPOUT_ITERS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_QUANTILES_ARG_NAME, type=int, required=False, default=0,
    help=USE_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ENSEMBLE_SIZE_ARG_NAME, type=int, required=False, default=1e10,
    help=MAX_ENSEMBLE_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _apply_nn_one_example_file(
        model_object, model_metadata_dict, example_file_name,
        num_dropout_iterations, use_quantiles):
    """Applies neural net to all examples in one file.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    :param example_file_name: Path to example file.
    :param num_dropout_iterations: See documentation at top of this script.
    :param use_quantiles: Same.
    :return: forecast_prob_matrix: See input doc for `prediction_io.write_file`.
    :return: target_class_matrix: Same.
    :return: data_dict: Dictionary returned by `neural_net.create_inputs`.
    """

    quantile_levels = model_metadata_dict[neural_net.QUANTILE_LEVELS_KEY]

    option_dict = copy.deepcopy(
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    option_dict[neural_net.EXAMPLE_FILE_KEY] = example_file_name
    data_dict = neural_net.create_inputs(option_dict)

    if data_dict[neural_net.TARGET_MATRIX_KEY] is None:
        return None, None, None

    target_class_matrix = numpy.argmax(
        data_dict[neural_net.TARGET_MATRIX_KEY], axis=-1
    )

    predictor_matrices = [
        m for m in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
        if m is not None
    ]

    if num_dropout_iterations > 1:
        forecast_prob_matrix = None
        ensemble_size = -1

        for k in range(num_dropout_iterations):
            new_prob_matrix = neural_net.apply_model(
                model_object=model_object,
                model_metadata_dict=model_metadata_dict,
                predictor_matrices=predictor_matrices,
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                use_dropout=True, verbose=True
            )

            if k == 0:
                ensemble_size = new_prob_matrix.shape[-1]

                forecast_prob_matrix = numpy.full(
                    new_prob_matrix.shape[:-1] +
                    (ensemble_size * num_dropout_iterations,),
                    numpy.nan
                )

            first_index = k * ensemble_size
            last_index = first_index + ensemble_size
            forecast_prob_matrix[..., first_index:last_index] = (
                new_prob_matrix + 0.
            )
    else:
        forecast_prob_matrix = neural_net.apply_model(
            model_object=model_object,
            model_metadata_dict=model_metadata_dict,
            predictor_matrices=predictor_matrices,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            use_dropout=False, verbose=True
        )

        if quantile_levels is not None and not use_quantiles:
            prediction_dict = {
                prediction_io.PROBABILITY_MATRIX_KEY: forecast_prob_matrix,
                prediction_io.QUANTILE_LEVELS_KEY: quantile_levels
            }
            forecast_prob_matrix = prediction_io.get_mean_predictions(
                prediction_dict
            )

            forecast_prob_matrix = numpy.expand_dims(
                forecast_prob_matrix, axis=-2
            )
            forecast_prob_matrix = numpy.concatenate(
                (1. - forecast_prob_matrix, forecast_prob_matrix), axis=-2
            )

            forecast_prob_matrix = numpy.expand_dims(
                forecast_prob_matrix, axis=-1
            )

    return forecast_prob_matrix, target_class_matrix, data_dict


def _run(model_file_name, example_dir_name, years, num_dropout_iterations,
         use_quantiles, max_ensemble_size, output_dir_name):
    """Applies trained neural net in inference mode.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param years: Same.
    :param num_dropout_iterations: Same.
    :param use_quantiles: Same.
    :param max_ensemble_size: Same.
    :param output_dir_name: Same.
    """

    max_ensemble_size = max([max_ensemble_size, 1])

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    cyclone_id_string_by_file = example_io.find_cyclones(
        directory_name=example_dir_name, raise_error_if_all_missing=True
    )
    cyclone_year_by_file = numpy.array([
        satellite_utils.parse_cyclone_id(c)[0]
        for c in cyclone_id_string_by_file
    ], dtype=int)

    good_flags = numpy.array(
        [c in years for c in cyclone_year_by_file], dtype=float
    )
    good_indices = numpy.where(good_flags)[0]

    cyclone_id_string_by_file = [
        cyclone_id_string_by_file[k] for k in good_indices
    ]
    cyclone_id_string_by_file.sort()

    example_file_names = [
        example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        for c in cyclone_id_string_by_file
    ]

    target_class_matrix = None
    forecast_prob_matrix = None
    cyclone_id_string_by_example = []
    init_times_unix_sec = numpy.array([], dtype=int)
    storm_latitudes_deg_n = numpy.array([], dtype=float)
    storm_longitudes_deg_e = numpy.array([], dtype=float)

    quantile_levels = model_metadata_dict[neural_net.QUANTILE_LEVELS_KEY]
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    predict_td_to_ts = training_option_dict[neural_net.PREDICT_TD_TO_TS_KEY]

    if quantile_levels is None:
        use_quantiles = False
    else:
        num_dropout_iterations = 0

    if predict_td_to_ts:
        storm_intensity_changes_m_s01 = None
    else:
        storm_intensity_changes_m_s01 = numpy.array([], dtype=float)

    for i in range(len(example_file_names)):
        this_prob_matrix, this_target_matrix, this_data_dict = (
            _apply_nn_one_example_file(
                model_object=model_object,
                model_metadata_dict=model_metadata_dict,
                example_file_name=example_file_names[i],
                num_dropout_iterations=num_dropout_iterations,
                use_quantiles=use_quantiles
            )
        )

        cyclone_id_string_by_example += (
            [cyclone_id_string_by_file[i]] *
            len(this_data_dict[neural_net.INIT_TIMES_KEY])
        )
        init_times_unix_sec = numpy.concatenate(
            (init_times_unix_sec, this_data_dict[neural_net.INIT_TIMES_KEY]),
            axis=0
        )
        storm_latitudes_deg_n = numpy.concatenate((
            storm_latitudes_deg_n,
            this_data_dict[neural_net.STORM_LATITUDES_KEY]
        ), axis=0)

        storm_longitudes_deg_e = numpy.concatenate((
            storm_longitudes_deg_e,
            this_data_dict[neural_net.STORM_LONGITUDES_KEY]
        ), axis=0)

        if not predict_td_to_ts:
            storm_intensity_changes_m_s01 = numpy.concatenate((
                storm_intensity_changes_m_s01,
                this_data_dict[neural_net.STORM_INTENSITY_CHANGES_KEY]
            ), axis=0)

        ensemble_size = this_prob_matrix.shape[-1]
        if not use_quantiles and max_ensemble_size < ensemble_size:
            ensemble_indices = numpy.linspace(
                0, ensemble_size - 1, num=ensemble_size, dtype=int
            )
            ensemble_indices = numpy.random.choice(
                ensemble_indices, size=max_ensemble_size, replace=False
            )
            this_prob_matrix = this_prob_matrix[..., ensemble_indices]

        if forecast_prob_matrix is None:
            target_class_matrix = this_target_matrix + 0
            forecast_prob_matrix = this_prob_matrix + 0.
        else:
            forecast_prob_matrix = numpy.concatenate(
                (forecast_prob_matrix, this_prob_matrix), axis=0
            )
            target_class_matrix = numpy.concatenate(
                (target_class_matrix, this_target_matrix), axis=0
            )

        print(SEPARATOR_STRING)

    if quantile_levels is not None and not use_quantiles:
        quantile_levels_to_write = None
    else:
        quantile_levels_to_write = quantile_levels

    output_file_name = prediction_io.find_file(
        directory_name=output_dir_name, raise_error_if_missing=False
    )
    lead_times_hours = model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY][
        neural_net.LEAD_TIMES_KEY
    ]

    print('Writing predictions and target values to: "{0:s}"...'.format(
        output_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        forecast_probability_matrix=forecast_prob_matrix,
        target_class_matrix=target_class_matrix,
        cyclone_id_strings=cyclone_id_string_by_example,
        init_times_unix_sec=init_times_unix_sec,
        storm_latitudes_deg_n=storm_latitudes_deg_n,
        storm_longitudes_deg_e=storm_longitudes_deg_e,
        storm_intensity_changes_m_s01=storm_intensity_changes_m_s01,
        model_file_name=model_file_name,
        lead_times_hours=lead_times_hours,
        quantile_levels=quantile_levels_to_write,
        isotonic_model_file_name=None,
        uncertainty_calib_model_file_name=None
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int),
        num_dropout_iterations=getattr(
            INPUT_ARG_OBJECT, NUM_DROPOUT_ITERS_ARG_NAME
        ),
        use_quantiles=bool(getattr(INPUT_ARG_OBJECT, USE_QUANTILES_ARG_NAME)),
        max_ensemble_size=getattr(INPUT_ARG_OBJECT, MAX_ENSEMBLE_SIZE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
