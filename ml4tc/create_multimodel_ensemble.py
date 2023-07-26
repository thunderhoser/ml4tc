"""Creates multi-model ensemble."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import prediction_io

INPUT_FILES_ARG_NAME = 'input_prediction_file_names'
MAX_ENSEMBLE_SIZE_ARG_NAME = 'max_total_ensemble_size'
OUTPUT_FILE_ARG_NAME = 'output_prediction_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each file will be read by '
    '`prediction_io.read_file`, and predictions from all these files will be '
    'concatenated along the final (ensemble-member) axis.'
)
MAX_ENSEMBLE_SIZE_HELP_STRING = (
    'Maximum size of total ensemble, after concatenating predictions from all '
    'input files along the final axis.  In other words, the size of the final '
    'axis may not exceed {0:s}.  If it does, {0:s} predictions will be '
    'randomly selected.'
).format(MAX_ENSEMBLE_SIZE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  All predictions will be written here by '
    '`prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ENSEMBLE_SIZE_ARG_NAME, type=int, required=True,
    help=MAX_ENSEMBLE_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_names, max_ensemble_size, output_file_name):
    """Creates multi-model ensemble.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param max_ensemble_size: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(max_ensemble_size, 2)

    num_models = len(input_file_names)
    prediction_dicts = [dict()] * num_models
    model_file_names = [''] * num_models
    isotonic_model_file_names = [''] * num_models
    uncty_calib_model_file_names = [''] * num_models

    for i in range(num_models):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        prediction_dicts[i] = prediction_io.read_file(input_file_names[i])

        model_file_names[i] = copy.deepcopy(
            prediction_dicts[i][prediction_io.MODEL_FILE_KEY]
        )
        model_file_names[i] = copy.deepcopy(
            isotonic_model_file_names[i][prediction_io.ISOTONIC_MODEL_FILE_KEY]
        )
        uncty_calib_model_file_names[i] = copy.deepcopy(
            isotonic_model_file_names[i][
                prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY
            ]
        )

    prediction_dict = prediction_io.concat_over_ensemble_members(
        prediction_dicts
    )
    del prediction_dicts

    ensemble_size = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape[-1]
    )

    if ensemble_size > max_ensemble_size:
        member_indices = numpy.linspace(
            0, ensemble_size - 1, num=ensemble_size, dtype=int
        )
        desired_indices = numpy.random.choice(
            member_indices, size=max_ensemble_size, replace=False
        )
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = prediction_dict[
            prediction_io.PROBABILITY_MATRIX_KEY
        ][..., desired_indices]

    dummy_model_file_name = ' '.join(model_file_names)

    if all([m is None for m in isotonic_model_file_names]):
        dummy_iso_model_file_name = None
    else:
        dummy_iso_model_file_name = ' '.join(isotonic_model_file_names)

    if all([m is None for m in uncty_calib_model_file_names]):
        dummy_uncty_calib_model_file_name = None
    else:
        dummy_uncty_calib_model_file_name = ' '.join(
            uncty_calib_model_file_names
        )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    d = prediction_dict

    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
        target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
        cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
        init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
        storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
        storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
        storm_intensity_changes_m_s01=
        d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
        model_file_name=dummy_model_file_name,
        lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
        quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
        isotonic_model_file_name=dummy_iso_model_file_name,
        uncertainty_calib_model_file_name=dummy_uncty_calib_model_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        max_ensemble_size=getattr(INPUT_ARG_OBJECT, MAX_ENSEMBLE_SIZE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
