"""Applies uncertainty-calibration model to neural-net predictions."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import uncertainty_calibration

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
MODEL_FILE_ARG_NAME = 'input_model_file_name'
OUTPUT_PREDICTION_FILE_ARG_NAME = 'output_prediction_file_name'

INPUT_PREDICTION_FILE_HELP_STRING = (
    'Path to file containing neural-net predictions with no uncertainty '
    'calibration.  Will be read by `prediction_io.read_file`.'
)
MODEL_FILE_HELP_STRING = (
    'Path to file with trained uncertainty-calibration model.  Will be read'
    ' by `uncertainty_calibration.read_model`.'
)
OUTPUT_PREDICTION_FILE_HELP_STRING = (
    'Path to output file, containing model predictions after uncertainty '
    'calibration.  Will be written by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_PREDICTION_FILE_HELP_STRING
)


def _run(input_prediction_file_name, model_file_name,
         output_prediction_file_name):
    """Applies uncertainty-calibration model to neural-net predictions.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param model_file_name: Same.
    :param output_prediction_file_name: Same.
    :raises: ValueError: if predictions in `input_prediction_file_name` already
        have calibrated uncertainty.
    """

    print('Reading original predictions from: "{0:s}"...'.format(
        input_prediction_file_name
    ))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)

    if (
            prediction_dict[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
            is not None
    ):
        raise ValueError(
            'Input predictions already have calibrated uncertainty.'
        )

    print('Reading uncertainty-calibration model from: "{0:s}"...'.format(
        model_file_name
    ))
    bin_edge_prediction_stdevs, stdev_inflation_factors = (
        uncertainty_calibration.read_model(model_file_name)
    )
    prediction_dict = uncertainty_calibration.apply_model(
        prediction_dict=prediction_dict,
        bin_edge_prediction_stdevs=bin_edge_prediction_stdevs,
        stdev_inflation_factors=stdev_inflation_factors
    )

    print((
        'Writing predictions with calibrated uncertainty to: "{0:s}"...'
    ).format(output_prediction_file_name))

    d = prediction_dict

    prediction_io.write_file(
        netcdf_file_name=output_prediction_file_name,
        forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
        target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
        cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
        init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
        storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
        storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
        storm_intensity_changes_m_s01=
        d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
        model_file_name=d[prediction_io.MODEL_FILE_KEY],
        lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
        quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
        uncertainty_calib_model_file_name=model_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_PREDICTION_FILE_ARG_NAME
        ),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        output_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_PREDICTION_FILE_ARG_NAME
        )
    )
