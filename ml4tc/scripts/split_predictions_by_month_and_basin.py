"""Splits predictions by month, then by ocean basin, separately."""

import copy
import argparse
from ml4tc.io import prediction_io
from ml4tc.utils import satellite_utils

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions for all months and basins.  '
    'Will be read by `prediction_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Predictions for each month, then each basin, '
    'will be written here by `prediction_io.write_file`, to exact locations '
    'determined by `prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, output_dir_name):
    """Splits predictions by month, then by ocean basin, separately.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...\n'.format(input_file_name))
    prediction_dict = prediction_io.read_file(input_file_name)

    # Split by month.
    for k in range(1, 13):
        this_prediction_dict = prediction_io.subset_by_month(
            prediction_dict=copy.deepcopy(prediction_dict), desired_month=k
        )
        d = this_prediction_dict

        if len(d[prediction_io.INIT_TIMES_KEY]) == 0:
            continue

        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name, month=k,
            raise_error_if_missing=False
        )
        print('Writing {0:d} examples to: "{1:s}"...'.format(
            len(d[prediction_io.INIT_TIMES_KEY]),
            this_output_file_name
        ))

        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
            target_classes=d[prediction_io.TARGET_CLASSES_KEY],
            cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
            init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
            storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
            storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
            model_file_name=d[prediction_io.MODEL_FILE_KEY]
        )

    print('\n')

    for this_basin_id_string in satellite_utils.VALID_BASIN_ID_STRINGS:
        this_prediction_dict = prediction_io.subset_by_basin(
            prediction_dict=copy.deepcopy(prediction_dict),
            desired_basin_id_string=this_basin_id_string
        )
        d = this_prediction_dict

        if len(d[prediction_io.INIT_TIMES_KEY]) == 0:
            continue

        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name,
            basin_id_string=this_basin_id_string,
            raise_error_if_missing=False
        )
        print('Writing {0:d} examples to: "{1:s}"...'.format(
            len(d[prediction_io.INIT_TIMES_KEY]),
            this_output_file_name
        ))

        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
            target_classes=d[prediction_io.TARGET_CLASSES_KEY],
            cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
            init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
            storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
            storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
            model_file_name=d[prediction_io.MODEL_FILE_KEY]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
