"""Processes SHIPS TD-to-TS predictions into decent file type."""

import glob
import argparse
import numpy
from ml4tc.io import raw_ships_prediction_io
from ml4tc.io import ships_prediction_io

LEAD_TIMES_HOURS = numpy.array(
    [0, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168],
    dtype=int
)

INPUT_DIR_ARG_NAME = 'top_input_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory.  Raw files will be found in annual '
    'subdirectories and read by '
    '`raw_ships_prediction_io.read_td_to_ts_predictions`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  SHIPS predictions will be written here by '
    '`ships_prediction_io.write_td_to_ts_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_input_dir_name, output_file_name):
    """Processes SHIPS TD-to-TS predictions into decent file type.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param output_file_name: Same.
    """

    input_file_names = glob.glob(
        '{0:s}/19*/*_ships.txt'.format(top_input_dir_name)
    )
    input_file_names += glob.glob(
        '{0:s}/20*/*_ships.txt'.format(top_input_dir_name)
    )
    input_file_names.sort()

    num_examples = len(input_file_names)
    num_lead_times = len(LEAD_TIMES_HOURS)
    dimensions = (num_examples, num_lead_times)

    forecast_label_matrix_land = numpy.full(dimensions, -1, dtype=int)
    forecast_label_matrix_lge = numpy.full(dimensions, -1, dtype=int)
    cyclone_id_strings = [''] * num_examples
    init_times_unix_sec = numpy.full(num_examples, -1, dtype=int)

    for i in range(num_examples):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        this_prediction_dict = (
            raw_ships_prediction_io.read_td_to_ts_predictions(
                input_file_names[i]
            )
        )

        these_lead_times_hours = (
            this_prediction_dict[raw_ships_prediction_io.LEAD_TIMES_KEY]
        )
        assert numpy.array_equal(
            these_lead_times_hours,
            LEAD_TIMES_HOURS[:len(these_lead_times_hours)]
        )

        forecast_label_matrix_land[i, :len(these_lead_times_hours)] = (
            this_prediction_dict[
                raw_ships_prediction_io.FORECAST_LABELS_LAND_KEY
            ]
        )
        forecast_label_matrix_lge[i, :len(these_lead_times_hours)] = (
            this_prediction_dict[
                raw_ships_prediction_io.FORECAST_LABELS_LGE_KEY
            ]
        )
        cyclone_id_strings[i] = (
            this_prediction_dict[raw_ships_prediction_io.CYCLONE_ID_KEY]
        )
        init_times_unix_sec[i] = (
            this_prediction_dict[raw_ships_prediction_io.INIT_TIME_KEY]
        )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    ships_prediction_io.write_td_to_ts_file(
        netcdf_file_name=output_file_name,
        lead_times_hours=LEAD_TIMES_HOURS,
        forecast_label_matrix_land=forecast_label_matrix_land,
        forecast_label_matrix_lge=forecast_label_matrix_lge,
        cyclone_id_strings=cyclone_id_strings,
        init_times_unix_sec=init_times_unix_sec
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
