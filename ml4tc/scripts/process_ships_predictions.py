"""Processes SHIPS predictions into decent file type."""

import glob
import argparse
import numpy
from ml4tc.io import raw_ships_prediction_io
from ml4tc.io import ships_prediction_io

INPUT_DIR_ARG_NAME = 'top_input_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory.  Raw files will be found in annual '
    'subdirectories and read by `raw_ships_prediction_io.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  SHIPS predictions will be written here by '
    '`ships_prediction_io.write_file`.'
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
    """Processes SHIPS predictions into decent file type.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param output_file_name: Same.
    """

    input_file_pattern = '{0:s}/20*/*_ships.txt'.format(top_input_dir_name)
    input_file_names = glob.glob(input_file_pattern)
    input_file_names.sort()

    num_examples = len(input_file_names)
    ri_probability_matrix = numpy.full((num_examples, 2), numpy.nan)
    cyclone_id_strings = [''] * num_examples
    init_times_unix_sec = numpy.full(num_examples, -1, dtype=int)

    for i in range(num_examples):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        these_probs, cyclone_id_strings[i], init_times_unix_sec[i] = (
            raw_ships_prediction_io.read_file(input_file_names[i])
        )

        ri_probability_matrix[i, :len(these_probs)] = these_probs

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    ships_prediction_io.write_file(
        netcdf_file_name=output_file_name,
        ri_probability_matrix=ri_probability_matrix,
        cyclone_id_strings=cyclone_id_strings,
        init_times_unix_sec=init_times_unix_sec
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
