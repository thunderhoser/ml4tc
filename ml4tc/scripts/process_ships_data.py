"""Processes SHIPS data (converts from raw format to my format)."""

import os
import argparse
import numpy
from ml4tc.io import ships_io
from ml4tc.io import raw_ships_io
from ml4tc.utils import general_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_file_name'
SEVEN_DAY_ARG_NAME = 'seven_day'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `raw_ships_io.read_file`.'
)
SEVEN_DAY_HELP_STRING = (
    'Boolean flag.  If 1 (0), will assume 7-day (5-day) files.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by '
    '`ships_io.write_file`, to exact locations determined by '
    '`ships_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SEVEN_DAY_ARG_NAME, type=int, required=True,
    help=SEVEN_DAY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, seven_day, output_dir_name):
    """Processes SHIPS data (converts from raw format to my format).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param seven_day: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    ships_table_xarray = raw_ships_io.read_file(
        ascii_file_name=input_file_name, seven_day=seven_day
    )
    print(SEPARATOR_STRING)

    cyclone_id_strings = ships_table_xarray[ships_io.CYCLONE_ID_KEY].values
    unique_cyclone_id_strings, orig_to_unique_indices = numpy.unique(
        numpy.array(cyclone_id_strings), return_inverse=True
    )
    num_cyclones = len(unique_cyclone_id_strings)

    for i in range(num_cyclones):
        this_index_dict = {
            ships_io.STORM_OBJECT_DIM:
                numpy.where(orig_to_unique_indices == i)[0]
        }
        this_ships_table_xarray = ships_table_xarray.isel(
            indexers=this_index_dict, drop=False
        )

        this_output_file_name = ships_io.find_file(
            directory_name=output_dir_name,
            cyclone_id_string=unique_cyclone_id_strings[i],
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(this_output_file_name))
        ships_io.write_file(
            ships_table_xarray=this_ships_table_xarray,
            netcdf_file_name=this_output_file_name
        )

        general_utils.compress_file(this_output_file_name)
        os.remove(this_output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        seven_day=bool(getattr(INPUT_ARG_OBJECT, SEVEN_DAY_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
