"""Processes SHIPS data (converts from raw format to my format)."""

import argparse
from ml4tc.io import ships_io
from ml4tc.io import raw_ships_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILES_ARG_NAME = 'input_file_names'
SEVEN_DAY_ARG_NAME = 'seven_day'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each will be read by '
    '`raw_ships_io.read_file`, and the resulting tables will be concatenated.'
)
SEVEN_DAY_HELP_STRING = (
    'Boolean flag.  If 1 (0), will assume 7-day (5-day) files.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  This will be a NetCDF file written by '
    '`ships_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SEVEN_DAY_ARG_NAME, type=int, required=True,
    help=SEVEN_DAY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_names, seven_day, output_file_name):
    """Processes SHIPS data (converts from raw format to my format).

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param seven_day: Same.
    :param output_file_name: Same.
    """

    ships_tables_xarray = []

    for this_file_name in input_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        ships_tables_xarray.append(
            raw_ships_io.read_file(
                ascii_file_name=this_file_name, seven_day=seven_day
            )
        )

        print(SEPARATOR_STRING)

    ships_table_xarray = ships_io.concat_tables_over_storm_object(
        ships_tables_xarray
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    ships_io.write_file(
        ships_table_xarray=ships_table_xarray, netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        seven_day=bool(getattr(INPUT_ARG_OBJECT, SEVEN_DAY_ARG_NAME)),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
