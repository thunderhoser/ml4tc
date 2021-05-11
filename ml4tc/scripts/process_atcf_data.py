"""Processes ATCF data (converts from raw format to my format)."""

import argparse
from ml4tc.io import raw_atcf_io
from ml4tc.io import atcf_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_dir_name'
YEAR_ARG_NAME = 'year'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with raw files.  Files therein will be found by '
    '`raw_atcf_io.find_file` and read by `raw_atcf_io.read_file`.'
)
YEAR_HELP_STRING = (
    'Will process all data (all cyclones in all basins) for this year.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by '
    '`atcf_io.write_file`, to exact locations determined by '
    '`atcf_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, year, output_dir_name):
    """Processes ATCF data (converts from raw format to my format).

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param year: Same.
    :param output_dir_name: Same.
    """

    cyclone_id_strings = raw_atcf_io.find_cyclones_one_year(
        directory_name=input_dir_name, year=year,
        raise_error_if_all_missing=True
    )

    num_cyclones = len(cyclone_id_strings)
    atcf_tables_xarray = [None] * num_cyclones

    for i in range(len(cyclone_id_strings)):
        this_file_name = raw_atcf_io.find_file(
            directory_name=input_dir_name,
            cyclone_id_string=cyclone_id_strings[i],
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        atcf_tables_xarray[i] = raw_atcf_io.read_file(this_file_name)

    atcf_table_xarray = atcf_io.concat_tables_over_storm_object(
        atcf_tables_xarray
    )
    output_file_name = atcf_io.find_file(
        directory_name=output_dir_name, year=year, raise_error_if_missing=False
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    atcf_io.write_file(
        atcf_table_xarray=atcf_table_xarray, netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
