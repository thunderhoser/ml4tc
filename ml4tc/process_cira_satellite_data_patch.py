"""Processes CIRA satellite data (converts from raw format to my format)."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import cira_satellite_io
import satellite_io
import satellite_utils
import general_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_dir_name'
YEAR_ARG_NAME = 'year'
CYCLONE_INDEX_ARG_NAME = 'cyclone_index'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with raw files.  Files therein will be found '
    'by `cira_satellite_io.find_file` and read by '
    '`cira_satellite_io.read_file`.'
)
YEAR_HELP_STRING = (
    'Will process all data (all cyclones in all basins) for this year.'
)
CYCLONE_INDEX_HELP_STRING = 'Will process data for this cyclone only.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by '
    '`satellite_io.write_file`, to exact locations determined by '
    '`satellite_io.find_file`.'
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
    '--' + CYCLONE_INDEX_ARG_NAME, type=int, required=True,
    help=CYCLONE_INDEX_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_input_dir_name, year, cyclone_index, output_dir_name):
    """Processes CIRA satellite data (converts from raw format to my format).

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param year: Same.
    :param cyclone_index: Same.
    :param output_dir_name: Same.
    """

    cyclone_id_strings = cira_satellite_io.find_cyclones_one_year(
        top_directory_name=top_input_dir_name, year=year,
        raise_error_if_all_missing=True
    )
    cyclone_id_strings = [cyclone_id_strings[cyclone_index]]

    for this_id_string in cyclone_id_strings:
        input_file_names = cira_satellite_io.find_files_one_cyclone(
            top_directory_name=top_input_dir_name,
            cyclone_id_string=this_id_string,
            raise_error_if_all_missing=True
        )

        satellite_tables_xarray = []

        for this_input_file_name in input_file_names:
            print('Reading data from: "{0:s}"...'.format(this_input_file_name))
            this_table_xarray = cira_satellite_io.read_file(
                netcdf_file_name=this_input_file_name, raise_error_if_fail=False
            )

            if this_table_xarray is None:
                continue

            satellite_tables_xarray.append(this_table_xarray)

        satellite_table_xarray = satellite_utils.concat_tables_over_time(
            satellite_tables_xarray
        )
        output_file_name = satellite_io.find_file(
            directory_name=output_dir_name, cyclone_id_string=this_id_string,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        satellite_io.write_file(
            satellite_table_xarray=satellite_table_xarray,
            netcdf_file_name=output_file_name
        )

        general_utils.compress_file(output_file_name)
        os.remove(output_file_name)

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME),
        cyclone_index=getattr(INPUT_ARG_OBJECT, CYCLONE_INDEX_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
