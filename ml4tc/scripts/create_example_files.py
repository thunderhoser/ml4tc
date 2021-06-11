"""Creates example files by merging satellite and SHIPS files."""

import os
import argparse
from ml4tc.io import satellite_io
from ml4tc.io import ships_io
from ml4tc.io import example_io
from ml4tc.utils import example_utils
from ml4tc.utils import satellite_utils
from ml4tc.utils import general_utils

SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
SHIPS_DIR_ARG_NAME = 'input_ships_dir_name'
YEAR_ARG_NAME = 'year'
TIME_INTERVAL_ARG_NAME = 'satellite_time_interval_sec'
COMPRESS_ARG_NAME = 'compress_output_files'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SATELLITE_DIR_HELP_STRING = (
    'Name of top-level directory with satellite data.  Files therein will be '
    'found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
SHIPS_DIR_HELP_STRING = (
    'Name of top-level directory with SHIPS data.  Files therein will be found '
    'by `ships_io.find_file` and read by `ships_io.read_file`.'
)
YEAR_HELP_STRING = 'Example files will be created only for this year.'
TIME_INTERVAL_HELP_STRING = 'Time interval for satellite data (seconds).'
COMPRESS_HELP_STRING = 'Boolean flag.  If 1 (0), will (not) gzip output files.'
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for learning examples.  Files will be written '
    'here by `example_io.write_file`, with exact names determined by '
    '`example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHIPS_DIR_ARG_NAME, type=str, required=True,
    help=SHIPS_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TIME_INTERVAL_ARG_NAME, type=int, required=True,
    help=TIME_INTERVAL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPRESS_ARG_NAME, type=int, required=True,
    help=COMPRESS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_satellite_dir_name, top_ships_dir_name, year,
         satellite_time_interval_sec, compress_output_files,
         top_output_dir_name):
    """Creates example files by merging satellite and SHIPS files.

    This is effectively the main method.

    :param top_satellite_dir_name: See documentation at top of file.
    :param top_ships_dir_name: Same.
    :param year: Same.
    :param satellite_time_interval_sec: Same.
    :param compress_output_files: Same.
    :param top_output_dir_name: Same.
    """

    satellite_cyclone_id_strings = satellite_io.find_cyclones(
        directory_name=top_satellite_dir_name, raise_error_if_all_missing=True
    )
    satellite_cyclone_id_strings = set([
        s for s in satellite_cyclone_id_strings
        if satellite_utils.parse_cyclone_id(s)[0] == year
    ])

    ships_cyclone_id_strings = ships_io.find_cyclones(
        directory_name=top_ships_dir_name, raise_error_if_all_missing=True
    )
    ships_cyclone_id_strings = set([
        s for s in ships_cyclone_id_strings
        if satellite_utils.parse_cyclone_id(s)[0] == year
    ])

    cyclone_id_strings = list(
        satellite_cyclone_id_strings.intersection(ships_cyclone_id_strings)
    )
    cyclone_id_strings.sort()

    for this_cyclone_id_string in cyclone_id_strings:
        this_satellite_file_name = satellite_io.find_file(
            directory_name=top_satellite_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        this_ships_file_name = ships_io.find_file(
            directory_name=top_ships_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        this_example_file_name = example_io.find_file(
            directory_name=top_output_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        print('Reading data from: "{0:s}"...'.format(this_satellite_file_name))
        this_satellite_table_xarray = satellite_io.read_file(
            this_satellite_file_name
        )

        print('Reading data from: "{0:s}"...'.format(this_ships_file_name))
        this_ships_table_xarray = ships_io.read_file(this_ships_file_name)

        this_example_table_xarray = example_utils.merge_data(
            satellite_table_xarray=this_satellite_table_xarray,
            ships_table_xarray=this_ships_table_xarray
        )
        this_example_table_xarray = example_utils.subset_satellite_times(
            example_table_xarray=this_example_table_xarray,
            time_interval_sec=satellite_time_interval_sec
        )

        print('Writing data to: "{0:s}"...\n'.format(this_example_file_name))
        example_io.write_file(
            example_table_xarray=this_example_table_xarray,
            netcdf_file_name=this_example_file_name
        )

        if compress_output_files:
            general_utils.compress_file(this_example_file_name)
            os.remove(this_example_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME
        ),
        top_ships_dir_name=getattr(INPUT_ARG_OBJECT, SHIPS_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME),
        satellite_time_interval_sec=getattr(
            INPUT_ARG_OBJECT, TIME_INTERVAL_ARG_NAME
        ),
        compress_output_files=bool(
            getattr(INPUT_ARG_OBJECT, COMPRESS_ARG_NAME)
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
