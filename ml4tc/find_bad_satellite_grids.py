"""Finds bad satellite grids.

USE ONCE AND DESTROY.
"""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_io
import example_utils
import satellite_utils

EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'

EXAMPLE_DIR_HELP_STRING = (
    'Name of input directory, containing learning examples.  The relevant file '
    '(for the given cyclone) will be found by `example_io.find_file` and read '
    'by `example_io.read_file`.'
)
CYCLONE_ID_HELP_STRING = 'Will check satellite grids for this cyclone only.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)


def _run(example_dir_name, cyclone_id_string):
    """Finds bad satellite grids.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of file.
    :param cyclone_id_string: Same.
    """

    # Read data.
    example_file_name = example_io.find_file(
        directory_name=example_dir_name, cyclone_id_string=cyclone_id_string,
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(example_file_name))
    example_table_xarray = example_io.read_file(example_file_name)
    xt = example_table_xarray

    num_times = xt[satellite_utils.GRID_LATITUDE_KEY].values.shape[0]

    for i in range(num_times):
        this_flag = satellite_utils.is_regular_grid_valid(
            latitudes_deg_n=
            xt[satellite_utils.GRID_LATITUDE_KEY].values[i, ...],
            longitudes_deg_e=
            xt[satellite_utils.GRID_LONGITUDE_KEY].values[i, ...]
        )

        if this_flag:
            continue

        print('FOUND INVALID GRID')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME)
    )
