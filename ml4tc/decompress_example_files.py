"""Decompresses gzipped example files."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_io
import general_utils
import satellite_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

EXAMPLE_DIR_ARG_NAME = 'example_dir_name'
YEAR_ARG_NAME = 'year'

EXAMPLE_DIR_HELP_STRING = (
    'Name of working directory.  Compressed files therein will be found by '
    '`example_io.find_file` and then decompressed.'
)
YEAR_HELP_STRING = 'Will decompress example files for this year.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)


def _run(example_dir_name, year):
    """Decompresses gzipped example files.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of file.
    :param year: Same.
    """

    cyclone_id_strings = example_io.find_cyclones(
        directory_name=example_dir_name, raise_error_if_all_missing=True
    )
    cyclone_id_strings = set([
        c for c in cyclone_id_strings
        if satellite_utils.parse_cyclone_id(c)[0] == year
    ])
    cyclone_id_strings = list(cyclone_id_strings)
    cyclone_id_strings.sort()

    for this_cyclone_id_string in cyclone_id_strings:
        this_example_file_name = example_io.find_file(
            directory_name=example_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=True, allow_other_format=False,
            raise_error_if_missing=False
        )

        if not os.path.isfile(this_example_file_name):
            continue

        print('Decompressing file: "{0:s}"...'.format(this_example_file_name))
        general_utils.decompress_file(this_example_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME)
    )
