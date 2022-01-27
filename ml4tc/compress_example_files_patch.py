"""Compresses example files with gzip."""

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
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'

EXAMPLE_DIR_HELP_STRING = (
    'Name of working directory.  Uncompressed files therein will be found by '
    '`example_io.find_file` and then compressed.'
)
CYCLONE_ID_HELP_STRING = (
    'Will handle only learning examples for this tropical cyclone.'
)

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
    """Compresses example files with gzip.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of file.
    :param cyclone_id_string: Same.
    """

    cyclone_id_strings = [cyclone_id_string]

    for this_cyclone_id_string in cyclone_id_strings:
        this_example_file_name = example_io.find_file(
            directory_name=example_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        if not os.path.isfile(this_example_file_name):
            continue

        print('Compressing file: "{0:s}"...'.format(this_example_file_name))
        general_utils.compress_file(this_example_file_name)
        os.remove(this_example_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME)
    )
