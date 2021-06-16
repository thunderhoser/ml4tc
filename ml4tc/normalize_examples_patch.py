"""Normalizes learning examples."""

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
import normalization

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
YEAR_ARG_NAME = 'year'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
COMPRESS_ARG_NAME = 'compress_output_files'
CYCLONE_INDEX_ARG_NAME = 'cyclone_index'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing unnormalized examples.  Files therein '
    'will be found by `example_io.find_file` and read by '
    '`example_io.read_file`.'
)
YEAR_HELP_STRING = 'Will normalize learning examples in this year.'
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be read by '
    '`normalization.read_file`).'
)
COMPRESS_HELP_STRING = 'Boolean flag.  If 1 (0), will (not) gzip output files.'
CYCLONE_INDEX_HELP_STRING = 'Will process data for this cyclone only.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Normalized examples will be written here by '
    '`example_io.write_file`, to exact locations determined by '
    '`example_io.find_file`.'
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
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPRESS_ARG_NAME, type=int, required=True,
    help=COMPRESS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_INDEX_ARG_NAME, type=int, required=True,
    help=CYCLONE_INDEX_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_example_dir_name, year, normalization_file_name,
         compress_output_files, cyclone_index, output_example_dir_name):
    """Normalizes learning examples.

    This is effectively the main method.

    :param input_example_dir_name: See documentation at top of file.
    :param year: Same.
    :param normalization_file_name: Same.
    :param compress_output_files: Same.
    :param cyclone_index: Same.
    :param output_example_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    cyclone_id_strings = example_io.find_cyclones(
        directory_name=input_example_dir_name, raise_error_if_all_missing=True
    )
    cyclone_id_strings = set([
        c for c in cyclone_id_strings
        if satellite_utils.parse_cyclone_id(c)[0] == year
    ])
    cyclone_id_strings = list(cyclone_id_strings)
    cyclone_id_strings.sort()
    cyclone_id_strings = [cyclone_id_strings[cyclone_index]]

    for this_cyclone_id_string in cyclone_id_strings:
        input_example_file_name = example_io.find_file(
            directory_name=input_example_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        output_example_file_name = example_io.find_file(
            directory_name=output_example_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        print('Reading unnormalized examples from: "{0:s}"...'.format(
            input_example_file_name
        ))
        example_table_xarray = example_io.read_file(input_example_file_name)

        example_table_xarray = normalization.normalize_data(
            example_table_xarray=example_table_xarray,
            normalization_table_xarray=normalization_table_xarray
        )

        print('Writing normalized examples to: "{0:s}"...'.format(
            output_example_file_name
        ))
        example_io.write_file(
            example_table_xarray=example_table_xarray,
            netcdf_file_name=output_example_file_name
        )

        if compress_output_files:
            general_utils.compress_file(output_example_file_name)
            os.remove(output_example_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        compress_output_files=bool(
            getattr(INPUT_ARG_OBJECT, COMPRESS_ARG_NAME)
        ),
        cyclone_index=getattr(INPUT_ARG_OBJECT, CYCLONE_INDEX_ARG_NAME),
        output_example_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
