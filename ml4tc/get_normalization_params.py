"""Computes normalization parameters."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import example_io
import satellite_utils
import normalization

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
YEARS_ARG_NAME = 'years'
NUM_FILES_ARG_NAME = 'num_example_files'
NUM_VALUES_PER_GRIDDED_ARG_NAME = 'num_values_per_gridded'
NUM_VALUES_PER_UNGRIDDED_ARG_NAME = 'num_values_per_ungridded'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
YEARS_HELP_STRING = 'Will use learning examples for these years (list).'
NUM_FILES_HELP_STRING = (
    'Will randomly select this many example files from the years in `{0:s}`.'
).format(YEARS_ARG_NAME)

NUM_VALUES_PER_GRIDDED_HELP_STRING = (
    'Number of reference values to save for each gridded predictor.'
)
NUM_VALUES_PER_UNGRIDDED_HELP_STRING = (
    'Number of reference values to save for each ungridded predictor.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Normalization parameters (i.e., reference values for'
    ' uniformization) will be saved here by `normalization.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEARS_ARG_NAME, type=int, nargs='+', required=True,
    help=YEARS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_FILES_ARG_NAME, type=int, required=True,
    help=NUM_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALUES_PER_GRIDDED_ARG_NAME, type=int, required=True,
    help=NUM_VALUES_PER_GRIDDED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALUES_PER_UNGRIDDED_ARG_NAME, type=int, required=True,
    help=NUM_VALUES_PER_UNGRIDDED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(example_dir_name, years, num_example_files, num_values_per_gridded,
         num_values_per_ungridded, output_file_name):
    """Computes normalization parameters.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of file.
    :param years: Same.
    :param num_example_files: Same.
    :param num_values_per_gridded: Same.
    :param num_values_per_ungridded: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(num_example_files, 10)

    cyclone_id_strings = example_io.find_cyclones(
        directory_name=example_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [satellite_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )

    good_flags = numpy.array([c in years for c in cyclone_years], dtype=float)
    good_indices = numpy.where(good_flags)[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    cyclone_id_strings.sort()

    example_file_names = [
        example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        for c in cyclone_id_strings
    ]

    if len(example_file_names) > num_example_files:
        file_indices = numpy.linspace(
            0, len(example_file_names) - 1, num=len(example_file_names),
            dtype=int
        )
        file_indices = numpy.random.choice(
            file_indices, size=num_example_files, replace=False
        )

        example_file_names = [example_file_names[k] for k in file_indices]

    normalization_table_xarray = normalization.get_normalization_params(
        example_file_names=example_file_names,
        num_values_per_ungridded=num_values_per_ungridded,
        num_values_per_gridded=num_values_per_gridded
    )
    print(SEPARATOR_STRING)

    print('Writing normalization params to file: "{0:s}"...'.format(
        output_file_name
    ))
    normalization.write_file(
        normalization_table_xarray=normalization_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int),
        num_example_files=getattr(INPUT_ARG_OBJECT, NUM_FILES_ARG_NAME),
        num_values_per_gridded=getattr(
            INPUT_ARG_OBJECT, NUM_VALUES_PER_GRIDDED_ARG_NAME
        ),
        num_values_per_ungridded=getattr(
            INPUT_ARG_OBJECT, NUM_VALUES_PER_UNGRIDDED_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
