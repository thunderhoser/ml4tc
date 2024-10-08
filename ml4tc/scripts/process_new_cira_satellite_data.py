"""Processes new CIRA IR data (converts from raw format to my format)."""

import os
import argparse
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import new_cira_satellite_io
from ml4tc.io import satellite_io
from ml4tc.utils import satellite_utils
from ml4tc.utils import general_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_dir_name'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
NUM_CROPPED_ROWS_ARG_NAME = 'num_cropped_rows'
NUM_CROPPED_COLUMNS_ARG_NAME = 'num_cropped_columns'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with raw files.  Files therein will be found '
    'by `cira_satellite_io.find_file` and read by '
    '`cira_satellite_io.read_file`.'
)
CYCLONE_IDS_HELP_STRING = (
    'List of cyclone IDs.  Will process data for these cyclones.'
)
NUM_CROPPED_ROWS_HELP_STRING = (
    'Number of rows in brightness-temperature images, after centering each '
    'image at storm center.  If you do not want to recenter images, leave this '
    'argument alone.'
)
NUM_CROPPED_COLUMNS_HELP_STRING = 'Same as `{0:s}` but for columns.'.format(
    NUM_CROPPED_ROWS_ARG_NAME
)
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
    '--' + CYCLONE_IDS_ARG_NAME, type=str, nargs='+', required=True,
    help=CYCLONE_IDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CROPPED_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_CROPPED_ROWS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CROPPED_COLUMNS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_CROPPED_COLUMNS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_input_dir_name, cyclone_id_strings, num_cropped_rows,
         num_cropped_columns, output_dir_name):
    """Processes new CIRA IR data (converts from raw format to my format).

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param cyclone_id_strings: Same.
    :param num_cropped_rows: Same.
    :param num_cropped_columns: Same.
    :param output_dir_name: Same.
    """

    if num_cropped_rows <= 0 or num_cropped_columns <= 0:
        num_cropped_rows = None
        num_cropped_columns = None

    if num_cropped_rows is not None:
        error_checking.assert_is_less_than(
            num_cropped_rows, satellite_utils.DEFAULT_NUM_GRID_ROWS
        )
        error_checking.assert_is_less_than(
            num_cropped_columns, satellite_utils.DEFAULT_NUM_GRID_COLUMNS
        )

    cyclone_id_strings = list(set(cyclone_id_strings))

    for this_id_string in cyclone_id_strings:
        input_file_names = new_cira_satellite_io.find_files_one_cyclone(
            top_directory_name=top_input_dir_name,
            cyclone_id_string=this_id_string,
            raise_error_if_all_missing=True
        )

        satellite_tables_xarray = []

        for this_input_file_name in input_file_names:
            print('Reading data from: "{0:s}"...'.format(this_input_file_name))
            this_table_xarray = new_cira_satellite_io.read_file(
                netcdf_file_name=this_input_file_name, raise_error_if_fail=False
            )

            if this_table_xarray is None:
                continue

            if num_cropped_rows is not None:
                this_table_xarray = (
                    satellite_utils.crop_images_around_storm_centers(
                        satellite_table_xarray=this_table_xarray,
                        num_cropped_rows=num_cropped_rows,
                        num_cropped_columns=num_cropped_columns
                    )
                )

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
        cyclone_id_strings=getattr(INPUT_ARG_OBJECT, CYCLONE_IDS_ARG_NAME),
        num_cropped_rows=getattr(INPUT_ARG_OBJECT, NUM_CROPPED_ROWS_ARG_NAME),
        num_cropped_columns=getattr(
            INPUT_ARG_OBJECT, NUM_CROPPED_COLUMNS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
