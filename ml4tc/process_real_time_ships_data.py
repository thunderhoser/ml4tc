"""Processes real-time SHIPS data (converts from raw format to my format)."""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import ships_io
import raw_ships_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
HOURS_TO_SECONDS = 3600

INPUT_DIR_ARG_NAME = 'input_dir_name'
YEAR_ARG_NAME = 'year'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with raw real-time SHIPS data.  Files therein '
    'will be found by `raw_ships_io.find_real_time_file` and read by '
    '`raw_ships_io.read_file`.'
)
YEAR_HELP_STRING = 'Will convert data for this year.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by '
    '`ships_io.write_file`, to exact locations determined by '
    '`ships_io.find_file`.'
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


def _process_data_one_cyclone(input_dir_name, cyclone_id_string,
                              output_dir_name):
    """Processes real-time SHIPS data for one cyclone.

    :param input_dir_name: See documentation at top of file.
    :param cyclone_id_string: Cyclone ID.
    :param output_dir_name: See documentation at top of file.
    """

    input_file_names = raw_ships_io.find_real_time_files_1cyclone(
        top_directory_name=input_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )

    num_files = len(input_file_names)
    ships_tables_xarray = [xarray.Dataset()] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        ships_tables_xarray[i] = raw_ships_io.read_file(
            ascii_file_name=input_file_names[i], real_time_flag=True,
            seven_day_flag=False
        )

    ships_table_xarray = xarray.concat(
        ships_tables_xarray, dim=ships_io.STORM_OBJECT_DIM, data_vars='all',
        coords='minimal', compat='identical', join='exact'
    )

    sort_indices = numpy.argsort(
        ships_table_xarray[ships_io.VALID_TIME_KEY].values
    )
    ships_table_xarray = ships_table_xarray.isel(
        indexers={ships_io.STORM_OBJECT_DIM: sort_indices},
        drop=False
    )

    # TODO(thunderhoser): Note that I compute the intensity change only at
    # analysis time -- not at forecast times -- because I do not use true
    # forecast intensity changes (at lead time > 0) in ML anyways!

    # Add 6-hour intensity changes to table.
    intensity_changes_m_s01 = numpy.concatenate((
        numpy.array([numpy.nan]),
        numpy.diff(ships_table_xarray[ships_io.STORM_INTENSITY_KEY].values)
    ))
    time_diffs_sec = numpy.concatenate((
        numpy.array([0], dtype=int),
        numpy.diff(ships_table_xarray[ships_io.VALID_TIME_KEY].values)
    ))

    storm_object_indices = numpy.where(
        time_diffs_sec == 6 * HOURS_TO_SECONDS
    )[0]
    fcst_hour_index = numpy.where(
        ships_table_xarray.coords[ships_io.FORECAST_HOUR_DIM].values == 0
    )[0][0]
    ships_table_xarray[ships_io.INTENSITY_CHANGE_6HOURS_KEY].values[
        storm_object_indices, fcst_hour_index
    ] = intensity_changes_m_s01[storm_object_indices]

    # Add intensity change since (init time minus 12 hours) to table.
    intensity_changes_m_s01 = numpy.concatenate((
        numpy.array([numpy.nan, numpy.nan]),
        ships_table_xarray[ships_io.STORM_INTENSITY_KEY].values[2:] -
        ships_table_xarray[ships_io.STORM_INTENSITY_KEY].values[:-2]
    ))
    time_diffs_sec = numpy.concatenate((
        numpy.array([0, 0], dtype=int),
        ships_table_xarray[ships_io.VALID_TIME_KEY].values[2:] -
        ships_table_xarray[ships_io.VALID_TIME_KEY].values[:-2]
    ))

    storm_object_indices = numpy.where(
        time_diffs_sec == 12 * HOURS_TO_SECONDS
    )[0]
    ships_table_xarray[ships_io.INTENSITY_CHANGE_M12HOURS_KEY].values[
        storm_object_indices, fcst_hour_index
    ] = intensity_changes_m_s01[storm_object_indices]

    output_file_name = ships_io.find_file(
        directory_name=output_dir_name,
        cyclone_id_string=cyclone_id_string,
        prefer_zipped=False, allow_other_format=False,
        raise_error_if_missing=False
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    ships_io.write_file(
        ships_table_xarray=ships_table_xarray,
        netcdf_file_name=output_file_name
    )


def _run(input_dir_name, year, output_dir_name):
    """Processes real-time SHIPS data (converts from raw format to my format).

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param year: Same.
    :param output_dir_name: Same.
    """

    cyclone_id_strings = raw_ships_io.find_real_time_cyclones_1year(
        top_directory_name=input_dir_name, year=year,
        raise_error_if_all_missing=True
    )

    for this_cyclone_id_string in cyclone_id_strings:
        _process_data_one_cyclone(
            input_dir_name=input_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            output_dir_name=output_dir_name
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
