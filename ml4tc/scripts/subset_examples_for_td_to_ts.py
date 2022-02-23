"""Subsets examples for TD-to-TS prediction.

'TD-to-TS' means intensification from tropical depression to tropical storm.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import example_io
from ml4tc.utils import example_utils
from ml4tc.utils import satellite_utils
from ml4tc.machine_learning import neural_net

LARGE_INTEGER = int(1e10)
DUMMY_LEAD_TIME_SEC = 666 * 3600

HOURS_TO_SECONDS = 3600
MINUTES_TO_SECONDS = 60

SATELLITE_KEYS_TO_REPLACE = [
    example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY,
    example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY,
    example_utils.GRID_LATITUDE_KEY, example_utils.GRID_LONGITUDE_KEY
]
SHIPS_KEYS_TO_REPLACE = [
    example_utils.SHIPS_PREDICTORS_FORECAST_KEY,
    example_utils.SHIPS_PREDICTORS_LAGGED_KEY
]

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
YEAR_ARG_NAME = 'year'
SHIPS_LAG_TIMES_ARG_NAME = 'ships_lag_times_hours'
SATELLITE_LAG_TIMES_ARG_NAME = 'satellite_lag_times_minutes'
SHIPS_TIME_TOLERANCE_ARG_NAME = 'ships_time_tolerance_sec'
SATELLITE_TIME_TOLERANCE_ARG_NAME = 'satellite_time_tolerance_sec'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing examples before subsetting.  Files '
    'therein will be found by `example_io.find_file` and read by '
    '`example_io.read_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Will subset examples for this cyclone.  If you want to subset examples '
    'for a whole year instead, leave this argument alone.'
)
YEAR_HELP_STRING = (
    'Will subset examples for this year.  If you want to subset examples for a '
    'specific cyclone instead, leave this argument alone.'
)
SHIPS_LAG_TIMES_HELP_STRING = 'List of model lag times for SHIPS data.'
SATELLITE_LAG_TIMES_HELP_STRING = 'List of model lag times for satellite data.'
SHIPS_TIME_TOLERANCE_HELP_STRING = 'Time tolerance for SHIPS data.'
SATELLITE_TIME_TOLERANCE_HELP_STRING = 'Time tolerance for satellite data.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory, for examples after subsetting.  Files will be '
    'written by `example_io.write_file`, to exact locations determined by '
    '`example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=False, default='',
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEAR_ARG_NAME, type=int, required=False, default=-1,
    help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHIPS_LAG_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=SHIPS_LAG_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_LAG_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=SATELLITE_LAG_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHIPS_TIME_TOLERANCE_ARG_NAME, type=int, required=True,
    help=SHIPS_TIME_TOLERANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_TIME_TOLERANCE_ARG_NAME, type=int, required=True,
    help=SATELLITE_TIME_TOLERANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_example_dir_name, cyclone_id_string, year,
         ships_lag_times_hours, satellite_lag_times_minutes,
         ships_time_tolerance_sec, satellite_time_tolerance_sec,
         output_example_dir_name):
    """Subsets examples for TD-to-TS prediction.

    This is effectively the main method.

    :param input_example_dir_name: See documentation at top of file.
    :param cyclone_id_string: Same.
    :param year: Same.
    :param ships_lag_times_hours: Same.
    :param satellite_lag_times_minutes: Same.
    :param ships_time_tolerance_sec: Same.
    :param satellite_time_tolerance_sec: Same.
    :param output_example_dir_name: Same.
    """

    error_checking.assert_is_geq_numpy_array(ships_lag_times_hours, 0)
    error_checking.assert_is_geq_numpy_array(satellite_lag_times_minutes, 0)
    error_checking.assert_is_geq(ships_time_tolerance_sec, 0)
    error_checking.assert_is_geq(satellite_time_tolerance_sec, 0)

    ships_lag_times_sec = ships_lag_times_hours * HOURS_TO_SECONDS
    satellite_lag_times_sec = satellite_lag_times_minutes * MINUTES_TO_SECONDS

    if year > 0:
        cyclone_id_strings = example_io.find_cyclones(
            directory_name=input_example_dir_name,
            raise_error_if_all_missing=True
        )
        cyclone_id_strings = set([
            c for c in cyclone_id_strings
            if satellite_utils.parse_cyclone_id(c)[0] == year
        ])
        cyclone_id_strings = list(cyclone_id_strings)
        cyclone_id_strings.sort()
    else:
        cyclone_id_strings = [cyclone_id_string]

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

        print('Reading data from: "{0:s}"...'.format(input_example_file_name))
        example_table_xarray = example_io.read_file(input_example_file_name)
        xt = example_table_xarray

        data_dict = neural_net._read_non_predictors_one_file(
            example_table_xarray=xt,
            num_examples_desired=LARGE_INTEGER,
            num_positive_examples_desired=LARGE_INTEGER,
            num_negative_examples_desired=LARGE_INTEGER,
            lead_time_sec=DUMMY_LEAD_TIME_SEC,
            satellite_lag_times_sec=satellite_lag_times_sec,
            ships_lag_times_sec=ships_lag_times_sec,
            predict_td_to_ts=True,
            satellite_time_tolerance_sec=satellite_time_tolerance_sec,
            satellite_max_missing_times=LARGE_INTEGER,
            ships_time_tolerance_sec=ships_time_tolerance_sec,
            ships_max_missing_times=LARGE_INTEGER, use_climo_as_backup=True,
            all_init_times_unix_sec=None
        )

        satellite_rows_to_keep = numpy.concatenate([
            r for r in data_dict[neural_net.SATELLITE_ROWS_KEY] if r is not None
        ], axis=0)

        ships_rows_to_keep = numpy.concatenate([
            r for r in data_dict[neural_net.SHIPS_ROWS_KEY] if r is not None
        ], axis=0)

        satellite_rows_to_keep = satellite_rows_to_keep[
            satellite_rows_to_keep != neural_net.MISSING_INDEX
        ]
        ships_rows_to_keep = ships_rows_to_keep[
            ships_rows_to_keep != neural_net.MISSING_INDEX
        ]

        if len(satellite_rows_to_keep) == 0 or len(ships_rows_to_keep) == 0:
            continue

        satellite_rows_to_keep = numpy.unique(satellite_rows_to_keep)
        ships_rows_to_keep = numpy.unique(ships_rows_to_keep)

        key_to_values = dict()
        key_to_dimensions = dict()

        for this_key in SATELLITE_KEYS_TO_REPLACE:
            key_to_values[this_key] = (
                xt[this_key].values[satellite_rows_to_keep, ...]
            )
            key_to_dimensions[this_key] = xt[this_key].dims

        satellite_valid_times_unix_sec = (
            xt.coords[example_utils.SATELLITE_TIME_DIM].values
        )
        orig_num_satellite_rows = len(satellite_valid_times_unix_sec)
        satellite_valid_times_unix_sec = (
            satellite_valid_times_unix_sec[satellite_rows_to_keep]
        )
        print('Removing {0:d} of {1:d} satellite valid times...'.format(
            orig_num_satellite_rows - len(satellite_rows_to_keep),
            orig_num_satellite_rows
        ))

        xt = xt.rename({
            example_utils.SATELLITE_TIME_DIM:
                example_utils.SATELLITE_METADATA_TIME_DIM
        })
        xt = xt.drop(labels=SATELLITE_KEYS_TO_REPLACE)
        xt = xt.assign_coords({
            example_utils.SATELLITE_TIME_DIM: satellite_valid_times_unix_sec
        })
        for this_key in SATELLITE_KEYS_TO_REPLACE:
            xt = xt.assign({
                this_key: (key_to_dimensions[this_key], key_to_values[this_key])
            })

        key_to_values = dict()
        key_to_dimensions = dict()

        for this_key in SHIPS_KEYS_TO_REPLACE:
            key_to_values[this_key] = (
                xt[this_key].values[ships_rows_to_keep, ...]
            )
            key_to_dimensions[this_key] = xt[this_key].dims

        ships_init_times_unix_sec = (
            xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values
        )
        orig_num_ships_rows = len(ships_init_times_unix_sec)
        ships_init_times_unix_sec = (
            ships_init_times_unix_sec[ships_rows_to_keep]
        )
        print('Removing {0:d} of {1:d} SHIPS init times...'.format(
            orig_num_ships_rows - len(ships_rows_to_keep),
            orig_num_ships_rows
        ))

        xt = xt.rename({
            example_utils.SHIPS_VALID_TIME_DIM:
                example_utils.SHIPS_METADATA_TIME_DIM,
            'ships_storm_object_index': example_utils.SHIPS_METADATA_TIME_DIM
        })
        xt = xt.drop(labels=SHIPS_KEYS_TO_REPLACE)
        xt = xt.assign_coords({
            example_utils.SHIPS_VALID_TIME_DIM: ships_init_times_unix_sec
        })
        for this_key in SHIPS_KEYS_TO_REPLACE:
            xt = xt.assign({
                this_key: (key_to_dimensions[this_key], key_to_values[this_key])
            })

        print('Writing subset examples to: "{0:s}"...'.format(
            output_example_file_name
        ))
        example_io.write_file(
            example_table_xarray=xt, netcdf_file_name=output_example_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME),
        ships_lag_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, SHIPS_LAG_TIMES_ARG_NAME), dtype=int
        ),
        satellite_lag_times_minutes=numpy.array(
            getattr(INPUT_ARG_OBJECT, SATELLITE_LAG_TIMES_ARG_NAME), dtype=int
        ),
        ships_time_tolerance_sec=getattr(
            INPUT_ARG_OBJECT, SHIPS_TIME_TOLERANCE_ARG_NAME
        ),
        satellite_time_tolerance_sec=getattr(
            INPUT_ARG_OBJECT, SATELLITE_TIME_TOLERANCE_ARG_NAME
        ),
        output_example_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
