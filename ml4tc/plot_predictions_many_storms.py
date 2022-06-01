"""Plots predictions for many storms, one map per time step."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import example_io
import prediction_io
import satellite_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H'
TIME_INTERVAL_SEC = 21600

CYCLONE_IDS_KEY = 'cyclone_id_strings'
FORECAST_PROBS_KEY = 'forecast_probabilities'

MODEL_METAFILE_ARG_NAME = 'input_model_metafile_name'
EXAMPLE_DIR_ARG_NAME = 'input_norm_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
MIN_LATITUDE_ARG_NAME = 'min_latitude_deg_n'
MAX_LATITUDE_ARG_NAME = 'max_latitude_deg_n'
MIN_LONGITUDE_ARG_NAME = 'min_longitude_deg_e'
MAX_LONGITUDE_ARG_NAME = 'max_longitude_deg_e'
FIRST_TIME_ARG_NAME = 'first_init_time_string'
LAST_TIME_ARG_NAME = 'last_init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_METAFILE_HELP_STRING = (
    'Path to metafile for model.  Will be read by `neural_net.read_metafile`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with normalized learning examples.  Files therein will '
    'be found by `example_io.find_file` and read by `example_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be used to denormalize '
    'brightness-temperature maps before plotting).  Will be read by '
    '`normalization.read_file`.'
)
PREDICTION_FILE_HELP_STRING = (
    'Path to file with predictions and targets.  Will be read by '
    '`prediction_io.read_file`.'
)
MIN_LATITUDE_HELP_STRING = 'Minimum latitude (deg north) in map.'
MAX_LATITUDE_HELP_STRING = 'Max latitude (deg north) in map.'
MIN_LONGITUDE_HELP_STRING = 'Minimum longitude (deg east) in map.'
MAX_LONGITUDE_HELP_STRING = 'Max longitude (deg east) in map.'
FIRST_TIME_HELP_STRING = (
    'First initialization time to plot (format "yyyy-mm-dd-HH").'
)
LAST_TIME_HELP_STRING = (
    'Last initialization time to plot (format "yyyy-mm-dd-HH").'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_METAFILE_ARG_NAME, type=str, required=True,
    help=MODEL_METAFILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=False, default='',
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LATITUDE_ARG_NAME, type=float, required=True,
    help=MIN_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LATITUDE_ARG_NAME, type=float, required=True,
    help=MAX_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LONGITUDE_ARG_NAME, type=float, required=True,
    help=MIN_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LONGITUDE_ARG_NAME, type=float, required=True,
    help=MAX_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True,
    help=LAST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _subset_data(
        data_dict, min_latitude_deg_n, max_latitude_deg_n, min_longitude_deg_e,
        max_longitude_deg_e, longitude_positive_in_west, cyclone_id_string,
        first_init_time_unix_sec, last_init_time_unix_sec):
    """Subsets data by time and location.

    E = number of examples

    :param data_dict: Dictionary returned by `neural_net.create_inputs`.
    :param min_latitude_deg_n: See documentation at top of file.
    :param max_latitude_deg_n: Same.
    :param min_longitude_deg_e: Same.
    :param max_longitude_deg_e: Same.
    :param longitude_positive_in_west: Boolean flag, indicating longitude
        format.
    :param cyclone_id_string: Cyclone ID.
    :param first_init_time_unix_sec: See documentation at top of file.
    :param last_init_time_unix_sec: Same.
    :return: data_dict: Subset version of input, containing fewer examples and
        an extra key.
    data_dict['cyclone_id_strings']: length-E list of cyclone IDs.
    """

    good_latitude_flags = numpy.logical_and(
        data_dict[neural_net.STORM_LATITUDES_KEY] >= min_latitude_deg_n,
        data_dict[neural_net.STORM_LATITUDES_KEY] <= max_latitude_deg_n
    )

    if longitude_positive_in_west:
        storm_longitudes_deg_e = (
            lng_conversion.convert_lng_positive_in_west(
                data_dict[neural_net.STORM_LONGITUDES_KEY]
            )
        )
    else:
        storm_longitudes_deg_e = (
            lng_conversion.convert_lng_negative_in_west(
                data_dict[neural_net.STORM_LONGITUDES_KEY]
            )
        )

    good_longitude_flags = numpy.logical_and(
        storm_longitudes_deg_e >= min_longitude_deg_e,
        storm_longitudes_deg_e <= max_longitude_deg_e
    )
    good_location_flags = numpy.logical_and(
        good_latitude_flags, good_longitude_flags
    )
    good_time_flags = numpy.logical_and(
        data_dict[neural_net.INIT_TIMES_KEY] >= first_init_time_unix_sec,
        data_dict[neural_net.INIT_TIMES_KEY] <= last_init_time_unix_sec
    )
    good_indices = numpy.where(
        numpy.logical_and(good_location_flags, good_time_flags)
    )[0]

    if len(good_indices) == 0:
        return None

    data_dict[neural_net.PREDICTOR_MATRICES_KEY] = [
        a[good_indices, ...] for a in
        data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]

    for this_key in [
            neural_net.TARGET_ARRAY_KEY, neural_net.INIT_TIMES_KEY,
            neural_net.STORM_LATITUDES_KEY, neural_net.STORM_LONGITUDES_KEY,
            neural_net.GRID_LATITUDE_MATRIX_KEY,
            neural_net.GRID_LONGITUDE_MATRIX_KEY
    ]:
        data_dict[this_key] = data_dict[this_key][good_indices, ...]

    data_dict[CYCLONE_IDS_KEY] = [cyclone_id_string] * len(good_indices)

    return data_dict


def _concat_data(data_dicts):
    """Concatenates many examples into the same dictionary.

    :param data_dicts: 1-D list of dictionaries returned by
        `_subset_data`.
    :return: data_dict: Single dictionary, created by concatenating inputs.
    """

    num_matrices = len(data_dicts[0][neural_net.PREDICTOR_MATRICES_KEY])
    data_dict = {
        neural_net.PREDICTOR_MATRICES_KEY: []
    }

    for k in range(num_matrices):
        data_dict[neural_net.PREDICTOR_MATRICES_KEY][k] = numpy.concatenate(
            [d[neural_net.PREDICTOR_MATRICES_KEY][k] for d in data_dicts],
            axis=0
        )

    for this_key in [
            neural_net.TARGET_ARRAY_KEY, neural_net.INIT_TIMES_KEY,
            neural_net.STORM_LATITUDES_KEY, neural_net.STORM_LONGITUDES_KEY,
            neural_net.GRID_LATITUDE_MATRIX_KEY,
            neural_net.GRID_LONGITUDE_MATRIX_KEY
    ]:
        data_dict[this_key] = numpy.concatenate(
            [d[this_key] for d in data_dicts], axis=0
        )

    data_dict[CYCLONE_IDS_KEY] = numpy.concatenate(
        [numpy.array(d[CYCLONE_IDS_KEY]) for d in data_dicts], axis=0
    )
    data_dict[CYCLONE_IDS_KEY] = data_dict[CYCLONE_IDS_KEY].tolist()

    return data_dict


def _match_predictors_to_predictions(data_dict, prediction_file_name):
    """Matches predictors to predictions.

    E = number of examples

    :param data_dict: Dictionary returned by `_concat_data`.
    :param prediction_file_name: See documentation at top of file.
    :return: data_dict: Same as input but with an extra key.
    data_dict['forecast_probabilities']: length-E numpy array of forecast event
        probabilities.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    prediction_dict[prediction_io.CYCLONE_IDS_KEY] = numpy.array(
        prediction_dict[prediction_io.CYCLONE_IDS_KEY]
    )
    all_forecast_probs = prediction_io.get_mean_predictions(prediction_dict)

    good_indices = []
    num_examples = len(data_dict[neural_net.INIT_TIMES_KEY])

    for i in range(num_examples):
        this_index = numpy.where(numpy.logical_and(
            prediction_dict[prediction_io.INIT_TIMES_KEY] ==
            data_dict[neural_net.INIT_TIMES_KEY][i],
            prediction_dict[prediction_io.CYCLONE_IDS_KEY] ==
            data_dict[CYCLONE_IDS_KEY][i]
        ))[0][0]

        good_indices.append(this_index)

    good_indices = numpy.array(good_indices, dtype=int)
    data_dict[FORECAST_PROBS_KEY] = all_forecast_probs[good_indices]
    return data_dict


def _run(model_metafile_name, norm_example_dir_name, normalization_file_name,
         prediction_file_name, min_latitude_deg_n, max_latitude_deg_n,
         min_longitude_deg_e, max_longitude_deg_e, first_init_time_string,
         last_init_time_string, output_dir_name):
    """Plots predictions for many storms, one map per time step.

    This is effectively the main method.

    :param model_metafile_name: See documentation at top of file.
    :param norm_example_dir_name: Same.
    :param normalization_file_name: Same.
    :param prediction_file_name: Same.
    :param min_latitude_deg_n: Same.
    :param max_latitude_deg_n: Same.
    :param min_longitude_deg_e: Same.
    :param max_longitude_deg_e: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_valid_latitude(min_latitude_deg_n)
    error_checking.assert_is_valid_latitude(max_latitude_deg_n)
    error_checking.assert_is_greater(max_latitude_deg_n, min_latitude_deg_n)

    min_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg_e, allow_nan=False
    )
    max_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        max_longitude_deg_e, allow_nan=False
    )
    longitude_positive_in_west = True

    if max_longitude_deg_e <= min_longitude_deg_e:
        min_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            min_longitude_deg_e, allow_nan=False
        )
        max_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            max_longitude_deg_e, allow_nan=False
        )
        error_checking.assert_is_greater(
            max_longitude_deg_e, min_longitude_deg_e
        )
        longitude_positive_in_west = False

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Find example files.
    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True
    )

    first_year = int(time_conversion.unix_sec_to_string(
        init_times_unix_sec[0], '%Y'
    ))
    last_year = int(time_conversion.unix_sec_to_string(
        init_times_unix_sec[-1], '%Y'
    ))
    years = numpy.linspace(
        first_year, last_year, num=last_year - first_year + 1, dtype=int
    )

    cyclone_id_strings = example_io.find_cyclones(
        directory_name=norm_example_dir_name,
        raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array([
        satellite_utils.parse_cyclone_id(c)[0]
        for c in cyclone_id_strings
    ], dtype=int)

    good_flags = numpy.array(
        [c in years for c in cyclone_years], dtype=float
    )
    good_indices = numpy.where(good_flags)[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    cyclone_id_strings.sort()

    example_file_names = [
        example_io.find_file(
            directory_name=norm_example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        for c in cyclone_id_strings
    ]

    # Read model metadata.
    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY][
        neural_net.USE_TIME_DIFFS_KEY
    ] = False

    # Read examples.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    data_dicts = []

    for i in range(len(example_file_names)):
        print(SEPARATOR_STRING)

        this_option_dict = copy.deepcopy(validation_option_dict)
        this_option_dict[neural_net.EXAMPLE_FILE_KEY] = example_file_names[i]
        this_data_dict = neural_net.create_inputs(this_option_dict)

        this_data_dict = _subset_data(
            data_dict=this_data_dict, min_latitude_deg_n=min_latitude_deg_n,
            max_latitude_deg_n=max_latitude_deg_n,
            min_longitude_deg_e=min_longitude_deg_e,
            max_longitude_deg_e=max_longitude_deg_e,
            longitude_positive_in_west=longitude_positive_in_west,
            cyclone_id_string=
            example_io.file_name_to_cyclone_id(example_file_names[i]),
            first_init_time_unix_sec=first_init_time_unix_sec,
            last_init_time_unix_sec=last_init_time_unix_sec
        )

        if this_data_dict is None:
            continue

        data_dicts.append(this_data_dict)

    print(SEPARATOR_STRING)
    data_dict = _concat_data(data_dicts)
    del data_dicts

    data_dict = _match_predictors_to_predictions(
        data_dict=data_dict, prediction_file_name=prediction_file_name
    )
    print(SEPARATOR_STRING)

    # TODO(thunderhoser): Now do the plotting.


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_metafile_name=getattr(INPUT_ARG_OBJECT, MODEL_METAFILE_ARG_NAME),
        norm_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        min_latitude_deg_n=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_latitude_deg_n=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_longitude_deg_e=getattr(INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_longitude_deg_e=getattr(INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        first_init_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_init_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
