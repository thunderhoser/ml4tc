"""Matches MCA loading with local solar time.

MCA loading = standardized expansion coefficient for brightness temperature
(not Shapley value), based on maximum-covariance analysis
"""

import os
import sys
import pickle
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import ships_io
import general_utils
import saliency
import run_mca_for_shapley_maps as run_mca

SHAPLEY_FILES_ARG_NAME = 'input_shapley_file_names'
MCA_FILE_ARG_NAME = 'input_mca_file_name'
SHIPS_DIR_ARG_NAME = 'input_ships_dir_name'
MODE_NUM_ARG_NAME = 'mode_num'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SHAPLEY_FILES_HELP_STRING = (
    'List of paths to Shapley files (used to find cyclone ID for each TC '
    'object).  These files will be read by `saliency.read_file`.'
)
MCA_FILE_HELP_STRING = (
    'Path to file with MCA results, created by run_mca_for_shapley_maps.py.'
)
SHIPS_DIR_HELP_STRING = (
    'Name of directory with processed SHIPS data (used to find longitude of '
    'each TC object).  Files therein will be found by `ships_io.find_file` and '
    'read by `ships_io.read_file`.'
)
MODE_NUM_HELP_STRING = (
    'Will use loadings for the [k]th-leading mode of the MCA, where k = {0:s}.'
    '  Use one-based indexing, so the first mode is 1.'
).format(MODE_NUM_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (Pickle).  Results will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SHAPLEY_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=SHAPLEY_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MCA_FILE_ARG_NAME, type=str, required=True,
    help=MCA_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHIPS_DIR_ARG_NAME, type=str, required=True,
    help=SHIPS_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODE_NUM_ARG_NAME, type=int, required=True,
    help=MODE_NUM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(shapley_file_names, mca_file_name, ships_dir_name, mode_num,
         output_file_name):
    """Matches MCA loading with local solar time.

    This is effectively the main method.

    :param shapley_file_names: See documentation at top of file.
    :param mca_file_name: Same.
    :param ships_dir_name: Same.
    :param mode_num: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(mode_num, 1)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    cyclone_id_strings = numpy.array([])
    init_times_unix_sec = numpy.array([], dtype=int)

    for this_file_name in shapley_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_saliency_dict = saliency.read_file(this_file_name)

        cyclone_id_strings = numpy.concatenate((
            cyclone_id_strings,
            this_saliency_dict[saliency.CYCLONE_IDS_KEY]
        ))
        init_times_unix_sec = numpy.concatenate((
            init_times_unix_sec,
            this_saliency_dict[saliency.INIT_TIMES_KEY]
        ))

    print('Reading data from: "{0:s}"...'.format(mca_file_name))
    mca_table_xarray = xarray.open_zarr(mca_file_name)
    predictor_expansion_coeffs = mca_table_xarray[
        run_mca.PREDICTOR_EXPANSION_COEFF_KEY
    ].values[:, mode_num - 1]

    assert len(cyclone_id_strings) == len(predictor_expansion_coeffs)

    num_examples = len(cyclone_id_strings)
    longitudes_deg_e = numpy.full(num_examples, numpy.nan)

    for unique_cyclone_id_string in numpy.unique(cyclone_id_strings):
        ships_file_name = ships_io.find_file(
            directory_name=ships_dir_name,
            cyclone_id_string=unique_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(ships_file_name))
        ships_table_xarray = ships_io.read_file(ships_file_name)

        example_indices = numpy.where(
            cyclone_id_strings == unique_cyclone_id_string
        )[0]

        ships_indices = numpy.array([
            numpy.where(ships_table_xarray[ships_io.VALID_TIME_KEY] == t)[0][0]
            for t in init_times_unix_sec[example_indices]
        ], dtype=int)

        longitudes_deg_e[example_indices] = ships_table_xarray[
            ships_io.STORM_LONGITUDE_KEY
        ].values[ships_indices]

    assert not numpy.any(numpy.isnan(longitudes_deg_e))

    solar_times_sec = general_utils.get_solar_times(
        valid_times_unix_sec=init_times_unix_sec,
        longitudes_deg_e=longitudes_deg_e
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))

    pickle_file_handle = open(output_file_name, 'wb')
    pickle.dump(cyclone_id_strings, pickle_file_handle)
    pickle.dump(init_times_unix_sec, pickle_file_handle)
    pickle.dump(predictor_expansion_coeffs, pickle_file_handle)
    pickle.dump(solar_times_sec, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        shapley_file_names=getattr(INPUT_ARG_OBJECT, SHAPLEY_FILES_ARG_NAME),
        mca_file_name=getattr(INPUT_ARG_OBJECT, MCA_FILE_ARG_NAME),
        ships_dir_name=getattr(INPUT_ARG_OBJECT, SHIPS_DIR_ARG_NAME),
        mode_num=getattr(INPUT_ARG_OBJECT, MODE_NUM_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
