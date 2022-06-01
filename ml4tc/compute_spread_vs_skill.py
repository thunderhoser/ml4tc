"""Computes spread vs. skill to determine quality of uncertainty estimates."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import prediction_io
import uq_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
BIN_EDGES_ARG_NAME = 'bin_edge_prediction_stdevs'
USE_FANCY_QUANTILES_ARG_NAME = 'use_fancy_quantile_method_for_stdev'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and target values.  Will be '
    'read by `prediction_io.read_file`.'
)
BIN_EDGES_HELP_STRING = (
    'List of bin cutoffs -- ranging from (0, 1) -- each a standard deviation '
    'of the predictive distribution.  This script will automatically use 0 and '
    '1 as the lowest and highest cutoffs.'
)
USE_FANCY_QUANTILES_HELP_STRING = (
    '[used only if model does quantile regression] Boolean flag.  If 1, will '
    'use fancy quantile-based method to compute standard deviation of '
    'predictive distribution.  If False, will treat each quantile-based '
    'estimate as a Monte Carlo estimate.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results will be written here by '
    '`uq_evaluation.write_spread_vs_skill`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BIN_EDGES_ARG_NAME, type=float, nargs='+', required=True,
    help=BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_FANCY_QUANTILES_ARG_NAME, type=int, required=True, default=1,
    help=USE_FANCY_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_name, bin_edge_prediction_stdevs,
         use_fancy_quantile_method_for_stdev, output_file_name):
    """Runs discard test to determine quality of uncertainty estimates.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param bin_edge_prediction_stdevs: Same.
    :param use_fancy_quantile_method_for_stdev: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_file_name
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    result_dict = uq_evaluation.get_spread_vs_skill(
        prediction_dict=prediction_dict,
        bin_edge_prediction_stdevs=bin_edge_prediction_stdevs,
        use_median=False,
        use_fancy_quantile_method_for_stdev=use_fancy_quantile_method_for_stdev
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    uq_evaluation.write_spread_vs_skill(
        netcdf_file_name=output_file_name, result_dict=result_dict,
        use_median=False,
        use_fancy_quantile_method_for_stdev=use_fancy_quantile_method_for_stdev
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        bin_edge_prediction_stdevs=numpy.array(
            getattr(INPUT_ARG_OBJECT, BIN_EDGES_ARG_NAME), dtype=float
        ),
        use_fancy_quantile_method_for_stdev=bool(
            getattr(INPUT_ARG_OBJECT, USE_FANCY_QUANTILES_ARG_NAME)
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
