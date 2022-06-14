"""Runs discard test to determine quality of uncertainty estimates."""

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
DISCARD_FRACTIONS_ARG_NAME = 'discard_fractions'
USE_FANCY_QUANTILES_ARG_NAME = 'use_fancy_quantile_method_for_stdev'
LEAD_TIMES_ARG_NAME = 'lead_times_hours'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and target values.  Will be '
    'read by `prediction_io.read_file`.'
)
DISCARD_FRACTIONS_HELP_STRING = (
    'List of discard fractions, ranging from (0, 1).  This script will '
    'automatically use 0 as the lowest discard fraction.'
)
USE_FANCY_QUANTILES_HELP_STRING = (
    '[used only if model does quantile regression] Boolean flag.  If 1, will '
    'use fancy quantile-based method to compute standard deviation of '
    'predictive distribution.  If False, will treat each quantile-based '
    'estimate as a Monte Carlo estimate.'
)
LEAD_TIMES_HELP_STRING = (
    'List of lead times.  Will run discard test aggregated over these lead '
    'times.  If you want to aggregate over all lead times, leave this argument '
    'alone.'
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
        '--' + DISCARD_FRACTIONS_ARG_NAME, type=float, nargs='+', required=True,
    help=DISCARD_FRACTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_FANCY_QUANTILES_ARG_NAME, type=int, required=True, default=1,
    help=USE_FANCY_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=LEAD_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_name, discard_fractions,
         use_fancy_quantile_method_for_stdev, lead_times_hours,
         output_file_name):
    """Runs discard test to determine quality of uncertainty estimates.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param discard_fractions: Same.
    :param use_fancy_quantile_method_for_stdev: Same.
    :param lead_times_hours: Same.
    :param output_file_name: Same.
    """

    if len(lead_times_hours) == 1 and lead_times_hours[0] <= 0:
        lead_times_hours = None

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_file_name
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    if lead_times_hours is not None:
        prediction_dict = prediction_io.subset_by_lead_time(
            prediction_dict=prediction_dict, lead_times_hours=lead_times_hours
        )

    uncertainty_function = uq_evaluation.get_stdev_uncertainty_function(
        use_fancy_quantile_method=use_fancy_quantile_method_for_stdev
    )
    error_function = uq_evaluation.get_xentropy_error_function(use_median=False)

    result_dict = uq_evaluation.run_discard_test(
        prediction_dict=prediction_dict, discard_fractions=discard_fractions,
        error_function=error_function,
        uncertainty_function=uncertainty_function,
        use_median=False, is_error_pos_oriented=False
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    uq_evaluation.write_discard_results(
        netcdf_file_name=output_file_name, result_dict=result_dict,
        error_function_name='cross-entropy',
        uncertainty_function_name='pixelwise stdev',
        use_fancy_quantile_method_for_stdev=
        use_fancy_quantile_method_for_stdev
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        discard_fractions=numpy.array(
            getattr(INPUT_ARG_OBJECT, DISCARD_FRACTIONS_ARG_NAME), dtype=float
        ),
        use_fancy_quantile_method_for_stdev=bool(
            getattr(INPUT_ARG_OBJECT, USE_FANCY_QUANTILES_ARG_NAME)
        ),
        lead_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, LEAD_TIMES_ARG_NAME), dtype=int
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
