"""Contains list of input arguments for training a neural net."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net

DEFAULT_TRAINING_YEARS = numpy.concatenate((
    numpy.linspace(1993, 2004, num=12, dtype=int),
    numpy.linspace(2015, 2019, num=5, dtype=int)
))
DEFAULT_VALIDATION_YEARS = numpy.linspace(2005, 2009, num=5, dtype=int)

TEMPLATE_FILE_ARG_NAME = 'input_template_file_name'
OUTPUT_DIR_ARG_NAME = 'output_model_dir_name'
TRAINING_DIR_ARG_NAME = 'training_example_dir_name'
VALIDATION_DIR_ARG_NAME = 'validation_example_dir_name'
TRAINING_YEARS_ARG_NAME = 'training_years'
VALIDATION_YEARS_ARG_NAME = 'validation_years'
LEAD_TIME_ARG_NAME = 'lead_time_hours'
SATELLITE_LAG_TIMES_ARG_NAME = 'satellite_lag_times_minutes'
SHIPS_LAG_TIMES_ARG_NAME = 'ships_lag_times_hours'
SATELLITE_PREDICTORS_ARG_NAME = 'satellite_predictor_names'
SHIPS_PREDICTORS_LAGGED_ARG_NAME = 'ships_predictor_names_lagged'
SHIPS_PREDICTORS_FORECAST_ARG_NAME = 'ships_predictor_names_forecast'
TRAINING_SAT_TIME_TOLERANCE_ARG_NAME = 'satellite_time_tolerance_training_sec'
TRAINING_SAT_MAX_MISSING_ARG_NAME = 'satellite_max_missing_times_training'
TRAINING_SHIPS_TIME_TOLERANCE_ARG_NAME = 'ships_time_tolerance_training_sec'
TRAINING_SHIPS_MAX_MISSING_ARG_NAME = 'ships_max_missing_times_training'
VALIDATION_SAT_TIME_TOLERANCE_ARG_NAME = (
    'satellite_time_tolerance_validation_sec'
)
VALIDATION_SHIPS_TIME_TOLERANCE_ARG_NAME = 'ships_time_tolerance_validation_sec'
NUM_POSITIVE_EXAMPLES_ARG_NAME = 'num_positive_examples_per_batch'
NUM_NEGATIVE_EXAMPLES_ARG_NAME = 'num_negative_examples_per_batch'
MAX_EXAMPLES_PER_CYCLONE_ARG_NAME = 'max_examples_per_cyclone_in_batch'
CLASS_CUTOFFS_ARG_NAME = 'class_cutoffs_kt'
PREDICT_TD_TO_TS_ARG_NAME = 'predict_td_to_ts'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
PLATEAU_LR_MULTIPLIER_ARG_NAME = 'plateau_lr_multiplier'

TEMPLATE_FILE_HELP_STRING = (
    'Path to template file, containing compiled but untrained model.  Will be '
    'read by `neural_net.read_model`.'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Model will be saved here.'
TRAINING_DIR_HELP_STRING = (
    'Name of directory with training examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
VALIDATION_DIR_HELP_STRING = (
    'Same as `{0:s}` but for validation data.'.format(TRAINING_DIR_ARG_NAME)
)
TRAINING_YEARS_HELP_STRING = 'List of training years.'
VALIDATION_YEARS_HELP_STRING = 'List of validation years.'
LEAD_TIME_HELP_STRING = 'Lead time for predicting storm intensity.'
SATELLITE_LAG_TIMES_HELP_STRING = (
    'List of lag times for satellite predictors.  If you do not want satellite '
    'predictors (brightness-temperature grids or scalars), make this a one-item'
    ' list with a negative value.'
)
SHIPS_LAG_TIMES_HELP_STRING = (
    'List of lag times for SHIPS predictors.  If you do not want SHIPS '
    'predictors, make this a one-item list with a negative value.'
)
SATELLITE_PREDICTORS_HELP_STRING = (
    'List with names of scalar satellite predictors to use.  If you do not want'
    ' scalar satellite predictors, make this a one-item list with the empty '
    'string, "".'
)
SHIPS_PREDICTORS_LAGGED_HELP_STRING = (
    'List with names of lagged SHIPS predictors to use.  If you do not want '
    'lagged SHIPS predictors, make this a one-item list with the empty '
    'string, "".'
)
SHIPS_PREDICTORS_FORECAST_HELP_STRING = (
    'List with names of forecast SHIPS predictors to use.  If you do not want '
    'forecast SHIPS predictors, make this a one-item list with the empty '
    'string, "".'
)

TRAINING_SAT_TIME_TOLERANCE_HELP_STRING = (
    'Time tolerance for satellite data.  For desired time t, if no data can be '
    'found within `{0:s}` of t, then missing data will be interpolated.'
).format(TRAINING_SAT_TIME_TOLERANCE_ARG_NAME)

TRAINING_SAT_MAX_MISSING_HELP_STRING = (
    'Max number of missing times for satellite data.  If more times are missing'
    ' for example e, then e will not be used for training.'
)

TRAINING_SHIPS_TIME_TOLERANCE_HELP_STRING = (
    'Time tolerance for SHIPS data.  For desired time t, if no data can be '
    'found within `{0:s}` of t, then missing data will be interpolated.'
).format(TRAINING_SHIPS_TIME_TOLERANCE_ARG_NAME)

TRAINING_SHIPS_MAX_MISSING_HELP_STRING = (
    'Max number of missing times for SHIPS data.  If more times are missing'
    ' for example e, then e will not be used for training.'
)

VALIDATION_SAT_TIME_TOLERANCE_HELP_STRING = (
    'Same as `{0:s}` but for validation data.'
).format(TRAINING_SAT_TIME_TOLERANCE_ARG_NAME)

VALIDATION_SHIPS_TIME_TOLERANCE_HELP_STRING = (
    'Same as `{0:s}` but for validation data.'
).format(TRAINING_SHIPS_TIME_TOLERANCE_ARG_NAME)

NUM_POSITIVE_EXAMPLES_HELP_STRING = (
    'Number of positive examples (in highest class) per batch.'
)
NUM_NEGATIVE_EXAMPLES_HELP_STRING = (
    'Number of negative examples (not in highest class) per batch.'
)
MAX_EXAMPLES_PER_CYCLONE_HELP_STRING = (
    'Max number of examples (time steps) from one cyclone in a batch.'
)
PREDICT_TD_TO_TS_HELP_STRING = (
    'Boolean flag.  If 1, will predict intensification of tropical depression '
    'to tropical storm.  If 0, will predict rapid intensification.'
)
CLASS_CUTOFFS_HELP_STRING = (
    'List of class cutoffs (intensification in knots).  List must have length '
    'K - 1, where K = number of classes.'
)
NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
NUM_VALIDATION_BATCHES_HELP_STRING = 'Number of validation batches per epoch.'
PLATEAU_LR_MULTIPLIER_HELP_STRING = (
    'Multiplier for learning rate.  Learning rate will be multiplied by this '
    'factor upon plateau in validation performance.'
)


def add_input_args(parser_object):
    """Adds input args to ArgumentParser object.

    :param parser_object: Instance of `argparse.ArgumentParser` (may already
        contain some input args).
    :return: parser_object: Same as input but with new args added.
    """

    parser_object.add_argument(
        '--' + TEMPLATE_FILE_ARG_NAME, type=str, required=True,
        help=TEMPLATE_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
        help=OUTPUT_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_DIR_ARG_NAME, type=str, required=True,
        help=TRAINING_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDATION_DIR_ARG_NAME, type=str, required=True,
        help=VALIDATION_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_YEARS_ARG_NAME, type=int, nargs='+', required=False,
        default=DEFAULT_TRAINING_YEARS, help=TRAINING_YEARS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDATION_YEARS_ARG_NAME, type=int, nargs='+', required=False,
        default=DEFAULT_VALIDATION_YEARS, help=VALIDATION_YEARS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LEAD_TIME_ARG_NAME, type=int, required=True,
        help=LEAD_TIME_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SATELLITE_LAG_TIMES_ARG_NAME, type=int, nargs='+', required=True,
        help=SATELLITE_LAG_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SHIPS_LAG_TIMES_ARG_NAME, type=int, nargs='+', required=True,
        help=SHIPS_LAG_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SATELLITE_PREDICTORS_ARG_NAME, type=str, nargs='+',
        required=False, default=neural_net.DEFAULT_SATELLITE_PREDICTOR_NAMES,
        help=SATELLITE_PREDICTORS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SHIPS_PREDICTORS_LAGGED_ARG_NAME, type=str, nargs='+',
        required=False, default=neural_net.DEFAULT_SHIPS_PREDICTOR_NAMES_LAGGED,
        help=SHIPS_PREDICTORS_LAGGED_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SHIPS_PREDICTORS_FORECAST_ARG_NAME, type=str, nargs='+',
        required=False,
        default=neural_net.DEFAULT_SHIPS_PREDICTOR_NAMES_FORECAST,
        help=SHIPS_PREDICTORS_FORECAST_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_SAT_TIME_TOLERANCE_ARG_NAME, type=int, required=False,
        default=930, help=TRAINING_SAT_TIME_TOLERANCE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_SAT_MAX_MISSING_ARG_NAME, type=int, required=True,
        help=TRAINING_SAT_MAX_MISSING_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_SHIPS_TIME_TOLERANCE_ARG_NAME, type=int, required=False,
        default=0, help=TRAINING_SHIPS_TIME_TOLERANCE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_SHIPS_MAX_MISSING_ARG_NAME, type=int, required=True,
        help=TRAINING_SHIPS_MAX_MISSING_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDATION_SAT_TIME_TOLERANCE_ARG_NAME, type=int, required=False,
        default=3630, help=VALIDATION_SAT_TIME_TOLERANCE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDATION_SHIPS_TIME_TOLERANCE_ARG_NAME, type=int,
        required=False, default=21610,
        help=VALIDATION_SHIPS_TIME_TOLERANCE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_POSITIVE_EXAMPLES_ARG_NAME, type=int, required=True,
        help=NUM_POSITIVE_EXAMPLES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_NEGATIVE_EXAMPLES_ARG_NAME, type=int, required=True,
        help=NUM_NEGATIVE_EXAMPLES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + MAX_EXAMPLES_PER_CYCLONE_ARG_NAME, type=int, required=True,
        help=MAX_EXAMPLES_PER_CYCLONE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PREDICT_TD_TO_TS_ARG_NAME, type=int, required=False, default=0,
        help=PREDICT_TD_TO_TS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + CLASS_CUTOFFS_ARG_NAME, type=float, nargs='+', required=False,
        default=[30], help=CLASS_CUTOFFS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False, default=1000,
        help=NUM_EPOCHS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
        default=32, help=NUM_TRAINING_BATCHES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
        default=16, help=NUM_VALIDATION_BATCHES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PLATEAU_LR_MULTIPLIER_ARG_NAME, type=float, required=False,
        default=0.6, help=PLATEAU_LR_MULTIPLIER_HELP_STRING
    )

    return parser_object
