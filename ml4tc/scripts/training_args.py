"""Contains list of input arguments for training a neural net."""

import numpy
from ml4tc.machine_learning import neural_net

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
LEAD_TIMES_ARG_NAME = 'lead_times_hours'
SATELLITE_PREDICTORS_ARG_NAME = 'satellite_predictor_names'
SATELLITE_LAG_TIMES_ARG_NAME = 'satellite_lag_times_minutes'
SHIPS_GOES_PREDICTORS_ARG_NAME = 'ships_goes_predictor_names'
SHIPS_GOES_LAG_TIMES_ARG_NAME = 'ships_goes_lag_times_hours'
SHIPS_FORECAST_PREDICTORS_ARG_NAME = 'ships_forecast_predictor_names'
SHIPS_MAX_FORECAST_HOUR_ARG_NAME = 'ships_max_forecast_hour'
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
NUM_ROWS_ARG_NAME = 'num_grid_rows'
NUM_COLUMNS_ARG_NAME = 'num_grid_columns'
USE_TIME_DIFFS_ARG_NAME = 'use_time_diffs_gridded_sat'
PREDICT_TD_TO_TS_ARG_NAME = 'predict_td_to_ts'
DATA_AUG_NUM_TRANS_ARG_NAME = 'data_aug_num_translations'
DATA_AUG_MAX_TRANS_ARG_NAME = 'data_aug_max_translation_px'
DATA_AUG_NUM_ROTATIONS_ARG_NAME = 'data_aug_num_rotations'
DATA_AUG_MAX_ROTATION_ARG_NAME = 'data_aug_max_rotation_deg'
DATA_AUG_NUM_NOISINGS_ARG_NAME = 'data_aug_num_noisings'
DATA_AUG_NOISE_STDEV_ARG_NAME = 'data_aug_noise_stdev'
WEST_PACIFIC_WEIGHT_ARG_NAME = 'west_pacific_weight'

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
LEAD_TIMES_HELP_STRING = 'Lead times for predicting storm intensity.'
SATELLITE_PREDICTORS_HELP_STRING = (
    'List with names of scalar satellite predictors to use.  If you do not want'
    ' scalar satellite predictors, make this a one-item list with the empty '
    'string, "".'
)
SATELLITE_LAG_TIMES_HELP_STRING = (
    'List of lag times for satellite predictors.  If you do not want satellite '
    'predictors (brightness-temperature grids or scalars), make this a one-item'
    ' list with a negative value.'
)
SHIPS_GOES_PREDICTORS_HELP_STRING = (
    'List with names of SHIPS GOES predictors to use.  If you do not want '
    'SHIPS GOES predictors, make this a one-item list with the empty '
    'string, "".'
)
SHIPS_GOES_LAG_TIMES_HELP_STRING = (
    'List of lag times for SHIPS GOES predictors.  If you do not want SHIPS '
    'GOES predictors, make this a one-item list with a negative value.'
)
SHIPS_FORECAST_PREDICTORS_HELP_STRING = (
    'List with names of forecast SHIPS predictors to use.  If you do not want '
    'forecast SHIPS predictors, make this a one-item list with the empty '
    'string, "".'
)
SHIPS_MAX_FORECAST_HOUR_HELP_STRING = (
    'Max forecast hour to include in SHIPS predictors.'
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
NUM_ROWS_HELP_STRING = (
    'Number of rows to keep in brightness-temperature grid.  If you want to '
    'keep all rows, leave this alone.'
)
NUM_COLUMNS_HELP_STRING = (
    'Number of columns to keep in brightness-temperature grid.  If you want to '
    'keep all columns, leave this alone.'
)
USE_TIME_DIFFS_HELP_STRING = (
    'Boolean flag.  If 1, will turn gridded satellite data at non-zero lag '
    'times into temporal differences.'
)
DATA_AUG_NUM_TRANS_HELP_STRING = (
    'Number of translations per example for data augmentation.  You can make '
    'this 0.'
)
DATA_AUG_MAX_TRANS_HELP_STRING = (
    'Max translation (pixels) for data augmentation.'
)
DATA_AUG_NUM_ROTATIONS_HELP_STRING = (
    'Number of rotations per example for data augmentation.  You can make this '
    '0.'
)
DATA_AUG_MAX_ROTATION_HELP_STRING = (
    'Max absolute rotation angle (degrees) for data augmentation.'
)
DATA_AUG_NUM_NOISINGS_HELP_STRING = (
    'Number of noisings per example for data augmentation.  You can make this '
    '0.'
)
DATA_AUG_NOISE_STDEV_HELP_STRING = (
    'Standard deviation of Gaussian noise for data augmentation.'
)
WEST_PACIFIC_WEIGHT_HELP_STRING = (
    'Loss-function weight for cyclones in the western Pacific.  All other '
    'cyclones will receive a weight of 1.0.  If you do not want different '
    'weights, leave this alone.'
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
        '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=True,
        help=LEAD_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SATELLITE_PREDICTORS_ARG_NAME, type=str, nargs='+',
        required=False, default=neural_net.DEFAULT_SATELLITE_PREDICTOR_NAMES,
        help=SATELLITE_PREDICTORS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SATELLITE_LAG_TIMES_ARG_NAME, type=int, nargs='+', required=True,
        help=SATELLITE_LAG_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SHIPS_GOES_PREDICTORS_ARG_NAME, type=str, nargs='+',
        required=False, default=neural_net.DEFAULT_SHIPS_GOES_PREDICTOR_NAMES,
        help=SHIPS_GOES_PREDICTORS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SHIPS_GOES_LAG_TIMES_ARG_NAME, type=int, nargs='+',
        required=True, help=SHIPS_GOES_LAG_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SHIPS_FORECAST_PREDICTORS_ARG_NAME, type=str, nargs='+',
        required=False,
        default=neural_net.DEFAULT_SHIPS_FORECAST_PREDICTOR_NAMES,
        help=SHIPS_FORECAST_PREDICTORS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SHIPS_MAX_FORECAST_HOUR_ARG_NAME, type=int, required=False,
        default=0, help=SHIPS_MAX_FORECAST_HOUR_HELP_STRING
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
        '--' + DATA_AUG_NUM_TRANS_ARG_NAME, type=int, required=False, default=0,
        help=DATA_AUG_NUM_TRANS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + DATA_AUG_MAX_TRANS_ARG_NAME, type=int, required=False,
        default=-1, help=DATA_AUG_MAX_TRANS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + DATA_AUG_NUM_ROTATIONS_ARG_NAME, type=int, required=False,
        default=0, help=DATA_AUG_NUM_ROTATIONS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + DATA_AUG_MAX_ROTATION_ARG_NAME, type=float, required=False,
        default=-1, help=DATA_AUG_MAX_ROTATION_HELP_STRING
    )
    parser_object.add_argument(
        '--' + DATA_AUG_NUM_NOISINGS_ARG_NAME, type=int, required=False,
        default=0, help=DATA_AUG_NUM_NOISINGS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + DATA_AUG_NOISE_STDEV_ARG_NAME, type=float, required=False,
        default=-1, help=DATA_AUG_NOISE_STDEV_HELP_STRING
    )
    parser_object.add_argument(
        '--' + WEST_PACIFIC_WEIGHT_ARG_NAME, type=float, required=False,
        default=-1, help=WEST_PACIFIC_WEIGHT_HELP_STRING
    )
    parser_object.add_argument(
        '--' + CLASS_CUTOFFS_ARG_NAME, type=float, nargs='+', required=False,
        default=[30], help=CLASS_CUTOFFS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_ROWS_ARG_NAME, type=int, required=False, default=-1,
        help=NUM_ROWS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_COLUMNS_ARG_NAME, type=int, required=False, default=-1,
        help=NUM_COLUMNS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + USE_TIME_DIFFS_ARG_NAME, type=int, required=False, default=0,
        help=USE_TIME_DIFFS_HELP_STRING
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
