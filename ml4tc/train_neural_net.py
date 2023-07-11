"""Trains neural net."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net
import training_args

NONE_STRINGS = ['', 'none', 'None']
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _run(template_file_name, output_dir_name, training_example_dir_name,
         validation_example_dir_name, training_years, validation_years,
         lead_times_hours,
         satellite_predictor_names, satellite_lag_times_minutes,
         ships_goes_predictor_names, ships_goes_lag_times_hours,
         ships_forecast_predictor_names, ships_max_forecast_hour,
         satellite_time_tolerance_training_sec,
         satellite_max_missing_times_training,
         ships_time_tolerance_training_sec, ships_max_missing_times_training,
         satellite_time_tolerance_validation_sec,
         ships_time_tolerance_validation_sec,
         num_positive_examples_per_batch,
         num_negative_examples_per_batch, max_examples_per_cyclone_in_batch,
         predict_td_to_ts, class_cutoffs_kt, num_grid_rows, num_grid_columns,
         use_time_diffs_gridded_sat, data_aug_num_translations,
         data_aug_max_translation_px, data_aug_num_rotations,
         data_aug_max_rotation_deg, data_aug_num_noisings, data_aug_noise_stdev,
         west_pacific_weight, num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, plateau_lr_multiplier):
    """Trains neural net.

    This is effectively the main method.

    :param template_file_name: See documentation at top of training_args.py.
    :param output_dir_name: Same.
    :param training_example_dir_name: Same.
    :param validation_example_dir_name: Same.
    :param training_years: Same.
    :param validation_years: Same.
    :param lead_times_hours: Same.
    :param satellite_predictor_names: Same.
    :param satellite_lag_times_minutes: Same.
    :param ships_goes_predictor_names: Same.
    :param ships_goes_lag_times_hours: Same.
    :param ships_forecast_predictor_names: Same.
    :param ships_max_forecast_hour: Same.
    :param satellite_time_tolerance_training_sec: Same.
    :param satellite_max_missing_times_training: Same.
    :param ships_time_tolerance_training_sec: Same.
    :param ships_max_missing_times_training: Same.
    :param satellite_time_tolerance_validation_sec: Same.
    :param ships_time_tolerance_validation_sec: Same.
    :param num_positive_examples_per_batch: Same.
    :param num_negative_examples_per_batch: Same.
    :param max_examples_per_cyclone_in_batch: Same.
    :param predict_td_to_ts: Same.
    :param class_cutoffs_kt: Same.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param use_time_diffs_gridded_sat: Same.
    :param data_aug_num_translations: Same.
    :param data_aug_max_translation_px: Same.
    :param data_aug_num_rotations: Same.
    :param data_aug_max_rotation_deg: Same.
    :param data_aug_num_noisings: Same.
    :param data_aug_noise_stdev: Same.
    :param west_pacific_weight: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param plateau_lr_multiplier: Same.
    """

    if num_grid_rows <= 0:
        num_grid_rows = None
    if num_grid_columns <= 0:
        num_grid_columns = None
    if west_pacific_weight <= 0:
        west_pacific_weight = None

    if (
            len(satellite_lag_times_minutes) == 1
            and satellite_lag_times_minutes[0] < 0
    ):
        satellite_lag_times_minutes = None

    if (
            len(satellite_predictor_names) == 1
            and satellite_predictor_names[0] in NONE_STRINGS
    ):
        satellite_predictor_names = None

    if (
            len(ships_goes_lag_times_hours) == 1 and
            ships_goes_lag_times_hours[0] < 0
    ):
        ships_goes_lag_times_hours = None

    if (
            len(ships_goes_predictor_names) == 1
            and ships_goes_predictor_names[0] in NONE_STRINGS
    ):
        ships_goes_predictor_names = None

    if (
            len(ships_forecast_predictor_names) == 1
            and ships_forecast_predictor_names[0] in NONE_STRINGS
    ):
        ships_forecast_predictor_names = None

    training_option_dict = {
        neural_net.EXAMPLE_DIRECTORY_KEY: training_example_dir_name,
        neural_net.YEARS_KEY: training_years,
        neural_net.LEAD_TIMES_KEY: lead_times_hours,
        neural_net.SATELLITE_LAG_TIMES_KEY: satellite_lag_times_minutes,
        neural_net.SATELLITE_PREDICTORS_KEY: satellite_predictor_names,
        neural_net.SHIPS_GOES_PREDICTORS_KEY: ships_goes_predictor_names,
        neural_net.SHIPS_GOES_LAG_TIMES_KEY: ships_goes_lag_times_hours,
        neural_net.SHIPS_FORECAST_PREDICTORS_KEY:
            ships_forecast_predictor_names,
        neural_net.SHIPS_MAX_FORECAST_HOUR_KEY: ships_max_forecast_hour,
        neural_net.NUM_POSITIVE_EXAMPLES_KEY: num_positive_examples_per_batch,
        neural_net.NUM_NEGATIVE_EXAMPLES_KEY: num_negative_examples_per_batch,
        neural_net.MAX_EXAMPLES_PER_CYCLONE_KEY:
            max_examples_per_cyclone_in_batch,
        neural_net.PREDICT_TD_TO_TS_KEY: predict_td_to_ts,
        neural_net.CLASS_CUTOFFS_KEY:
            class_cutoffs_kt * KT_TO_METRES_PER_SECOND,
        neural_net.NUM_GRID_ROWS_KEY: num_grid_rows,
        neural_net.NUM_GRID_COLUMNS_KEY: num_grid_columns,
        neural_net.USE_TIME_DIFFS_KEY: use_time_diffs_gridded_sat,
        neural_net.SATELLITE_TIME_TOLERANCE_KEY:
            satellite_time_tolerance_training_sec,
        neural_net.SATELLITE_MAX_MISSING_TIMES_KEY:
            satellite_max_missing_times_training,
        neural_net.SHIPS_TIME_TOLERANCE_KEY: ships_time_tolerance_training_sec,
        neural_net.SHIPS_MAX_MISSING_TIMES_KEY:
            ships_max_missing_times_training,
        neural_net.USE_CLIMO_KEY: False,
        neural_net.DATA_AUG_NUM_TRANS_KEY: data_aug_num_translations,
        neural_net.DATA_AUG_MAX_TRANS_KEY: data_aug_max_translation_px,
        neural_net.DATA_AUG_NUM_ROTATIONS_KEY: data_aug_num_rotations,
        neural_net.DATA_AUG_MAX_ROTATION_KEY: data_aug_max_rotation_deg,
        neural_net.DATA_AUG_NUM_NOISINGS_KEY: data_aug_num_noisings,
        neural_net.DATA_AUG_NOISE_STDEV_KEY: data_aug_noise_stdev,
        neural_net.WEST_PACIFIC_WEIGHT_KEY: west_pacific_weight
    }

    validation_option_dict = {
        neural_net.EXAMPLE_DIRECTORY_KEY: validation_example_dir_name,
        neural_net.YEARS_KEY: validation_years,
        neural_net.SATELLITE_TIME_TOLERANCE_KEY:
            satellite_time_tolerance_validation_sec,
        neural_net.SHIPS_TIME_TOLERANCE_KEY:
            ships_time_tolerance_validation_sec,
        neural_net.USE_CLIMO_KEY: True
    }

    print('Reading model template from: "{0:s}"...'.format(template_file_name))
    model_object = neural_net.read_model(hdf5_file_name=template_file_name)

    model_metafile_name = neural_net.find_metafile(
        model_file_name=template_file_name, raise_error_if_missing=True
    )
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    neural_net.train_model(
        model_object=model_object, output_dir_name=output_dir_name,
        num_epochs=num_epochs,
        use_crps_loss=model_metadata_dict[neural_net.USE_CRPS_LOSS_KEY],
        quantile_levels=model_metadata_dict[neural_net.QUANTILE_LEVELS_KEY],
        central_loss_function_weight=
        model_metadata_dict[neural_net.CENTRAL_LOSS_WEIGHT_KEY],
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        do_early_stopping=True,
        plateau_lr_multiplier=plateau_lr_multiplier,
        bnn_architecture_dict=
        model_metadata_dict[neural_net.BNN_ARCHITECTURE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        template_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.TEMPLATE_FILE_ARG_NAME
        ),
        output_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.OUTPUT_DIR_ARG_NAME
        ),
        training_example_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_DIR_ARG_NAME
        ),
        validation_example_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDATION_DIR_ARG_NAME
        ),
        training_years=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.TRAINING_YEARS_ARG_NAME),
            dtype=int
        ),
        validation_years=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.VALIDATION_YEARS_ARG_NAME),
            dtype=int
        ),
        lead_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.LEAD_TIMES_ARG_NAME),
            dtype=int
        ),
        satellite_predictor_names=getattr(
            INPUT_ARG_OBJECT, training_args.SATELLITE_PREDICTORS_ARG_NAME
        ),
        satellite_lag_times_minutes=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.SATELLITE_LAG_TIMES_ARG_NAME
            ),
            dtype=int
        ),
        ships_goes_predictor_names=getattr(
            INPUT_ARG_OBJECT, training_args.SHIPS_GOES_PREDICTORS_ARG_NAME
        ),
        ships_goes_lag_times_hours=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.SHIPS_GOES_LAG_TIMES_ARG_NAME
            ),
            dtype=int
        ),
        ships_forecast_predictor_names=getattr(
            INPUT_ARG_OBJECT, training_args.SHIPS_FORECAST_PREDICTORS_ARG_NAME
        ),
        ships_max_forecast_hour=getattr(
            INPUT_ARG_OBJECT, training_args.SHIPS_MAX_FORECAST_HOUR_ARG_NAME
        ),
        satellite_time_tolerance_training_sec=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_SAT_TIME_TOLERANCE_ARG_NAME
        ),
        satellite_max_missing_times_training=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_SAT_MAX_MISSING_ARG_NAME
        ),
        ships_time_tolerance_training_sec=getattr(
            INPUT_ARG_OBJECT,
            training_args.TRAINING_SHIPS_TIME_TOLERANCE_ARG_NAME
        ),
        ships_max_missing_times_training=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_SHIPS_MAX_MISSING_ARG_NAME
        ),
        satellite_time_tolerance_validation_sec=getattr(
            INPUT_ARG_OBJECT,
            training_args.VALIDATION_SAT_TIME_TOLERANCE_ARG_NAME
        ),
        ships_time_tolerance_validation_sec=getattr(
            INPUT_ARG_OBJECT,
            training_args.VALIDATION_SHIPS_TIME_TOLERANCE_ARG_NAME
        ),
        num_positive_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_POSITIVE_EXAMPLES_ARG_NAME
        ),
        num_negative_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_NEGATIVE_EXAMPLES_ARG_NAME
        ),
        max_examples_per_cyclone_in_batch=getattr(
            INPUT_ARG_OBJECT, training_args.MAX_EXAMPLES_PER_CYCLONE_ARG_NAME
        ),
        predict_td_to_ts=bool(getattr(
            INPUT_ARG_OBJECT, training_args.PREDICT_TD_TO_TS_ARG_NAME
        )),
        class_cutoffs_kt=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.CLASS_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        num_grid_rows=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_ROWS_ARG_NAME
        ),
        num_grid_columns=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_COLUMNS_ARG_NAME
        ),
        use_time_diffs_gridded_sat=bool(getattr(
            INPUT_ARG_OBJECT, training_args.USE_TIME_DIFFS_ARG_NAME
        )),
        data_aug_num_translations=getattr(
            INPUT_ARG_OBJECT, training_args.DATA_AUG_NUM_TRANS_ARG_NAME
        ),
        data_aug_max_translation_px=getattr(
            INPUT_ARG_OBJECT, training_args.DATA_AUG_MAX_TRANS_ARG_NAME
        ),
        data_aug_num_rotations=getattr(
            INPUT_ARG_OBJECT, training_args.DATA_AUG_NUM_ROTATIONS_ARG_NAME
        ),
        data_aug_max_rotation_deg=getattr(
            INPUT_ARG_OBJECT, training_args.DATA_AUG_MAX_ROTATION_ARG_NAME
        ),
        data_aug_num_noisings=getattr(
            INPUT_ARG_OBJECT, training_args.DATA_AUG_NUM_NOISINGS_ARG_NAME
        ),
        data_aug_noise_stdev=getattr(
            INPUT_ARG_OBJECT, training_args.DATA_AUG_NOISE_STDEV_ARG_NAME
        ),
        west_pacific_weight=getattr(
            INPUT_ARG_OBJECT, training_args.WEST_PACIFIC_WEIGHT_ARG_NAME
        ),
        num_epochs=getattr(INPUT_ARG_OBJECT, training_args.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_TRAINING_BATCHES_ARG_NAME
        ),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_VALIDATION_BATCHES_ARG_NAME
        ),
        plateau_lr_multiplier=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_LR_MULTIPLIER_ARG_NAME
        )
    )
