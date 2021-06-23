"""Trains neural net."""

import argparse
import numpy
from ml4tc.machine_learning import neural_net
from ml4tc.scripts import training_args

KT_TO_METRES_PER_SECOND = 1.852 / 3.6

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _run(template_file_name, output_dir_name, training_example_dir_name,
         validation_example_dir_name, training_years, validation_years,
         lead_time_hours, satellite_lag_times_minutes, ships_lag_times_hours,
         satellite_predictor_names,
         ships_predictor_names_lagged, ships_predictor_names_forecast,
         satellite_time_tolerance_training_sec,
         satellite_max_missing_times_training,
         ships_time_tolerance_training_sec, ships_max_missing_times_training,
         satellite_time_tolerance_validation_sec,
         ships_time_tolerance_validation_sec,
         num_positive_examples_per_batch,
         num_negative_examples_per_batch, max_examples_per_cyclone_in_batch,
         class_cutoffs_kt, num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, plateau_lr_multiplier):
    """Trains neural net.

    This is effectively the main method.

    :param template_file_name: See documentation at top of training_args.py.
    :param output_dir_name: Same.
    :param training_example_dir_name: Same.
    :param validation_example_dir_name: Same.
    :param training_years: Same.
    :param validation_years: Same.
    :param lead_time_hours: Same.
    :param satellite_lag_times_minutes: Same.
    :param ships_lag_times_hours: Same.
    :param satellite_predictor_names: Same.
    :param ships_predictor_names_lagged: Same.
    :param ships_predictor_names_forecast: Same.
    :param satellite_time_tolerance_training_sec: Same.
    :param satellite_max_missing_times_training: Same.
    :param ships_time_tolerance_training_sec: Same.
    :param ships_max_missing_times_training: Same.
    :param satellite_time_tolerance_validation_sec: Same.
    :param ships_time_tolerance_validation_sec: Same.
    :param num_positive_examples_per_batch: Same.
    :param num_negative_examples_per_batch: Same.
    :param max_examples_per_cyclone_in_batch: Same.
    :param class_cutoffs_kt: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param plateau_lr_multiplier: Same.
    """

    training_option_dict = {
        neural_net.EXAMPLE_DIRECTORY_KEY: training_example_dir_name,
        neural_net.YEARS_KEY: training_years,
        neural_net.LEAD_TIME_KEY: lead_time_hours,
        neural_net.SATELLITE_LAG_TIMES_KEY: satellite_lag_times_minutes,
        neural_net.SHIPS_LAG_TIMES_KEY: ships_lag_times_hours,
        neural_net.SATELLITE_PREDICTORS_KEY: satellite_predictor_names,
        neural_net.SHIPS_PREDICTORS_LAGGED_KEY: ships_predictor_names_lagged,
        neural_net.SHIPS_PREDICTORS_FORECAST_KEY:
            ships_predictor_names_forecast,
        neural_net.NUM_POSITIVE_EXAMPLES_KEY: num_positive_examples_per_batch,
        neural_net.NUM_NEGATIVE_EXAMPLES_KEY: num_negative_examples_per_batch,
        neural_net.MAX_EXAMPLES_PER_CYCLONE_KEY:
            max_examples_per_cyclone_in_batch,
        neural_net.CLASS_CUTOFFS_KEY:
            class_cutoffs_kt * KT_TO_METRES_PER_SECOND,
        neural_net.SATELLITE_TIME_TOLERANCE_KEY:
            satellite_time_tolerance_training_sec,
        neural_net.SATELLITE_MAX_MISSING_TIMES_KEY:
            satellite_max_missing_times_training,
        neural_net.SHIPS_TIME_TOLERANCE_KEY: ships_time_tolerance_training_sec,
        neural_net.SHIPS_MAX_MISSING_TIMES_KEY: ships_max_missing_times_training
    }

    validation_option_dict = {
        neural_net.EXAMPLE_DIRECTORY_KEY: validation_example_dir_name,
        neural_net.YEARS_KEY: validation_years,
        neural_net.SATELLITE_TIME_TOLERANCE_KEY:
            satellite_time_tolerance_validation_sec,
        neural_net.SHIPS_TIME_TOLERANCE_KEY: ships_time_tolerance_validation_sec
    }

    print('Reading model template from: "{0:s}"...'.format(template_file_name))
    model_object = neural_net.read_model(hdf5_file_name=template_file_name)

    neural_net.train_model(
        model_object=model_object, output_dir_name=output_dir_name,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        do_early_stopping=True,
        plateau_lr_multiplier=plateau_lr_multiplier
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
        lead_time_hours=getattr(
            INPUT_ARG_OBJECT, training_args.LEAD_TIME_ARG_NAME
        ),
        satellite_lag_times_minutes=numpy.array(
            getattr(
                INPUT_ARG_OBJECT, training_args.SATELLITE_LAG_TIMES_ARG_NAME
            ),
            dtype=int
        ),
        ships_lag_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.SHIPS_LAG_TIMES_ARG_NAME),
            dtype=int
        ),
        satellite_predictor_names=getattr(
            INPUT_ARG_OBJECT, training_args.SATELLITE_PREDICTORS_ARG_NAME
        ),
        ships_predictor_names_lagged=getattr(
            INPUT_ARG_OBJECT, training_args.SHIPS_PREDICTORS_LAGGED_ARG_NAME
        ),
        ships_predictor_names_forecast=getattr(
            INPUT_ARG_OBJECT, training_args.SHIPS_PREDICTORS_FORECAST_ARG_NAME
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
        class_cutoffs_kt=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.CLASS_CUTOFFS_ARG_NAME),
            dtype=float
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
