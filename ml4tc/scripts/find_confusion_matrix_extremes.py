"""Finds confusion-matrix extremes in prediction set.

There are 4 types of confusion-matrix extremes: best hits, worst false alarms,
worst misses, and best correct nulls.

This script also finds 2 more types of extremes: low-probability and high-
probability examples, regardless of the label (correct answer)
"""

import copy
import argparse
import numpy
from gewittergefahr.deep_learning import model_activation as gg_model_activation
from ml4tc.io import prediction_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
SUBSET_SIZE_ARG_NAME = 'num_examples_per_subset'
UNIQUE_CYCLONES_ARG_NAME = 'enforce_unique_cyclones'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and targets (correct answers) '
    'for many examples.  Will be read by `prediction_io.read_file`.'
)
SUBSET_SIZE_HELP_STRING = (
    'Number of examples in each subset (best hits, worst false alarms, worst '
    'misses, best correct nulls).'
)
UNIQUE_CYCLONES_HELP_STRING = (
    'Boolean flag.  If 1, will ensure that each subset contains only unique '
    'cyclones.  If 0, will allow non-unique cyclones (i.e., multiple time '
    'steps from the same cyclone) in the same subset.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  For each subset, one file will be written to '
    'this directory by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SUBSET_SIZE_ARG_NAME, type=int, required=True,
    help=SUBSET_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + UNIQUE_CYCLONES_ARG_NAME, type=int, required=True,
    help=UNIQUE_CYCLONES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_prediction_file_name, num_examples_per_subset,
         enforce_unique_cyclones, output_dir_name):
    """Finds confusion-matrix extremes in prediction set.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param num_examples_per_subset: Same.
    :param enforce_unique_cyclones: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if multiple lead times are found in prediction file.
    """

    print('Reading data from: "{0:s}"...'.format(input_prediction_file_name))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)
    mean_forecast_prob_matrix = prediction_io.get_mean_predictions(
        prediction_dict
    )

    if mean_forecast_prob_matrix.shape[1] > 1:
        error_string = (
            'This script works for predictions with only one lead time.  Found '
            '{0:d} lead times.'
        ).format(mean_forecast_prob_matrix.shape[1])

        raise ValueError(error_string)

    mean_forecast_probs = mean_forecast_prob_matrix[:, 0]

    index_dict = gg_model_activation.get_contingency_table_extremes(
        storm_activations=mean_forecast_probs,
        storm_target_values=
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][:, 0],
        num_hits=num_examples_per_subset,
        num_misses=num_examples_per_subset,
        num_false_alarms=num_examples_per_subset,
        num_correct_nulls=num_examples_per_subset,
        unique_storm_cells=enforce_unique_cyclones,
        full_storm_id_strings=prediction_dict[prediction_io.CYCLONE_IDS_KEY]
    )

    best_hit_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=index_dict[gg_model_activation.HIT_INDICES_KEY]
    )
    best_hit_file_name = '{0:s}/predictions_best_hits.nc'.format(
        output_dir_name
    )
    d = best_hit_prediction_dict

    print(SEPARATOR_STRING)
    print('Forecast probabilities for best hits:')
    for k in index_dict[gg_model_activation.HIT_INDICES_KEY]:
        print('{0:.4f}'.format(mean_forecast_probs[k]))

    print('Writing {0:d} examples to: "{1:s}"...'.format(
        len(index_dict[gg_model_activation.HIT_INDICES_KEY]),
        best_hit_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=best_hit_file_name,
        forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
        target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
        cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
        init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
        storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
        storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
        storm_intensity_changes_m_s01=
        d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
        model_file_name=d[prediction_io.MODEL_FILE_KEY],
        lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
        quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
        uncertainty_calib_model_file_name=
        d[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
    )

    worst_false_alarm_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=index_dict[gg_model_activation.FALSE_ALARM_INDICES_KEY]
    )
    worst_false_alarm_file_name = (
        '{0:s}/predictions_worst_false_alarms.nc'
    ).format(output_dir_name)

    d = worst_false_alarm_prediction_dict

    print(SEPARATOR_STRING)
    print('Forecast probabilities for worst false alarms:')
    for k in index_dict[gg_model_activation.FALSE_ALARM_INDICES_KEY]:
        print('{0:.4f}'.format(mean_forecast_probs[k]))

    print('Writing {0:d} examples to: "{1:s}"...'.format(
        len(index_dict[gg_model_activation.FALSE_ALARM_INDICES_KEY]),
        worst_false_alarm_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=worst_false_alarm_file_name,
        forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
        target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
        cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
        init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
        storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
        storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
        storm_intensity_changes_m_s01=
        d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
        model_file_name=d[prediction_io.MODEL_FILE_KEY],
        lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
        quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
        uncertainty_calib_model_file_name=
        d[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
    )

    worst_miss_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=index_dict[gg_model_activation.MISS_INDICES_KEY]
    )
    worst_miss_file_name = '{0:s}/predictions_worst_misses.nc'.format(
        output_dir_name
    )
    d = worst_miss_prediction_dict

    print(SEPARATOR_STRING)
    print('Forecast probabilities for worst misses:')
    for k in index_dict[gg_model_activation.MISS_INDICES_KEY]:
        print('{0:.4f}'.format(mean_forecast_probs[k]))

    print('Writing {0:d} examples to: "{1:s}"...'.format(
        len(index_dict[gg_model_activation.MISS_INDICES_KEY]),
        worst_miss_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=worst_miss_file_name,
        forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
        target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
        cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
        init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
        storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
        storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
        storm_intensity_changes_m_s01=
        d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
        model_file_name=d[prediction_io.MODEL_FILE_KEY],
        lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
        quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
        uncertainty_calib_model_file_name=
        d[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
    )

    best_correct_null_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=index_dict[gg_model_activation.CORRECT_NULL_INDICES_KEY]
    )
    best_correct_null_file_name = (
        '{0:s}/predictions_best_correct_nulls.nc'
    ).format(output_dir_name)

    d = best_correct_null_prediction_dict

    print(SEPARATOR_STRING)
    print('Forecast probabilities for best correct nulls:')
    for k in index_dict[gg_model_activation.CORRECT_NULL_INDICES_KEY]:
        print('{0:.4f}'.format(mean_forecast_probs[k]))

    print('Writing {0:d} examples to: "{1:s}"...'.format(
        len(index_dict[gg_model_activation.CORRECT_NULL_INDICES_KEY]),
        best_correct_null_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=best_correct_null_file_name,
        forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
        target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
        cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
        init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
        storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
        storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
        storm_intensity_changes_m_s01=
        d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
        model_file_name=d[prediction_io.MODEL_FILE_KEY],
        lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
        quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
        uncertainty_calib_model_file_name=
        d[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
    )

    high_prob_indices, low_prob_indices = (
        gg_model_activation.get_hilo_activation_examples(
            storm_activations=mean_forecast_probs,
            num_high_activation_examples=num_examples_per_subset,
            num_low_activation_examples=num_examples_per_subset,
            unique_storm_cells=enforce_unique_cyclones,
            full_storm_id_strings=prediction_dict[prediction_io.CYCLONE_IDS_KEY]
        )
    )

    low_prob_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=low_prob_indices
    )
    low_prob_file_name = '{0:s}/predictions_low_prob.nc'.format(output_dir_name)
    d = low_prob_prediction_dict

    print(SEPARATOR_STRING)
    print('Forecast probabilities for low-prob cases:')
    for k in low_prob_indices:
        print('{0:.4f}'.format(mean_forecast_probs[k]))

    print('Writing {0:d} examples to: "{1:s}"...'.format(
        len(low_prob_indices), low_prob_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=low_prob_file_name,
        forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
        target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
        cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
        init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
        storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
        storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
        storm_intensity_changes_m_s01=
        d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
        model_file_name=d[prediction_io.MODEL_FILE_KEY],
        lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
        quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
        uncertainty_calib_model_file_name=
        d[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
    )

    high_prob_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=high_prob_indices
    )
    high_prob_file_name = '{0:s}/predictions_high_prob.nc'.format(
        output_dir_name
    )
    d = high_prob_prediction_dict

    print(SEPARATOR_STRING)
    print('Forecast probabilities for high-prob cases:')
    for k in high_prob_indices:
        print('{0:.4f}'.format(mean_forecast_probs[k]))

    print('Writing {0:d} examples to: "{1:s}"...'.format(
        len(high_prob_indices), high_prob_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=high_prob_file_name,
        forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
        target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
        cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
        init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
        storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
        storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
        storm_intensity_changes_m_s01=
        d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
        model_file_name=d[prediction_io.MODEL_FILE_KEY],
        lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
        quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
        uncertainty_calib_model_file_name=
        d[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
    )

    positive_event_indices = numpy.where(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][:, 0] == 1
    )[0]
    negative_event_indices = numpy.where(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][:, 0] == 0
    )[0]

    positive_event_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=positive_event_indices
    )
    positive_event_file_name = '{0:s}/predictions_positive_events.nc'.format(
        output_dir_name
    )
    d = positive_event_prediction_dict

    print('Writing {0:d} examples to: "{1:s}"...'.format(
        len(positive_event_indices), positive_event_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=positive_event_file_name,
        forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
        target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
        cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
        init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
        storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
        storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
        storm_intensity_changes_m_s01=
        d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
        model_file_name=d[prediction_io.MODEL_FILE_KEY],
        lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
        quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
        uncertainty_calib_model_file_name=
        d[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
    )

    negative_event_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=negative_event_indices
    )
    negative_event_file_name = '{0:s}/predictions_negative_events.nc'.format(
        output_dir_name
    )
    d = negative_event_prediction_dict

    print('Writing {0:d} examples to: "{1:s}"...'.format(
        len(negative_event_indices), negative_event_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=negative_event_file_name,
        forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
        target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
        cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
        init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
        storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
        storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
        storm_intensity_changes_m_s01=
        d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
        model_file_name=d[prediction_io.MODEL_FILE_KEY],
        lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
        quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
        uncertainty_calib_model_file_name=
        d[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        num_examples_per_subset=getattr(INPUT_ARG_OBJECT, SUBSET_SIZE_ARG_NAME),
        enforce_unique_cyclones=bool(
            getattr(INPUT_ARG_OBJECT, UNIQUE_CYCLONES_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
