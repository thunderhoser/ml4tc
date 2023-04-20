#!/bin/sh

template_file_name=$1
output_dir_name=$2

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_standalone/ml4tc"

TRAINING_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_project/learning_examples/rotated_with_storm_motion/imputed/normalized"
VALIDATION_DIR_NAME="${TRAINING_DIR_NAME}"

echo $output_dir_name

python3 -u "${CODE_DIR_NAME}/train_neural_net.py" \
--input_template_file_name="${template_file_name}" \
--output_model_dir_name="${output_dir_name}" \
--training_example_dir_name="${TRAINING_DIR_NAME}" \
--validation_example_dir_name="${VALIDATION_DIR_NAME}" \
--lead_times_hours 24 \
--satellite_lag_times_minutes 0 \
--satellite_predictor_names "satellite_brightness_temp_kelvins" \
--use_time_diffs_gridded_sat=0 \
--ships_lag_times_hours 0 6 12 18 24 \
--ships_predictor_names_lagged "" \
--ships_builtin_lag_times_hours nan \
--ships_predictor_names_forecast "ships_forecast_solar_zenith_angle_deg" "ships_intensity_change_6hours_m_s01" "ships_temp_gradient_850to700mb_0to500km_k_m01" "ships_shear_850to200mb_gnrl_0to500km_no_vortex_m_s01" "ships_temp_200mb_200to800km_kelvins" "ships_shear_850to500mb_eastward_m_s01" "ships_w_wind_0to15km_agl_0to500km_no_vortex_m_s01" "ships_ocean_age_seconds" "ships_max_tangential_wind_850mb_m_s01" "ships_intensity_m_s01" "merged_sea_surface_temp_kelvins" "merged_ocean_heat_content_j_m02" "ships_forecast_latitude_deg_n" "ships_max_pttl_intensity_m_s01" \
--ships_max_forecast_hour=24 \
--satellite_time_tolerance_training_sec=7200 \
--satellite_max_missing_times_training=0 \
--ships_time_tolerance_training_sec=0 \
--ships_max_missing_times_training=1 \
--satellite_time_tolerance_validation_sec=7200 \
--ships_time_tolerance_validation_sec=21610 \
--num_positive_examples_per_batch=4 \
--num_negative_examples_per_batch=12 \
--max_examples_per_cyclone_in_batch=3 \
--class_cutoffs_kt 30 \
--num_epochs=1000 \
--num_training_batches_per_epoch=32 \
--num_validation_batches_per_epoch=16 \
--data_aug_num_translations=2 \
--data_aug_max_translation_px=6 \
--data_aug_num_rotations=2 \
--data_aug_max_rotation_deg=50 \
--data_aug_num_noisings=3 \
--data_aug_noise_stdev=0.5 \
--plateau_lr_multiplier=0.6
