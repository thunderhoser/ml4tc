#!/bin/csh

set CODE_DIR_NAME="/home/ralager/ml4tc/ml4tc"
set EXAMPLE_DIR_NAME="/home/ralager/condo/swatwork/ralager/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_project/learning_examples/rotated_with_storm_motion/imputed/normalized"
set MODEL_DIR_NAME="/home/ralager/condo/swatwork/ralager/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_models/crps_experiment02_predictor_types/dropout-rates=0.100-0.500-0.300_num-satellite-lag-times=1_num-ships-forecast-predictors=00_satellite-use-temporal_diffs=0"

set noise_stdev="1.0"
set cyclone_id_string="2005AL12"
set model_file_name="${MODEL_DIR_NAME}/model.h5"

set j=1

while ($j <= 9)
    set j_string=`printf "%03d" $j`
    set output_file_name="${MODEL_DIR_NAME}/smoothgrad_shapley_experiment/noise-stdev=${noise_stdev}/shapley_maps_${cyclone_id_string}_sample${j_string}.nc"
    
    ~/anaconda3/bin/python3.7 -u "${CODE_DIR_NAME}/scripts/make_shapley_maps.py" \
    --input_model_file_name="${model_file_name}" \
    --input_example_dir_name="${EXAMPLE_DIR_NAME}" \
    --baseline_cyclone_id_strings 2015AL09 2015CP02 2015EP16 2015WP10 2015WP19 \
    --new_cyclone_id_strings 2005AL12 \
    --num_smoothgrad_samples=1 \
    --smoothgrad_noise_stdev=${noise_stdev} \
    --output_file_name="${output_file_name}"
    
    @ j = $j + 1
end
