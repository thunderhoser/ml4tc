#!/bin/sh

model_file_name=$1
cyclone_id_string=$2
noise_stdev=$3
output_file_name=$4

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_standalone/ml4tc"
EXAMPLE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_project/learning_examples/rotated_with_storm_motion/imputed/normalized"

echo $output_file_name

python3 -u "${CODE_DIR_NAME}/make_shapley_maps.py" \
--input_model_file_name="${model_file_name}" \
--input_example_dir_name="${EXAMPLE_DIR_NAME}" \
--baseline_cyclone_id_strings 2018WP26 1999AL14 2002WP10 2016EP18 2015EP21 2004AL11 2003AL06 2017WP03 2019WP24 2002WP04 \
--max_num_baseline_examples=50 \
--new_cyclone_id_strings ${cyclone_id_string} \
--num_smoothgrad_samples=1 \
--smoothgrad_noise_stdev=${noise_stdev} \
--output_file_name="${output_file_name}"
