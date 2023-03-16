#!/bin/sh

model_file_name=$1
cyclone_id_string=$2
noise_stdev=$3
output_file_name_sans_sample_num=$4

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_standalone/ml4tc"
EXAMPLE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_project/learning_examples/rotated_with_storm_motion/imputed/normalized"

echo $output_file_name_sans_sample_num

python3 -u "${CODE_DIR_NAME}/make_shapley_maps.py" \
--input_model_file_name="${model_file_name}" \
--input_example_dir_name="${EXAMPLE_DIR_NAME}" \
--baseline_cyclone_id_strings 2018WP26 1999AL14 2002WP10 2016EP18 2015EP21 2004AL11 2003AL06 2017WP03 2019WP24 2002WP04 2019SH17 2000AL13 2002WP18 2004WP25 2018WP30 2003EP01 2018SH20 2019AL01 2018SH03 1995AL06 2002EP01 2000AL09 1995AL09 2002AL02 1996AL05 2018SH13 2004EP14 2001AL11 2003AL19 2017EP12 2018EP23 2019WP03 2019EP11 2015WP08 2002AL12 1999AL12 2016AL11 2004WP04 2019SH26 2000AL11 2017WP25 2015CP07 2019SH15 2015WP26 2004EP10 2004WP31 2019EP09 2004WP10 2017EP15 2019WP17 \
--max_num_baseline_examples=50 \
--new_cyclone_id_strings ${cyclone_id_string} \
--num_smoothgrad_samples=100 \
--smoothgrad_noise_stdev=${noise_stdev} \
--output_file_name_sans_sample_num="${output_file_name_sans_sample_num}"
