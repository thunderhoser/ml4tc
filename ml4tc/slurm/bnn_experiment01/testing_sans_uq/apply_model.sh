#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_standalone/ml4tc"
EXAMPLE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_project/learning_examples/rotated_with_storm_motion/imputed/normalized"

echo $model_dir_name

model_file_name="${model_dir_name}/model.h5"
output_dir_name="testing"

python3 -u "${CODE_DIR_NAME}/apply_neural_net.py" \
--input_model_file_name="${model_file_name}" \
--input_example_dir_name="${EXAMPLE_DIR_NAME}" \
--years 2010 2011 2012 2013 2014 \
--output_dir_name="${output_dir_name}"
