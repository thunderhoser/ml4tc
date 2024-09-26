#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_standalone/ml4tc"

model_file_name="${model_dir_name}/isotonic_regression/isotonic_regression.dill"
input_prediction_file_name="${model_dir_name}/training/predictions.nc"
output_prediction_file_name="${model_dir_name}/training/isotonic_regression/predictions.nc"

python3 -u "${CODE_DIR_NAME}/apply_isotonic_regression.py" \
--input_prediction_file_name="${input_prediction_file_name}" \
--input_model_file_name="${model_file_name}" \
--output_prediction_file_name="${output_prediction_file_name}"
