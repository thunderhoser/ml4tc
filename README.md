# ml4tc

NOTE: If you have any questions about this code, please contact me at ryan dot lagerquist at noaa dot gov.

ml4tc uses convolutional neural networks (CNN), a type of machine-learning model, to predict the intensity of tropical cyclones (TC).  ml4tc solves two specific problems:

 - Rapid intensification (RI).  The standard definition of RI is an intensity (*i.e.*, maximum sustained surface wind speed) increase of at least 30 kt over the next 24 hours, but ml4tc allows for a custom definition of RI (*e.g.*, 50 kt in the next 48 hours or whatever else you might want).
 - Intensification of tropical depression to tropical storm (TD-to-TS).  A tropical depression is a TC with intensity $<$ 34 kt, and a tropical storm is a TC with intensity $\in \left[ 34, 64 \right)$ kt.
 
Both problems are examples of binary classification, *i.e.*, predicting a yes-or-no event.  The CNN outputs a number from $\left[ 0, 1 \right]$ for each data sample (*i.e.*, each TC at each time step), which can be treated as a probability.  Depending on the problem, this number is a probability of either RI or TD-to-TS.

Inputs to the CNN (*i.e.*, predictors) come from three different sources:

 - Satellite images.  Specifically, a recent time series of TC-centered satellite images from the CIRA (Cooperative Institute for Research in the Atmosphere) IR (infrared) dataset.  These images contain data from only one channel -- *i.e.*, one wavelength -- which is around 10.8 microns.  (The exact wavelength depends on which satellite covers the TC.  CIRA IR is a global dataset, combining data from satellites around the globe.)  Letting the forecast-issue time be $t_0$, satellite images in the predictor data must come from times $t \le t_0$.  For example, the time series might be $\lbrace t_0 - 2\textrm{hours}, t_0 - 1\textrm{hour}, $t_0$ \rbrace$.  In other words, one would say that the predictors include satellite images at *lag times* of 0, 1, and 2 hours.
 - Scalar satellite data.  Specifically, a recent time series *summary statistics* for the full satellite image, also from the CIRA IR dataset.
 - SHIPS (Statistical Hurricane-intensity-prediction Scheme) developmental data.  This dataset contains scalars describing the near-storm environment, *i.e.*, the larger-scale atmospheric and oceanic environment around the TC.

To train the CNN, we need correct answers (or "labels" or "ground truth").  We obtain correct answers from the best-track dataset.  Correct answers are used to compute the loss function -- *i.e.*, the error of the CNN -- and are not used as predictors.  Since both problems (RI and TD-to-TS) are binary classification, the correct answer is always yes (1) or no (0), while the CNN predictions are real-number probabilities ranging continuously from $\left[ 0, 1 \right]$.

Documentation for important scripts, which you can run from the Unix command line, is provided below.  Please note that this package is not intended for Windows and I provide no support for Windows.  Also, though I have included many unit tests (every file ending in `_test.py`), I provide no guarantee that the code is free of bugs.  If you choose to use this package, you do so at your own risk.

# Setting up a CNN

Before training a CNN (or any model in Keras), you must set up the model.  "Setting up" includes four things: choosing the architecture, choosing the loss function, choosing the metrics (evaluation scores other than the loss function, which, in addition to the loss function, are used to monitor the model's performance after each training epoch), and compiling the model.  For the TD-to-TS problem, I have created a script that sets up my current favourite CNN (*i.e.*, my favourite architecture and loss function).  This script, which you can find in the directory `ml4tc/scripts`, is called `make_best_td_to_ts_architecture.py`.  The script will set up the model (`model_object`) and print the model's architecture in a text-only flow chart to the command window, using the command `model_object.summary()`.  If you want to save the model (which is still untrained) to a file, add the following command, replacing `output_path` with the desired file name.

`model_object.save(filepath=output_path, overwrite=True, include_optimizer=True)`

The resulting CNN predicts TD-to-TS probability for each data sample (*i.e.*, each TD at each time step) at 28 different lead times (6, 12, $\ldots$, 168 hours).  For uncertainty quantification (UQ), this CNN produces 100 different answers for each lead time -- *i.e.*, an ensemble of 100 different TD-to-TS probabilities.  To ensure that this ensemble adequately captures the uncertainty in the TD-to-TS intensification process -- *e.g.*, that the model is not extremely underdispersive or overdispersive -- this CNN is trained with the continuous ranked probability score (CRPS) as its loss function.

# Training the CNN

Once you have set up a CNN, you can train the CNN, using the script `train_neural_net.py` in the directory `ml4tc/scripts`.  Below is an example of how you would call `train_neural_net.py` from a Unix terminal.  For some input arguments I have suggested a default (where I include an actual value), and for some I have not.  In this case, the lead times are $\lbrace 6, 12, \ldots, 168 \rbrace$ hours; the lag times for satellite data (both imagery and summary statistics from the CIRA IR dataset) are $\lbrace 0, 24 \rbrace$ hours; and the lag times for SHIPS data are $\lbrace 0, 6, 12, 18, 24 \rbrace$ hours.  Thus, if the forecast issue time is 1200 UTC 2 Jan, the satellite-based predictors will come from 1200 UTC 1 Jan and 1200 UTC 2 Jan; while the SHIPS predictors will come from 1200 UTC 1 Jan, 1800 UTC 1 Jan, 0000 UTC 2 Jan, 0600 UTC 2 Jan, and 1200 UTC 2 Jan.

```
python train_neural_net.py \
    --input_template_file_name="your file name here" \
    --output_model_dir_name="your directory name here" \
    --training_example_dir_name="your directory name here" \
    --validation_example_dir_name="your directory name here" \
    --west_pacific_weight="your weight here" \
    --lead_times_hours 6 12 18 24 30 36 42 48 54 60 66 72 78 84 90 96 102 108 114 120 126 132 138 144 150 156 162 168 \
    --satellite_lag_times_minutes 0 1440 \
    --satellite_predictor_names "satellite_brightness_temp_kelvins" \
    --ships_lag_times_hours 0 6 12 18 24 \
    --ships_predictor_names_lagged "ships_goes_ch4_fraction_temp_below_m10c_50to200km" "ships_goes_ch4_fraction_temp_below_m20c_50to200km" "ships_goes_ch4_fraction_temp_below_m30c_50to200km" "ships_goes_ch4_
fraction_temp_below_m40c_50to200km" "ships_goes_ch4_fraction_temp_below_m50c_50to200km" "ships_goes_ch4_fraction_temp_below_m60c_50to200km" "ships_goes_ch4_temp_0to200km_kelvins" "ships_goes_ch4_temp_std
ev_0to200km_kelvins" "ships_goes_ch4_temp_100to300km_kelvins" "ships_goes_ch4_temp_stdev_100to300km_kelvins" "ships_goes_ch4_max_temp_0to30km_kelvins" "ships_goes_ch4_mean_temp_0to30km_kelvins" "ships_go
es_ch4_max_temp_radius_metres" "ships_goes_ch4_min_temp_20to120km_kelvins" "ships_goes_ch4_mean_temp_20to120km_kelvins" "ships_goes_ch4_min_temp_radius_metres" \
    --ships_builtin_lag_times_hours nan \
    --ships_predictor_names_forecast "ships_intensity_change_6hours_m_s01" "ships_temp_gradient_850to700mb_0to500km_k_m01" "ships_shear_850to200mb_gnrl_0to500km_no_vortex_m_s01" "ships_temp_200mb_200to800km_
kelvins" "ships_shear_850to500mb_eastward_m_s01" "ships_w_wind_0to15km_agl_0to500km_no_vortex_m_s01" "ships_ocean_age_seconds" "ships_max_tangential_wind_850mb_m_s01" "ships_intensity_m_s01" "merged_sea_
surface_temp_kelvins" "merged_ocean_heat_content_j_m02" "ships_forecast_latitude_deg_n" "ships_max_pttl_intensity_m_s01" \
    --ships_max_forecast_hour=120 \
    --satellite_time_tolerance_training_sec=43200 \
    --satellite_max_missing_times_training=0 \
    --ships_time_tolerance_training_sec=0 \
    --ships_max_missing_times_training=1 \
    --satellite_time_tolerance_validation_sec=43200 \
    --ships_time_tolerance_validation_sec=21610 \
    --num_positive_examples_per_batch=8 \
    --num_negative_examples_per_batch=8 \
    --max_examples_per_cyclone_in_batch=3 \
    --predict_td_to_ts=1 \
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
```

More details on the input arguments are provided below.

 - `training_predictor_dir_name` is a string, naming the directory with predictor files (containing brightness-temperature maps).  Files therein will be found by `example_io.find_predictor_file` and read by `example_io.read_predictor_file`, where `example_io.py` is in the directory `ml4convection/io`.  `example_io.find_predictor_file` will only look for files named like `[training_predictor_dir_name]/[yyyy]/predictors_[yyyymmdd]_radar[k].nc` and `[training_predictor_dir_name]/[yyyy]/predictors_[yyyymmdd]_radar[k].nc.gz`, where `[yyyy]` is the 4-digit year; `[yyyymmdd]` is the date; and `[k]` is the radar number, ranging from 1-3.  An example of a good file name, assuming the top-level directory is `foo`, is `foo/2016/predictors_20160101_radar1.nc`.
