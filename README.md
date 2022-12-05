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

# Training a CNN

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

 - `input_template_file_name` is a string, the path to an HDF5 file containing an untrained CNN.  See "Setting up a CNN," above, for instructions on creating this file.
 - `output_model_dir_name` is a string, the path to the directory where you want the trained model to be saved.
 - `training_example_dir_name` is a string, the path to the directory with training data (both predictors and ground truth).  Files therein will be found by `example_io.find_file` and read by `example_io.read_file`, where `example_io.py` is in the directory `ml4tc/io`.  `example_io.find_file` will only look for files named like `[training_example_dir_name]/learning_examples_[cyclone-id].nc` and `[training_example_dir_name]/learning_examples_[cyclone-id].nc`.  An example of a valid file name, assuming the top-level directory is `foo`, is `foo/learning_examples_2005AL12.nc`.  NOTE: Although ml4tc is an end-to-end library and therefore includes pre-processing code to create such training file, this documentation does not provide details on the pre-processing code.  Pre-processing is a tricky business, and I recommend that you just use files I have already created (available from me upon request).
 - `validation_example_dir_name`: Same as above but for validation data.
 - `west_pacific_weight` is a real number (float) $>$ 1, used to weight TCs in the western Pacific more heavily than TCs in other basins.  For applications where you care only about the western Pacific, you might ask: "Why train the model with data from anywhere else, then?"  The answer is that training with TCs from all basins increases the size of the dataset, which is small because there are not many TCs in a given year.  Many relationships learned by the CNN will generalize across basins.  However, to ensure that the CNN still cares *most* about the western Pacific, this weight emphasizes western-Pacific TCs in the loss function.  All TCs in another basin receive a weight of 1.0.
 - `lead_times_hours` is the set of lead times.  The same CNN makes predictions at all lead times.  For RI, there is usually one lead time (24 hours).  But for TD-to-TS, there are usually multiple lead times.  The specific question being answered by the CNN, with a probabilistic prediction, is: "Will the TD intensity to TS strength at *any* time in the next $K$ hours?"  Thus, as lead time increases, the correct answer can flip from no (0) to yes (1), but it can never flip from yes (1) to no (0).  For example, if the TD intensifies to TS strength at a lead time of 12 hours, the correct answer for all lead times $\ge$ 12 hours is yes, even if the TS subsequently dissipates at 60 hours.  Because the correct answer is non-decreasing with lead time, the CNN probabilities are also non-decreasing with lead time.  The CNN's architecture ensures this, using a "trick" that is beyond the scope of this documentation.
 - `satellite_lag_times_minutes`: Satellite data (both images and summary statistics, from the CIRA IR dataset) at these lag times are used in the predictors.
 - `satellite_predictor_names`: List of satellite-based predictors to use.  The default argument provided above is a single-item list with "satellite_brightness_temp_kelvins," meaning that only the images (containing gridded brightness temperatures) are used, while none of the summary statistics are used.
 - `ships_lag_times_hours`: SHIPS data at these lag times are used in the predictors.
 - `ships_predictor_names_lagged`: Some SHIPS variables are available at multiple forecast times (mainly those forecast by the GFS [Global Forecast System] weather model), and some are available at multiple lag times.  This argument is a list of lagged SHIPS predictors to use.
 - `ships_builtin_lag_times_hours` is a list of lag times to use for the lagged SHIPS variables.  The default argument provided above is a single-item list with "nan" (not a number).  "nan" means that, for each lagged predictor, the most recent time will be used.  I strongly recommend always using "nan".
 - `ships_predictor_names_forecast`: List of forecast SHIPS variables to use.
 - `ships_max_forecast_hour`: Max forecast hour to use for the forecast SHIPS variables.  A WORD OF CAUTION: In the archived SHIPS files, the "forecasts" are not actually forecasts; they are actually 0-hour analyses.  For example, the "120-hour forecast" is actually the 0-hour analysis at 120-hour lead time.  In other words, using any forecast hour $>$ 0 for SHIPS variables is tantamount to leaking future information to the CNN, *i.e.*, cheating.  However, I've been told by tropical-cyclone experts that at lead times up to $\sim$5 days, the true forecasts and fake forecasts (0-hour analyses) should not be *too* different.  But also, for some variables my code uses only 0-hour forecasts, because having future information on these variables makes the RI or TD-to-TS problem too easy:
   - Any variable related to storm intensity (current intensity, change over last 6 hours, change over last 12 hours, etc.)
   - Central pressure (*i.e.*, minimum sea-level pressure inside TC)
   - Storm type (tropical depression, tropical storm, extratropical, etc.)
   - Z850 (average 850-mb vorticity from 0-1000 km outside TC center)
   - D200 (average 200-mb divergence from 0-1000 km outside TC center)
   - PENC (surface pressure at vortex edge)
   - DIVC (vortex-centered version of D200)
   - PENV (surface pressure averaged from 200-800 km outside TC center)
   - HE07 (storm-relative helicity averaged from 200-800 km outside TC center and 700-1000 mb)
   - HE05 (same as HE07 but from 500-1000 mb)
 - `satellite_time_tolerance_training_sec`: The CIRA IR dataset does not have a consistent time interval.  Sometimes the interval is 10 min; sometimes it is 15 min; and sometimes it is much longer, due to missing data.  Thus, you might not get satellite data at the desired lag times.  This argument is a tolerance.  For each desired lag time $t_{\textrm{lag}}$, my code will find satellite data at the nearest time $t$ and use said data in the predictors, unless the absolute difference between $t_{\textrm{lag}}$ and $t$ exceeds this tolerance.
 - `satellite_max_missing_times_training`: Tolerance for missing lag times in satellite data.  For a given data sample $S$, if the number of missing lag times is $>$ `satellite_max_missing_times_training`, data sample $S$ will not be used for training.
 - `ships_time_tolerance_training_sec`: Same as `satellite_time_tolerance_training_sec` but for SHIPS data.
 - `ships_max_missing_times_training`: Same as `satellite_max_missing_times_training` but for SHIPS data.
 - `satellite_time_tolerance_validation_sec`: Same as `satellite_time_tolerance_training_sec` but for validation data.  NOTE: This option is not available for post-training evaluation on testing data, which are meant to simulate model deployment in the real world.  In the real world, missing data happen and you cannot simply refuse to make predictions when there is missing data.
 - `ships_time_tolerance_validation_sec`: Same as `ships_time_tolerance_training_sec` but for validation data.
 - `num_positive_examples_per_batch`: Number of positive examples (with correct answer = yes = 1) per batch of data samples.
 - `num_negative_examples_per_batch`: Number of negative examples (with correct answer = yes = 0) per batch of data samples.
 - `max_examples_per_cyclone_in_batch`: Max number of data samples for one TC in the same batch.  In other words, max number of time steps (snapshots) for one TC in the same batch.  Keeping this value lower than `num_positive_examples_per_batch` and `num_negative_examples_per_batch` ensures that each batch will contain a diversity of data samples, coming from many different TCs, thus avoiding a large amount of temporal autocorrelation.
 - `predict_td_to_ts`: Boolean flag.  If 1 (0), the CNN will be trained to predict TD-to-TS (RI).
 - `num_epochs`: Number of training epochs.  If the number of epochs is extremely high (*e.g.*, 1000), early stopping will almost definitely stop training long before 1000 epochs.
 - `num_training_batches_per_epoch`: Number of training batches in each epoch.
 - `num_validation_batches_per_epoch`: Number of validation batches in each epoch.
 - `data_aug_num_translations`: This and the next 5 arguments all pertain to data augmentation.  Data augmentation is the practice of randomly perturbing the predictor data while assuming that the correct answer remains the same.  This increases the size of the dataset and often improves model performance -- even on validation data, which are not perturbed.  Data augmentation affects only the training data.  `data_aug_num_translations` is the number of random translations to apply to each data sample (affects satellite images only).
 - `data_aug_max_translation_px`: Maximum translation (number of pixels) for satellite images.  Note that grid spacing is 4 km, *i.e.*, each pixel is 4 $\times$ 4 km.  The actual translation distance in both the $x$- and $y$-directions will be sampled from a uniform distribution over $\left[ -\textrm{data-aug-max-translation-px}, \textrm{data-aug-max-translation-px} \right]$ for each data sample.
 - `data_aug_num_rotations`: Number of random rotations to apply to each data sample (affects satellite images only).
 - `data_aug_max_rotation_deg`: Maximum rotation (degrees) for satellite images.  The actual rotation angle will be sampled from a uniform distribution over $\left[ -\textrm{data-aug-num-rotations}, \textrm{data-aug-num-rotations} \right]$ for each data sample.
 - `data_aug_num_noisings`: Number of times to add Gaussian noise to each data sample (applies to all predictors, not just satellite images).
 - `data_aug_noise_stdev`: Standard deviation of Gaussian noise, in normalized units.  Note that all predictors are transformed to $z$-scores (number of standard deviations away from the mean) before training the CNN, so every predictor has the same (non-physical) scale, ranging from about $\left[ -3, 3 \right]$.
 - `plateau_lr_multiplier`: Multiplier used to reduce learning rate if validation loss has not improved over 10 epochs.

# Inference mode: Applying a trained CNN

Once you have trained a CNN, you can use it to make predictions on new data.  This is called the "inference phase," as opposed to the "training phase".  You can do this with the script `apply_neural_net.py` in the directory `ml4tc/scripts`.  Below is an example of how you would call `apply_neural_net.py` from a Unix terminal.

```
python apply_neural_net.py \
    --input_model_file_name="file with trained model" \
    --input_example_dir_name="your directory name here" \
    --years 2010 2011 2012 2013 2014 \  # This is just an example, not a suggested default.
    --num_dropout_iterations=100 \  # This is just an example, not a suggested default.
    --use_quantiles=[0 or 1] \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_model_file_name` is a string, containing the full path to the trained model.  This file will be read by `neural_net.read_model`.
 - `input_example_dir_name` is a string, naming the directory with input data.  For more details, see documentation for the inputs `training_example_dir_name` and `validation_example_dir_name` to `train_neural_net.py`.
 - `years` is a list of years.  `apply_neural_net.py` will make predictions only for data samples in these years.
 - `num_dropout_iterations`: This argument is used only for CNNs trained with Monte Carlo dropout.  In this case, it specifies how many times to run the CNN for each data sample.  If `num_dropout_iterations` is $D$, then `apply_neural_net.py` will create an ensemble of $D$ predictions ($D$ probabilities) for each data sample.  The full ensemble will be written to the output file.
 - `use_quantiles`: This argument is used only for CNNs trained with quantile regression.  It is a Boolean flag.  If the flag is 0 (`False`), `apply_neural_net.py` will create only one deterministic prediction for each data sample.  If the flag is 1 (`True`), `apply_neural_net.py` will create $Q$ quantile-based estimates for each data sample -- where $Q$ is the number of quantile levels -- as well as a deterministic mean prediction.  All these predictions (mean and quantiles) will be written to the output file.
 - `output_dir_name` is a string, naming the output directory.  CNN predictions (along with corresponding targets [correct answers]) will be written here in NetCDF format.

# Plotting predictions and inputs (TD-to-TS only)

Once you have run `apply_neural_net.py` to make predictions, you can plot the predictions with the script `plot_predictions_with_gridsat.py` in the directory `ml4tc/scripts`.  Below is an example of how you would call `plot_predictions_with_gridsat.py` from a Unix terminal.  This will plot figures like the one shown below, containing GridSat data in the background.  You can download GridSat data easily from here: https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/

```
python plot_predictions_with_gridsat.py \
    --input_model_metafile_name="file with metadata for trained model" \
    --input_prediction_file_name="file with predictions from trained model, created by `apply_neural_net.py`" \
    --input_gridsat_dir_name="directory with GridSat data" \
    --confidence_level=0.95 \  # This default will produce 95% confidence intervals.
    --min_latitude_deg_n=0 \
    --max_latitude_deg_n=60 \
    --min_longitude_deg_e=105 \
    --max_longitude_deg_e=179.999999 \
    --first_init_time_string="2005-11-09-00" \  # This is just an example, not a suggested default.
    --last_init_time_string="2005-11-16-18" \  # This is just an example, not a suggested default.
    --output_dir_name="your directory name here" \
```

 - `input_model_metafile_name` is a string, containing the full path to the metadata file for the trained model.  This file will be read by `neural_net.read_metafile`.
 - `input_prediction_file_name` is a string, containing the full path to a prediction file for the same model, created by `apply_neural_net.py`.
 - `input_gridsat_dir_name` is a string, naming the directory with GridSat data.  When you download GridSat data from the above website, put the files in this directory (no subdirectories) and do not rename the files.  As long as you do this, `plot_predictions_with_gridsat.py` will find the GridSat files.
 - `confidence_level`: For models with uncertainty quantification (via, *e.g.*, Monte Carlo dropout or quantile regression), this number tells `plot_predictions_with_gridsat.py` the width of the confidence interval to plot for the forecast TD-to-TS probabilities.  In the above example, the confidence level is 0.95 or 95%.
 - `min_latitude_deg_n`: Minimum latitude in left panel, in degrees north.
 - `max_latitude_deg_n`: Max latitude in left panel, in degrees north.
 - `min_longitude_deg_e`: Minimum longitude in left panel, in degrees east.
 - `max_longitude_deg_e`: Max longitude in left panel, in degrees east.
 - `first_init_time_string`: First time to plot.  `plot_predictions_with_gridsat.py` will create a figure for every 6-hour time step between `first_init_time_string` and `last_init_time_string`.
 - `last_init_time_string`: See documentation above for `first_init_time_string`.
 - `output_dir_name` is a string, naming the output directory.  Figures will be saved here in JPEG format.

# Evaluating a trained CNN: mean predictions only (not uncertainty estimates)

Below is an example of how you would call `evaluate_model.py` from a Unix terminal.

```
python evaluate_model.py \
    --input_prediction_file_name="your file name here" \
    --lead_times_hours 6 12 18 24 30 36 42 48 54 60 66 72 78 84 90 96 102 108 114 120 126 132 138 144 150 156 162 168 \
    --event_freq_in_training=0.50886938 \
    --num_prob_thresholds=1001 \
    --num_reliability_bins=20 \
    --num_bootstrap_reps=1000 \
    --output_eval_file_name="your file name here"
```

More details on the input arguments are provided below.

 - `input_prediction_file_name` is a string, containing the full path to a prediction file created by `apply_neural_net.py`.
 - `lead_times_hours` is the set of lead times at which to evaluate the model.  In the example above, `evaluate_model.py` is told to evaluate the model at all lead times (6, 12, ..., 128 hours).  However, if you'd like you can evaluate at just one lead times -- or just a few lead times.
 - `event_freq_in_training`: Event frequency in training data.  This is used to create the "climatological model," which is used to compute Brier skill score.  In the example above, the event frequency is 0.50886938 or 50.886 938%, which means that TD-to-TS occurs for 50.886 938% of data samples (where one data sample is one tropical depression at one time), averaged over all lead times.  For the 28 lead times in the above list, the training event frequencies are: 0.15196328, 0.27001530, 0.35135135, 0.40566038, 0.43982662, 0.46736359, 0.48878123, 0.50458950, 0.51835798, 0.52830189, 0.53646099, 0.54258032, 0.54793473, 0.55277919, 0.55609383, 0.55864355, 0.56093830, 0.56348802, 0.56527282, 0.56731260, 0.56884243, 0.57011729, 0.57113717, 0.57190209, 0.57215706, 0.57215706, 0.57215706, 0.57215706.  In other words, 57.215 706% of TD samples intensify to TS strength within the next 168 hours.
 - `num_prob_thresholds`: Number of probability thresholds, used to create receiver operating characteristic (ROC) curve and performance diagram.  I suggest leaving this at 1001.
 - `num_reliability_bins`: Number of bins for attributes diagram, which is a fancy reliability curve.  I suggest leaving this at 10.
 - `num_bootstrap_reps`: Number of replicates for bootstrapping, used to create confidence intervals in the ROC curve and performance diagram and attributes diagram.  If you do not want confidence intervals, make this 1.
 - `output_eval_file_name` is a string, containing the full path to the output file.  Model-evaluation results will be written here in NetCDF format.

# Evaluating a trained CNN: uncertainty estimates only (not mean predictions)

Below is an example of how you would call `compute_spread_vs_skill.py` from a Unix terminal, to compute the spread-skill plot.

```
python compute_spread_vs_skill.py \
    --input_prediction_file_name="your file name here" \
    --bin_edge_prediction_stdevs 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.40 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.50 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.60 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 \
    --lead_times_hours 6 12 18 24 30 36 42 48 54 60 66 72 78 84 90 96 102 108 114 120 126 132 138 144 150 156 162 168 \
    --output_file_name="your file name here"
```

More details on the input arguments are provided below.

 - `input_prediction_file_name` is a string, containing the full path to a prediction file created by `apply_neural_net.py`.
 - `bin_edge_prediction_stdevs` is the set of standard deviations to use as bin edges for the spread-skill plot.  These are the standard deviations of the CNN's predicted distribution, *i.e.*, of the CNN's predicted TD-to-TS probabilities.  I suggest using the default shown above.
 - `lead_times_hours` is the set of lead times at which to evaluate the model.  For more details, see the above documentation for `evaluate_model.py`.
 - `output_file_name` is a string, containing the full path to the output file.  Results will be written here in NetCDF format.

And below is an example of how you would call `run_discard_test.py` from a Unix terminal.

```
python run_discard_test.py \
    --input_prediction_file_name="your file name here" \
    --discard_fractions 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 \
    --lead_times_hours 6 12 18 24 30 36 42 48 54 60 66 72 78 84 90 96 102 108 114 120 126 132 138 144 150 156 162 168 \
    --output_file_name="your file name here"
```

More details on the input arguments are provided below.

 - `input_prediction_file_name` is a string, containing the full path to a prediction file created by `apply_neural_net.py`.
 - `discard_fractions` is the set of discard fractions to use in the discard test.
 - `lead_times_hours` is the set of lead times at which to evaluate the model.  For more details, see the above documentation for `evaluate_model.py`.
 - `output_file_name` is a string, containing the full path to the output file.  Results will be written here in NetCDF format.

# Plotting evaluation results: mean predictions only (not uncertainty estimates)

Below is an example of how you would call `plot_evaluation.py` from a Unix terminal.

```
python plot_evaluation.py \
    --input_evaluation_file_name="your file name here" \
    --confidence_level=0.95 \
    --output_dir_name="your directory name here"
```

More details on the input arguments are provided below.

 - `input_evaluation_file_name` is a string, containing the full path to an evaluation file created by `evaluate_model.py`.
 - `confidence_level`: If the input file contains bootstrap-resampled evaluation results -- *i.e.*, if you called `evaluate_model.py` with `num_bootstrap_reps > 1` -- then you can plot confidence intervals in the evaluation graphics (ROC curve, performance diagram, and attributes diagram).  This argument indicates the confidence level, ranging from $\left[ 0, 1 \right]$.  Thus, 0.95 corresponds to the 95% confidence level.
 - `output_dir_name` is a string, naming the output directory.  Evaluation graphics will be saved here.

# Plotting evaluation results: uncertainty estimates only (not mean predictions)

Below is an example of how you would call `plot_spread_vs_skill.py` from a Unix terminal.

```
python plot_spread_vs_skill.py \
    --input_file_name="your file name here" \
    --output_file_name="your file name here"
```

More details on the input arguments are provided below.

 - `input_file_name` is a string, containing the full path to a result file created by `compute_spread_vs_skill.py`.
 - `output_file_name` is a string, containing the full path to the output file.  The spread-skill plot will be written here as a JPEG image.

And is an example of how you would call `plot_discard_test.py` from a Unix terminal.

```
python plot_discard_test.py \
    --input_file_name="your file name here" \
    --output_file_name="your file name here"
```

More details on the input arguments are provided below.

 - `input_file_name` is a string, containing the full path to a result file created by `run_discard_test.py`.
 - `output_file_name` is a string, containing the full path to the output file.  The spread-skill plot will be written here as a JPEG image.
