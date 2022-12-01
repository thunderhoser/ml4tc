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
