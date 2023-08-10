"""Plots case study with predictors, NN predictions, and baseline predictions.

The three baselines are SHIPS-RII, SHIPS consensus, and DTOPS.
"""

import os
import sys
import argparse
from PIL import Image
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import gg_plotting_utils
import imagemagick_utils
import prediction_io
import example_io
import ships_io
import border_io
import satellite_utils
import general_utils
import normalization
import neural_net
import predictor_plotting
import ships_plotting
import satellite_plotting

TIME_FORMAT = '%Y-%m-%d-%H'

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
METRES_PER_SECOND_TO_KT = 3.6 / 1.852
HOURS_TO_SECONDS = 3600

BASELINE_DESCRIPTION_STRINGS = ['basic', 'consensus', 'dtops']
BASELINE_DESCRIPTION_STRINGS_FANCY = ['SHIPS-RII', 'SHIPS consensus', 'DTOPS']

SHIPS_LAGGED_PREDICTOR_NAMES = [
    ships_io.SATELLITE_TEMP_0TO200KM_KEY,
    ships_io.SATELLITE_TEMP_0TO200KM_STDEV_KEY,
    ships_io.SATELLITE_TEMP_100TO300KM_KEY,
    ships_io.SATELLITE_TEMP_100TO300KM_STDEV_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M10C_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M20C_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M30C_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M40C_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M50C_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M60C_KEY,
    ships_io.SATELLITE_MAX_TEMP_0TO30KM_KEY,
    ships_io.SATELLITE_MEAN_TEMP_0TO30KM_KEY,
    ships_io.SATELLITE_MAX_TEMP_RADIUS_KEY,
    ships_io.SATELLITE_MIN_TEMP_20TO120KM_KEY,
    ships_io.SATELLITE_MEAN_TEMP_20TO120KM_KEY,
    ships_io.SATELLITE_MIN_TEMP_RADIUS_KEY
]

SHIPS_FORECAST_PREDICTOR_NAMES = [
    ships_io.INTENSITY_KEY,
    ships_io.INTENSITY_CHANGE_6HOURS_KEY,
    ships_io.SHEAR_850TO500MB_U_KEY,
    ships_io.SHEAR_850TO200MB_INNER_RING_GNRL_KEY,
    ships_io.TEMP_200MB_OUTER_RING_KEY,
    ships_io.TEMP_GRADIENT_850TO700MB_INNER_RING_KEY,
    ships_io.MAX_TAN_WIND_850MB_KEY,
    ships_io.W_WIND_0TO15KM_INNER_RING_KEY,
    ships_io.FORECAST_LATITUDE_KEY,
    ships_io.MAX_PTTL_INTENSITY_KEY,
    ships_io.OCEAN_AGE_KEY,
    ships_io.MERGED_SST_KEY,
    ships_io.MERGED_OHC_KEY
]

SHIPS_MAX_FORECAST_HOUR = 24
RAPID_INTENSIFN_CUTOFF_M_S01 = 30 * KT_TO_METRES_PER_SECOND

BASELINE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
BASELINE_MARKER_TYPE = 'o'
BASELINE_MARKER_SIZE = 16

VIOLIN_LINE_COLOUR = numpy.full(3, 0.)
VIOLIN_FACE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
VIOLIN_FACE_COLOUR = matplotlib.colors.to_rgba(c=VIOLIN_FACE_COLOUR, alpha=0.4)
VIOLIN_LINE_WIDTH = 2.
VIOLIN_EDGE_COLOUR = VIOLIN_FACE_COLOUR
VIOLIN_EDGE_WIDTH = 0.

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

DEFAULT_FONT_SIZE = 36
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

NN_MODEL_DIRS_ARG_NAME = 'input_nn_model_dir_names'
NN_MODEL_DESCRIPTIONS_ARG_NAME = 'nn_model_description_strings'
NORM_EXAMPLE_DIR_ARG_NAME = 'input_norm_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
SHIPS_DIR_ARG_NAME = 'input_ships_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
INIT_TIME_ARG_NAME = 'init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

NN_MODEL_DIRS_HELP_STRING = (
    'List of input directories, one for each selected NN.  Each directory '
    'should be the top-level directory for the given NN.  This script will '
    'find results on the testing data, using isotonic regression.'
)
NN_MODEL_DESCRIPTIONS_HELP_STRING = (
    'List of NN descriptions, one per input directory.'
)
NORM_EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with normalized learning examples.  Predictors will be '
    'read by `example_io.read_file`, from files at locations determined by '
    '`example_io.find_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file.  This will be read by `example_io.read_file` '
    'and used to denormalize predictors.'
)
SHIPS_DIR_HELP_STRING = (
    'Name of directory with processed SHIPS data.  Actual intensities will be '
    'read from here.'
)
CYCLONE_ID_HELP_STRING = (
    'This script will plot the case study for one TC object, specified by this '
    '8-character cyclone ID and {0:s}'
).format(INIT_TIME_ARG_NAME)

INIT_TIME_HELP_STRING = (
    'This script will plot the case study for one TC object, specified by this '
    'forecast-init time (format "yyyy-mm-dd-HH") and {0:s}.'
).format(CYCLONE_ID_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NN_MODEL_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=NN_MODEL_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NN_MODEL_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=NN_MODEL_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORM_EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=NORM_EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHIPS_DIR_ARG_NAME, type=str, required=True,
    help=SHIPS_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_ARG_NAME, type=str, required=True,
    help=INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_forecast_probs(
        top_model_dir_names, model_description_strings, ships_dir_name,
        cyclone_id_string, init_time_unix_sec, output_file_name):
    """Plots forecast probs from all models in one figure.

    :param top_model_dir_names: See documentation at top of file.
    :param model_description_strings: Same.
    :param ships_dir_name: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    num_models = len(model_description_strings)
    num_baseline_models = len(BASELINE_DESCRIPTION_STRINGS)
    num_nn_models = num_models - num_baseline_models
    nn_model_description_strings = model_description_strings[:num_nn_models]

    nn_forecast_prob_matrix = numpy.array([])
    baseline_forecast_probs = []
    target_class = -1

    for i in range(num_models):
        if model_description_strings[i] in BASELINE_DESCRIPTION_STRINGS:
            prediction_file_name = (
                '{0:s}/real_time_testing_matched_with_ships/'
                'isotonic_regression/ships_predictions_{1:s}.nc'
            ).format(
                top_model_dir_names[0], model_description_strings[i]
            )
        else:
            prediction_file_name = (
                '{0:s}/real_time_testing_matched_with_ships/'
                'isotonic_regression/cnn_predictions_cf_dtops.nc'
            ).format(top_model_dir_names[i])

        print('Reading data from: "{0:s}"...'.format(prediction_file_name))
        prediction_dict = prediction_io.read_file(prediction_file_name)

        good_index = numpy.where(numpy.logical_and(
            numpy.array(prediction_dict[prediction_io.CYCLONE_IDS_KEY])
            == cyclone_id_string,
            prediction_dict[prediction_io.INIT_TIMES_KEY] == init_time_unix_sec
        ))[0][0]

        if model_description_strings[i] in BASELINE_DESCRIPTION_STRINGS:
            this_prob = prediction_io.get_mean_predictions(prediction_dict)[
                good_index, 0
            ]
            baseline_forecast_probs.append(this_prob)
        else:
            these_probs = prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][
                good_index, -1, 0, :
            ]

            if nn_forecast_prob_matrix.size == 0:
                nn_ensemble_size = len(these_probs)
                nn_forecast_prob_matrix = numpy.full(
                    (num_nn_models, nn_ensemble_size), numpy.nan
                )

            nn_forecast_prob_matrix[i, :] = these_probs

        this_target_class = prediction_dict[prediction_io.TARGET_MATRIX_KEY][
            good_index, 0
        ]
        if target_class == -1:
            target_class = this_target_class + 0

        assert target_class == this_target_class

    ships_file_name = ships_io.find_file(
        directory_name=ships_dir_name, cyclone_id_string=cyclone_id_string,
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(ships_file_name))
    ships_table_xarray = ships_io.read_file(ships_file_name)
    stx = ships_table_xarray

    good_index = numpy.where(
        stx[ships_io.VALID_TIME_KEY].values == init_time_unix_sec
    )[0][0]
    current_intensity_kt = (
        METRES_PER_SECOND_TO_KT *
        stx[ships_io.STORM_INTENSITY_KEY].values[good_index]
    )

    good_indices = numpy.where(numpy.logical_and(
        stx[ships_io.VALID_TIME_KEY].values >= init_time_unix_sec,
        stx[ships_io.VALID_TIME_KEY].values <=
        init_time_unix_sec + 24 * HOURS_TO_SECONDS
    ))[0]
    future_intensity_kt = numpy.max(
        METRES_PER_SECOND_TO_KT *
        stx[ships_io.STORM_INTENSITY_KEY].values[good_indices]
    )

    title_string = 'RI probs for {0:s} at {1:s}\nCurrent $I$ = '.format(
        cyclone_id_string,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
    )
    title_string += '{0:.0f} kt'.format(current_intensity_kt)
    title_string += '; max future $I$ = '
    title_string += '{0:.0f} kt; {1:s} RI'.format(
        future_intensity_kt, 'yes' if target_class == 1 else 'no'
    )

    print(title_string)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    x_tick_values = numpy.linspace(
        1, num_nn_models, num=num_nn_models, dtype=float
    )
    violin_handles = axes_object.violinplot(
        numpy.transpose(nn_forecast_prob_matrix),
        positions=x_tick_values,
        vert=True, widths=0.8, showmeans=True, showmedians=False,
        showextrema=False
    )

    for part_name in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        try:
            this_handle = violin_handles[part_name]
        except:
            continue

        this_handle.set_edgecolor(VIOLIN_LINE_COLOUR)
        this_handle.set_linewidth(VIOLIN_LINE_WIDTH)

    for this_handle in violin_handles['bodies']:
        this_handle.set_facecolor(VIOLIN_FACE_COLOUR)
        this_handle.set_edgecolor(VIOLIN_EDGE_COLOUR)
        this_handle.set_linewidth(VIOLIN_EDGE_WIDTH)
        this_handle.set_alpha(1.)

    for i in range(num_baseline_models):
        for this_x in x_tick_values:
            if BASELINE_DESCRIPTION_STRINGS[i] == 'basic':
                label_string = 'RII'
            elif BASELINE_DESCRIPTION_STRINGS[i] == 'consensus':
                label_string = 'Cons.'
            else:
                label_string = 'DTOPS'

            axes_object.plot(
                this_x, baseline_forecast_probs[i], linestyle='None',
                marker=BASELINE_MARKER_TYPE,
                markersize=BASELINE_MARKER_SIZE, markeredgewidth=0,
                markerfacecolor=BASELINE_COLOUR,
                markeredgecolor=BASELINE_COLOUR
            )
            axes_object.text(
                this_x + 0.05, baseline_forecast_probs[i], label_string,
                color=BASELINE_COLOUR, fontsize=DEFAULT_FONT_SIZE,
                horizontalalignment='left', verticalalignment='center'
            )

    axes_object.set_ylabel('RI probability')
    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(nn_model_description_strings, rotation=90)
    axes_object.set_title(title_string)

    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(top_nn_model_dir_names, nn_model_description_strings,
         norm_example_dir_name, normalization_file_name, ships_dir_name,
         cyclone_id_string, init_time_string, output_dir_name):
    """Plots case study with predictors, NN predictions, and baseline predictions.

    This is effectively the main method.

    :param top_nn_model_dir_names: See documentation at top of file.
    :param nn_model_description_strings: Same.
    :param norm_example_dir_name: Same.
    :param normalization_file_name: Same.
    :param ships_dir_name: Same.
    :param cyclone_id_string: Same.
    :param init_time_string: Same.
    :param output_dir_name: Same.
    """

    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_nn_models = len(top_nn_model_dir_names)
    assert len(nn_model_description_strings) == num_nn_models
    nn_model_description_strings = [
        s.replace('_', ' ') for s in nn_model_description_strings
    ]

    model_description_strings = (
        nn_model_description_strings + BASELINE_DESCRIPTION_STRINGS
    )
    top_model_dir_names = (
        top_nn_model_dir_names + [''] * len(BASELINE_DESCRIPTION_STRINGS)
    )

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    norm_example_file_name = example_io.find_file(
        directory_name=norm_example_dir_name,
        cyclone_id_string=cyclone_id_string,
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_missing=True
    )

    generator_option_dict = {
        neural_net.EXAMPLE_FILE_KEY: norm_example_file_name,
        neural_net.LEAD_TIMES_KEY: numpy.array([24], dtype=int),
        neural_net.SATELLITE_PREDICTORS_KEY:
            [satellite_utils.BRIGHTNESS_TEMPERATURE_KEY],
        neural_net.SATELLITE_LAG_TIMES_KEY: numpy.array([0], dtype=int),
        neural_net.SHIPS_GOES_LAG_TIMES_KEY: numpy.array([0], dtype=int),
        neural_net.SHIPS_GOES_PREDICTORS_KEY: SHIPS_LAGGED_PREDICTOR_NAMES,
        neural_net.SHIPS_FORECAST_PREDICTORS_KEY:
            SHIPS_FORECAST_PREDICTOR_NAMES,
        neural_net.SHIPS_MAX_FORECAST_HOUR_KEY: SHIPS_MAX_FORECAST_HOUR,
        neural_net.NUM_POSITIVE_EXAMPLES_KEY: 2,
        neural_net.NUM_NEGATIVE_EXAMPLES_KEY: 2,
        neural_net.MAX_EXAMPLES_PER_CYCLONE_KEY: 2,
        neural_net.PREDICT_TD_TO_TS_KEY: False,
        neural_net.CLASS_CUTOFFS_KEY:
            numpy.array([RAPID_INTENSIFN_CUTOFF_M_S01]),
        neural_net.NUM_GRID_ROWS_KEY: None,
        neural_net.NUM_GRID_COLUMNS_KEY: None,
        neural_net.USE_TIME_DIFFS_KEY: False,
        neural_net.SATELLITE_TIME_TOLERANCE_KEY: 86400,
        neural_net.SATELLITE_MAX_MISSING_TIMES_KEY: 1,
        neural_net.SHIPS_TIME_TOLERANCE_KEY: 0,
        neural_net.SHIPS_MAX_MISSING_TIMES_KEY: 0,
        neural_net.USE_CLIMO_KEY: False,
        neural_net.DATA_AUG_NUM_TRANS_KEY: 0,
        neural_net.DATA_AUG_NUM_ROTATIONS_KEY: 0,
        neural_net.DATA_AUG_NUM_NOISINGS_KEY: 0
    }

    data_dict = neural_net.create_inputs(generator_option_dict)
    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    all_init_times_unix_sec = data_dict[neural_net.INIT_TIMES_KEY]
    grid_latitude_matrix_deg_n = data_dict[neural_net.GRID_LATITUDE_MATRIX_KEY]
    grid_longitude_matrix_deg_e = (
        data_dict[neural_net.GRID_LONGITUDE_MATRIX_KEY]
    )

    time_index = general_utils.find_exact_times(
        actual_times_unix_sec=all_init_times_unix_sec,
        desired_times_unix_sec=numpy.array([init_time_unix_sec], dtype=int)
    )[0]

    predictor_matrices = [
        None if m is None else m[[time_index], ...] for m in predictor_matrices
    ]
    grid_latitude_matrix_deg_n = grid_latitude_matrix_deg_n[time_index, ...]
    grid_longitude_matrix_deg_e = grid_longitude_matrix_deg_e[time_index, ...]

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    (
        figure_objects, axes_objects, _
    ) = predictor_plotting.plot_brightness_temp_one_example(
        predictor_matrices_one_example=predictor_matrices,
        model_metadata_dict=
        {neural_net.VALIDATION_OPTIONS_KEY: generator_option_dict},
        cyclone_id_string=cyclone_id_string,
        init_time_unix_sec=init_time_unix_sec,
        normalization_table_xarray=normalization_table_xarray,
        grid_latitude_matrix_deg_n=grid_latitude_matrix_deg_n,
        grid_longitude_matrix_deg_e=grid_longitude_matrix_deg_e,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        plot_time_diffs_at_lags=False
    )

    figure_object = figure_objects[0]
    axes_object = axes_objects[0]
    axes_object.set_title('CIRA IR brightness temperature (K)')

    dummy_values = numpy.array([
        ships_plotting.MIN_NORMALIZED_VALUE,
        ships_plotting.MAX_NORMALIZED_VALUE
    ])

    this_cmap_object, this_cnorm_object = (
        satellite_plotting.get_colour_scheme()
    )
    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=dummy_values,
        colour_map_object=this_cmap_object,
        colour_norm_object=this_cnorm_object,
        orientation_string='vertical',
        extend_min=True, extend_max=True,
        fraction_of_axis_length=1., font_size=DEFAULT_FONT_SIZE
    )

    tick_values = numpy.linspace(190, 310, num=13, dtype=float)
    tick_strings = ['{0:.0f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')

    panel_file_names = [
        '{0:s}/cira_ir_image.jpg'.format(output_dir_name)
    ]
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    forecast_hours = numpy.linspace(
        0, SHIPS_MAX_FORECAST_HOUR,
        num=int(numpy.round(SHIPS_MAX_FORECAST_HOUR / 6)) + 1,
        dtype=int
    )

    (
        figure_objects, axes_objects, _
    ) = predictor_plotting.plot_lagged_ships_one_example(
        predictor_matrices_one_example=predictor_matrices,
        model_metadata_dict=
        {neural_net.VALIDATION_OPTIONS_KEY: generator_option_dict},
        cyclone_id_string=cyclone_id_string,
        forecast_hours=forecast_hours,
        init_time_unix_sec=init_time_unix_sec
    )

    figure_object = figure_objects[0]
    axes_object = axes_objects[0]
    axes_object.set_ylabel('')
    axes_object.set_yticks([], [])
    axes_object.tick_params(axis='x', labelsize=30)
    axes_object.set_title('GOES-based SHIPS predictors')

    gg_plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=dummy_values,
        colour_map_object=ships_plotting.COLOUR_MAP_OBJECT,
        min_value=ships_plotting.MIN_NORMALIZED_VALUE,
        max_value=ships_plotting.MAX_NORMALIZED_VALUE,
        orientation_string='vertical',
        extend_min=True, extend_max=True,
        fraction_of_axis_length=0.3, font_size=DEFAULT_FONT_SIZE
    )

    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    panel_file_names.append(
        '{0:s}/goes_based_predictors.jpg'.format(output_dir_name)
    )
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    (
        figure_objects, axes_objects, _
    ) = predictor_plotting.plot_forecast_ships_one_example(
        predictor_matrices_one_example=predictor_matrices,
        model_metadata_dict=
        {neural_net.VALIDATION_OPTIONS_KEY: generator_option_dict},
        cyclone_id_string=cyclone_id_string,
        forecast_hours=forecast_hours,
        init_time_unix_sec=init_time_unix_sec
    )

    figure_object = figure_objects[0]
    axes_object = axes_objects[0]
    axes_object.tick_params(axis='x', labelsize=30)
    axes_object.set_title('Enviro/hist SHIPS predictors')

    gg_plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=dummy_values,
        colour_map_object=ships_plotting.COLOUR_MAP_OBJECT,
        min_value=ships_plotting.MIN_NORMALIZED_VALUE,
        max_value=ships_plotting.MAX_NORMALIZED_VALUE,
        orientation_string='vertical',
        extend_min=True, extend_max=True,
        fraction_of_axis_length=0.5, font_size=DEFAULT_FONT_SIZE
    )

    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(c)')

    panel_file_names.append(
        '{0:s}/enviro_and_hist_predictors.jpg'.format(output_dir_name)
    )
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    panel_file_names.append(
        '{0:s}/forecast_probs.jpg'.format(output_dir_name)
    )
    _plot_forecast_probs(
        top_model_dir_names=top_model_dir_names,
        model_description_strings=model_description_strings,
        ships_dir_name=ships_dir_name,
        cyclone_id_string=cyclone_id_string,
        init_time_unix_sec=init_time_unix_sec,
        output_file_name=panel_file_names[-1]
    )

    for i in range(len(panel_file_names)):
        if i == 1:
            continue

        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[i],
            output_file_name=panel_file_names[i],
            output_size_pixels=PANEL_SIZE_PX
        )

    image_matrix = Image.open(panel_file_names[3])
    desired_width_px, _ = image_matrix.size

    image_matrix = Image.open(panel_file_names[1])
    current_width_px, current_height_px = image_matrix.size
    resize_factor = (float(desired_width_px) / current_width_px) ** 2
    desired_size_px = int(numpy.round(
        current_width_px * current_height_px * resize_factor
    ))

    imagemagick_utils.resize_image(
        input_file_name=panel_file_names[1],
        output_file_name=panel_file_names[1],
        output_size_pixels=desired_size_px
    )

    concat_figure_file_name = '{0:s}/case_study.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name, border_width_pixels=10
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_nn_model_dir_names=getattr(
            INPUT_ARG_OBJECT, NN_MODEL_DIRS_ARG_NAME
        ),
        nn_model_description_strings=getattr(
            INPUT_ARG_OBJECT, NN_MODEL_DESCRIPTIONS_ARG_NAME
        ),
        norm_example_dir_name=getattr(
            INPUT_ARG_OBJECT, NORM_EXAMPLE_DIR_ARG_NAME
        ),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        ships_dir_name=getattr(INPUT_ARG_OBJECT, SHIPS_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        init_time_string=getattr(INPUT_ARG_OBJECT, INIT_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
