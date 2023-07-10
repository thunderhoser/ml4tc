"""Plots all predictors (scalars and brightness-temp maps) for a given model."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import imagemagick_utils
from ml4tc.io import example_io
from ml4tc.io import border_io
from ml4tc.io import prediction_io
from ml4tc.utils import example_utils
from ml4tc.utils import satellite_utils
from ml4tc.utils import general_utils
from ml4tc.utils import normalization
from ml4tc.machine_learning import neural_net
from ml4tc.plotting import plotting_utils
from ml4tc.plotting import satellite_plotting
from ml4tc.plotting import scalar_satellite_plotting
from ml4tc.plotting import ships_plotting
from ml4tc.plotting import predictor_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

HOURS_TO_SECONDS = 3600
METRES_PER_SECOND_TO_KT = 3.6 / 1.852

TITLE_FONT_SIZE = 16
COLOUR_BAR_FONT_SIZE = 20
FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)

MODEL_METAFILE_ARG_NAME = 'input_model_metafile_name'
EXAMPLE_FILE_ARG_NAME = 'input_norm_example_file_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
INIT_TIMES_ARG_NAME = 'init_time_strings'
FIRST_TIME_ARG_NAME = 'first_init_time_string'
LAST_TIME_ARG_NAME = 'last_init_time_string'
PLOT_TIME_DIFFS_ARG_NAME = 'plot_time_diffs_if_used'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_METAFILE_HELP_STRING = (
    'Path to metafile for model.  Will be read by `neural_net.read_metafile`.'
)
EXAMPLE_FILE_HELP_STRING = (
    'Path to file with normalized learning examples for one cyclone.  Will be '
    'read by `example_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be used to denormalize '
    'brightness-temperature maps before plotting).  Will be read by '
    '`normalization.read_file`.'
)
PREDICTION_FILE_HELP_STRING = (
    'Path to file with predictions and targets.  Will be read by '
    '`prediction_io.read_file`.  If you do not want to plot predictions and '
    'targets, leave this argument alone.'
)
INIT_TIMES_HELP_STRING = (
    'List of initialization times (format "yyyy-mm-dd-HHMMSS").  '
    'Predictors will be plotted for each of these init times.'
)
FIRST_TIME_HELP_STRING = (
    '[used only if `{0:s}` is left alone] First init time (format '
    '"yyyy-mm-dd-HHMMSS").'
).format(INIT_TIMES_ARG_NAME)

LAST_TIME_HELP_STRING = (
    '[used only if `{0:s}` is left alone] Last init time (format '
    '"yyyy-mm-dd-HHMMSS").'
).format(INIT_TIMES_ARG_NAME)

PLOT_TIME_DIFFS_HELP_STRING = (
    'Boolean flag.  If 1, will plot temporal differences of satellite images '
    'at non-zero lag times, assuming temporal diffs were used in training.'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Images will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_METAFILE_ARG_NAME, type=str, required=True,
    help=MODEL_METAFILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=False, default='',
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIMES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=INIT_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=False, default='',
    help=FIRST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=False, default='',
    help=LAST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_TIME_DIFFS_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_TIME_DIFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_intensities(example_table_xarray, init_times_unix_sec,
                     model_metadata_dict):
    """Returns current and future storm intensities.

    T = number of forecast-initialization times

    :param example_table_xarray: xarray table returned by
        `example_io.read_file`.
    :param init_times_unix_sec: length-T numpy array of forecast-initialization
        times.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    :return: current_intensities_kt: length-T numpy array of current intensities
        (knots).
    :return: future_intensities_kt: length-T numpy array of future intensities
        (knots).
    """

    xt = example_table_xarray
    these_times_unix_sec = (
        xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values
    )
    good_indices = numpy.array([
        numpy.where(these_times_unix_sec == t)[0][0]
        for t in init_times_unix_sec
    ], dtype=int)

    current_intensities_kt = METRES_PER_SECOND_TO_KT * (
        xt[example_utils.STORM_INTENSITY_KEY].values[good_indices]
    )
    current_intensities_kt = numpy.round(current_intensities_kt).astype(int)

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    lead_time_sec = (
        HOURS_TO_SECONDS * validation_option_dict[neural_net.LEAD_TIMES_KEY][0]
    )
    good_indices_2d_list = [
        numpy.where(these_times_unix_sec == t)[0]
        for t in init_times_unix_sec + lead_time_sec
    ]
    good_indices = numpy.array([
        -1 if len(idcs) == 0 else idcs[0]
        for idcs in good_indices_2d_list
    ], dtype=int)

    future_intensities_kt = METRES_PER_SECOND_TO_KT * (
        xt[example_utils.STORM_INTENSITY_KEY].values[good_indices]
    )
    future_intensities_kt[good_indices == -1] = 0.
    future_intensities_kt = numpy.round(future_intensities_kt).astype(int)

    return current_intensities_kt, future_intensities_kt


def get_predictions_and_targets(prediction_file_name, cyclone_id_string,
                                init_times_unix_sec):
    """Returns prediction and target for each forecast-initialization time.

    T = number of forecast-initialization times

    :param prediction_file_name: Path to input file.  Will be read by
        `prediction_io.read_file`.
    :param cyclone_id_string: ID for desired cyclone.
    :param init_times_unix_sec: length-T numpy array of desired init times.
    :return: forecast_probabilities: length-T numpy array of forecast class
        probabilities.
    :return: target_classes: length-T numpy array of target classes (integers in
        range 0...1).
    """

    error_checking.assert_is_string(prediction_file_name)
    error_checking.assert_is_string(cyclone_id_string)
    error_checking.assert_is_integer_numpy_array(init_times_unix_sec)
    error_checking.assert_is_numpy_array(init_times_unix_sec, num_dimensions=1)

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    good_flags = numpy.array([
        cid == cyclone_id_string
        for cid in prediction_dict[prediction_io.CYCLONE_IDS_KEY]
    ], dtype=bool)

    good_indices = numpy.where(good_flags)[0]
    these_times_unix_sec = (
        prediction_dict[prediction_io.INIT_TIMES_KEY][good_indices]
    )
    good_subindices = numpy.array([
        numpy.where(these_times_unix_sec == t)[0][0]
        for t in init_times_unix_sec
    ], dtype=int)

    good_indices = good_indices[good_subindices]
    target_classes = (
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][good_indices, -1]
    )
    forecast_probabilities = prediction_io.get_mean_predictions(
        prediction_dict
    )[good_indices, -1]

    return forecast_probabilities, target_classes


def _finish_figure_scalar_satellite(figure_object, output_dir_name,
                                    init_time_unix_sec, cyclone_id_string):
    """Finishes one figure for scalar (ungridded) satellite-based predictors.

    One figure corresponds to one forecast-initialization time.

    :param figure_object: Figure handle (instances of
        `matplotlib.figure.Figure`).
    :param output_dir_name: Name of output directory.
    :param init_time_unix_sec: Forecast-initialization time.
    :param cyclone_id_string: Cyclone ID.
    """

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    output_file_name = '{0:s}/{1:s}_{2:s}_scalar_satellite.jpg'.format(
        output_dir_name, cyclone_id_string, init_time_string
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=scalar_satellite_plotting.MIN_NORMALIZED_VALUE,
        vmax=scalar_satellite_plotting.MAX_NORMALIZED_VALUE
    )
    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=scalar_satellite_plotting.COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='', tick_label_format_string='{0:.2g}'
    )


def _finish_figure_brightness_temp(
        figure_objects, pathless_panel_file_names, output_dir_name,
        init_time_unix_sec, cyclone_id_string, plotted_time_diffs):
    """Finishes one figure for brightness temperature.

    One figure corresponds to one forecast-initialization time.

    L = number of model lag times

    :param figure_objects: length-L list of figure handles (instances of
        `matplotlib.figure.Figure`).
    :param pathless_panel_file_names: length-L list of pathless file names for
        panels.
    :param output_dir_name: Name of output directory.
    :param init_time_unix_sec: Forecast-initialization time.
    :param cyclone_id_string: Cyclone ID.
    :param plotted_time_diffs: Boolean flag.  If True, temporal differences were
        plotted at lag times before the most recent one.
    """

    num_model_lag_times = len(figure_objects)
    panel_file_names = [
        '{0:s}/{1:s}'.format(output_dir_name, p)
        for p in pathless_panel_file_names
    ]

    for k in range(num_model_lag_times):
        print('Saving figure to file: "{0:s}"...'.format(
            panel_file_names[k]
        ))
        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[k],
            output_file_name=panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    concat_figure_file_name = (
        '{0:s}/{1:s}_{2:s}_brightness_temp_concat.jpg'
    ).format(
        output_dir_name, cyclone_id_string, init_time_string
    )
    plotting_utils.concat_panels(
        panel_file_names=panel_file_names,
        concat_figure_file_name=concat_figure_file_name
    )

    if plotted_time_diffs:
        this_cmap_object, this_cnorm_object = (
            satellite_plotting.get_diff_colour_scheme()
        )
        plotting_utils.add_colour_bar(
            figure_file_name=concat_figure_file_name,
            colour_map_object=this_cmap_object,
            colour_norm_object=this_cnorm_object,
            orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
            cbar_label_string='Brightness-temp difference (K)',
            tick_label_format_string='{0:d}'
        )

    this_cmap_object, this_cnorm_object = (
        satellite_plotting.get_colour_scheme()
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=this_cmap_object,
        colour_norm_object=this_cnorm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Brightness temperature (K)',
        tick_label_format_string='{0:d}'
    )


def _finish_figure_lagged_ships(
        figure_objects, pathless_panel_file_names, output_dir_name,
        init_time_unix_sec, cyclone_id_string):
    """Finishes one figure for lagged SHIPS predictors.

    One figure corresponds to one forecast-initialization time.

    :param figure_objects: See doc for `_finish_figure_brightness_temp`.
    :param pathless_panel_file_names: Same.
    :param output_dir_name: Same.
    :param init_time_unix_sec: Same.
    :param cyclone_id_string: Same.
    """

    num_model_lag_times = len(figure_objects)
    panel_file_names = [
        '{0:s}/{1:s}'.format(output_dir_name, p)
        for p in pathless_panel_file_names
    ]

    for k in range(num_model_lag_times):
        print('Saving figure to file: "{0:s}"...'.format(
            panel_file_names[k]
        ))
        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[k],
            output_file_name=panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    concat_figure_file_name = (
        '{0:s}/{1:s}_{2:s}_ships_lagged_concat.jpg'
    ).format(
        output_dir_name, cyclone_id_string, init_time_string
    )
    plotting_utils.concat_panels(
        panel_file_names=panel_file_names,
        concat_figure_file_name=concat_figure_file_name
    )

    colour_norm_object = pyplot.Normalize(
        vmin=ships_plotting.MIN_NORMALIZED_VALUE,
        vmax=ships_plotting.MAX_NORMALIZED_VALUE
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=ships_plotting.COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='', tick_label_format_string='{0:.2g}'
    )


def _finish_figure_forecast_ships(
        figure_objects, pathless_panel_file_names, output_dir_name,
        init_time_unix_sec, cyclone_id_string):
    """Finishes one figure for forecast SHIPS predictors.

    One figure corresponds to one forecast-initialization time.

    :param figure_objects: See doc for `_finish_figure_brightness_temp`.
    :param pathless_panel_file_names: Same.
    :param output_dir_name: Same.
    :param init_time_unix_sec: Same.
    :param cyclone_id_string: Same.
    """

    num_model_lag_times = len(figure_objects)
    panel_file_names = [
        '{0:s}/{1:s}'.format(output_dir_name, p)
        for p in pathless_panel_file_names
    ]

    for k in range(num_model_lag_times):
        print('Saving figure to file: "{0:s}"...'.format(
            panel_file_names[k]
        ))
        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[k],
            output_file_name=panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    concat_figure_file_name = (
        '{0:s}/{1:s}_{2:s}_ships_forecast_concat.jpg'
    ).format(
        output_dir_name, cyclone_id_string, init_time_string
    )
    plotting_utils.concat_panels(
        panel_file_names=panel_file_names,
        concat_figure_file_name=concat_figure_file_name
    )

    colour_norm_object = pyplot.Normalize(
        vmin=ships_plotting.MIN_NORMALIZED_VALUE,
        vmax=ships_plotting.MAX_NORMALIZED_VALUE
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=ships_plotting.COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='', tick_label_format_string='{0:.2g}'
    )


def _run(model_metafile_name, norm_example_file_name, normalization_file_name,
         prediction_file_name, init_time_strings, first_init_time_string,
         last_init_time_string, plot_time_diffs_if_used, output_dir_name):
    """Plots all predictors (scalars and brightness temps) for a given model.

    This is effectively the main method.

    :param model_metafile_name: See documentation at top of file.
    :param norm_example_file_name: Same.
    :param normalization_file_name: Same.
    :param prediction_file_name: Same.
    :param init_time_strings: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param plot_time_diffs_if_used: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    v = model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]

    plot_time_diffs = (
        v[neural_net.USE_TIME_DIFFS_KEY]
        and len(v[neural_net.SATELLITE_LAG_TIMES_KEY]) > 1
        and plot_time_diffs_if_used
    )
    v[neural_net.USE_TIME_DIFFS_KEY] = False
    model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY] = v

    validation_option_dict = model_metadata_dict[
        neural_net.VALIDATION_OPTIONS_KEY
    ]
    validation_option_dict[neural_net.EXAMPLE_FILE_KEY] = norm_example_file_name

    print('Reading data from: "{0:s}"...'.format(norm_example_file_name))
    example_table_xarray = example_io.read_file(norm_example_file_name)
    cyclone_id_string = (
        example_table_xarray[satellite_utils.CYCLONE_ID_KEY].values[0]
    )
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    print(SEPARATOR_STRING)

    data_dict = neural_net.create_inputs(validation_option_dict)
    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    all_init_times_unix_sec = data_dict[neural_net.INIT_TIMES_KEY]
    grid_latitude_matrix_deg_n = data_dict[neural_net.GRID_LATITUDE_MATRIX_KEY]
    grid_longitude_matrix_deg_e = (
        data_dict[neural_net.GRID_LONGITUDE_MATRIX_KEY]
    )
    print(SEPARATOR_STRING)

    if len(init_time_strings) == 1 and init_time_strings[0] == '':
        first_init_time_unix_sec = time_conversion.string_to_unix_sec(
            first_init_time_string, TIME_FORMAT
        )
        last_init_time_unix_sec = time_conversion.string_to_unix_sec(
            last_init_time_string, TIME_FORMAT
        )
        time_indices = general_utils.find_exact_times(
            actual_times_unix_sec=all_init_times_unix_sec,
            first_desired_time_unix_sec=first_init_time_unix_sec,
            last_desired_time_unix_sec=last_init_time_unix_sec
        )
    else:
        init_times_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, TIME_FORMAT)
            for t in init_time_strings
        ], dtype=int)

        time_indices = general_utils.find_exact_times(
            actual_times_unix_sec=all_init_times_unix_sec,
            desired_times_unix_sec=init_times_unix_sec
        )

    init_times_unix_sec = all_init_times_unix_sec[time_indices]
    predictor_matrices = [
        None if m is None else m[time_indices, ...] for m in predictor_matrices
    ]

    if grid_latitude_matrix_deg_n is not None:
        grid_latitude_matrix_deg_n = (
            grid_latitude_matrix_deg_n[time_indices, ...]
        )
        grid_longitude_matrix_deg_e = (
            grid_longitude_matrix_deg_e[time_indices, ...]
        )

    sort_indices = numpy.argsort(init_times_unix_sec)
    init_times_unix_sec = init_times_unix_sec[sort_indices]
    predictor_matrices = [
        None if m is None else m[sort_indices, ...] for m in predictor_matrices
    ]

    if grid_latitude_matrix_deg_n is not None:
        grid_latitude_matrix_deg_n = (
            grid_latitude_matrix_deg_n[sort_indices, ...]
        )
        grid_longitude_matrix_deg_e = (
            grid_longitude_matrix_deg_e[sort_indices, ...]
        )

    num_init_times = len(init_times_unix_sec)
    predict_td_to_ts = validation_option_dict[neural_net.PREDICT_TD_TO_TS_KEY]

    if predict_td_to_ts:
        info_strings = [''] * num_init_times
    else:
        current_intensities_kt, future_intensities_kt = _get_intensities(
            example_table_xarray=example_table_xarray,
            init_times_unix_sec=init_times_unix_sec,
            model_metadata_dict=model_metadata_dict
        )
        info_strings = [
            r'$I$ = {0:d} to {1:d} kt; '.format(c, f) for c, f in
            zip(current_intensities_kt, future_intensities_kt)
        ]

    if prediction_file_name != '':
        forecast_probabilities, target_classes = get_predictions_and_targets(
            prediction_file_name=prediction_file_name,
            cyclone_id_string=cyclone_id_string,
            init_times_unix_sec=init_times_unix_sec
        )

        for i in range(num_init_times):
            if predict_td_to_ts:
                info_strings[i] += 'future TS = {0:s}'.format(
                    'yes' if target_classes[i] == 1 else 'no'
                )
                info_strings[i] += r'; $p_{TS}$ = '
            else:
                info_strings[i] += 'RI = {0:s}'.format(
                    'yes' if target_classes[i] == 1 else 'no'
                )
                info_strings[i] += r'; $p_{RI}$ = '

            info_strings[i] += '{0:.2f}'.format(forecast_probabilities[i])

    for i in range(num_init_times):
        if predictor_matrices[1] is not None:
            these_predictor_matrices = [
                None if m is None else m[[i], ...] for m in predictor_matrices
            ]

            figure_object, axes_object = (
                predictor_plotting.plot_scalar_satellite_one_example(
                    predictor_matrices_one_example=these_predictor_matrices,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=cyclone_id_string,
                    init_time_unix_sec=init_times_unix_sec[i]
                )[:2]
            )

            title_string = '{0:s}; {1:s}'.format(
                axes_object.get_title(), info_strings[i]
            )
            axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

            _finish_figure_scalar_satellite(
                figure_object=figure_object, output_dir_name=output_dir_name,
                init_time_unix_sec=init_times_unix_sec[i],
                cyclone_id_string=cyclone_id_string
            )

        if predictor_matrices[0] is not None:
            these_predictor_matrices = [
                None if m is None else m[[i], ...] for m in predictor_matrices
            ]

            figure_objects, axes_objects, pathless_panel_file_names = (
                predictor_plotting.plot_brightness_temp_one_example(
                    predictor_matrices_one_example=these_predictor_matrices,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=cyclone_id_string,
                    init_time_unix_sec=init_times_unix_sec[i],
                    normalization_table_xarray=normalization_table_xarray,
                    grid_latitude_matrix_deg_n=
                    grid_latitude_matrix_deg_n[i, ...],
                    grid_longitude_matrix_deg_e=
                    grid_longitude_matrix_deg_e[i, ...],
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    plot_time_diffs_at_lags=plot_time_diffs
                )
            )

            title_string = '{0:s}; {1:s}'.format(
                axes_objects[0].get_title(), info_strings[i]
            )
            axes_objects[0].set_title(title_string, fontsize=TITLE_FONT_SIZE)

            _finish_figure_brightness_temp(
                figure_objects=figure_objects,
                pathless_panel_file_names=pathless_panel_file_names,
                output_dir_name=output_dir_name,
                init_time_unix_sec=init_times_unix_sec[i],
                cyclone_id_string=cyclone_id_string,
                plotted_time_diffs=plot_time_diffs
            )

        v = validation_option_dict

        if v[neural_net.SHIPS_GOES_PREDICTORS_KEY] is not None:
            these_predictor_matrices = [
                None if m is None else m[[i], ...] for m in predictor_matrices
            ]

            max_forecast_hour = (
                v[neural_net.SHIPS_MAX_FORECAST_HOUR_KEY]
            )
            forecast_hours = numpy.linspace(
                0, max_forecast_hour,
                num=int(numpy.round(max_forecast_hour / 6)) + 1, dtype=int
            )

            figure_objects, axes_objects, pathless_panel_file_names = (
                predictor_plotting.plot_lagged_ships_one_example(
                    predictor_matrices_one_example=these_predictor_matrices,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=cyclone_id_string,
                    forecast_hours=forecast_hours,
                    init_time_unix_sec=init_times_unix_sec[i]
                )
            )

            title_string = '{0:s}; {1:s}'.format(
                axes_objects[0].get_title(), info_strings[i]
            )
            axes_objects[0].set_title(title_string, fontsize=TITLE_FONT_SIZE)

            _finish_figure_lagged_ships(
                figure_objects=figure_objects,
                pathless_panel_file_names=pathless_panel_file_names,
                output_dir_name=output_dir_name,
                init_time_unix_sec=init_times_unix_sec[i],
                cyclone_id_string=cyclone_id_string
            )

        if v[neural_net.SHIPS_FORECAST_PREDICTORS_KEY] is not None:
            these_predictor_matrices = [
                None if m is None else m[[i], ...] for m in predictor_matrices
            ]

            max_forecast_hour = (
                v[neural_net.SHIPS_MAX_FORECAST_HOUR_KEY]
            )
            forecast_hours = numpy.linspace(
                0, max_forecast_hour,
                num=int(numpy.round(max_forecast_hour / 6)) + 1, dtype=int
            )

            figure_objects, axes_objects, pathless_panel_file_names = (
                predictor_plotting.plot_forecast_ships_one_example(
                    predictor_matrices_one_example=these_predictor_matrices,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=cyclone_id_string,
                    forecast_hours=forecast_hours,
                    init_time_unix_sec=init_times_unix_sec[i]
                )
            )

            title_string = '{0:s}; {1:s}'.format(
                axes_objects[0].get_title(), info_strings[i]
            )
            axes_objects[0].set_title(title_string, fontsize=TITLE_FONT_SIZE)

            _finish_figure_forecast_ships(
                figure_objects=figure_objects,
                pathless_panel_file_names=pathless_panel_file_names,
                output_dir_name=output_dir_name,
                init_time_unix_sec=init_times_unix_sec[i],
                cyclone_id_string=cyclone_id_string
            )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_metafile_name=getattr(INPUT_ARG_OBJECT, MODEL_METAFILE_ARG_NAME),
        norm_example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        init_time_strings=getattr(INPUT_ARG_OBJECT, INIT_TIMES_ARG_NAME),
        first_init_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_init_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        plot_time_diffs_if_used=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_TIME_DIFFS_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
