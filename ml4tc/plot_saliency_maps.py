"""Plots saliency maps."""

import os
import sys
import copy
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_general_utils
import time_conversion
import file_system_utils
import error_checking
import imagemagick_utils
import example_io
import border_io
import normalization
import saliency
import neural_net
import plotting_utils
import ships_plotting
import satellite_plotting
import scalar_satellite_plotting
import predictor_plotting
import plot_predictors

TOLERANCE = 1e-10
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

COLOUR_BAR_FONT_SIZE = 12
FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
PLOT_INPUT_GRAD_ARG_NAME = 'plot_input_times_grad'
SPATIAL_COLOUR_MAP_ARG_NAME = 'spatial_colour_map_name'
NONSPATIAL_COLOUR_MAP_ARG_NAME = 'nonspatial_colour_map_name'
MIN_COLOUR_PERCENTILE_ARG_NAME = 'min_colour_percentile'
MAX_COLOUR_PERCENTILE_ARG_NAME = 'max_colour_percentile'
PLOT_IN_LOG_SPACE_ARG_NAME = 'plot_in_log_space'
PLOT_TIME_DIFFS_ARG_NAME = 'plot_time_diffs_if_used'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file.  Will be read by `saliency.read_file`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with input examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be used to denormalize '
    'brightness-temperature maps before plotting).  Will be read by '
    '`normalization.read_file`.'
)
PREDICTION_FILE_HELP_STRING = (
    'Path to file with predictions and targets, to be shown in figure titles.  '
    'Will be read by `prediction_io.read_file`.  If you do not want fancy '
    'figure titles, leave this argument alone.'
)
PLOT_INPUT_GRAD_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot input * gradient (saliency).'
)
SPATIAL_COLOUR_MAP_HELP_STRING = (
    'Name of colour scheme for spatial saliency maps.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
NONSPATIAL_COLOUR_MAP_HELP_STRING = (
    'Name of colour scheme for non-spatial saliency maps.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
MIN_COLOUR_PERCENTILE_HELP_STRING = (
    'Determines minimum absolute saliency in colour bar.  For each example, '
    'the minimum will be the [k]th percentile of all absolute saliency values '
    'for said example, where k = `{0:s}`.'
).format(MIN_COLOUR_PERCENTILE_ARG_NAME)

MAX_COLOUR_PERCENTILE_HELP_STRING = (
    'Same as `{0:s}` but for max absolute saliency in colour bar.'
).format(MIN_COLOUR_PERCENTILE_ARG_NAME)

PLOT_IN_LOG_SPACE_HELP_STRING = (
    'Boolean flag.  If 1 (0), colours for gridded saliency will be scaled '
    'logarithmically (linearly).'
)
PLOT_TIME_DIFFS_HELP_STRING = (
    'Boolean flag.  If 1, will plot temporal differences of satellite images '
    'at non-zero lag times, assuming temporal diffs were used in training.'
)
SMOOTHING_RADIUS_HELP_STRING = (
    'Smoothing radius (number of pixels) for saliency maps.  If you do not want'
    ' to smooth, make this 0 or negative.'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Images will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_FILE_ARG_NAME, type=str, required=True,
    help=SALIENCY_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
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
    '--' + PLOT_INPUT_GRAD_ARG_NAME, type=int, required=True,
    help=PLOT_INPUT_GRAD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SPATIAL_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default='BuGn', help=SPATIAL_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NONSPATIAL_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default='binary', help=NONSPATIAL_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_COLOUR_PERCENTILE_ARG_NAME, type=float, required=False,
    default=1, help=MIN_COLOUR_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_COLOUR_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99, help=MAX_COLOUR_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_IN_LOG_SPACE_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_IN_LOG_SPACE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_TIME_DIFFS_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_TIME_DIFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False, default=-1,
    help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _smooth_maps(saliency_dict, smoothing_radius_px, plot_input_times_grad):
    """Smooths saliency maps via Gaussian filter.

    :param saliency_dict: Dictionary returned by `saliency.read_file`.
    :param smoothing_radius_px: e-folding radius (num pixels).
    :param plot_input_times_grad: See documentation at top of file.
    :return: saliency_dict: Same as input but with smoothed maps.
    """

    if plot_input_times_grad:
        this_key = saliency.THREE_INPUT_GRAD_KEY
    else:
        this_key = saliency.THREE_SALIENCY_KEY

    brightness_temp_saliency_matrix = saliency_dict[this_key][0]
    if brightness_temp_saliency_matrix is None:
        return saliency_dict

    print((
        'Smoothing maps with Gaussian filter (e-folding radius of {0:.1f} grid '
        'cells)...'
    ).format(
        smoothing_radius_px
    ))

    num_examples = brightness_temp_saliency_matrix.shape[0]
    num_model_lag_times = brightness_temp_saliency_matrix.shape[-2]

    for i in range(num_examples):
        for j in range(num_model_lag_times):
            brightness_temp_saliency_matrix[i, ..., j, 0] = (
                gg_general_utils.apply_gaussian_filter(
                    input_matrix=brightness_temp_saliency_matrix[i, ..., j, 0],
                    e_folding_radius_grid_cells=smoothing_radius_px
                )
            )

    saliency_dict[this_key][0] = brightness_temp_saliency_matrix
    return saliency_dict


def _plot_scalar_satellite_saliency(
        data_dict, saliency_dict, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec, colour_map_object, min_colour_value,
        max_colour_value, plot_in_log_space, plot_input_times_grad,
        info_string, output_dir_name):
    """Plots saliency for scalar satellite for each lag time at one init time.

    :param data_dict: Dictionary returned by `neural_net.create_inputs`.
    :param saliency_dict: Dictionary returned by `saliency.read_file`.
    :param model_metadata_dict: See doc for
        `predictor_plotting.plot_scalar_satellite_one_example`.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param min_colour_value: Minimum absolute value in colour scheme.
    :param max_colour_value: Max absolute value in colour scheme.
    :param plot_in_log_space: Boolean flag.  If True (False), colours will be
        scaled logarithmically (linearly).
    :param plot_input_times_grad: Boolean flag.  If True (False), will plot
        input * gradient (saliency).
    :param info_string: Info string (will be appended to title).
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    predictor_example_index = numpy.where(
        data_dict[neural_net.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]
    saliency_example_index = numpy.where(
        saliency_dict[saliency.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]

    predictor_matrices_one_example = [
        None if p is None else p[[predictor_example_index], ...]
        for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]

    if plot_input_times_grad:
        this_key = saliency.THREE_INPUT_GRAD_KEY
    else:
        this_key = saliency.THREE_SALIENCY_KEY

    saliency_matrix = saliency_dict[this_key][1][saliency_example_index, ...]

    figure_object, axes_object = (
        predictor_plotting.plot_scalar_satellite_one_example(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            init_time_unix_sec=init_time_unix_sec
        )[:2]
    )

    scalar_satellite_plotting.plot_raw_numbers_multi_times(
        data_matrix=saliency_matrix,
        axes_object=axes_object,
        font_size=17.5, colour_map_object=colour_map_object,
        min_colour_value=min_colour_value,
        max_colour_value=max_colour_value,
        number_format_string='.1e',
        plot_in_log_space=plot_in_log_space
    )

    if info_string != '':
        axes_object.set_title('{0:s}; {1:s}'.format(
            axes_object.get_title(), info_string
        ))

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
        cbar_label_string='Predictor', tick_label_format_string='{0:.2g}'
    )

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    label_string = 'Absolute {0:s}'.format(
        'input times gradient' if plot_input_times_grad else 'saliency'
    )

    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string=label_string, tick_label_format_string='{0:.2g}',
        log_space=plot_in_log_space
    )


def _plot_brightness_temp_saliency(
        data_dict, saliency_dict, model_metadata_dict,
        cyclone_id_string, init_time_unix_sec, normalization_table_xarray,
        border_latitudes_deg_n, border_longitudes_deg_e, colour_map_object,
        min_colour_value, max_colour_value, plot_in_log_space,
        plot_time_diffs_at_lags, plot_input_times_grad, info_string,
        output_dir_name):
    """Plots saliency for brightness temp for each lag time at one init time.

    P = number of points in border set

    :param data_dict: See doc for `_plot_scalar_satellite_saliency`.
    :param saliency_dict: Same.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param normalization_table_xarray: xarray table returned by
        `normalization.read_file`.
    :param border_latitudes_deg_n: length-P numpy array of latitudes
        (deg north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes
        (deg east).
    :param colour_map_object: See doc for `_plot_scalar_satellite_saliency`.
    :param min_colour_value: Same.
    :param max_colour_value: Same.
    :param plot_in_log_space: Same.
    :param plot_time_diffs_at_lags: Boolean flag.  If True, at each lag time t
        before the most recent one, will plot temporal difference:
        (brightness temp at most recent lag time) - (brightness temp at t).
    :param plot_input_times_grad: See doc for `_plot_scalar_satellite_saliency`.
    :param info_string: Same.
    :param output_dir_name: Same.
    """

    predictor_example_index = numpy.where(
        data_dict[neural_net.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]
    saliency_example_index = numpy.where(
        saliency_dict[saliency.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]

    predictor_matrices_one_example = [
        None if p is None else p[[predictor_example_index], ...]
        for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]

    if plot_input_times_grad:
        this_key = saliency.THREE_INPUT_GRAD_KEY
    else:
        this_key = saliency.THREE_SALIENCY_KEY

    saliency_matrix = saliency_dict[this_key][0][saliency_example_index, ..., 0]

    grid_latitude_matrix_deg_n = data_dict[
        neural_net.GRID_LATITUDE_MATRIX_KEY
    ][predictor_example_index, ...]

    grid_longitude_matrix_deg_e = data_dict[
        neural_net.GRID_LONGITUDE_MATRIX_KEY
    ][predictor_example_index, ...]

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    num_model_lag_times = len(
        validation_option_dict[neural_net.SATELLITE_LAG_TIMES_KEY]
    )
    plot_time_diffs_at_lags = (
        plot_time_diffs_at_lags and num_model_lag_times > 1
    )

    figure_objects, axes_objects, pathless_output_file_names = (
        predictor_plotting.plot_brightness_temp_one_example(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            init_time_unix_sec=init_time_unix_sec,
            grid_latitude_matrix_deg_n=grid_latitude_matrix_deg_n,
            grid_longitude_matrix_deg_e=grid_longitude_matrix_deg_e,
            normalization_table_xarray=normalization_table_xarray,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            plot_motion_arrow=True,
            plot_time_diffs_at_lags=plot_time_diffs_at_lags
        )
    )

    panel_file_names = [''] * num_model_lag_times

    for k in range(num_model_lag_times):
        satellite_plotting.plot_saliency(
            saliency_matrix=saliency_matrix[..., k],
            axes_object=axes_objects[k],
            latitude_array_deg_n=grid_latitude_matrix_deg_n[..., k],
            longitude_array_deg_e=grid_longitude_matrix_deg_e[..., k],
            min_abs_contour_value=min_colour_value,
            max_abs_contour_value=max_colour_value,
            half_num_contours=10, plot_in_log_space=plot_in_log_space,
            colour_map_object=colour_map_object
        )

        if info_string != '':
            axes_objects[k].set_title('{0:s}; {1:s}'.format(
                axes_objects[k].get_title(), info_string
            ))

        panel_file_names[k] = '{0:s}/{1:s}'.format(
            output_dir_name, pathless_output_file_names[k]
        )
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

    if plot_time_diffs_at_lags:
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
        cbar_label_string='Brightness temp (K)',
        tick_label_format_string='{0:d}'
    )

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    label_string = 'Absolute {0:s}'.format(
        'input times gradient' if plot_input_times_grad else 'saliency'
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string=label_string, tick_label_format_string='{0:.2g}',
        log_space=plot_in_log_space
    )


def _plot_lagged_ships_saliency(
        data_dict, saliency_dict, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec, colour_map_object, min_colour_value,
        max_colour_value, plot_in_log_space, plot_input_times_grad,
        info_string, output_dir_name):
    """Plots saliency for lagged SHIPS for each lag time at one init time.

    :param data_dict: See doc for `_plot_scalar_satellite_saliency`.
    :param saliency_dict: Same.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param colour_map_object: Same.
    :param min_colour_value: Same.
    :param max_colour_value: Same.
    :param plot_in_log_space: Same.
    :param plot_input_times_grad: Same.
    :param info_string: Same.
    :param output_dir_name: Same.
    """

    predictor_example_index = numpy.where(
        data_dict[neural_net.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]
    saliency_example_index = numpy.where(
        saliency_dict[saliency.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]

    predictor_matrices_one_example = [
        None if p is None else p[[predictor_example_index], ...]
        for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]

    if plot_input_times_grad:
        this_key = saliency.THREE_INPUT_GRAD_KEY
    else:
        this_key = saliency.THREE_SALIENCY_KEY

    saliency_matrix = saliency_dict[this_key][2][saliency_example_index, ...]

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    num_lagged_predictors = len(
        validation_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
    )
    num_model_lag_times = len(
        validation_option_dict[neural_net.SHIPS_LAG_TIMES_KEY]
    )

    forecast_predictor_names = (
        validation_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
    )
    num_forecast_predictors = (
        0 if forecast_predictor_names is None else len(forecast_predictor_names)
    )

    max_forecast_hour = (
        validation_option_dict[neural_net.SHIPS_MAX_FORECAST_HOUR_KEY]
    )
    forecast_hours = numpy.linspace(
        0, max_forecast_hour, num=int(numpy.round(max_forecast_hour / 6)) + 1,
        dtype=int
    )
    builtin_lag_times_hours = (
        validation_option_dict[neural_net.SHIPS_BUILTIN_LAG_TIMES_KEY]
    )

    figure_objects, axes_objects, pathless_output_file_names = (
        predictor_plotting.plot_lagged_ships_one_example(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            builtin_lag_times_hours=builtin_lag_times_hours,
            forecast_hours=forecast_hours,
            init_time_unix_sec=init_time_unix_sec
        )
    )

    if len(builtin_lag_times_hours) == 1:
        orig_saliency_matrix = saliency_matrix + 0.
        saliency_matrix = numpy.array([], dtype=float)

        for j in range(num_model_lag_times):
            new_saliency_matrix = numpy.expand_dims(
                orig_saliency_matrix[j, :], axis=0
            )
            new_saliency_matrix = numpy.expand_dims(new_saliency_matrix, axis=0)
            new_saliency_matrix = neural_net.ships_predictors_3d_to_4d(
                predictor_matrix_3d=new_saliency_matrix,
                num_lagged_predictors=num_lagged_predictors,
                num_builtin_lag_times=len(builtin_lag_times_hours),
                num_forecast_predictors=num_forecast_predictors,
                num_forecast_hours=len(forecast_hours)
            )[0][:, 0, ...]

            if saliency_matrix.size == 0:
                saliency_matrix = new_saliency_matrix + 0.
            else:
                saliency_matrix = numpy.concatenate(
                    (saliency_matrix, new_saliency_matrix), axis=1
                )
    else:
        saliency_matrix = neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=numpy.expand_dims(saliency_matrix, axis=0),
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=len(builtin_lag_times_hours),
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=len(forecast_hours)
        )[0][0, ...]

    panel_file_names = [''] * len(axes_objects)

    for k in range(len(axes_objects)):
        ships_plotting.plot_raw_numbers_one_init_time(
            data_matrix=saliency_matrix[k, ...],
            axes_object=axes_objects[k],
            font_size=17.5, colour_map_object=colour_map_object,
            min_colour_value=min_colour_value,
            max_colour_value=max_colour_value,
            number_format_string='.1e',
            plot_in_log_space=plot_in_log_space
        )

        if info_string != '':
            axes_objects[k].set_title('{0:s}; {1:s}'.format(
                axes_objects[k].get_title(), info_string
            ))

        panel_file_names[k] = '{0:s}/{1:s}'.format(
            output_dir_name, pathless_output_file_names[k]
        )
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

    concat_figure_file_name = (
        '{0:s}/{1:s}_{2:s}_ships_lagged_concat.jpg'
    ).format(
        output_dir_name, cyclone_id_string,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
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
        cbar_label_string='Predictor', tick_label_format_string='{0:.2g}'
    )

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    label_string = 'Absolute {0:s}'.format(
        'input times gradient' if plot_input_times_grad else 'saliency'
    )

    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string=label_string, tick_label_format_string='{0:.2g}',
        log_space=plot_in_log_space
    )


def _plot_forecast_ships_saliency(
        data_dict, saliency_dict, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec, colour_map_object, min_colour_value,
        max_colour_value, plot_in_log_space, plot_input_times_grad,
        info_string, output_dir_name):
    """Plots saliency for forecast SHIPS for each lag time at one init time.

    :param data_dict: See doc for `_plot_scalar_satellite_saliency`.
    :param saliency_dict: Same.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param colour_map_object: Same.
    :param min_colour_value: Same.
    :param max_colour_value: Same.
    :param plot_in_log_space: Same.
    :param plot_input_times_grad: Same.
    :param info_string: Same.
    :param output_dir_name: Same.
    """

    predictor_example_index = numpy.where(
        data_dict[neural_net.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]
    saliency_example_index = numpy.where(
        saliency_dict[saliency.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]

    predictor_matrices_one_example = [
        None if p is None else p[[predictor_example_index], ...]
        for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]

    if plot_input_times_grad:
        this_key = saliency.THREE_INPUT_GRAD_KEY
    else:
        this_key = saliency.THREE_SALIENCY_KEY

    saliency_matrix = saliency_dict[this_key][2][saliency_example_index, ...]

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    num_forecast_predictors = len(
        validation_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
    )
    num_model_lag_times = len(
        validation_option_dict[neural_net.SHIPS_LAG_TIMES_KEY]
    )

    lagged_predictor_names = (
        validation_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
    )
    num_lagged_predictors = (
        0 if lagged_predictor_names is None else len(lagged_predictor_names)
    )

    max_forecast_hour = (
        validation_option_dict[neural_net.SHIPS_MAX_FORECAST_HOUR_KEY]
    )
    forecast_hours = numpy.linspace(
        0, max_forecast_hour, num=int(numpy.round(max_forecast_hour / 6)) + 1,
        dtype=int
    )
    builtin_lag_times_hours = (
        validation_option_dict[neural_net.SHIPS_BUILTIN_LAG_TIMES_KEY]
    )

    figure_objects, axes_objects, pathless_output_file_names = (
        predictor_plotting.plot_forecast_ships_one_example(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            builtin_lag_times_hours=builtin_lag_times_hours,
            forecast_hours=forecast_hours,
            init_time_unix_sec=init_time_unix_sec
        )
    )

    saliency_matrix = neural_net.ships_predictors_3d_to_4d(
        predictor_matrix_3d=numpy.expand_dims(saliency_matrix, axis=0),
        num_lagged_predictors=num_lagged_predictors,
        num_builtin_lag_times=len(builtin_lag_times_hours),
        num_forecast_predictors=num_forecast_predictors,
        num_forecast_hours=len(forecast_hours)
    )[1][0, ...]

    panel_file_names = [''] * num_model_lag_times

    for k in range(num_model_lag_times):
        ships_plotting.plot_raw_numbers_one_init_time(
            data_matrix=saliency_matrix[k, ...],
            axes_object=axes_objects[k],
            font_size=17.5, colour_map_object=colour_map_object,
            min_colour_value=min_colour_value,
            max_colour_value=max_colour_value,
            number_format_string='.1e',
            plot_in_log_space=plot_in_log_space
        )

        if info_string != '':
            axes_objects[k].set_title('{0:s}; {1:s}'.format(
                axes_objects[k].get_title(), info_string
            ))

        panel_file_names[k] = '{0:s}/{1:s}'.format(
            output_dir_name, pathless_output_file_names[k]
        )
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

    concat_figure_file_name = (
        '{0:s}/{1:s}_{2:s}_ships_forecast_concat.jpg'
    ).format(
        output_dir_name, cyclone_id_string,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
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
        cbar_label_string='Predictor', tick_label_format_string='{0:.2g}'
    )

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value,
        vmax=max_colour_value
    )
    label_string = 'Absolute {0:s}'.format(
        'input times gradient' if plot_input_times_grad else 'saliency'
    )

    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string=label_string, tick_label_format_string='{0:.2g}',
        log_space=plot_in_log_space
    )


def _run(saliency_file_name, example_dir_name, normalization_file_name,
         prediction_file_name, plot_input_times_grad, spatial_colour_map_name,
         nonspatial_colour_map_name, min_colour_percentile,
         max_colour_percentile, plot_in_log_space, plot_time_diffs_if_used,
         smoothing_radius_px, output_dir_name):
    """Plots saliency maps.

    This is effectively the main method.

    :param saliency_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param normalization_file_name: Same.
    :param prediction_file_name: Same.
    :param plot_input_times_grad: Same.
    :param spatial_colour_map_name: Same.
    :param nonspatial_colour_map_name: Same.
    :param min_colour_percentile: Same.
    :param max_colour_percentile: Same.
    :param plot_in_log_space: Same.
    :param plot_time_diffs_if_used: Same.
    :param smoothing_radius_px: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(min_colour_percentile, 0.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)
    error_checking.assert_is_greater(
        max_colour_percentile, min_colour_percentile
    )

    if prediction_file_name == '':
        prediction_file_name = None

    spatial_colour_map_object = pyplot.get_cmap(spatial_colour_map_name)
    nonspatial_colour_map_object = pyplot.get_cmap(nonspatial_colour_map_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read files.
    print('Reading data from: "{0:s}"...'.format(saliency_file_name))
    saliency_dict = saliency.read_file(saliency_file_name)

    if smoothing_radius_px > 0:
        saliency_dict = _smooth_maps(
            saliency_dict=saliency_dict,
            smoothing_radius_px=smoothing_radius_px,
            plot_input_times_grad=plot_input_times_grad
        )

    model_file_name = saliency_dict[saliency.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    base_option_dict = model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    plot_time_diffs = (
        base_option_dict[neural_net.USE_TIME_DIFFS_KEY]
        and plot_time_diffs_if_used
    )
    base_option_dict[neural_net.USE_TIME_DIFFS_KEY] = False
    model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY] = base_option_dict

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    # Find example files.
    unique_cyclone_id_strings = numpy.unique(
        numpy.array(saliency_dict[saliency.CYCLONE_IDS_KEY])
    )
    num_cyclones = len(unique_cyclone_id_strings)

    unique_example_file_names = [
        example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        for c in unique_cyclone_id_strings
    ]

    # Plot saliency maps.
    for i in range(num_cyclones):
        option_dict = copy.deepcopy(base_option_dict)
        option_dict[neural_net.EXAMPLE_FILE_KEY] = unique_example_file_names[i]

        print(SEPARATOR_STRING)
        data_dict = neural_net.create_inputs(option_dict)
        print(SEPARATOR_STRING)

        example_indices = numpy.where(
            numpy.array(saliency_dict[saliency.CYCLONE_IDS_KEY]) ==
            unique_cyclone_id_strings[i]
        )[0]

        info_strings = [''] * len(example_indices)

        if prediction_file_name is not None:
            forecast_probabilities, target_classes = (
                plot_predictors.get_predictions_and_targets(
                    prediction_file_name=prediction_file_name,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    init_times_unix_sec=saliency_dict[saliency.INIT_TIMES_KEY]
                )
            )

            for j in range(len(example_indices)):
                info_strings[j] = 'RI = {0:s}; '.format(
                    'yes' if target_classes[j] == 1 else 'no'
                )
                info_strings[j] += r'$p_{RI}$'
                info_strings[j] += ' = {0:.2f}'.format(
                    forecast_probabilities[j]
                )

        for j in range(len(example_indices)):
            k = example_indices[j]

            if plot_input_times_grad:
                this_key = saliency.THREE_INPUT_GRAD_KEY
            else:
                this_key = saliency.THREE_SALIENCY_KEY

            tensor_saliency_matrices_example_j = [
                s[k, ...] for s in saliency_dict[this_key]
                if s is not None and len(s.shape) > 3
            ]

            if len(tensor_saliency_matrices_example_j) > 0:
                tensor_saliency_values_example_j = numpy.concatenate([
                    numpy.ravel(t) for t in tensor_saliency_matrices_example_j
                ])
                if plot_in_log_space:
                    tensor_saliency_values_example_j = numpy.log10(
                        1 + numpy.absolute(tensor_saliency_values_example_j)
                    )

                min_colour_value_for_tensors = numpy.percentile(
                    numpy.absolute(tensor_saliency_values_example_j),
                    min_colour_percentile
                )
                max_colour_value_for_tensors = numpy.percentile(
                    numpy.absolute(tensor_saliency_values_example_j),
                    max_colour_percentile
                )
                max_colour_value_for_tensors = max([
                    max_colour_value_for_tensors,
                    min_colour_value_for_tensors + TOLERANCE
                ])
            else:
                min_colour_value_for_tensors = 0.
                max_colour_value_for_tensors = 1.

            scalar_saliency_matrices_example_j = [
                s[k, ...] for s in saliency_dict[this_key]
                if s is not None and len(s.shape) <= 3
            ]

            if len(scalar_saliency_matrices_example_j) > 0:
                scalar_saliency_values_example_j = numpy.concatenate([
                    numpy.ravel(s) for s in scalar_saliency_matrices_example_j
                ])
                if plot_in_log_space:
                    scalar_saliency_values_example_j = numpy.log10(
                        1 + numpy.absolute(scalar_saliency_values_example_j)
                    )

                min_colour_value_for_scalars = numpy.percentile(
                    numpy.absolute(scalar_saliency_values_example_j),
                    min_colour_percentile
                )
                max_colour_value_for_scalars = numpy.percentile(
                    numpy.absolute(scalar_saliency_values_example_j),
                    max_colour_percentile
                )
                max_colour_value_for_scalars = max([
                    max_colour_value_for_scalars,
                    min_colour_value_for_scalars + TOLERANCE
                ])
            else:
                min_colour_value_for_scalars = 0.
                max_colour_value_for_scalars = 1.

            if data_dict[neural_net.PREDICTOR_MATRICES_KEY][0] is not None:
                _plot_brightness_temp_saliency(
                    data_dict=data_dict, saliency_dict=saliency_dict,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    init_time_unix_sec=
                    saliency_dict[saliency.INIT_TIMES_KEY][k],
                    normalization_table_xarray=normalization_table_xarray,
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    colour_map_object=spatial_colour_map_object,
                    min_colour_value=min_colour_value_for_tensors,
                    max_colour_value=max_colour_value_for_tensors,
                    plot_in_log_space=plot_in_log_space,
                    plot_time_diffs_at_lags=plot_time_diffs,
                    plot_input_times_grad=plot_input_times_grad,
                    info_string=info_strings[j],
                    output_dir_name=output_dir_name
                )

            if data_dict[neural_net.PREDICTOR_MATRICES_KEY][1] is not None:
                _plot_scalar_satellite_saliency(
                    data_dict=data_dict, saliency_dict=saliency_dict,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    init_time_unix_sec=
                    saliency_dict[saliency.INIT_TIMES_KEY][k],
                    colour_map_object=nonspatial_colour_map_object,
                    min_colour_value=min_colour_value_for_scalars,
                    max_colour_value=max_colour_value_for_scalars,
                    plot_in_log_space=plot_in_log_space,
                    plot_input_times_grad=plot_input_times_grad,
                    info_string=info_strings[j],
                    output_dir_name=output_dir_name
                )

            if option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY] is not None:
                _plot_lagged_ships_saliency(
                    data_dict=data_dict, saliency_dict=saliency_dict,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    init_time_unix_sec=
                    saliency_dict[saliency.INIT_TIMES_KEY][k],
                    colour_map_object=nonspatial_colour_map_object,
                    min_colour_value=min_colour_value_for_scalars,
                    max_colour_value=max_colour_value_for_scalars,
                    plot_in_log_space=plot_in_log_space,
                    plot_input_times_grad=plot_input_times_grad,
                    info_string=info_strings[j],
                    output_dir_name=output_dir_name
                )

            if (
                    option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
                    is not None
            ):
                _plot_forecast_ships_saliency(
                    data_dict=data_dict, saliency_dict=saliency_dict,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    init_time_unix_sec=
                    saliency_dict[saliency.INIT_TIMES_KEY][k],
                    colour_map_object=nonspatial_colour_map_object,
                    min_colour_value=min_colour_value_for_scalars,
                    max_colour_value=max_colour_value_for_scalars,
                    plot_in_log_space=plot_in_log_space,
                    plot_input_times_grad=plot_input_times_grad,
                    info_string=info_strings[j],
                    output_dir_name=output_dir_name
                )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_name=getattr(INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        plot_input_times_grad=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_INPUT_GRAD_ARG_NAME
        )),
        spatial_colour_map_name=getattr(
            INPUT_ARG_OBJECT, SPATIAL_COLOUR_MAP_ARG_NAME
        ),
        nonspatial_colour_map_name=getattr(
            INPUT_ARG_OBJECT, NONSPATIAL_COLOUR_MAP_ARG_NAME
        ),
        min_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_COLOUR_PERCENTILE_ARG_NAME
        ),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_COLOUR_PERCENTILE_ARG_NAME
        ),
        plot_in_log_space=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_IN_LOG_SPACE_ARG_NAME
        )),
        plot_time_diffs_if_used=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_TIME_DIFFS_ARG_NAME
        )),
        smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
