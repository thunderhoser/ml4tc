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

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MAX_COLOUR_PERCENTILE = 99.
SHIPS_FORECAST_HOURS = numpy.linspace(-12, 120, num=23, dtype=int)
SHIPS_BUILTIN_LAG_TIMES_HOURS = numpy.array([numpy.nan, 0, 1.5, 3])

COLOUR_BAR_FONT_SIZE = 12
SCALAR_SATELLITE_FONT_SIZE = 20
LAGGED_SHIPS_FONT_SIZE = 20
FORECAST_SHIPS_FONT_SIZE = 10

FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
PLOT_INPUT_GRAD_ARG_NAME = 'plot_input_times_grad'
SPATIAL_COLOUR_MAP_ARG_NAME = 'spatial_colour_map_name'
NONSPATIAL_COLOUR_MAP_ARG_NAME = 'nonspatial_colour_map_name'
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
        init_time_unix_sec, colour_map_object, plot_input_times_grad,
        output_dir_name):
    """Plots saliency for scalar satellite for each lag time at one init time.

    :param data_dict: Dictionary returned by `neural_net.create_inputs`.
    :param saliency_dict: Dictionary returned by `saliency.read_file`.
    :param model_metadata_dict: See doc for
        `predictor_plotting.plot_scalar_satellite_one_example`.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param plot_input_times_grad: Boolean flag.  If 1 (0), will plot input *
        gradient (saliency).
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

    saliency_matrices_one_example = [
        None if s is None else s[[saliency_example_index], ...]
        for s in saliency_dict[this_key]
    ]

    figure_object, axes_object = (
        predictor_plotting.plot_scalar_satellite_one_example(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            init_time_unix_sec=init_time_unix_sec
        )[:2]
    )

    all_saliency_values = numpy.concatenate([
        numpy.ravel(s) for s in saliency_matrices_one_example if s is not None
    ])
    max_absolute_colour_value = numpy.percentile(
        numpy.absolute(all_saliency_values), MAX_COLOUR_PERCENTILE
    )
    scalar_satellite_plotting.plot_pm_signs_multi_times(
        data_matrix=saliency_matrices_one_example[1][0, ...],
        axes_object=axes_object,
        font_size=SCALAR_SATELLITE_FONT_SIZE,
        colour_map_object=colour_map_object,
        max_absolute_colour_value=max_absolute_colour_value
    )

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
        vmin=0., vmax=max_absolute_colour_value
    )
    label_string = 'Absolute {0:s}'.format(
        'input times gradient' if plot_input_times_grad else 'saliency'
    )
    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string=label_string, tick_label_format_string='{0:.2g}'
    )


def _plot_brightness_temp_saliency(
        data_dict, saliency_dict, model_metadata_dict,
        cyclone_id_string, init_time_unix_sec, normalization_table_xarray,
        border_latitudes_deg_n, border_longitudes_deg_e, colour_map_object,
        plot_input_times_grad, output_dir_name):
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
    :param plot_input_times_grad: Same.
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

    saliency_matrices_one_example = [
        None if s is None else s[[saliency_example_index], ...]
        for s in saliency_dict[this_key]
    ]

    grid_latitude_matrix_deg_n = data_dict[
        neural_net.GRID_LATITUDE_MATRIX_KEY
    ][predictor_example_index, ...]

    grid_longitude_matrix_deg_e = data_dict[
        neural_net.GRID_LONGITUDE_MATRIX_KEY
    ][predictor_example_index, ...]

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
            border_longitudes_deg_e=border_longitudes_deg_e
        )
    )

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    num_model_lag_times = len(
        validation_option_dict[neural_net.SATELLITE_LAG_TIMES_KEY]
    )

    all_saliency_values = numpy.concatenate([
        numpy.ravel(s) for s in saliency_matrices_one_example if s is not None
    ])
    min_abs_contour_value = numpy.percentile(
        numpy.absolute(all_saliency_values), 100. - MAX_COLOUR_PERCENTILE
    )
    max_abs_contour_value = numpy.percentile(
        numpy.absolute(all_saliency_values), MAX_COLOUR_PERCENTILE
    )

    panel_file_names = [''] * num_model_lag_times

    for k in range(num_model_lag_times):
        min_abs_contour_value, max_abs_contour_value = (
            satellite_plotting.plot_saliency(
                saliency_matrix=saliency_matrices_one_example[0][0, ..., k, 0],
                axes_object=axes_objects[k],
                latitudes_deg_n=grid_latitude_matrix_deg_n[:, k],
                longitudes_deg_e=grid_longitude_matrix_deg_e[:, k],
                min_abs_contour_value=min_abs_contour_value,
                max_abs_contour_value=max_abs_contour_value,
                half_num_contours=10,
                colour_map_object=colour_map_object
            )
        )

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
        vmin=min_abs_contour_value, vmax=max_abs_contour_value
    )
    label_string = 'Absolute {0:s}'.format(
        'input times gradient' if plot_input_times_grad else 'saliency'
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string=label_string, tick_label_format_string='{0:.2g}'
    )


def _plot_lagged_ships_saliency(
        data_dict, saliency_dict, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec, colour_map_object, plot_input_times_grad,
        output_dir_name):
    """Plots saliency for lagged SHIPS for each lag time at one init time.

    :param data_dict: See doc for `_plot_scalar_satellite_saliency`.
    :param saliency_dict: Same.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param colour_map_object: Same.
    :param plot_input_times_grad: Same.
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

    saliency_matrices_one_example = [
        None if s is None else s[[saliency_example_index], ...]
        for s in saliency_dict[this_key]
    ]

    figure_objects, axes_objects, pathless_output_file_names = (
        predictor_plotting.plot_lagged_ships_one_example(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            builtin_lag_times_hours=SHIPS_BUILTIN_LAG_TIMES_HOURS,
            forecast_hours=SHIPS_FORECAST_HOURS,
            init_time_unix_sec=init_time_unix_sec
        )
    )

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

    saliency_matrix = neural_net.ships_predictors_3d_to_4d(
        predictor_matrix_3d=saliency_matrices_one_example[2][[0], ...],
        num_lagged_predictors=num_lagged_predictors,
        num_builtin_lag_times=len(SHIPS_BUILTIN_LAG_TIMES_HOURS),
        num_forecast_predictors=num_forecast_predictors,
        num_forecast_hours=len(SHIPS_FORECAST_HOURS)
    )[0][0, ...]

    all_saliency_values = numpy.concatenate([
        numpy.ravel(s) for s in saliency_matrices_one_example if s is not None
    ])
    max_absolute_colour_value = numpy.percentile(
        numpy.absolute(all_saliency_values), MAX_COLOUR_PERCENTILE
    )

    panel_file_names = [''] * num_model_lag_times

    for k in range(num_model_lag_times):
        ships_plotting.plot_pm_signs_one_init_time(
            data_matrix=saliency_matrix[k, ...],
            axes_object=axes_objects[k],
            font_size=LAGGED_SHIPS_FONT_SIZE,
            colour_map_object=colour_map_object,
            max_absolute_colour_value=max_absolute_colour_value
        )

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
        vmin=0., vmax=max_absolute_colour_value
    )
    label_string = 'Absolute {0:s}'.format(
        'input times gradient' if plot_input_times_grad else 'saliency'
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string=label_string, tick_label_format_string='{0:.2g}'
    )


def _plot_forecast_ships_saliency(
        data_dict, saliency_dict, model_metadata_dict, cyclone_id_string,
        init_time_unix_sec, colour_map_object, plot_input_times_grad,
        output_dir_name):
    """Plots saliency for forecast SHIPS for each lag time at one init time.

    :param data_dict: See doc for `_plot_scalar_satellite_saliency`.
    :param saliency_dict: Same.
    :param model_metadata_dict: Same.
    :param cyclone_id_string: Same.
    :param init_time_unix_sec: Same.
    :param colour_map_object: Same.
    :param plot_input_times_grad: Same.
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

    saliency_matrices_one_example = [
        None if s is None else s[[saliency_example_index], ...]
        for s in saliency_dict[this_key]
    ]

    figure_objects, axes_objects, pathless_output_file_names = (
        predictor_plotting.plot_forecast_ships_one_example(
            predictor_matrices_one_example=predictor_matrices_one_example,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            builtin_lag_times_hours=SHIPS_BUILTIN_LAG_TIMES_HOURS,
            forecast_hours=SHIPS_FORECAST_HOURS,
            init_time_unix_sec=init_time_unix_sec
        )
    )

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

    saliency_matrix = neural_net.ships_predictors_3d_to_4d(
        predictor_matrix_3d=saliency_matrices_one_example[2][[0], ...],
        num_lagged_predictors=num_lagged_predictors,
        num_builtin_lag_times=len(SHIPS_BUILTIN_LAG_TIMES_HOURS),
        num_forecast_predictors=num_forecast_predictors,
        num_forecast_hours=len(SHIPS_FORECAST_HOURS)
    )[1][0, ...]

    all_saliency_values = numpy.concatenate([
        numpy.ravel(s) for s in saliency_matrices_one_example if s is not None
    ])
    max_absolute_colour_value = numpy.percentile(
        numpy.absolute(all_saliency_values), MAX_COLOUR_PERCENTILE
    )

    panel_file_names = [''] * num_model_lag_times

    for k in range(num_model_lag_times):
        ships_plotting.plot_pm_signs_one_init_time(
            data_matrix=saliency_matrix[k, ...],
            axes_object=axes_objects[k],
            font_size=FORECAST_SHIPS_FONT_SIZE,
            colour_map_object=colour_map_object,
            max_absolute_colour_value=max_absolute_colour_value
        )

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
        vmin=0., vmax=max_absolute_colour_value
    )
    label_string = 'Absolute {0:s}'.format(
        'input times gradient' if plot_input_times_grad else 'saliency'
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string=label_string, tick_label_format_string='{0:.2g}'
    )


def _run(saliency_file_name, example_dir_name, normalization_file_name,
         plot_input_times_grad, spatial_colour_map_name,
         nonspatial_colour_map_name, smoothing_radius_px, output_dir_name):
    """Plots saliency maps.

    This is effectively the main method.

    :param saliency_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param normalization_file_name: Same.
    :param plot_input_times_grad: Same.
    :param spatial_colour_map_name: Same.
    :param nonspatial_colour_map_name: Same.
    :param smoothing_radius_px: Same.
    :param output_dir_name: Same.
    """

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
    base_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )

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

        for j in example_indices:
            if data_dict[neural_net.PREDICTOR_MATRICES_KEY][0] is not None:
                _plot_brightness_temp_saliency(
                    data_dict=data_dict, saliency_dict=saliency_dict,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    init_time_unix_sec=
                    saliency_dict[saliency.INIT_TIMES_KEY][j],
                    normalization_table_xarray=normalization_table_xarray,
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    colour_map_object=spatial_colour_map_object,
                    plot_input_times_grad=plot_input_times_grad,
                    output_dir_name=output_dir_name
                )

            if data_dict[neural_net.PREDICTOR_MATRICES_KEY][1] is not None:
                _plot_scalar_satellite_saliency(
                    data_dict=data_dict, saliency_dict=saliency_dict,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    init_time_unix_sec=
                    saliency_dict[saliency.INIT_TIMES_KEY][j],
                    colour_map_object=nonspatial_colour_map_object,
                    plot_input_times_grad=plot_input_times_grad,
                    output_dir_name=output_dir_name
                )

            if option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY] is not None:
                _plot_lagged_ships_saliency(
                    data_dict=data_dict, saliency_dict=saliency_dict,
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    init_time_unix_sec=
                    saliency_dict[saliency.INIT_TIMES_KEY][j],
                    colour_map_object=nonspatial_colour_map_object,
                    plot_input_times_grad=plot_input_times_grad,
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
                    saliency_dict[saliency.INIT_TIMES_KEY][j],
                    colour_map_object=nonspatial_colour_map_object,
                    plot_input_times_grad=plot_input_times_grad,
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
        plot_input_times_grad=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_INPUT_GRAD_ARG_NAME
        )),
        spatial_colour_map_name=getattr(
            INPUT_ARG_OBJECT, SPATIAL_COLOUR_MAP_ARG_NAME
        ),
        nonspatial_colour_map_name=getattr(
            INPUT_ARG_OBJECT, NONSPATIAL_COLOUR_MAP_ARG_NAME
        ),
        smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
