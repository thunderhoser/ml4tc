"""Plots class-activation maps."""

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
import gradcam
import neural_net
import plotting_utils
import satellite_plotting
import predictor_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MAX_COLOUR_PERCENTILE = 99.
COLOUR_BAR_FONT_SIZE = 12

FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)

GRADCAM_FILE_ARG_NAME = 'input_gradcam_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

GRADCAM_FILE_HELP_STRING = (
    'Path to file with Grad-CAM results.  Will be read by `gradcam.read_file`.'
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
COLOUR_MAP_HELP_STRING = (
    'Name of colour scheme for class activation.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
SMOOTHING_RADIUS_HELP_STRING = (
    'Smoothing radius (number of pixels) for class-activation maps.  If you do '
    'not want to smooth, make this 0 or negative.'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Images will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRADCAM_FILE_ARG_NAME, type=str, required=True,
    help=GRADCAM_FILE_HELP_STRING
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
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='BuGn',
    help=COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False, default=-1,
    help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _smooth_maps(class_activation_dict, smoothing_radius_px):
    """Smooths class-activation maps via Gaussian filter.

    :param class_activation_dict: Dictionary returned by `gradcam.read_file`.
    :param smoothing_radius_px: e-folding radius (num pixels).
    :return: class_activation_dict: Same as input but with smoothed maps.
    """

    print((
        'Smoothing class-activation maps with Gaussian filter (e-folding radius'
        ' of {0:.1f} grid cells)...'
    ).format(
        smoothing_radius_px
    ))

    class_activation_matrix = (
        class_activation_dict[gradcam.CLASS_ACTIVATION_KEY]
    )
    num_examples = class_activation_matrix.shape[0]

    for i in range(num_examples):
        class_activation_matrix[i, ...] = (
            gg_general_utils.apply_gaussian_filter(
                input_matrix=class_activation_matrix[i, ...],
                e_folding_radius_grid_cells=smoothing_radius_px
            )
        )

    class_activation_dict[gradcam.CLASS_ACTIVATION_KEY] = (
        class_activation_matrix
    )
    return class_activation_dict


def _plot_cam_one_example(
        data_dict, class_activation_dict, model_metadata_dict,
        cyclone_id_string, init_time_unix_sec, normalization_table_xarray,
        border_latitudes_deg_n, border_longitudes_deg_e, colour_map_object,
        output_dir_name):
    """Plots class-activation map for one example.

    P = number of points in border set

    :param data_dict: Dictionary returned by `neural_net.create_inputs`.
    :param class_activation_dict: Dictionary returned by `gradcam.read_file`.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `satellite_utils.parse_cyclone_id`).
    :param init_time_unix_sec: Forecast-initialization time.
    :param normalization_table_xarray: xarray table returned by
        `normalization.read_file`.
    :param border_latitudes_deg_n: length-P numpy array of latitudes
        (deg north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes
        (deg east).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    predictor_example_index = numpy.where(
        data_dict[neural_net.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]
    cam_example_index = numpy.where(
        class_activation_dict[gradcam.INIT_TIMES_KEY] == init_time_unix_sec
    )[0][0]

    predictor_matrices_one_example = [
        None if p is None else p[[predictor_example_index], ...]
        for p in data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    ]
    cam_matrix_one_example = (
        class_activation_dict[gradcam.CLASS_ACTIVATION_KEY][
            [cam_example_index], ...
        ]
    )
    cam_matrix_one_example_log10 = numpy.log10(
        numpy.maximum(cam_matrix_one_example, 1e-6)
    )

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
    max_contour_value = numpy.percentile(
        cam_matrix_one_example_log10, MAX_COLOUR_PERCENTILE
    )
    min_contour_value = numpy.percentile(
        cam_matrix_one_example_log10, 100 - MAX_COLOUR_PERCENTILE
    )

    panel_file_names = [''] * num_model_lag_times

    for k in range(num_model_lag_times):
        satellite_plotting.plot_class_activation(
            class_activation_matrix=cam_matrix_one_example_log10[0, ...],
            axes_object=axes_objects[k],
            latitudes_deg_n=grid_latitude_matrix_deg_n[:, k],
            longitudes_deg_e=grid_longitude_matrix_deg_e[:, k],
            min_contour_value=min_contour_value,
            max_contour_value=max_contour_value,
            num_contours=15, colour_map_object=colour_map_object
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
    concat_figure_file_name = '{0:s}/{1:s}_{2:s}_brightness_temp.jpg'.format(
        output_dir_name, cyclone_id_string, init_time_string
    )
    plotting_utils.concat_panels(
        panel_file_names=panel_file_names,
        concat_figure_file_name=concat_figure_file_name
    )

    this_colour_map_object, colour_norm_object = (
        satellite_plotting.get_colour_scheme()
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=this_colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string='Brightness temp (K)',
        tick_label_format_string='{0:d}'
    )

    colour_norm_object = pyplot.Normalize(
        vmin=min_contour_value, vmax=max_contour_value
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=COLOUR_BAR_FONT_SIZE,
        cbar_label_string=r'log$_{10}$(class activation)',
        tick_label_format_string='{0:.2g}'
    )


def _run(gradcam_file_name, example_dir_name, normalization_file_name,
         colour_map_name, smoothing_radius_px, output_dir_name):
    """Plots class-activation maps.

    This is effectively the main method.

    :param gradcam_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param normalization_file_name: Same.
    :param colour_map_name: Same.
    :param smoothing_radius_px: Same.
    :param output_dir_name: Same.
    """

    colour_map_object = pyplot.get_cmap(colour_map_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read files.
    print('Reading data from: "{0:s}"...'.format(gradcam_file_name))
    class_activation_dict = gradcam.read_file(gradcam_file_name)

    if smoothing_radius_px > 0:
        class_activation_dict = _smooth_maps(
            class_activation_dict=class_activation_dict,
            smoothing_radius_px=smoothing_radius_px
        )

    model_file_name = class_activation_dict[gradcam.MODEL_FILE_KEY]
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
        numpy.array(class_activation_dict[gradcam.CYCLONE_IDS_KEY])
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

    # Plot class-activation maps.
    for i in range(num_cyclones):
        option_dict = copy.deepcopy(base_option_dict)
        option_dict[neural_net.EXAMPLE_FILE_KEY] = unique_example_file_names[i]

        print(SEPARATOR_STRING)
        data_dict = neural_net.create_inputs(option_dict)
        print(SEPARATOR_STRING)

        example_indices = numpy.where(
            numpy.array(class_activation_dict[gradcam.CYCLONE_IDS_KEY]) ==
            unique_cyclone_id_strings[i]
        )[0]

        for j in example_indices:
            _plot_cam_one_example(
                data_dict=data_dict,
                class_activation_dict=class_activation_dict,
                model_metadata_dict=model_metadata_dict,
                cyclone_id_string=unique_cyclone_id_strings[i],
                init_time_unix_sec=
                class_activation_dict[gradcam.INIT_TIMES_KEY][j],
                normalization_table_xarray=normalization_table_xarray,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                colour_map_object=colour_map_object,
                output_dir_name=output_dir_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        gradcam_file_name=getattr(INPUT_ARG_OBJECT, GRADCAM_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
