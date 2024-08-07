"""Plots composite saliency map."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import general_utils as gg_general_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4tc.utils import normalization
from ml4tc.machine_learning import saliency
from ml4tc.machine_learning import neural_net
from ml4tc.plotting import plotting_utils
from ml4tc.plotting import satellite_plotting
from ml4tc.plotting import predictor_plotting

TOLERANCE = 1e-10
MAX_COLOUR_PERCENTILE = 99.
COLOUR_BAR_FONT_SIZE = 12

FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
PLOT_INPUT_GRAD_ARG_NAME = 'plot_input_times_grad'
SPATIAL_COLOUR_MAP_ARG_NAME = 'spatial_colour_map_name'
NONSPATIAL_COLOUR_MAP_ARG_NAME = 'nonspatial_colour_map_name'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file.  Will be read by `saliency.read_composite_file`.'
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


def _plot_brightness_temp_saliency(
        saliency_dict, model_metadata_dict, normalization_table_xarray,
        colour_map_object, plot_input_times_grad, output_dir_name):
    """Plots saliency for brightness temp for each lag time at one init time.

    :param saliency_dict: See doc for `_plot_scalar_satellite_saliency`.
    :param model_metadata_dict: Same.
    :param normalization_table_xarray: xarray table returned by
        `normalization.read_file`.
    :param colour_map_object: See doc for `_plot_scalar_satellite_saliency`.
    :param plot_input_times_grad: Same.
    :param output_dir_name: Same.
    """

    predictor_matrices = [
        None if p is None else numpy.expand_dims(p, axis=0)
        for p in saliency_dict[saliency.THREE_PREDICTORS_KEY]
    ]

    if plot_input_times_grad:
        this_key = saliency.THREE_INPUT_GRAD_KEY
    else:
        this_key = saliency.THREE_SALIENCY_KEY

    saliency_matrices = [
        None if p is None else numpy.expand_dims(p, axis=0)
        for p in saliency_dict[this_key]
    ]
    num_lag_times = predictor_matrices[0].shape[3]

    grid_latitudes_deg_n = numpy.linspace(
        -10, 10, num=predictor_matrices[0].shape[1], dtype=float
    )
    grid_latitude_matrix_deg_n = numpy.expand_dims(grid_latitudes_deg_n, axis=1)
    grid_latitude_matrix_deg_n = numpy.repeat(
        grid_latitude_matrix_deg_n, axis=1, repeats=num_lag_times
    )

    grid_longitudes_deg_e = numpy.linspace(
        300, 320, num=predictor_matrices[0].shape[2], dtype=float
    )
    grid_longitude_matrix_deg_e = numpy.expand_dims(
        grid_longitudes_deg_e, axis=1
    )
    grid_longitude_matrix_deg_e = numpy.repeat(
        grid_longitude_matrix_deg_e, axis=1, repeats=num_lag_times
    )

    figure_objects, axes_objects, pathless_output_file_names = (
        predictor_plotting.plot_brightness_temp_one_example(
            predictor_matrices_one_example=predictor_matrices,
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string='2005AL12', init_time_unix_sec=0,
            grid_latitude_matrix_deg_n=grid_latitude_matrix_deg_n,
            grid_longitude_matrix_deg_e=grid_longitude_matrix_deg_e,
            normalization_table_xarray=normalization_table_xarray,
            border_latitudes_deg_n=numpy.array([20.]),
            border_longitudes_deg_e=numpy.array([330.])
        )
    )

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    num_model_lag_times = len(
        validation_option_dict[neural_net.SATELLITE_LAG_TIMES_KEY]
    )

    tensor_saliency_matrices = [
        sm for sm in saliency_matrices
        if sm is not None and len(sm.shape) > 3
    ]

    if len(tensor_saliency_matrices) > 0:
        tensor_saliency_values = numpy.concatenate([
            numpy.ravel(t) for t in tensor_saliency_matrices
        ])

        min_colour_value_for_tensors = numpy.percentile(
            numpy.absolute(tensor_saliency_values),
            100. - MAX_COLOUR_PERCENTILE
        )
        max_colour_value_for_tensors = numpy.percentile(
            numpy.absolute(tensor_saliency_values),
            MAX_COLOUR_PERCENTILE
        )
        max_colour_value_for_tensors = max([
            max_colour_value_for_tensors,
            min_colour_value_for_tensors + TOLERANCE
        ])
    else:
        min_colour_value_for_tensors = 0.
        max_colour_value_for_tensors = 1.

    # TODO(thunderhoser): If I ever decide to plot saliency maps for scalars
    # here, I will need to fix the dimensions in composite_saliency_maps.py.
    # Currently, channel axes for SHIPS forecast variables (for example)
    # combines both forecast time and variable, so PMM will not work properly.
    # Channel axis should correspond only to variables.

    # scalar_saliency_matrices = [
    #     sm for sm in saliency_matrices
    #     if sm is not None and len(sm.shape) <= 3
    # ]
    #
    # if len(scalar_saliency_matrices) > 0:
    #     scalar_saliency_values = numpy.concatenate([
    #         numpy.ravel(t) for t in scalar_saliency_matrices
    #     ])
    #
    #     min_colour_value_for_scalars = numpy.percentile(
    #         numpy.absolute(scalar_saliency_values),
    #         100. - MAX_COLOUR_PERCENTILE
    #     )
    #     max_colour_value_for_scalars = numpy.percentile(
    #         numpy.absolute(scalar_saliency_values),
    #         MAX_COLOUR_PERCENTILE
    #     )
    #     max_colour_value_for_scalars = max([
    #         max_colour_value_for_scalars,
    #         min_colour_value_for_scalars + TOLERANCE
    #     ])
    # else:
    #     min_colour_value_for_scalars = 0.
    #     max_colour_value_for_scalars = 1.

    panel_file_names = [''] * num_model_lag_times

    for k in range(num_model_lag_times):
        satellite_plotting.plot_saliency(
            saliency_matrix=saliency_matrices[0][0, ..., k, 0],
            axes_object=axes_objects[k],
            latitude_array_deg_n=grid_latitude_matrix_deg_n[:, k],
            longitude_array_deg_e=grid_longitude_matrix_deg_e[:, k],
            min_abs_contour_value=min_colour_value_for_tensors,
            max_abs_contour_value=max_colour_value_for_tensors,
            half_num_contours=10,
            colour_map_object=colour_map_object
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

    concat_figure_file_name = '{0:s}/brightness_temp_concat.jpg'.format(
        output_dir_name
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
        vmin=min_colour_value_for_tensors, vmax=max_colour_value_for_tensors
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


def _run(saliency_file_name, normalization_file_name, plot_input_times_grad,
         spatial_colour_map_name, nonspatial_colour_map_name,
         smoothing_radius_px, output_dir_name):
    """Plots composite saliency map.

    This is effectively the main method.

    :param saliency_file_name: See documentation at top of file.
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
    saliency_dict = saliency.read_composite_file(saliency_file_name)

    if plot_input_times_grad:
        this_key = saliency.THREE_INPUT_GRAD_KEY
    else:
        this_key = saliency.THREE_SALIENCY_KEY

    if smoothing_radius_px > 0 and saliency_dict[this_key][0] is not None:
        print((
            'Smoothing maps with Gaussian filter (e-folding radius of {0:.1f} '
            'pixels)...'
        ).format(smoothing_radius_px))

        num_lag_times = saliency_dict[this_key][0].shape[-2]

        for k in range(num_lag_times):
            saliency_dict[this_key][0][..., k, 0] = (
                gg_general_utils.apply_gaussian_filter(
                    input_matrix=saliency_dict[this_key][0][..., k, 0],
                    e_folding_radius_grid_cells=smoothing_radius_px
                )
            )

    model_file_name = saliency_dict[saliency.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )
    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    # Plot saliency map.
    _plot_brightness_temp_saliency(
        saliency_dict=saliency_dict, model_metadata_dict=model_metadata_dict,
        normalization_table_xarray=normalization_table_xarray,
        colour_map_object=spatial_colour_map_object,
        plot_input_times_grad=plot_input_times_grad,
        output_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_name=getattr(INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
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
