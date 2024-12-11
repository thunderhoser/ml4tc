"""Helper methods for plotting (mostly 2-D georeferenced maps)."""

import os
import sys
import shutil
import numpy
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import gg_plotting_utils
import imagemagick_utils

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

GRID_LINE_WIDTH = 1.
GRID_LINE_COLOUR = numpy.full(3, 0.)
DEFAULT_PARALLEL_SPACING_DEG = 2.
DEFAULT_MERIDIAN_SPACING_DEG = 2.

DEFAULT_BORDER_WIDTH = 2.
DEFAULT_BORDER_Z_ORDER = -1e8
DEFAULT_BORDER_COLOUR = numpy.array([139, 69, 19], dtype=float) / 255

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)


def plot_grid_lines(
        plot_latitudes_deg_n, plot_longitudes_deg_e, axes_object,
        parallel_spacing_deg=DEFAULT_PARALLEL_SPACING_DEG,
        meridian_spacing_deg=DEFAULT_MERIDIAN_SPACING_DEG,
        font_size=DEFAULT_FONT_SIZE):
    """Adds grid lines (parallels and meridians) to plot.

    :param plot_latitudes_deg_n: 1-D numpy array of latitudes in plot (deg N).
    :param plot_longitudes_deg_e: 1-D numpy array of longitudes in plot (deg E).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param parallel_spacing_deg: Spacing between adjacent parallels.
    :param meridian_spacing_deg: Spacing between adjacent meridians.
    :param font_size: Font size.
    """

    error_checking.assert_is_numpy_array(plot_latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(plot_latitudes_deg_n)
    error_checking.assert_is_numpy_array(
        plot_longitudes_deg_e, num_dimensions=1
    )
    plot_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        plot_longitudes_deg_e, allow_nan=False
    )

    error_checking.assert_is_greater(parallel_spacing_deg, 0.)
    error_checking.assert_is_greater(meridian_spacing_deg, 0.)
    error_checking.assert_is_greater(font_size, 0.)

    parallels_deg_n = numpy.unique(number_rounding.round_to_nearest(
        plot_latitudes_deg_n, parallel_spacing_deg
    ))
    parallels_deg_n = parallels_deg_n[
        parallels_deg_n >= numpy.min(plot_latitudes_deg_n)
    ]
    parallels_deg_n = parallels_deg_n[
        parallels_deg_n <= numpy.max(plot_latitudes_deg_n)
    ]
    parallel_label_strings = [
        '{0:.1f}'.format(p) if parallel_spacing_deg < 1.
        else '{0:d}'.format(int(numpy.round(p)))
        for p in parallels_deg_n
    ]
    parallel_label_strings = [
        s + r'$^{\circ}$' for s in parallel_label_strings
    ]

    meridians_deg_e = numpy.unique(
        number_rounding.round_to_nearest(
            plot_longitudes_deg_e, meridian_spacing_deg
        )
    )
    meridians_deg_e = meridians_deg_e[
        meridians_deg_e >= numpy.min(plot_longitudes_deg_e)
    ]
    meridians_deg_e = meridians_deg_e[
        meridians_deg_e <= numpy.max(plot_longitudes_deg_e)
    ]
    meridian_label_strings = [
        '{0:.1f}'.format(m) if meridian_spacing_deg < 1.
        else '{0:d}'.format(int(numpy.round(m)))
        for m in meridians_deg_e
    ]
    meridian_label_strings = [
        s + r'$^{\circ}$' for s in meridian_label_strings
    ]

    axes_object.set_yticks(parallels_deg_n)
    axes_object.set_yticklabels(
        parallel_label_strings, fontdict={'fontsize': font_size}
    )

    axes_object.set_xticks(meridians_deg_e)
    axes_object.set_xticklabels(
        meridian_label_strings, fontdict={'fontsize': font_size},
        rotation=90.
    )

    axes_object.grid(
        which='major', axis='both', linestyle='--',
        linewidth=GRID_LINE_WIDTH, color=GRID_LINE_COLOUR
    )

    axes_object.set_xlim(
        numpy.min(plot_longitudes_deg_e), numpy.max(plot_longitudes_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(plot_latitudes_deg_n), numpy.max(plot_latitudes_deg_n)
    )


def plot_borders(
        border_latitudes_deg_n, border_longitudes_deg_e, axes_object,
        line_colour=DEFAULT_BORDER_COLOUR, line_width=DEFAULT_BORDER_WIDTH,
        z_order=DEFAULT_BORDER_Z_ORDER):
    """Adds borders to plot.

    P = number of points in border set

    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param line_colour: Line colour.
    :param line_width: Line width.
    :param z_order: z-order (lower values put borders near "back" of plot, and
        higher values put borders near "front").
    """

    error_checking.assert_is_numpy_array(
        border_latitudes_deg_n, num_dimensions=1
    )
    error_checking.assert_is_valid_lat_numpy_array(
        border_latitudes_deg_n, allow_nan=True
    )

    expected_dim = numpy.array([len(border_latitudes_deg_n)], dtype=int)
    error_checking.assert_is_numpy_array(
        border_longitudes_deg_e, exact_dimensions=expected_dim
    )
    border_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        border_longitudes_deg_e, allow_nan=True
    )

    axes_object.plot(
        border_longitudes_deg_e, border_latitudes_deg_n, color=line_colour,
        linestyle='solid', linewidth=line_width, zorder=z_order
    )


def add_colour_bar(
        figure_file_name, colour_map_object, colour_norm_object,
        orientation_string, font_size, cbar_label_string,
        tick_label_format_string='{0:.2g}', log_space=False,
        temporary_cbar_file_name=None):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_map_object: See doc for `gg_plotting_utils.plot_colour_bar`.
    :param colour_norm_object: Same.
    :param orientation_string: Same.
    :param font_size: Same.
    :param cbar_label_string: Label for colour bar.
    :param tick_label_format_string: Number format for tick labels.  A valid
        example is '{0:.2g}'.
    :param log_space: Boolean flag.  If True (False), values are scaled
        logarithmically (linearly).
    :param temporary_cbar_file_name: Path to temporary image file where colour
        bar will be saved.  If None, will determine this on the fly.
    """

    error_checking.assert_is_boolean(log_space)

    this_image_matrix = Image.open(figure_file_name)
    figure_width_px, figure_height_px = this_image_matrix.size
    figure_width_inches = float(figure_width_px) / FIGURE_RESOLUTION_DPI
    figure_height_inches = float(figure_height_px) / FIGURE_RESOLUTION_DPI

    extra_figure_object, extra_axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )
    extra_axes_object.axis('off')

    if hasattr(colour_norm_object, 'boundaries'):
        dummy_values = numpy.array([
            colour_norm_object.boundaries[0], colour_norm_object.boundaries[-1]
        ])
    else:
        dummy_values = numpy.array([
            colour_norm_object.vmin, colour_norm_object.vmax
        ])

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=extra_axes_object, data_matrix=dummy_values,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=orientation_string,
        extend_min=False, extend_max=False, fraction_of_axis_length=1.25,
        font_size=font_size
    )

    tick_values = colour_bar_object.get_ticks()
    if 'd' in tick_label_format_string:
        tick_values = numpy.round(tick_values).astype(int)

    if log_space:
        tick_strings = [
            tick_label_format_string.format(10 ** v - 1) for v in tick_values
        ]
    else:
        tick_strings = [tick_label_format_string.format(v) for v in tick_values]

    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)
    colour_bar_object.set_label(cbar_label_string, fontsize=font_size)

    if temporary_cbar_file_name is None:
        temporary_cbar_file_name = '{0:s}_cbar.jpg'.format(
            '.'.join(figure_file_name.split('.')[:-1])
        )

    extra_figure_object.savefig(
        temporary_cbar_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(extra_figure_object)

    print('Concatenating colour bar to: "{0:s}"...'.format(figure_file_name))

    if orientation_string == 'vertical':
        num_panel_rows = 1
        num_panel_columns = 2
    else:
        num_panel_rows = 2
        num_panel_columns = 1

    imagemagick_utils.concatenate_images(
        input_file_names=[figure_file_name, temporary_cbar_file_name],
        output_file_name=figure_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
        extra_args_string='-gravity Center'
    )

    os.remove(temporary_cbar_file_name)
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name
    )


def concat_panels(panel_file_names, concat_figure_file_name):
    """Concatenates panels into one figure.

    :param panel_file_names: 1-D list of paths to input image files.
    :param concat_figure_file_name: Path to output image file.
    """

    error_checking.assert_is_string_list(panel_file_names)
    error_checking.assert_is_string(concat_figure_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=concat_figure_file_name
    )

    print('Concatenating panels to: "{0:s}"...'.format(
        concat_figure_file_name
    ))

    num_panels = len(panel_file_names)
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_panels)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_panels) / num_panel_rows
    ))

    if num_panels == 1:
        shutil.move(panel_file_names[0], concat_figure_file_name)
    else:
        imagemagick_utils.concatenate_images(
            input_file_names=panel_file_names,
            num_panel_rows=num_panel_rows,
            num_panel_columns=num_panel_columns,
            output_file_name=concat_figure_file_name
        )

    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name
    )

    if num_panels == 1:
        return

    for this_panel_file_name in panel_file_names:
        os.remove(this_panel_file_name)
