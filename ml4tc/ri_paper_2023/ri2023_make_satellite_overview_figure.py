"""Makes satellite-overview figure for 2023 RI (rapid intensification) paper."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4tc.io import border_io
from ml4tc.io import cira_satellite_io
from ml4tc.utils import example_utils
from ml4tc.utils import satellite_utils
from ml4tc.plotting import plotting_utils
from ml4tc.plotting import satellite_plotting

NUM_CROPPED_ROWS = 380
NUM_CROPPED_COLUMNS = 540
CLIMO_MEAN_KELVINS = 269.80128466

IMAGE_CENTER_MARKER = 's'
IMAGE_CENTER_MARKER_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
IMAGE_CENTER_MARKER_SIZE = 14
IMAGE_CENTER_MARKER_EDGE_WIDTH = 2
IMAGE_CENTER_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

ACTUAL_CENTER_MARKER = '*'
ACTUAL_CENTER_MARKER_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
ACTUAL_CENTER_MARKER_SIZE = 24
ACTUAL_CENTER_MARKER_EDGE_WIDTH = 2
ACTUAL_CENTER_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

INPUT_FILE_ARG_NAME = 'input_raw_satellite_file_name'
OUTPUT_DIR_ARG_NAME = 'output_figure_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `cira_satellite_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(raw_satellite_file_name, output_dir_name):
    """Makes satellite-overview figure for 2023 RI (rapid intensification) ppr.

    This is effectively the main method.

    :param raw_satellite_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    print('Reading data from: "{0:s}"...'.format(raw_satellite_file_name))
    satellite_table_xarray = cira_satellite_io.read_file(
        raw_satellite_file_name
    )
    st = satellite_table_xarray

    good_latitude_flags = numpy.all(
        numpy.isfinite(st[satellite_utils.GRID_LATITUDE_KEY].values), axis=1
    )
    good_longitude_flags = numpy.all(
        numpy.isfinite(st[satellite_utils.GRID_LONGITUDE_KEY].values), axis=1
    )
    good_coord_indices = numpy.where(
        numpy.logical_and(good_latitude_flags, good_longitude_flags)
    )[0]

    time_subindex = int(numpy.floor(
        0.25 * len(good_coord_indices)
    ))
    time_index = good_coord_indices[time_subindex]
    st = st.isel(indexers={
        satellite_utils.TIME_DIM: numpy.array([time_index], dtype=int)
    })

    brightness_temp_matrix_kelvins = (
        st[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values[0, ...]
    )
    grid_latitudes_deg_n = (
        st[satellite_utils.GRID_LATITUDE_KEY].values[0, :]
    )
    grid_longitudes_deg_e = (
        st[satellite_utils.GRID_LONGITUDE_KEY].values[0, :]
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    satellite_plotting.plot_2d_grid(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        latitude_array_deg_n=grid_latitudes_deg_n,
        longitude_array_deg_e=grid_longitudes_deg_e,
        cbar_orientation_string='vertical',
        font_size=FONT_SIZE, plot_motion_arrow=True,
        u_motion_m_s01=st[satellite_utils.STORM_MOTION_U_KEY].values[0],
        v_motion_m_s01=st[satellite_utils.STORM_MOTION_V_KEY].values[0]
    )
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=numpy.ravel(grid_latitudes_deg_n),
        plot_longitudes_deg_e=numpy.ravel(grid_longitudes_deg_e),
        axes_object=axes_object, parallel_spacing_deg=2.,
        meridian_spacing_deg=2., font_size=FONT_SIZE
    )
    image_center_handle = axes_object.plot(
        0.5, 0.5, linestyle='None',
        marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
        markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgewidth=IMAGE_CENTER_MARKER_EDGE_WIDTH,
        markeredgecolor=IMAGE_CENTER_MARKER_EDGE_COLOUR,
        transform=axes_object.transAxes, zorder=1e10
    )[0]
    actual_center_handle = axes_object.plot(
        st[satellite_utils.STORM_LONGITUDE_KEY].values[0],
        st[satellite_utils.STORM_LATITUDE_KEY].values[0],
        linestyle='None',
        marker=ACTUAL_CENTER_MARKER, markersize=ACTUAL_CENTER_MARKER_SIZE,
        markerfacecolor=ACTUAL_CENTER_MARKER_COLOUR,
        markeredgewidth=ACTUAL_CENTER_MARKER_EDGE_WIDTH,
        markeredgecolor=ACTUAL_CENTER_MARKER_EDGE_COLOUR,
        zorder=1e10
    )[0]

    legend_handles = [image_center_handle, actual_center_handle]
    legend_strings = ['Image center', 'TC center']
    axes_object.legend(
        legend_handles, legend_strings,
        loc='lower left', bbox_to_anchor=(0, 0),
        fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )
    axes_object.set_title('Raw image (roughly TC-centered)')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')

    panel_file_names = ['{0:s}/uncropped.jpg'.format(output_dir_name)]
    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    st = satellite_utils.crop_images_around_storm_centers(
        satellite_table_xarray=st,
        num_cropped_rows=NUM_CROPPED_ROWS,
        num_cropped_columns=NUM_CROPPED_COLUMNS
    )
    brightness_temp_matrix_kelvins = (
        st[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values[0, ...]
    )
    grid_latitudes_deg_n = (
        st[satellite_utils.GRID_LATITUDE_KEY].values[0, :]
    )
    grid_longitudes_deg_e = (
        st[satellite_utils.GRID_LONGITUDE_KEY].values[0, :]
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    satellite_plotting.plot_2d_grid(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        latitude_array_deg_n=grid_latitudes_deg_n,
        longitude_array_deg_e=grid_longitudes_deg_e,
        cbar_orientation_string='vertical',
        font_size=FONT_SIZE, plot_motion_arrow=True,
        u_motion_m_s01=st[satellite_utils.STORM_MOTION_U_KEY].values[0],
        v_motion_m_s01=st[satellite_utils.STORM_MOTION_V_KEY].values[0]
    )
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=numpy.ravel(grid_latitudes_deg_n),
        plot_longitudes_deg_e=numpy.ravel(grid_longitudes_deg_e),
        axes_object=axes_object, parallel_spacing_deg=2.,
        meridian_spacing_deg=2., font_size=FONT_SIZE
    )

    axes_object.set_title('Cropped image (exactly TC-centered)')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    panel_file_names.append('{0:s}/cropped.jpg'.format(output_dir_name))
    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    print('u = {0:.2f} m/s ... v = {1:.2f} m/s'.format(
        st[satellite_utils.STORM_MOTION_U_KEY].values[0],
        st[satellite_utils.STORM_MOTION_V_KEY].values[0]
    ))

    (
        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e
    ) = example_utils.rotate_satellite_grid(
        center_latitude_deg_n=st[satellite_utils.STORM_LATITUDE_KEY].values[0],
        center_longitude_deg_e=
        st[satellite_utils.STORM_LONGITUDE_KEY].values[0],
        east_velocity_m_s01=st[satellite_utils.STORM_MOTION_U_KEY].values[0],
        north_velocity_m_s01=st[satellite_utils.STORM_MOTION_V_KEY].values[0],
        num_rows=len(grid_latitudes_deg_n),
        num_columns=len(grid_longitudes_deg_e)
    )

    brightness_temp_matrix_kelvins = example_utils.rotate_satellite_image(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        orig_latitudes_deg_n=grid_latitudes_deg_n,
        orig_longitudes_deg_e=grid_longitudes_deg_e,
        new_latitude_matrix_deg_n=grid_latitude_matrix_deg_n,
        new_longitude_matrix_deg_e=grid_longitude_matrix_deg_e,
        fill_value=CLIMO_MEAN_KELVINS
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    satellite_plotting.plot_2d_grid(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        latitude_array_deg_n=grid_latitude_matrix_deg_n,
        longitude_array_deg_e=grid_longitude_matrix_deg_e,
        cbar_orientation_string='vertical',
        font_size=FONT_SIZE, plot_motion_arrow=True
    )
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=numpy.ravel(grid_latitudes_deg_n),
        plot_longitudes_deg_e=numpy.ravel(grid_longitudes_deg_e),
        axes_object=axes_object, parallel_spacing_deg=2.,
        meridian_spacing_deg=2., font_size=FONT_SIZE
    )

    x_min, x_max = axes_object.get_xlim()
    axes_object.set_xlim(x_min - 4, x_max + 4)
    y_min, y_max = axes_object.get_ylim()
    axes_object.set_ylim(y_min - 5, y_max + 5)

    axes_object.set_title('Rotated image')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(c)')

    panel_file_names.append('{0:s}/rotated.jpg'.format(output_dir_name))
    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    satellite_plotting.plot_2d_grid(
        brightness_temp_matrix_kelvins=numpy.flip(
            brightness_temp_matrix_kelvins, axis=0
        ),
        axes_object=axes_object,
        latitude_array_deg_n=grid_latitude_matrix_deg_n,
        longitude_array_deg_e=grid_longitude_matrix_deg_e,
        cbar_orientation_string='vertical',
        font_size=FONT_SIZE, plot_motion_arrow=False
    )
    axes_object.set_xticks([], [])
    axes_object.set_yticks([], [])

    axes_object.set_title(
        'Flipped image\n(done only for southern-hemisphere TCs)'
    )
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')

    panel_file_names.append('{0:s}/flipped.jpg'.format(output_dir_name))
    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    for this_file_name in panel_file_names:
        imagemagick_utils.resize_image(
            input_file_name=this_file_name, output_file_name=this_file_name,
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/satellite_overview.jpg'.format(
        output_dir_name
    )
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
        raw_satellite_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
