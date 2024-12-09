"""Plots SHIPS data."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import ships_io
from ml4tc.utils import general_utils
from ml4tc.utils import example_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

VARIABLE_ABBREV_TO_VERBOSE = {
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M10C_KEY:
        r'Pct T$_{b} <$ -10$^{\circ}$C, 50-200 km',
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M20C_KEY:
        r'Pct T$_{b} <$ -20$^{\circ}$C, 50-200 km',
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M30C_KEY:
        r'Pct T$_{b} <$ -30$^{\circ}$C, 50-200 km',
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M40C_KEY:
        r'Pct T$_{b} <$ -40$^{\circ}$C, 50-200 km',
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M50C_KEY:
        r'Pct T$_{b} <$ -50$^{\circ}$C, 50-200 km',
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M60C_KEY:
        r'Pct T$_{b} <$ -60$^{\circ}$C, 50-200 km',
    ships_io.SATELLITE_TEMP_0TO200KM_KEY: r'Mean T$_{b}$, 0-200 km',
    ships_io.SATELLITE_TEMP_0TO200KM_STDEV_KEY: r'Stdev T$_{b}$, 0-200 km',
    ships_io.SATELLITE_TEMP_100TO300KM_KEY: r'Mean T$_{b}$, 100-300 km',
    ships_io.SATELLITE_TEMP_100TO300KM_STDEV_KEY: r'Stdev T$_{b}$, 100-300 km',
    ships_io.SATELLITE_MAX_TEMP_0TO30KM_KEY: r'Max T$_{b}$, 0-30 km',
    ships_io.SATELLITE_MEAN_TEMP_0TO30KM_KEY: r'Mean T$_{b}$, 0-30 km',
    ships_io.SATELLITE_MIN_TEMP_20TO120KM_KEY: r'Min T$_{b}$, 20-120 km',
    ships_io.SATELLITE_MEAN_TEMP_20TO120KM_KEY: r'Mean T$_{b}$, 20-120 km',
    ships_io.SATELLITE_MIN_TEMP_RADIUS_KEY: r'Radius of min T$_{b}$',
    ships_io.SATELLITE_MAX_TEMP_RADIUS_KEY: r'Radius of max T$_{b}$',

    ships_io.INTENSITY_CHANGE_6HOURS_KEY: 'INCV (6-hour TC-intensity change)',
    ships_io.TEMP_GRADIENT_850TO700MB_INNER_RING_KEY:
        'TGRD (low-level, inner-ring temp gradient)',
    ships_io.SHEAR_850TO200MB_INNER_RING_GNRL_KEY:
        'SHGC (deep-layer, inner-ring, no-vortex shear)',
    ships_io.TEMP_200MB_OUTER_RING_KEY: 'T2OO (200-mb outer-ring temp)',
    ships_io.SHEAR_850TO500MB_U_KEY: r'SHRS/SHTS (low-to-mid-level $u$-shear)',
    ships_io.W_WIND_0TO15KM_INNER_RING_KEY:
        r'VVAC (full-column, inner-ring, no-vortex $w$-wind)',
    ships_io.OCEAN_AGE_KEY: 'OAGE (ocean age)',
    ships_io.MAX_TAN_WIND_850MB_KEY: 'TWXC (max 850-mb tangential wind)',
    ships_io.INTENSITY_KEY: 'VMAX (TC intensity)',
    ships_io.MERGED_OHC_KEY: 'Best OHC (ocean heat content)',
    ships_io.MERGED_SST_KEY: 'Best SST (sea-surface temp)',
    ships_io.FORECAST_LATITUDE_KEY: 'LAT (TC-center latitude)',
    ships_io.MAX_PTTL_INTENSITY_KEY: 'VMPI (max potential intensity)'
}

DEFAULT_FCST_HOUR_TICK_FONT_SIZE = 30
DEFAULT_LAG_TIME_TICK_FONT_SIZE = 30
DEFAULT_PREDICTOR_TICK_FONT_SIZE = 20

MIN_NORMALIZED_VALUE = -3.
MAX_NORMALIZED_VALUE = 3.
COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)


def plot_lagged_predictors_one_init_time(
        example_table_xarray, init_time_index, predictor_indices,
        lag_time_tick_font_size=DEFAULT_LAG_TIME_TICK_FONT_SIZE,
        predictor_tick_font_size=DEFAULT_PREDICTOR_TICK_FONT_SIZE,
        info_string=None, figure_object=None, axes_object=None
):
    """Plots lagged predictors for one initialization time.

    :param example_table_xarray: xarray table in format returned by
        `example_io.read_file`.
    :param init_time_index: Index of initial time to plot.
    :param predictor_indices: 1-D numpy array with indices of predictors to
        plot.
    :param lag_time_tick_font_size: Font size for tick labels on y-axis.
    :param predictor_tick_font_size: Font size for tick labels on x-axis.
    :param info_string: Info string (to be appended to title).
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).  If None, will create new figure.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If None, will create new
        axes.
    :return: figure_object: See input doc.
    :return: axes_object: See input doc.
    :return: pathless_output_file_name: Pathless name for output file.
    """

    xt = example_table_xarray

    lag_times_hours = xt.coords[example_utils.SHIPS_LAG_TIME_DIM].values
    predictor_matrix = (
        xt[example_utils.SHIPS_PREDICTORS_LAGGED_KEY].values[
            init_time_index, ...
        ][:, predictor_indices]
    )
    predictor_matrix = predictor_matrix.astype(float)

    if figure_object is None or axes_object is None:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    colour_norm_object = pyplot.Normalize(
        vmin=MIN_NORMALIZED_VALUE, vmax=MAX_NORMALIZED_VALUE
    )
    axes_object.imshow(
        predictor_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
        norm=colour_norm_object
    )

    y_tick_values = numpy.linspace(
        0, predictor_matrix.shape[0] - 1, num=predictor_matrix.shape[0],
        dtype=float
    )
    y_tick_labels = ['{0:.1f}'.format(t) for t in lag_times_hours]
    y_tick_labels = [l.replace('inf', 'IRXX') for l in y_tick_labels]
    y_tick_labels = [l.replace('nan', 'Best') for l in y_tick_labels]
    y_tick_labels = [l.replace('0.0', 'IR00') for l in y_tick_labels]
    y_tick_labels = [l.replace('1.5', 'IRM1') for l in y_tick_labels]
    y_tick_labels = [l.replace('3.0', 'IRM3') for l in y_tick_labels]

    pyplot.yticks(
        y_tick_values, y_tick_labels, fontsize=lag_time_tick_font_size
    )
    # axes_object.set_ylabel('Lag time (hours)')

    x_tick_values = numpy.linspace(
        0, predictor_matrix.shape[1] - 1, num=predictor_matrix.shape[1],
        dtype=float
    )
    x_tick_labels = (
        xt.coords[example_utils.SHIPS_PREDICTOR_LAGGED_DIM].values[
            predictor_indices
        ].tolist()
    )
    x_tick_labels = [
        s if s not in VARIABLE_ABBREV_TO_VERBOSE
        else VARIABLE_ABBREV_TO_VERBOSE[s]
        for s in x_tick_labels
    ]
    pyplot.xticks(
        x_tick_values, x_tick_labels, rotation=90.,
        fontsize=predictor_tick_font_size
    )

    init_time_unix_sec = (
        xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values[init_time_index]
    )
    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    cyclone_id_string = xt[ships_io.CYCLONE_ID_KEY].values[init_time_index]
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    title_string = 'SHIPS for {0:s} at {1:s}'.format(
        cyclone_id_string, init_time_string
    )
    if info_string is not None:
        title_string += '; {0:s}'.format(info_string)

    axes_object.set_title(title_string)

    pathless_output_file_name = '{0:s}_{1:s}_ships_lagged.jpg'.format(
        cyclone_id_string, init_time_string
    )

    return figure_object, axes_object, pathless_output_file_name


def plot_fcst_predictors_one_init_time(
        example_table_xarray, init_time_index, predictor_indices,
        forecast_hour_tick_font_size=DEFAULT_FCST_HOUR_TICK_FONT_SIZE,
        predictor_tick_font_size=DEFAULT_PREDICTOR_TICK_FONT_SIZE,
        info_string=None, figure_object=None, axes_object=None
):
    """Plots forecast predictors for one initialization time.

    :param example_table_xarray: See doc for
        `plot_lagged_predictors_one_init_time`.
    :param init_time_index: Same.
    :param predictor_indices: Same.
    :param forecast_hour_tick_font_size: Font size for tick labels on y-axis.
    :param predictor_tick_font_size: Font size for tick labels on x-axis.
    :param info_string: Same.
    :param figure_object: Same.
    :param axes_object: Same.
    :return: figure_object: Same.
    :return: axes_object: Same.
    :return: pathless_output_file_name: Same.
    """

    xt = example_table_xarray

    forecast_times_hours = numpy.round(
        xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values
    ).astype(int)

    predictor_matrix = (
        xt[example_utils.SHIPS_PREDICTORS_FORECAST_KEY].values[
            init_time_index, ...
        ][:, predictor_indices]
    ) + 0.
    predictor_matrix = predictor_matrix.astype(float)

    predictor_names = (
        xt.coords[example_utils.SHIPS_PREDICTOR_FORECAST_DIM].values[
            predictor_indices
        ].tolist()
    )

    for i in range(len(forecast_times_hours)):
        if forecast_times_hours[i] <= 0:
            continue

        for j in range(len(predictor_names)):
            if predictor_names[j] not in [
                    ships_io.INTENSITY_KEY, ships_io.INTENSITY_CHANGE_6HOURS_KEY
            ]:
                continue

            predictor_matrix[i, j] = numpy.nan

    if figure_object is None or axes_object is None:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    colour_norm_object = pyplot.Normalize(
        vmin=MIN_NORMALIZED_VALUE, vmax=MAX_NORMALIZED_VALUE
    )
    axes_object.imshow(
        predictor_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
        norm=colour_norm_object
    )

    y_tick_values = numpy.linspace(
        0, predictor_matrix.shape[0] - 1, num=predictor_matrix.shape[0],
        dtype=float
    )
    y_tick_labels = ['{0:d}'.format(t) for t in forecast_times_hours]
    pyplot.yticks(
        y_tick_values, y_tick_labels, fontsize=forecast_hour_tick_font_size
    )
    axes_object.set_ylabel('Forecast hour')

    x_tick_values = numpy.linspace(
        0, predictor_matrix.shape[1] - 1, num=predictor_matrix.shape[1],
        dtype=float
    )
    x_tick_labels = [
        s if s not in VARIABLE_ABBREV_TO_VERBOSE
        else VARIABLE_ABBREV_TO_VERBOSE[s]
        for s in predictor_names
    ]
    pyplot.xticks(
        x_tick_values, x_tick_labels, rotation=90.,
        fontsize=predictor_tick_font_size
    )

    init_time_unix_sec = (
        xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values[init_time_index]
    )
    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    cyclone_id_string = xt[ships_io.CYCLONE_ID_KEY].values[init_time_index]
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    title_string = 'SHIPS for {0:s} at {1:s}'.format(
        cyclone_id_string, init_time_string
    )
    if info_string is not None:
        title_string += '; {0:s}'.format(info_string)

    axes_object.set_title(title_string)

    pathless_output_file_name = '{0:s}_{1:s}_ships_forecast.jpg'.format(
        cyclone_id_string, init_time_string
    )

    return figure_object, axes_object, pathless_output_file_name


def plot_pm_signs_one_init_time(
        data_matrix, axes_object, font_size, colour_map_object,
        max_absolute_colour_value):
    """Plots data at one initialization time with plus-minus signs.

    Positive values will be plotted as plus signs; negative values will be
    plotted as minus signs; and the colour map will be applied to absolute
    values only.

    This method is good for plotting saliency, input * gradient, or any other
    explainable-ML result that can be either positive or negative.

    V = number of SHIPS variables
    T = number of lag times or forecast hours

    :param data_matrix: T-by-V numpy array of data values to plot.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param font_size: Font size for plus and minus signs.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param max_absolute_colour_value: Max absolute value in colour scheme.
    """

    error_checking.assert_is_numpy_array_without_nan(data_matrix)
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
    error_checking.assert_is_geq(max_absolute_colour_value, 0.)
    max_absolute_colour_value = max([max_absolute_colour_value, 0.001])

    num_grid_rows = data_matrix.shape[0]
    num_grid_columns = data_matrix.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = grids.get_xy_grid_points(
        x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
        x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    colour_norm_object = pyplot.Normalize(
        vmin=0., vmax=max_absolute_colour_value
    )
    rgb_matrix = colour_map_object(colour_norm_object(
        numpy.absolute(data_matrix)
    ))[..., :-1]

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if data_matrix[i, j] >= 0:
                axes_object.text(
                    x_coords[j], y_coords[i], '+', fontsize=font_size,
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='center', transform=axes_object.transAxes
                )
            else:
                axes_object.text(
                    x_coords[j], y_coords[i], '_', fontsize=font_size,
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='bottom', transform=axes_object.transAxes
                )


def plot_raw_numbers_one_init_time(
        data_matrix, axes_object, font_size, colour_map_object,
        min_colour_value, max_colour_value, number_format_string,
        plot_in_log_space=False):
    """Plots data at one initialization time with raw numbers (text).

    This method is good for plotting saliency, input * gradient, or any other
    explainable-ML result that can have only one sign (positive or negative).

    V = number of SHIPS variables
    T = number of lag times or forecast hours

    :param data_matrix: T-by-V numpy array of data values to plot.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param font_size: Font size.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param number_format_string: Format for printing numbers.  Valid examples
        are ".2f", "d", etc.
    :param plot_in_log_space: Boolean flag.  If True (False), colours will be
        scaled logarithmically (linearly).
    """

    error_checking.assert_is_numpy_array_without_nan(data_matrix)
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
    error_checking.assert_is_greater(max_colour_value, min_colour_value)
    error_checking.assert_is_string(number_format_string)
    error_checking.assert_is_boolean(plot_in_log_space)

    number_format_string = '{0:' + number_format_string + '}'

    num_grid_rows = data_matrix.shape[0]
    num_grid_columns = data_matrix.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = grids.get_xy_grid_points(
        x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
        x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )

    if plot_in_log_space:
        rgb_matrix = colour_map_object(colour_norm_object(
            numpy.log10(1 + numpy.absolute(data_matrix))
        ))[..., :-1]
    else:
        rgb_matrix = colour_map_object(colour_norm_object(
            numpy.absolute(data_matrix)
        ))[..., :-1]

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_string = general_utils.simplify_scientific_notation(
                number_format_string.format(data_matrix[i, j])
            )

            axes_object.text(
                x_coords[j], y_coords[i], this_string,
                fontsize=font_size,
                fontstyle='italic' if data_matrix[i, j] < 0 else 'normal',
                fontweight='bold',
                color=rgb_matrix[i, j, ...],
                horizontalalignment='center', verticalalignment='center',
                transform=axes_object.transAxes
            )
