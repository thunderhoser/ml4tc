"""Plots all predictors (scalars and brightness-temp maps) for a given model."""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import imagemagick_utils
import example_io
import border_io
import example_utils
import satellite_utils
import normalization
import neural_net
import plot_satellite

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MINUTES_TO_SECONDS = 60

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

MODEL_METAFILE_ARG_NAME = 'input_model_metafile_name'
EXAMPLE_FILE_ARG_NAME = 'input_norm_example_file_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
INIT_TIMES_ARG_NAME = 'init_time_strings'
FIRST_TIME_ARG_NAME = 'first_init_time_string'
LAST_TIME_ARG_NAME = 'last_init_time_string'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_brightness_temps(
        example_table_xarray, normalization_table_xarray, model_metadata_dict,
        predictor_matrices, init_times_unix_sec, border_latitudes_deg_n,
        border_longitudes_deg_e, output_dir_name):
    """Plots one brightness-temp map for each init time and lag time.

    P = number of points in border set

    :param example_table_xarray: xarray table returned by
        `example_io.read_file`.
    :param normalization_table_xarray: xarray table returned by
        `normalization.read_file`.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    :param predictor_matrices: See output doc for `neural_net.create_inputs`.
    :param init_times_unix_sec: Same.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    xt = example_table_xarray
    nt = normalization_table_xarray
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )

    # Denormalize brightness temperatures.
    predictor_names_norm = list(
        nt.coords[normalization.SATELLITE_PREDICTOR_GRIDDED_DIM].values
    )
    k = predictor_names_norm.index(satellite_utils.BRIGHTNESS_TEMPERATURE_KEY)
    training_values = (
        nt[normalization.SATELLITE_PREDICTORS_GRIDDED_KEY].values[:, k]
    )
    training_values = training_values[numpy.isfinite(training_values)]

    brightness_temp_matrix_kelvins = normalization._denorm_one_variable(
        normalized_values_new=predictor_matrices[0],
        actual_values_training=training_values
    )[..., 0]

    # Plot maps of denormalized brightness temp.
    num_init_times = len(init_times_unix_sec)
    lag_times_sec = (
        MINUTES_TO_SECONDS *
        validation_option_dict[neural_net.SATELLITE_LAG_TIMES_KEY]
    )
    num_lag_times = len(lag_times_sec)

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_lag_times)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_lag_times) / num_panel_rows
    ))

    num_grid_rows = brightness_temp_matrix_kelvins.shape[1]
    num_grid_columns = brightness_temp_matrix_kelvins.shape[2]
    grid_row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=int
    )
    grid_column_indices = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=int
    )

    for i in range(num_init_times):
        panel_file_names = [''] * num_lag_times

        for j in range(num_lag_times):
            desired_time_unix_sec = init_times_unix_sec[i] - lag_times_sec[j]
            k = numpy.argmin(numpy.absolute(
                xt.coords[example_utils.SATELLITE_TIME_DIM].values -
                desired_time_unix_sec
            ))

            valid_time_unix_sec = (
                xt.coords[example_utils.SATELLITE_TIME_DIM].values[k]
            )
            satellite_metadata_dict = {
                satellite_utils.GRID_ROW_DIM: grid_row_indices,
                satellite_utils.GRID_COLUMN_DIM: grid_column_indices,
                satellite_utils.TIME_DIM:
                    numpy.array([valid_time_unix_sec], dtype=int)
            }

            grid_latitudes_deg_n = (
                xt[satellite_utils.GRID_LATITUDE_KEY].values[k, :]
            )
            grid_longitudes_deg_e = (
                xt[satellite_utils.GRID_LONGITUDE_KEY].values[k, :]
            )
            cyclone_id_string = xt[satellite_utils.CYCLONE_ID_KEY].values[k]
            these_dim_3d = (
                satellite_utils.TIME_DIM,
                satellite_utils.GRID_ROW_DIM, satellite_utils.GRID_COLUMN_DIM
            )

            satellite_data_dict = {
                satellite_utils.CYCLONE_ID_KEY: (
                    (satellite_utils.TIME_DIM,),
                    [cyclone_id_string]
                ),
                satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
                    these_dim_3d,
                    brightness_temp_matrix_kelvins[[i], ..., j]
                ),
                satellite_utils.GRID_LATITUDE_KEY: (
                    (satellite_utils.TIME_DIM, satellite_utils.GRID_ROW_DIM),
                    numpy.expand_dims(grid_latitudes_deg_n, axis=0)
                ),
                satellite_utils.GRID_LONGITUDE_KEY: (
                    (satellite_utils.TIME_DIM, satellite_utils.GRID_COLUMN_DIM),
                    numpy.expand_dims(grid_longitudes_deg_e, axis=0)
                )
            }

            satellite_table_xarray = xarray.Dataset(
                data_vars=satellite_data_dict, coords=satellite_metadata_dict
            )
            panel_file_names[j] = plot_satellite.plot_one_satellite_image(
                satellite_table_xarray=satellite_table_xarray, time_index=0,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                output_dir_name=output_dir_name
            )
            imagemagick_utils.resize_image(
                input_file_name=panel_file_names[j],
                output_file_name=panel_file_names[j],
                output_size_pixels=PANEL_SIZE_PX
            )

        concat_figure_file_name = '{0:s}/{1:s}_brightness_temp.jpg'.format(
            output_dir_name,
            time_conversion.unix_sec_to_string(
                init_times_unix_sec[i], TIME_FORMAT
            )
        )

        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=panel_file_names, num_panel_rows=num_panel_rows,
            num_panel_columns=num_panel_columns,
            output_file_name=concat_figure_file_name
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=CONCAT_FIGURE_SIZE_PX
        )

        for j in range(num_lag_times):
            os.remove(panel_file_names[j])


def _run(model_metafile_name, norm_example_file_name, normalization_file_name,
         init_time_strings, first_init_time_string, last_init_time_string,
         output_dir_name):
    """Plots all predictors (scalars and brightness-temp maps) for a given model.

    This is effectively the main method.

    :param model_metafile_name: See documentation at top of file.
    :param norm_example_file_name: Same.
    :param normalization_file_name: Same.
    :param init_time_strings: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    validation_option_dict = model_metadata_dict[
        neural_net.VALIDATION_OPTIONS_KEY
    ]
    validation_option_dict[neural_net.EXAMPLE_FILE_KEY] = norm_example_file_name

    print('Reading data from: "{0:s}"...'.format(norm_example_file_name))
    example_table_xarray = example_io.read_file(norm_example_file_name)

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    print(SEPARATOR_STRING)

    predictor_matrices, target_array, all_init_times_unix_sec = (
        neural_net.create_inputs(validation_option_dict)
    )
    print(SEPARATOR_STRING)

    if len(init_time_strings) == 1 and init_time_strings[0] == '':
        first_init_time_unix_sec = time_conversion.string_to_unix_sec(
            first_init_time_string, TIME_FORMAT
        )
        last_init_time_unix_sec = time_conversion.string_to_unix_sec(
            last_init_time_string, TIME_FORMAT
        )
        time_indices = numpy.where(numpy.logical_and(
            all_init_times_unix_sec >= first_init_time_unix_sec,
            all_init_times_unix_sec <= last_init_time_unix_sec
        ))[0]

        if len(time_indices) == 0:
            error_string = (
                'Cannot find any init times in file "{0:s}" between {1:s} and '
                '{2:s}.'
            ).format(
                norm_example_file_name, first_init_time_string,
                last_init_time_string
            )

            raise ValueError(error_string)
    else:
        init_times_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, TIME_FORMAT)
            for t in init_time_strings
        ], dtype=int)

        time_indices = numpy.array([
            numpy.where(all_init_times_unix_sec == t)[0][0]
            for t in init_times_unix_sec
        ], dtype=int)

    predictor_matrices = [a[time_indices, ...] for a in predictor_matrices]
    target_array = target_array[time_indices, ...]
    init_times_unix_sec = all_init_times_unix_sec[time_indices]

    _plot_brightness_temps(
        example_table_xarray=example_table_xarray,
        normalization_table_xarray=normalization_table_xarray,
        model_metadata_dict=model_metadata_dict,
        predictor_matrices=predictor_matrices,
        init_times_unix_sec=init_times_unix_sec,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        output_dir_name=output_dir_name
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
        init_time_strings=getattr(INPUT_ARG_OBJECT, INIT_TIMES_ARG_NAME),
        first_init_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_init_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
