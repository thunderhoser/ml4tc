"""Plots saliency maps."""

import os
import copy
import shutil
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4tc.io import ships_io
from ml4tc.io import example_io
from ml4tc.utils import example_utils
from ml4tc.utils import satellite_utils
from ml4tc.machine_learning import saliency
from ml4tc.machine_learning import neural_net
from ml4tc.plotting import ships_plotting
from ml4tc.plotting import scalar_satellite_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MINUTES_TO_SECONDS = 60
HOURS_TO_SECONDS = 3600

SHIPS_FORECAST_HOURS = numpy.linspace(-12, 120, num=23, dtype=int)
SHIPS_BUILTIN_LAG_TIMES_HOURS = numpy.array([numpy.nan, 0, 1.5, 3])

FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _concat_panels(panel_file_names, concat_figure_file_name):
    """Concatenates panels into one figure.

    :param panel_file_names: 1-D list of paths to input image files.
    :param concat_figure_file_name: Path to output image file.
    """

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


def _plot_scalar_satellite_predictors(
        predictor_matrices, model_metadata_dict, cyclone_id_string,
        init_time_index, init_time_unix_sec):
    """Plots scalar satellite predictors for each lag time at one init time.

    :param predictor_matrices: FOO.
    :param model_metadata_dict: FOO.
    :param cyclone_id_string: FOO.
    :param init_time_index: FOO.
    :param init_time_unix_sec: FOO.
    :return: figure_object: FOO.
    :return: axes_object: FOO.
    :return: pathless_output_file_name: FOO.
    """

    # TODO(thunderhoser): Fuck with documentation.

    # Housekeeping.
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    lag_times_sec = (
        MINUTES_TO_SECONDS *
        validation_option_dict[neural_net.SATELLITE_LAG_TIMES_KEY]
    )

    num_predictors = predictor_matrices[1].shape[-1]
    predictor_indices = numpy.linspace(
        0, num_predictors - 1, num=num_predictors, dtype=int
    )

    valid_times_unix_sec = init_time_unix_sec - lag_times_sec
    num_valid_times = len(valid_times_unix_sec)
    valid_time_indices = numpy.linspace(
        0, num_valid_times - 1, num=num_valid_times, dtype=int
    )

    metadata_dict = {
        example_utils.SATELLITE_TIME_DIM: init_time_unix_sec - lag_times_sec,
        example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM:
            validation_option_dict[neural_net.SATELLITE_PREDICTORS_KEY]
    }

    these_dim_2d = (
        example_utils.SATELLITE_TIME_DIM,
        example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
    )
    main_data_dict = {
        example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY: (
            these_dim_2d,
            predictor_matrices[1][init_time_index, ...]
        ),
        satellite_utils.CYCLONE_ID_KEY: (
            (example_utils.SATELLITE_TIME_DIM,),
            [cyclone_id_string]
        )
    }

    example_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    return scalar_satellite_plotting.plot_colour_map_multi_times(
        example_table_xarray=example_table_xarray,
        time_indices=valid_time_indices, predictor_indices=predictor_indices
    )


def _plot_lagged_ships_predictors(
        predictor_matrices, model_metadata_dict, cyclone_id_string,
        builtin_lag_times_hours, forecast_hours, init_time_index,
        init_time_unix_sec):
    """Plots lagged SHIPS predictors for each lag time at one init time.

    :param predictor_matrices: FOO.
    :param model_metadata_dict: FOO.
    :param cyclone_id_string: FOO.
    :param builtin_lag_times_hours: FOO.
    :param forecast_hours: FOO.
    :param init_time_index: FOO.
    :param init_time_unix_sec: FOO.
    :return: figure_objects: FOO.
    :return: axes_objects: FOO.
    :return: pathless_output_file_names: FOO.
    """

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    model_lag_times_sec = (
        HOURS_TO_SECONDS *
        validation_option_dict[neural_net.SHIPS_LAG_TIMES_KEY]
    )

    lagged_predictor_names = (
        validation_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
    )
    num_lagged_predictors = len(lagged_predictor_names)
    lagged_predictor_indices = numpy.linspace(
        0, num_lagged_predictors - 1, num=num_lagged_predictors, dtype=int
    )

    num_forecast_predictors = len(
        validation_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
    )
    num_forecast_hours = len(forecast_hours)
    num_builtin_lag_times = len(builtin_lag_times_hours)
    num_model_lag_times = len(model_lag_times_sec)

    # Do actual stuff (plot 2-D colour maps with normalized predictors).
    figure_objects = [None] * num_model_lag_times
    axes_objects = [None] * num_model_lag_times
    pathless_output_file_names = [''] * num_model_lag_times

    for j in range(num_model_lag_times):
        valid_time_unix_sec = init_time_unix_sec - model_lag_times_sec[j]

        metadata_dict = {
            example_utils.SHIPS_LAG_TIME_DIM: builtin_lag_times_hours,
            example_utils.SHIPS_VALID_TIME_DIM:
                numpy.array([valid_time_unix_sec], dtype=int),
            example_utils.SHIPS_PREDICTOR_LAGGED_DIM: lagged_predictor_names
        }

        predictor_matrix = predictor_matrices[2][init_time_index, j, :]
        predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)
        predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)

        predictor_matrix = neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=predictor_matrix,
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=num_builtin_lag_times,
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=num_forecast_hours
        )[0]

        predictor_matrix = predictor_matrix[:, 0, ...]

        these_dim_3d = (
            example_utils.SHIPS_VALID_TIME_DIM,
            example_utils.SHIPS_LAG_TIME_DIM,
            example_utils.SHIPS_PREDICTOR_LAGGED_DIM
        )
        main_data_dict = {
            example_utils.SHIPS_PREDICTORS_LAGGED_KEY: (
                these_dim_3d, predictor_matrix
            ),
            ships_io.CYCLONE_ID_KEY: (
                (example_utils.SHIPS_VALID_TIME_DIM,),
                [cyclone_id_string]
            )
        }

        this_table_xarray = xarray.Dataset(
            data_vars=main_data_dict, coords=metadata_dict
        )
        figure_objects[j], axes_objects[j], pathless_output_file_names[j] = (
            ships_plotting.plot_lagged_predictors_one_init_time(
                example_table_xarray=this_table_xarray, init_time_index=0,
                predictor_indices=lagged_predictor_indices,
            )
        )

    return figure_objects, axes_objects, pathless_output_file_names


def _plot_forecast_ships_predictors(
        predictor_matrices, model_metadata_dict, cyclone_id_string,
        builtin_lag_times_hours, forecast_hours, init_time_index,
        init_time_unix_sec):
    """Plots lagged SHIPS predictors for each lag time at one init time.

    :param predictor_matrices: FOO.
    :param model_metadata_dict: FOO.
    :param cyclone_id_string: FOO.
    :param builtin_lag_times_hours: FOO.
    :param forecast_hours: FOO.
    :param init_time_index: FOO.
    :param init_time_unix_sec: FOO.
    :return: figure_objects: FOO.
    :return: axes_objects: FOO.
    :return: pathless_output_file_names: FOO.
    """

    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    model_lag_times_sec = (
        HOURS_TO_SECONDS *
        validation_option_dict[neural_net.SHIPS_LAG_TIMES_KEY]
    )

    forecast_predictor_names = (
        validation_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
    )
    num_forecast_predictors = len(forecast_predictor_names)
    forecast_predictor_indices = numpy.linspace(
        0, num_forecast_predictors - 1, num=num_forecast_predictors, dtype=int
    )

    num_lagged_predictors = len(
        validation_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
    )
    num_forecast_hours = len(forecast_hours)
    num_builtin_lag_times = len(builtin_lag_times_hours)
    num_model_lag_times = len(model_lag_times_sec)

    # Do actual stuff (plot 2-D colour maps with normalized predictors).
    figure_objects = [None] * num_model_lag_times
    axes_objects = [None] * num_model_lag_times
    pathless_output_file_names = [''] * num_model_lag_times

    # Do actual stuff (plot 2-D colour maps with normalized predictors).
    for j in range(num_model_lag_times):
        valid_time_unix_sec = init_time_unix_sec - model_lag_times_sec[j]

        metadata_dict = {
            example_utils.SHIPS_FORECAST_HOUR_DIM: forecast_hours,
            example_utils.SHIPS_VALID_TIME_DIM:
                numpy.array([valid_time_unix_sec], dtype=int),
            example_utils.SHIPS_PREDICTOR_FORECAST_DIM:
                forecast_predictor_names
        }

        predictor_matrix = predictor_matrices[2][init_time_index, j, :]
        predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)
        predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)

        predictor_matrix = neural_net.ships_predictors_3d_to_4d(
            predictor_matrix_3d=predictor_matrix,
            num_lagged_predictors=num_lagged_predictors,
            num_builtin_lag_times=num_builtin_lag_times,
            num_forecast_predictors=num_forecast_predictors,
            num_forecast_hours=num_forecast_hours
        )[1]

        predictor_matrix = predictor_matrix[:, 0, ...]

        these_dim_3d = (
            example_utils.SHIPS_VALID_TIME_DIM,
            example_utils.SHIPS_FORECAST_HOUR_DIM,
            example_utils.SHIPS_PREDICTOR_FORECAST_DIM
        )
        main_data_dict = {
            example_utils.SHIPS_PREDICTORS_FORECAST_KEY: (
                these_dim_3d, predictor_matrix
            ),
            ships_io.CYCLONE_ID_KEY: (
                (example_utils.SHIPS_VALID_TIME_DIM,),
                [cyclone_id_string]
            )
        }

        this_table_xarray = xarray.Dataset(
            data_vars=main_data_dict, coords=metadata_dict
        )
        figure_objects[j], axes_objects[j], pathless_output_file_names[j] = (
            ships_plotting.plot_fcst_predictors_one_init_time(
                example_table_xarray=this_table_xarray, init_time_index=0,
                predictor_indices=forecast_predictor_indices
            )
        )

    return figure_objects, axes_objects, pathless_output_file_names


def _run(saliency_file_name, example_dir_name, normalization_file_name,
         output_dir_name):
    """Plots saliency maps.

    This is effectively the main method.

    :param saliency_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param normalization_file_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(saliency_file_name))
    saliency_dict = saliency.read_file(saliency_file_name)

    model_file_name = saliency_dict[saliency.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    base_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )

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

    print(SEPARATOR_STRING)

    for i in range(num_cyclones):
        option_dict = copy.deepcopy(base_option_dict)
        option_dict[neural_net.EXAMPLE_FILE_KEY] = unique_example_file_names[i]
        data_dict = neural_net.create_inputs(option_dict)
        print(SEPARATOR_STRING)

        example_indices = numpy.where(
            numpy.array(saliency_dict[saliency.CYCLONE_IDS_KEY]) ==
            unique_cyclone_id_strings[i]
        )[0]

        for j in example_indices:
            init_time_index = numpy.where(
                data_dict[neural_net.INIT_TIMES_KEY] ==
                saliency_dict[saliency.INIT_TIMES_KEY][j]
            )[0][0]

            figure_object, axes_object, pathless_output_file_name = (
                _plot_scalar_satellite_predictors(
                    predictor_matrices=
                    data_dict[neural_net.PREDICTOR_MATRICES_KEY],
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    init_time_index=init_time_index,
                    init_time_unix_sec=
                    data_dict[neural_net.INIT_TIMES_KEY][init_time_index]
                )
            )

            all_saliency_values = numpy.concatenate([
                numpy.ravel(s[j, ...])
                for s in saliency_dict[saliency.SALIENCY_KEY]
            ])
            max_absolute_colour_value = numpy.percentile(
                numpy.absolute(all_saliency_values), 99.
            )
            scalar_satellite_plotting.plot_pm_signs_multi_times(
                data_matrix=saliency_dict[saliency.SALIENCY_KEY][1][j, ...],
                axes_object=axes_object, font_size=20,
                colour_map_object=pyplot.get_cmap('binary'),
                max_absolute_colour_value=max_absolute_colour_value
            )

            output_file_name = '{0:s}/{1:s}'.format(
                output_dir_name, pathless_output_file_name
            )
            print('Saving figure to file: "{0:s}"...'.format(output_file_name))
            figure_object.savefig(
                output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            figure_objects, axes_objects, pathless_output_file_names = (
                _plot_lagged_ships_predictors(
                    predictor_matrices=
                    data_dict[neural_net.PREDICTOR_MATRICES_KEY],
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    builtin_lag_times_hours=SHIPS_BUILTIN_LAG_TIMES_HOURS,
                    forecast_hours=SHIPS_FORECAST_HOURS,
                    init_time_index=init_time_index,
                    init_time_unix_sec=
                    data_dict[neural_net.INIT_TIMES_KEY][init_time_index]
                )
            )

            num_lagged_predictors = len(
                base_option_dict[neural_net.SHIPS_PREDICTORS_LAGGED_KEY]
            )
            num_forecast_predictors = len(
                base_option_dict[neural_net.SHIPS_PREDICTORS_FORECAST_KEY]
            )
            num_model_lag_times = len(
                base_option_dict[neural_net.SHIPS_LAG_TIMES_KEY]
            )
            this_saliency_matrix = neural_net.ships_predictors_3d_to_4d(
                predictor_matrix_3d=
                saliency_dict[saliency.SALIENCY_KEY][2][[j], ...],
                num_lagged_predictors=num_lagged_predictors,
                num_builtin_lag_times=len(SHIPS_BUILTIN_LAG_TIMES_HOURS),
                num_forecast_predictors=num_forecast_predictors,
                num_forecast_hours=len(SHIPS_FORECAST_HOURS)
            )[0][0, ...]

            panel_file_names = [''] * num_model_lag_times

            for k in range(num_model_lag_times):
                ships_plotting.plot_pm_signs_one_init_time(
                    data_matrix=this_saliency_matrix[k, ...],
                    axes_object=axes_objects[k], font_size=20,
                    colour_map_object=pyplot.get_cmap('binary'),
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

            concat_figure_file_name = '{0:s}/{1:s}_ships_lagged.jpg'.format(
                output_dir_name,
                time_conversion.unix_sec_to_string(
                    saliency_dict[saliency.INIT_TIMES_KEY][j], TIME_FORMAT
                )
            )
            _concat_panels(
                panel_file_names=panel_file_names,
                concat_figure_file_name=concat_figure_file_name
            )

            figure_objects, axes_objects, pathless_output_file_names = (
                _plot_forecast_ships_predictors(
                    predictor_matrices=
                    data_dict[neural_net.PREDICTOR_MATRICES_KEY],
                    model_metadata_dict=model_metadata_dict,
                    cyclone_id_string=unique_cyclone_id_strings[i],
                    builtin_lag_times_hours=SHIPS_BUILTIN_LAG_TIMES_HOURS,
                    forecast_hours=SHIPS_FORECAST_HOURS,
                    init_time_index=init_time_index,
                    init_time_unix_sec=
                    data_dict[neural_net.INIT_TIMES_KEY][init_time_index]
                )
            )

            this_saliency_matrix = neural_net.ships_predictors_3d_to_4d(
                predictor_matrix_3d=
                saliency_dict[saliency.SALIENCY_KEY][2][[j], ...],
                num_lagged_predictors=num_lagged_predictors,
                num_builtin_lag_times=len(SHIPS_BUILTIN_LAG_TIMES_HOURS),
                num_forecast_predictors=num_forecast_predictors,
                num_forecast_hours=len(SHIPS_FORECAST_HOURS)
            )[1][0, ...]

            panel_file_names = [''] * num_model_lag_times

            for k in range(num_model_lag_times):
                ships_plotting.plot_pm_signs_one_init_time(
                    data_matrix=this_saliency_matrix[k, ...],
                    axes_object=axes_objects[k], font_size=10,
                    colour_map_object=pyplot.get_cmap('binary'),
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

            concat_figure_file_name = '{0:s}/{1:s}_ships_forecast.jpg'.format(
                output_dir_name,
                time_conversion.unix_sec_to_string(
                    saliency_dict[saliency.INIT_TIMES_KEY][j], TIME_FORMAT
                )
            )
            _concat_panels(
                panel_file_names=panel_file_names,
                concat_figure_file_name=concat_figure_file_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_name=getattr(INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
