"""Plots saliency maps."""

import os
import sys
import copy
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import example_io
import example_utils
import satellite_utils
import saliency
import neural_net
import scalar_satellite_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MINUTES_TO_SECONDS = 60
FIGURE_RESOLUTION_DPI = 300

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

            saliency_matrix = saliency_dict[saliency.SALIENCY_KEY][1][j, ...]
            max_absolute_colour_value = numpy.percentile(
                numpy.absolute(saliency_matrix), 99.
            )
            scalar_satellite_plotting.plot_pm_signs_multi_times(
                data_matrix=saliency_matrix, axes_object=axes_object,
                font_size=20, colour_map_object=pyplot.get_cmap('binary'),
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
