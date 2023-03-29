"""Normalizes learning examples."""

import os
import shutil
import argparse
import numpy
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from ml4tc.io import ships_io
from ml4tc.io import example_io
from ml4tc.utils import example_utils
from ml4tc.utils import general_utils
from ml4tc.utils import satellite_utils
from ml4tc.utils import normalization

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
HOURS_TO_SECONDS = 3600

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
YEAR_ARG_NAME = 'year'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
ALTITUDE_ANGLE_EXE_ARG_NAME = 'altitude_angle_exe_name'
COMPRESS_ARG_NAME = 'compress_output_files'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing unnormalized examples.  Files therein '
    'will be found by `example_io.find_file` and read by '
    '`example_io.read_file`.'
)
YEAR_HELP_STRING = (
    'Data will be normalized only for this year.  If you would rather '
    'normalize data for specific cyclones, leave this argument alone.'
)
CYCLONE_IDS_HELP_STRING = (
    'List of strings.  Data will be normalized for these cyclones.  If you '
    'would rather normalize data for all cyclones in a year, leave this '
    'argument alone.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be read by '
    '`normalization.read_file`).'
)
TEMPORARY_DIR_HELP_STRING = (
    'Path to directory for temporary files with solar altitude angles.'
)
ALTITUDE_ANGLE_EXE_HELP_STRING = (
    'Path to Fortran executable (pathless file name should probably be '
    '"solarpos") that computes solar altitude angles.'
)
COMPRESS_HELP_STRING = 'Boolean flag.  If 1 (0), will (not) gzip output files.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Normalized examples will be written here by '
    '`example_io.write_file`, to exact locations determined by '
    '`example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEAR_ARG_NAME, type=int, required=False, default=-1,
    help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_IDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=CYCLONE_IDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ALTITUDE_ANGLE_EXE_ARG_NAME, type=str, required=False,
    default=general_utils.DEFAULT_EXE_NAME_FOR_ALTITUDE_ANGLE,
    help=ALTITUDE_ANGLE_EXE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPRESS_ARG_NAME, type=int, required=True,
    help=COMPRESS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _add_zenith_angles_to_example_table(
        example_table_xarray, temporary_dir_name, altitude_angle_exe_name):
    """Adds solar zenith angles to xarray table with learning examples.

    :param example_table_xarray: xarray table in format returned by
        `example_io.read_file`.
    :param temporary_dir_name: See documentation at top of file.
    :param altitude_angle_exe_name: Same.
    :return: example_table_xarray: Same as input but with solar zenith angle as
        a forecast SHIPS predictor.
    """

    xt = example_table_xarray

    init_times_unix_sec = xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values
    forecast_hours = xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values
    forecast_latitude_matrix_deg_n = (
        xt[example_utils.FORECAST_LATITUDE_KEY].values
    )
    forecast_longitude_matrix_deg_e = (
        xt[example_utils.FORECAST_LONGITUDE_KEY].values
    )

    num_examples = forecast_latitude_matrix_deg_n.shape[0]

    for i in range(num_examples):
        nan_flags = numpy.isnan(forecast_latitude_matrix_deg_n[i, :])
        nan_indices = numpy.where(nan_flags)[0]
        if len(nan_indices) == 0:
            continue

        real_indices = numpy.where(numpy.invert(nan_flags))[0]
        assert len(real_indices) > 0

        if len(real_indices) == 1:
            forecast_latitude_matrix_deg_n[i, nan_indices] = (
                forecast_latitude_matrix_deg_n[i, real_indices[0]]
            )
            continue

        interp_object = interp1d(
            x=forecast_hours[real_indices],
            y=forecast_latitude_matrix_deg_n[i, real_indices],
            kind='linear', bounds_error=False, assume_sorted=True,
            fill_value='extrapolate'
        )

        forecast_latitude_matrix_deg_n[i, nan_indices] = interp_object(
            forecast_hours[nan_indices]
        )

    forecast_latitude_matrix_deg_n = numpy.maximum(
        forecast_latitude_matrix_deg_n, -90.
    )
    forecast_latitude_matrix_deg_n = numpy.minimum(
        forecast_latitude_matrix_deg_n, 90.
    )

    for i in range(num_examples):
        nan_flags = numpy.isnan(forecast_longitude_matrix_deg_e[i, :])
        nan_indices = numpy.where(nan_flags)[0]
        if len(nan_indices) == 0:
            continue

        real_indices = numpy.where(numpy.invert(nan_flags))[0]
        assert len(real_indices) > 0

        if len(real_indices) == 1:
            forecast_longitude_matrix_deg_e[i, nan_indices] = (
                forecast_longitude_matrix_deg_e[i, real_indices[0]]
            )
            continue

        real_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            forecast_longitude_matrix_deg_e[i, real_indices], allow_nan=False
        )
        if numpy.any(numpy.absolute(numpy.diff(real_longitudes_deg_e)) > 50):
            real_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
                real_longitudes_deg_e, allow_nan=False
            )

        interp_object = interp1d(
            x=forecast_hours[real_indices], y=real_longitudes_deg_e,
            kind='linear', bounds_error=False, assume_sorted=True,
            fill_value='extrapolate'
        )

        forecast_longitude_matrix_deg_e[i, nan_indices] = interp_object(
            forecast_hours[nan_indices]
        )

    forecast_longitude_matrix_deg_e = numpy.maximum(
        forecast_longitude_matrix_deg_e, -179.999999
    )
    forecast_longitude_matrix_deg_e = numpy.minimum(
        forecast_longitude_matrix_deg_e, 359.999999
    )

    forecast_hour_matrix, init_time_matrix_unix_sec = numpy.meshgrid(
        forecast_hours, init_times_unix_sec
    )
    valid_time_matrix_unix_sec = (
        init_time_matrix_unix_sec + HOURS_TO_SECONDS * forecast_hour_matrix
    )

    forecast_altitude_angle_matrix_deg = (
        general_utils.get_solar_altitude_angles(
            valid_times_unix_sec=valid_time_matrix_unix_sec,
            latitudes_deg_n=forecast_latitude_matrix_deg_n,
            longitudes_deg_e=forecast_longitude_matrix_deg_e,
            temporary_dir_name=temporary_dir_name,
            fortran_exe_name=altitude_angle_exe_name
        )
    )

    forecast_zenith_angle_matrix_deg = 90. - forecast_altitude_angle_matrix_deg

    forecast_predictor_matrix = (
        xt[example_utils.SHIPS_PREDICTORS_FORECAST_KEY].values
    )
    forecast_predictor_names = (
        xt.coords[example_utils.SHIPS_PREDICTOR_FORECAST_DIM].values
    )
    xt = xt.drop_vars(names=example_utils.SHIPS_PREDICTORS_FORECAST_KEY)

    assert (
        ships_io.FORECAST_SOLAR_ZENITH_ANGLE_KEY not in forecast_predictor_names
    )
    forecast_predictor_names = numpy.concatenate((
        forecast_predictor_names,
        numpy.array([ships_io.FORECAST_SOLAR_ZENITH_ANGLE_KEY], dtype=object)
    ))

    forecast_predictor_matrix = numpy.concatenate((
        forecast_predictor_matrix,
        numpy.expand_dims(forecast_zenith_angle_matrix_deg, axis=-1)
    ), axis=-1)

    xt = xt.assign_coords({
        example_utils.SHIPS_PREDICTOR_FORECAST_DIM: forecast_predictor_names
    })

    these_dim = (
        example_utils.SHIPS_VALID_TIME_DIM,
        example_utils.SHIPS_FORECAST_HOUR_DIM,
        example_utils.SHIPS_PREDICTOR_FORECAST_DIM
    )
    xt = xt.assign({
        example_utils.SHIPS_PREDICTORS_FORECAST_KEY:
            (these_dim, forecast_predictor_matrix)
    })
    example_table_xarray = xt

    return example_table_xarray


def _run(input_example_dir_name, year, cyclone_id_strings,
         normalization_file_name, temporary_dir_name, altitude_angle_exe_name,
         compress_output_files, output_example_dir_name):
    """Normalizes learning examples.

    This is effectively the main method.

    :param input_example_dir_name: See documentation at top of file.
    :param year: Same.
    :param cyclone_id_strings: Same.
    :param normalization_file_name: Same.
    :param temporary_dir_name: Same.
    :param altitude_angle_exe_name: Same.
    :param compress_output_files: Same.
    :param output_example_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=temporary_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )
    nt = normalization_table_xarray

    forecast_predictor_matrix = (
        nt[normalization.SHIPS_PREDICTORS_FORECAST_KEY].values
    )
    forecast_predictor_names = (
        nt.coords[normalization.SHIPS_PREDICTOR_FORECAST_DIM].values
    )
    nt = nt.drop_vars(names=normalization.SHIPS_PREDICTORS_FORECAST_KEY)

    assert (
        ships_io.FORECAST_SOLAR_ZENITH_ANGLE_KEY not in forecast_predictor_names
    )
    forecast_predictor_names = numpy.concatenate((
        forecast_predictor_names,
        numpy.array([ships_io.FORECAST_SOLAR_ZENITH_ANGLE_KEY], dtype=object)
    ))

    num_reference_values = forecast_predictor_matrix.shape[0]
    reference_zenith_angles_deg = numpy.linspace(
        0, 180, num=num_reference_values, dtype=float
    )
    forecast_predictor_matrix = numpy.concatenate((
        forecast_predictor_matrix,
        numpy.expand_dims(reference_zenith_angles_deg, axis=1)
    ), axis=1)

    nt = nt.assign_coords({
        normalization.SHIPS_PREDICTOR_FORECAST_DIM: forecast_predictor_names
    })

    these_dim = (
        normalization.UNGRIDDED_INDEX_DIM,
        normalization.SHIPS_PREDICTOR_FORECAST_DIM
    )
    nt = nt.assign({
        normalization.SHIPS_PREDICTORS_FORECAST_KEY:
            (these_dim, forecast_predictor_matrix)
    })
    normalization_table_xarray = nt

    if len(cyclone_id_strings) == 1 and cyclone_id_strings[0] == '':
        cyclone_id_strings = None

    if cyclone_id_strings is None:
        cyclone_id_strings = example_io.find_cyclones(
            directory_name=input_example_dir_name,
            raise_error_if_all_missing=True
        )
        cyclone_id_strings = set([
            c for c in cyclone_id_strings
            if satellite_utils.parse_cyclone_id(c)[0] == year
        ])
        cyclone_id_strings = list(cyclone_id_strings)

    cyclone_id_strings.sort()

    for this_cyclone_id_string in cyclone_id_strings:
        input_example_file_name = example_io.find_file(
            directory_name=input_example_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        output_example_file_name = example_io.find_file(
            directory_name=output_example_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        print('Reading unnormalized examples from: "{0:s}"...'.format(
            input_example_file_name
        ))
        example_table_xarray = example_io.read_file(input_example_file_name)

        example_table_xarray = _add_zenith_angles_to_example_table(
            example_table_xarray=example_table_xarray,
            temporary_dir_name=temporary_dir_name,
            altitude_angle_exe_name=altitude_angle_exe_name
        )

        example_table_xarray = normalization.normalize_data(
            example_table_xarray=example_table_xarray,
            normalization_table_xarray=normalization_table_xarray
        )

        print('Writing normalized examples to: "{0:s}"...'.format(
            output_example_file_name
        ))
        example_io.write_file(
            example_table_xarray=example_table_xarray,
            netcdf_file_name=output_example_file_name
        )

        if compress_output_files:
            general_utils.compress_file(output_example_file_name)
            os.remove(output_example_file_name)

    shutil.rmtree(temporary_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME),
        cyclone_id_strings=getattr(INPUT_ARG_OBJECT, CYCLONE_IDS_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        altitude_angle_exe_name=getattr(
            INPUT_ARG_OBJECT, ALTITUDE_ANGLE_EXE_ARG_NAME
        ),
        compress_output_files=bool(
            getattr(INPUT_ARG_OBJECT, COMPRESS_ARG_NAME)
        ),
        output_example_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
