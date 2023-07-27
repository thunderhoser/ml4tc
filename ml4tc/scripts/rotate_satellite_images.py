"""Rotates satellite images to align with either storm motion or wind shear."""

import argparse
import numpy
from scipy.interpolate import interp1d
from ml4tc.io import example_io
from ml4tc.utils import example_utils
from ml4tc.utils import satellite_utils
from ml4tc.utils import normalization

MAX_INTERP_TIME_DIFF_SEC = 43200

INPUT_DIR_ARG_NAME = 'input_unnorm_unimputed_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
EASTWARD_SHEAR_VARIABLE_ARG_NAME = 'eastward_shear_variable_name'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing unnormalized and unimputed learning '
    'examples.  The relevant file (for the given cyclone) will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file, in which climo mean brightness temperature '
    'will be found.  This will be read by `normalization.read_file`.'
)
CYCLONE_ID_HELP_STRING = 'Will rotate satellite images for this cyclone only.'
EASTWARD_SHEAR_VARIABLE_HELP_STRING = (
    'Variable name for eastward component of shear vector.  If you want to '
    'align satellite images with storm motion instead, leave this argument '
    'alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Examples with rotated satellite images will be '
    'written here by `example_io.write_file`, to an exact location determined '
    'by `example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EASTWARD_SHEAR_VARIABLE_ARG_NAME, type=str, required=False,
    default='', help=EASTWARD_SHEAR_VARIABLE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, cyclone_id_string, normalization_file_name,
         eastward_shear_variable_name, output_dir_name):
    """Rotates satellite images to align with either storm motion or wind shear.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param cyclone_id_string: Same.
    :param normalization_file_name: Same.
    :param eastward_shear_variable_name: Same.
    :param output_dir_name: Same.
    """

    # Read data.
    input_file_name = example_io.find_file(
        directory_name=input_dir_name, cyclone_id_string=cyclone_id_string,
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    example_table_xarray = example_io.read_file(input_file_name)

    # Find climo mean brightness temperature.
    if normalization_file_name == '':
        climo_mean_kelvins = 269.80128466
    else:
        print('Reading data from: "{0:s}"...'.format(normalization_file_name))
        normalization_table_xarray = normalization.read_file(
            normalization_file_name
        )
        nt = normalization_table_xarray

        training_values_kelvins = (
            nt[normalization.SATELLITE_PREDICTORS_GRIDDED_KEY].values
        )
        training_values_kelvins = training_values_kelvins[
            numpy.isfinite(training_values_kelvins)
        ]
        climo_mean_kelvins = numpy.mean(training_values_kelvins)

    print('Climatological mean brightness temperature = {0:.1f} K'.format(
        climo_mean_kelvins
    ))

    xt = example_table_xarray

    # Find rotation vector for each satellite image (i.e., for each time).
    if eastward_shear_variable_name != '':
        assert '_eastward_' in eastward_shear_variable_name
        northward_shear_variable_name = eastward_shear_variable_name.replace(
            '_eastward_', '_northward_'
        )

        predictor_names = xt.coords[
            example_utils.SHIPS_PREDICTOR_FORECAST_DIM
        ].values.tolist()

        zero_hour_index = numpy.where(
            xt.coords[example_utils.SHIPS_FORECAST_HOUR_DIM].values == 0
        )[0][0]

        u_index = predictor_names.index(eastward_shear_variable_name)
        v_index = predictor_names.index(northward_shear_variable_name)

        orig_east_velocities_m_s01 = xt[
            example_utils.SHIPS_PREDICTORS_FORECAST_KEY
        ].values[:, zero_hour_index, u_index]

        orig_north_velocities_m_s01 = xt[
            example_utils.SHIPS_PREDICTORS_FORECAST_KEY
        ].values[:, zero_hour_index, v_index]

        orig_times_unix_sec = (
            xt.coords[example_utils.SHIPS_VALID_TIME_DIM].values
        )
    else:
        orig_east_velocities_m_s01 = (
            xt[satellite_utils.STORM_MOTION_U_KEY].values
        )
        orig_north_velocities_m_s01 = (
            xt[satellite_utils.STORM_MOTION_V_KEY].values
        )
        orig_times_unix_sec = xt.coords[example_utils.SATELLITE_TIME_DIM].values

    # Interpolate motion vectors to satellite times.
    valid_times_unix_sec = xt.coords[example_utils.SATELLITE_TIME_DIM].values

    good_indices = numpy.where(
        numpy.invert(numpy.logical_or(
            numpy.isnan(orig_east_velocities_m_s01),
            numpy.isnan(orig_north_velocities_m_s01)
        ))
    )[0]

    interp_object = interp1d(
        x=orig_times_unix_sec[good_indices],
        y=orig_east_velocities_m_s01[good_indices],
        kind='linear', bounds_error=False, fill_value='extrapolate'
    )
    east_velocities_m_s01 = interp_object(valid_times_unix_sec)

    interp_object = interp1d(
        x=orig_times_unix_sec[good_indices],
        y=orig_north_velocities_m_s01[good_indices],
        kind='linear', bounds_error=False, fill_value='extrapolate'
    )
    north_velocities_m_s01 = interp_object(valid_times_unix_sec)

    # Remove motion vectors that were interpolated over too much time.
    time_diffs_sec = numpy.array([
        numpy.min(numpy.absolute(t - orig_times_unix_sec[good_indices]))
        for t in valid_times_unix_sec
    ], dtype=int)

    bad_indices = numpy.where(time_diffs_sec > MAX_INTERP_TIME_DIFF_SEC)[0]
    east_velocities_m_s01[bad_indices] = numpy.nan
    north_velocities_m_s01[bad_indices] = numpy.nan

    # Rotate satellite images.
    brightness_temp_matrix_kelvins = xt[
        example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY
    ].values[..., 0]

    num_valid_times = len(valid_times_unix_sec)
    num_grid_rows = brightness_temp_matrix_kelvins.shape[1]
    num_grid_columns = brightness_temp_matrix_kelvins.shape[2]
    dimensions = (num_valid_times, num_grid_rows, num_grid_columns)

    grid_latitude_matrix_deg_n = numpy.full(dimensions, numpy.nan)
    grid_longitude_matrix_deg_e = numpy.full(dimensions, numpy.nan)

    for i in range(num_valid_times):
        if numpy.mod(i, 10) == 0:
            print((
                'Have rotated satellite images for {0:d} of {1:d} times...'
            ).format(
                i, num_valid_times
            ))

        if numpy.isnan(east_velocities_m_s01[i]):
            continue

        (
            grid_latitude_matrix_deg_n[i, ...],
            grid_longitude_matrix_deg_e[i, ...]
        ) = example_utils.rotate_satellite_grid(
            center_latitude_deg_n=
            xt[satellite_utils.STORM_LATITUDE_KEY].values[i],
            center_longitude_deg_e=
            xt[satellite_utils.STORM_LONGITUDE_KEY].values[i],
            east_velocity_m_s01=east_velocities_m_s01[i],
            north_velocity_m_s01=north_velocities_m_s01[i],
            num_rows=num_grid_rows, num_columns=num_grid_columns
        )

        try:
            brightness_temp_matrix_kelvins[i, ...] = (
                example_utils.rotate_satellite_image(
                    brightness_temp_matrix_kelvins=
                    brightness_temp_matrix_kelvins[i, ...],
                    orig_latitudes_deg_n=
                    xt[satellite_utils.GRID_LATITUDE_KEY].values[i, ...],
                    orig_longitudes_deg_e=
                    xt[satellite_utils.GRID_LONGITUDE_KEY].values[i, ...],
                    new_latitude_matrix_deg_n=
                    grid_latitude_matrix_deg_n[i, ...],
                    new_longitude_matrix_deg_e=
                    grid_longitude_matrix_deg_e[i, ...],
                    fill_value=climo_mean_kelvins
                )
            )
        except AssertionError:
            print('POTENTIAL ERROR: failed to rotate satellite image.')
            east_velocities_m_s01[i] = numpy.nan
            north_velocities_m_s01[i] = numpy.nan

    print('Have rotated satellite images for all {0:d} times!'.format(
        num_valid_times
    ))

    # Put rotated satellite images in xarray table.
    example_table_xarray = xt

    dimensions = (
        example_utils.SATELLITE_TIME_DIM,
        example_utils.SATELLITE_GRID_ROW_DIM,
        example_utils.SATELLITE_GRID_COLUMN_DIM,
        example_utils.SATELLITE_PREDICTOR_GRIDDED_DIM
    )
    example_table_xarray[example_utils.SATELLITE_PREDICTORS_GRIDDED_KEY] = (
        dimensions, numpy.expand_dims(brightness_temp_matrix_kelvins, axis=-1)
    )

    dimensions = (
        example_utils.SATELLITE_TIME_DIM,
        example_utils.SATELLITE_GRID_ROW_DIM,
        example_utils.SATELLITE_GRID_COLUMN_DIM
    )
    example_table_xarray[satellite_utils.GRID_LATITUDE_KEY] = (
        dimensions, grid_latitude_matrix_deg_n
    )
    example_table_xarray[satellite_utils.GRID_LONGITUDE_KEY] = (
        dimensions, grid_longitude_matrix_deg_e
    )

    # Remove rotated satellite images with NaN.
    good_indices = numpy.where(
        numpy.invert(numpy.logical_or(
            numpy.isnan(east_velocities_m_s01),
            numpy.isnan(north_velocities_m_s01)
        ))
    )[0]

    print('Removing {0:d} of {1:d} satellite images due to bad data...'.format(
        num_valid_times - len(good_indices), num_valid_times
    ))
    example_table_xarray = example_table_xarray.isel(
        indexers={example_utils.SATELLITE_TIME_DIM: good_indices}
    )

    # Write rotated satellite images.
    output_file_name = example_io.find_file(
        directory_name=output_dir_name, cyclone_id_string=cyclone_id_string,
        prefer_zipped=False, allow_other_format=False,
        raise_error_if_missing=False
    )

    print((
        'Writing examples with rotated satellite images to: "{0:s}"...'
    ).format(
        output_file_name
    ))
    example_io.write_file(
        example_table_xarray=example_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        eastward_shear_variable_name=getattr(
            INPUT_ARG_OBJECT, EASTWARD_SHEAR_VARIABLE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
