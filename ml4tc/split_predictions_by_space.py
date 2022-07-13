"""Splits predictions by space (geographic location)."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import longitude_conversion as lng_conversion
import prediction_io
import general_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MIN_LATITUDE_DEG_N = -60.
MAX_LATITUDE_DEG_N = 60.
MIN_LONGITUDE_DEG_E = 0.
MAX_LONGITUDE_DEG_E = 360.

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
LATITUDE_SPACING_ARG_NAME = 'latitude_spacing_deg'
LONGITUDE_SPACING_ARG_NAME = 'longitude_spacing_deg'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions for all locations.  Will be '
    'read by `prediction_io.read_file`.'
)
LATITUDE_SPACING_HELP_SPACING = 'Meridional grid spacing (degrees).'
LONGITUDE_SPACING_HELP_SPACING = 'Zonal grid spacing (degrees).'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  For each grid cell, predictions will be written'
    ' here by `prediction_io.write_file`, to an exact location determined by '
    '`prediction_io.find_file`.  Also, grid metadata will be written here by '
    '`prediction_io.write_grid_metafile`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LATITUDE_SPACING_ARG_NAME, type=float, required=False, default=1.,
    help=LATITUDE_SPACING_HELP_SPACING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LONGITUDE_SPACING_ARG_NAME, type=float, required=False, default=1.,
    help=LONGITUDE_SPACING_HELP_SPACING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, latitude_spacing_deg, longitude_spacing_deg,
         output_dir_name):
    """Splits predictions by space (geographic location).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param latitude_spacing_deg: Same.
    :param longitude_spacing_deg: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    prediction_dict = prediction_io.read_file(input_file_name)

    storm_latitudes_deg_n = prediction_dict[prediction_io.STORM_LATITUDES_KEY]
    good_indices = numpy.where(numpy.logical_and(
        storm_latitudes_deg_n >= MIN_LATITUDE_DEG_N,
        storm_latitudes_deg_n <= MAX_LATITUDE_DEG_N
    ))[0]
    prediction_dict = prediction_io.subset_by_index(
        prediction_dict=prediction_dict, desired_indices=good_indices
    )

    storm_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        prediction_dict[prediction_io.STORM_LONGITUDES_KEY]
    )
    good_indices = numpy.where(numpy.logical_and(
        storm_longitudes_deg_e >= MIN_LONGITUDE_DEG_E,
        storm_longitudes_deg_e <= MAX_LONGITUDE_DEG_E
    ))[0]
    prediction_dict = prediction_io.subset_by_index(
        prediction_dict=prediction_dict, desired_indices=good_indices
    )

    storm_latitudes_deg_n = prediction_dict[prediction_io.STORM_LATITUDES_KEY]
    storm_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        prediction_dict[prediction_io.STORM_LONGITUDES_KEY]
    )

    # Create grid.
    grid_latitudes_deg_n, grid_longitudes_deg_e = (
        general_utils.create_latlng_grid(
            min_latitude_deg_n=MIN_LATITUDE_DEG_N,
            max_latitude_deg_n=MAX_LATITUDE_DEG_N,
            latitude_spacing_deg=latitude_spacing_deg,
            min_longitude_deg_e=MIN_LONGITUDE_DEG_E,
            max_longitude_deg_e=MAX_LONGITUDE_DEG_E - longitude_spacing_deg,
            longitude_spacing_deg=longitude_spacing_deg
        )
    )

    grid_longitudes_deg_e += longitude_spacing_deg / 2
    print(grid_longitudes_deg_e)

    num_grid_rows = len(grid_latitudes_deg_n)
    num_grid_columns = len(grid_longitudes_deg_e)

    grid_edge_latitudes_deg, grid_edge_longitudes_deg = (
        grids.get_latlng_grid_cell_edges(
            min_latitude_deg=grid_latitudes_deg_n[0],
            min_longitude_deg=grid_longitudes_deg_e[0],
            lat_spacing_deg=numpy.diff(grid_latitudes_deg_n[:2])[0],
            lng_spacing_deg=numpy.diff(grid_longitudes_deg_e[:2])[0],
            num_rows=num_grid_rows, num_columns=num_grid_columns
        )
    )

    print(grid_edge_longitudes_deg)

    print(SEPARATOR_STRING)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            these_indices = grids.find_events_in_grid_cell(
                event_x_coords_metres=storm_longitudes_deg_e,
                event_y_coords_metres=storm_latitudes_deg_n,
                grid_edge_x_coords_metres=grid_edge_longitudes_deg,
                grid_edge_y_coords_metres=grid_edge_latitudes_deg,
                row_index=i, column_index=j, verbose=False
            )

            this_prediction_dict = prediction_io.subset_by_index(
                prediction_dict=copy.deepcopy(prediction_dict),
                desired_indices=these_indices
            )
            d = this_prediction_dict

            if len(d[prediction_io.INIT_TIMES_KEY]) == 0:
                continue

            this_output_file_name = prediction_io.find_file(
                directory_name=output_dir_name, grid_row=i, grid_column=j,
                raise_error_if_missing=False
            )
            print('Writing {0:d} examples to: "{1:s}"...'.format(
                len(d[prediction_io.INIT_TIMES_KEY]),
                this_output_file_name
            ))

            prediction_io.write_file(
                netcdf_file_name=this_output_file_name,
                forecast_probability_matrix=
                d[prediction_io.PROBABILITY_MATRIX_KEY],
                target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
                cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
                init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
                storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
                storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
                model_file_name=d[prediction_io.MODEL_FILE_KEY],
                lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
                quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
                uncertainty_calib_model_file_name=
                d[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
            )

    print(SEPARATOR_STRING)

    grid_metafile_name = prediction_io.find_grid_metafile(
        prediction_dir_name=output_dir_name, raise_error_if_missing=False
    )

    print('Writing grid metadata to: "{0:s}"...'.format(grid_metafile_name))
    prediction_io.write_grid_metafile(
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        netcdf_file_name=grid_metafile_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        latitude_spacing_deg=getattr(
            INPUT_ARG_OBJECT, LATITUDE_SPACING_ARG_NAME
        ),
        longitude_spacing_deg=getattr(
            INPUT_ARG_OBJECT, LONGITUDE_SPACING_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
