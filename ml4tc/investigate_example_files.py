"""Investigates predictor variables in example files."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_io
import example_utils
import satellite_utils

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
YEAR_ARG_NAME = 'year'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
YEAR_HELP_STRING = (
    'Summary statistics will be reported for this year, separately for the '
    'northern and southern hemispheres.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)


def _run(input_dir_name, year):
    """Investigates predictor variables in example files.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param year: Same.
    """

    cyclone_id_strings = example_io.find_cyclones(
        directory_name=input_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [satellite_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )

    good_indices = numpy.where(cyclone_years == year)[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    cyclone_id_strings.sort()

    forecast_ships_predictor_matrix_nh = numpy.array([])
    lagged_ships_predictor_matrix_nh = numpy.array([])
    satellite_predictor_matrix_nh = numpy.array([])
    forecast_ships_predictor_matrix_sh = numpy.array([])
    lagged_ships_predictor_matrix_sh = numpy.array([])
    satellite_predictor_matrix_sh = numpy.array([])
    forecast_ships_predictor_names = []
    lagged_ships_predictor_names = []
    satellite_predictor_names = []

    num_nh_files_read = 0
    num_sh_files_read = 0

    for this_cyclone_id_string in cyclone_id_strings:
        if num_nh_files_read >= 5 and num_sh_files_read >= 5:
            continue

        this_file_name = example_io.find_file(
            directory_name=input_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        this_basin_id_string = satellite_utils.parse_cyclone_id(
            this_cyclone_id_string
        )[1]

        if (
                this_basin_id_string ==
                satellite_utils.SOUTHERN_HEMISPHERE_ID_STRING
        ):
            if num_nh_files_read >= 5:
                continue

            forecast_ships_predictor_matrix = forecast_ships_predictor_matrix_sh
            lagged_ships_predictor_matrix = lagged_ships_predictor_matrix_sh
            satellite_predictor_matrix = satellite_predictor_matrix_sh
            num_nh_files_read += 1
        else:
            if num_sh_files_read >= 5:
                continue

            forecast_ships_predictor_matrix = forecast_ships_predictor_matrix_nh
            lagged_ships_predictor_matrix = lagged_ships_predictor_matrix_nh
            satellite_predictor_matrix = satellite_predictor_matrix_nh
            num_sh_files_read += 1

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(this_file_name)

        this_forecast_matrix = this_example_dict[
            example_utils.SHIPS_PREDICTORS_FORECAST_KEY
        ].values

        this_lagged_matrix = this_example_dict[
            example_utils.SHIPS_PREDICTORS_LAGGED_KEY
        ].values

        this_satellite_matrix = this_example_dict[
            example_utils.SATELLITE_PREDICTORS_UNGRIDDED_KEY
        ].values

        if forecast_ships_predictor_matrix.size == 0:
            forecast_ships_predictor_matrix = this_forecast_matrix + 0.
            lagged_ships_predictor_matrix = this_lagged_matrix + 0.
            satellite_predictor_matrix = this_satellite_matrix + 0.
        else:
            forecast_ships_predictor_matrix = numpy.concatenate(
                (forecast_ships_predictor_matrix, this_forecast_matrix), axis=0
            )
            lagged_ships_predictor_matrix = numpy.concatenate(
                (lagged_ships_predictor_matrix, this_lagged_matrix), axis=0
            )
            satellite_predictor_matrix = numpy.concatenate(
                (satellite_predictor_matrix, this_satellite_matrix), axis=0
            )

        if (
                this_basin_id_string ==
                satellite_utils.SOUTHERN_HEMISPHERE_ID_STRING
        ):
            forecast_ships_predictor_matrix_sh = forecast_ships_predictor_matrix
            lagged_ships_predictor_matrix_sh = lagged_ships_predictor_matrix
            satellite_predictor_matrix_sh = satellite_predictor_matrix
        else:
            forecast_ships_predictor_matrix_nh = forecast_ships_predictor_matrix
            lagged_ships_predictor_matrix_nh = lagged_ships_predictor_matrix
            satellite_predictor_matrix_nh = satellite_predictor_matrix

        if len(forecast_ships_predictor_names) == 0:
            forecast_ships_predictor_names = this_example_dict.coords[
                example_utils.SHIPS_PREDICTOR_FORECAST_DIM
            ].values.tolist()

            lagged_ships_predictor_names = this_example_dict.coords[
                example_utils.SHIPS_PREDICTOR_LAGGED_DIM
            ].values.tolist()

            satellite_predictor_names = this_example_dict.coords[
                example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
            ].values.tolist()
        else:
            assert (
                forecast_ships_predictor_names ==
                this_example_dict.coords[
                    example_utils.SHIPS_PREDICTOR_FORECAST_DIM
                ].values.tolist()
            )

            assert (
                lagged_ships_predictor_names ==
                this_example_dict.coords[
                    example_utils.SHIPS_PREDICTOR_LAGGED_DIM
                ].values.tolist()
            )

            assert (
                satellite_predictor_names ==
                this_example_dict.coords[
                    example_utils.SATELLITE_PREDICTOR_UNGRIDDED_DIM
                ].values.tolist()
            )

    for k in range(len(forecast_ships_predictor_names)):
        variable_name = forecast_ships_predictor_names[k]
        these_values = forecast_ships_predictor_matrix_nh[..., k]
        these_real_values = these_values[
            numpy.invert(numpy.isnan(these_values))
        ]

        print((
            'Variable "{0:s}" in northern hemi ... NaN frequency = {1:.4f} '
            '... mean = {2:.4f} ... median = {3:.4f} ... min = {4:.4f} '
            '... max = {5:.4f} ... negative frequency = {6:.4f} '
            '... positive frequency = {7:.4f}'
        ).format(
            variable_name,
            numpy.mean(numpy.isnan(these_values)),
            numpy.nanmean(these_values),
            numpy.nanmedian(these_values),
            numpy.nanmin(these_values),
            numpy.nanmax(these_values),
            numpy.mean(these_real_values < 0),
            numpy.mean(these_real_values > 0)
        ))

        these_values = forecast_ships_predictor_matrix_sh[..., k]
        these_real_values = these_values[
            numpy.invert(numpy.isnan(these_values))
        ]

        print((
            'Variable "{0:s}" in southern hemi ... NaN frequency = {1:.4f} '
            '... mean = {2:.4f} ... median = {3:.4f} ... min = {4:.4f} '
            '... max = {5:.4f} ... negative frequency = {6:.4f} '
            '... positive frequency = {7:.4f}\n'
        ).format(
            variable_name,
            numpy.mean(numpy.isnan(these_values)),
            numpy.nanmean(these_values),
            numpy.nanmedian(these_values),
            numpy.nanmin(these_values),
            numpy.nanmax(these_values),
            numpy.mean(these_real_values < 0),
            numpy.mean(these_real_values > 0)
        ))

    for k in range(len(lagged_ships_predictor_names)):
        variable_name = lagged_ships_predictor_names[k]
        these_values = lagged_ships_predictor_matrix_nh[..., k]
        these_real_values = these_values[
            numpy.invert(numpy.isnan(these_values))
        ]

        print((
            'Variable "{0:s}" in northern hemi ... NaN frequency = {1:.4f} '
            '... mean = {2:.4f} ... median = {3:.4f} ... min = {4:.4f} '
            '... max = {5:.4f} ... negative frequency = {6:.4f} '
            '... positive frequency = {7:.4f}'
        ).format(
            variable_name,
            numpy.mean(numpy.isnan(these_values)),
            numpy.nanmean(these_values),
            numpy.nanmedian(these_values),
            numpy.nanmin(these_values),
            numpy.nanmax(these_values),
            numpy.mean(these_real_values < 0),
            numpy.mean(these_real_values > 0)
        ))

        these_values = lagged_ships_predictor_matrix_sh[..., k]
        these_real_values = these_values[
            numpy.invert(numpy.isnan(these_values))
        ]

        print((
            'Variable "{0:s}" in southern hemi ... NaN frequency = {1:.4f} '
            '... mean = {2:.4f} ... median = {3:.4f} ... min = {4:.4f} '
            '... max = {5:.4f} ... negative frequency = {6:.4f} '
            '... positive frequency = {7:.4f}\n'
        ).format(
            variable_name,
            numpy.mean(numpy.isnan(these_values)),
            numpy.nanmean(these_values),
            numpy.nanmedian(these_values),
            numpy.nanmin(these_values),
            numpy.nanmax(these_values),
            numpy.mean(these_real_values < 0),
            numpy.mean(these_real_values > 0)
        ))

    for k in range(len(satellite_predictor_names)):
        variable_name = satellite_predictor_names[k]
        these_values = satellite_predictor_matrix_nh[..., k]
        these_real_values = these_values[
            numpy.invert(numpy.isnan(these_values))
        ]

        print((
            'Variable "{0:s}" in northern hemi ... NaN frequency = {1:.4f} '
            '... mean = {2:.4f} ... median = {3:.4f} ... min = {4:.4f} '
            '... max = {5:.4f} ... negative frequency = {6:.4f} '
            '... positive frequency = {7:.4f}'
        ).format(
            variable_name,
            numpy.mean(numpy.isnan(these_values)),
            numpy.nanmean(these_values),
            numpy.nanmedian(these_values),
            numpy.nanmin(these_values),
            numpy.nanmax(these_values),
            numpy.mean(these_real_values < 0),
            numpy.mean(these_real_values > 0)
        ))

        these_values = satellite_predictor_matrix_sh[..., k]
        these_real_values = these_values[
            numpy.invert(numpy.isnan(these_values))
        ]

        print((
            'Variable "{0:s}" in southern hemi ... NaN frequency = {1:.4f} '
            '... mean = {2:.4f} ... median = {3:.4f} ... min = {4:.4f} '
            '... max = {5:.4f} ... negative frequency = {6:.4f} '
            '... positive frequency = {7:.4f}\n'
        ).format(
            variable_name,
            numpy.mean(numpy.isnan(these_values)),
            numpy.nanmean(these_values),
            numpy.nanmedian(these_values),
            numpy.nanmin(these_values),
            numpy.nanmax(these_values),
            numpy.mean(these_real_values < 0),
            numpy.mean(these_real_values > 0)
        ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME)
    )
