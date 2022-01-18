"""Investigates SHIPS variables."""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import ships_io
import satellite_utils

INPUT_DIR_ARG_NAME = 'input_ships_dir_name'
YEAR_ARG_NAME = 'year'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`ships_io.find_file` and read by `ships_io.read_file`.'
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
    """Investigates SHIPS variables.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param year: Same.
    """

    cyclone_id_strings = ships_io.find_cyclones(
        directory_name=input_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [satellite_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )

    good_indices = numpy.where(cyclone_years == year)[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    cyclone_id_strings.sort()

    ships_tables_nh_xarray = []
    ships_tables_sh_xarray = []

    for this_cyclone_id_string in cyclone_id_strings:
        this_file_name = ships_io.find_file(
            directory_name=input_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )
        this_basin_id_string = satellite_utils.parse_cyclone_id(
            this_cyclone_id_string
        )[1]

        print('Reading data from: "{0:s}"...'.format(this_file_name))

        if (
                this_basin_id_string ==
                satellite_utils.SOUTHERN_HEMISPHERE_ID_STRING
        ):
            ships_tables_sh_xarray.append(
                ships_io.read_file(this_file_name)
            )
        else:
            ships_tables_nh_xarray.append(
                ships_io.read_file(this_file_name)
            )

    ships_table_nh_xarray = xarray.concat(
        objs=ships_tables_nh_xarray, dim=ships_io.STORM_OBJECT_DIM
    )
    del ships_tables_nh_xarray

    ships_table_sh_xarray = xarray.concat(
        objs=ships_tables_sh_xarray, dim=ships_io.STORM_OBJECT_DIM
    )
    del ships_tables_sh_xarray
    print('\n')

    for variable_name in ships_table_nh_xarray.data_vars:
        these_values = ships_table_nh_xarray[variable_name].values
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

    print('\n')

    for variable_name in ships_table_sh_xarray.data_vars:
        these_values = ships_table_sh_xarray[variable_name].values
        these_real_values = these_values[
            numpy.invert(numpy.isnan(these_values))
        ]

        print((
            'Variable "{0:s}" in southern hemi ... NaN frequency = {1:.4f} '
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


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME)
    )
