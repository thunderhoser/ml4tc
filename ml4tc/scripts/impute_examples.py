"""Imputes missing data in learning examples."""

import os
import argparse
from ml4tc.io import example_io
from ml4tc.utils import general_utils
from ml4tc.utils import satellite_utils
from ml4tc.utils import normalization
from ml4tc.utils import imputation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
YEAR_ARG_NAME = 'year'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
MIN_TEMPORAL_FRACTION_ARG_NAME = 'min_temporal_coverage_fraction'
MIN_NUM_TIMES_ARG_NAME = 'min_num_times'
MIN_SPATIAL_FRACTION_ARG_NAME = 'min_spatial_coverage_fraction'
MIN_NUM_PIXELS_ARG_NAME = 'min_num_pixels'
FILL_VALUE_ARG_NAME = 'fill_value_for_isotherm_stuff'
COMPRESS_ARG_NAME = 'compress_output_files'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing unnormalized examples with missing '
    'values.  Files therein will be found by `example_io.find_file` and read '
    'by `example_io.read_file`.'
)
YEAR_HELP_STRING = (
    'Data will be imputed only for this year.  If you would rather impute data '
    'for specific cyclones, leave this argument alone.'
)
CYCLONE_IDS_HELP_STRING = (
    'List of strings.  Data will be imputed for these cyclones.  If you would '
    'rather impute data for all cyclones in a year, leave this argument alone.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be read by '
    '`normalization.read_file`).'
)
MIN_TEMPORAL_FRACTION_HELP_STRING = (
    'Minimum coverage fraction for a temporal variable.  If fewer times have '
    'actual values (not NaN), then interpolation will not be done.'
)
MIN_NUM_TIMES_HELP_STRING = (
    'Minimum number of times for a temporal variable.  If fewer times have '
    'actual values (not NaN), then interpolation will not be done.'
)
MIN_SPATIAL_FRACTION_HELP_STRING = (
    'Minimum coverage fraction for a spatial variable.  If fewer pixels have '
    'actual values (not NaN), then interpolation will not be done.'
)
MIN_NUM_PIXELS_HELP_STRING = (
    'Minimum number of pixels for a spatial variable.  If fewer pixels have '
    'actual values (not NaN), then interpolation will not be done.'
)
FILL_VALUE_HELP_STRING = (
    'Fill value for isotherm-related variables.  If None, will use climatology '
    'as fill value.  I suggest setting a large negative fill value, because '
    'missing values for isotherm-related variables usually mean that the given '
    'isotherm does not exist in the ocean column -- not that the value is just '
    'unknown.'
)
COMPRESS_HELP_STRING = 'Boolean flag.  If 1 (0), will (not) gzip output files.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Unnormalized examples without missing values '
    'will be written here by `example_io.write_file`, to exact locations '
    'determined by `example_io.find_file`.'
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
    '--' + MIN_TEMPORAL_FRACTION_ARG_NAME, type=float, required=False,
    default=1. / 3, help=MIN_TEMPORAL_FRACTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_NUM_TIMES_ARG_NAME, type=int, required=False, default=4,
    help=MIN_NUM_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_SPATIAL_FRACTION_ARG_NAME, type=float, required=False,
    default=0.5, help=MIN_SPATIAL_FRACTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_NUM_PIXELS_ARG_NAME, type=int, required=False, default=10000,
    help=MIN_NUM_PIXELS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FILL_VALUE_ARG_NAME, type=float, required=False,
    default=imputation.LARGE_NEGATIVE_NUMBER, help=FILL_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPRESS_ARG_NAME, type=int, required=True,
    help=COMPRESS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_example_dir_name, year, cyclone_id_strings,
         normalization_file_name, min_temporal_coverage_fraction, min_num_times,
         min_spatial_coverage_fraction, min_num_pixels,
         fill_value_for_isotherm_stuff, compress_output_files,
         output_example_dir_name):
    """Imputes missing data in learning examples.

    This is effectively the main method.

    :param input_example_dir_name: See documentation at top of file.
    :param year: Same.
    :param cyclone_id_strings: Same.
    :param normalization_file_name: Same.
    :param min_temporal_coverage_fraction: Same.
    :param min_num_times: Same.
    :param min_spatial_coverage_fraction: Same.
    :param min_num_pixels: Same.
    :param fill_value_for_isotherm_stuff: Same.
    :param compress_output_files: Same.
    :param output_example_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_table_xarray = normalization.read_file(
        normalization_file_name
    )

    if len(cyclone_id_strings) == 1 and cyclone_id_strings[0] == '':
        cyclone_id_strings = None

    if cyclone_id_strings is None:
        cyclone_id_strings = example_io.find_cyclones(
            directory_name=input_example_dir_name, raise_error_if_all_missing=True
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

        print('Reading examples with missing data from: "{0:s}"...'.format(
            input_example_file_name
        ))
        example_table_xarray = example_io.read_file(input_example_file_name)

        example_table_xarray = imputation.impute_examples(
            example_table_xarray_unnorm=example_table_xarray,
            normalization_table_xarray=normalization_table_xarray,
            min_temporal_coverage_fraction=min_temporal_coverage_fraction,
            min_num_times=min_num_times,
            min_spatial_coverage_fraction=min_spatial_coverage_fraction,
            min_num_pixels=min_num_pixels,
            fill_value_for_isotherm_stuff=fill_value_for_isotherm_stuff
        )

        print('\nWriting examples with imputed data to: "{0:s}"...'.format(
            output_example_file_name
        ))
        example_io.write_file(
            example_table_xarray=example_table_xarray,
            netcdf_file_name=output_example_file_name
        )

        print(SEPARATOR_STRING)

        if compress_output_files:
            general_utils.compress_file(output_example_file_name)
            os.remove(output_example_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME),
        cyclone_id_strings=getattr(INPUT_ARG_OBJECT, CYCLONE_IDS_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        min_temporal_coverage_fraction=getattr(
            INPUT_ARG_OBJECT, MIN_TEMPORAL_FRACTION_ARG_NAME
        ),
        min_num_times=getattr(INPUT_ARG_OBJECT, MIN_NUM_TIMES_ARG_NAME),
        min_spatial_coverage_fraction=getattr(
            INPUT_ARG_OBJECT, MIN_SPATIAL_FRACTION_ARG_NAME
        ),
        min_num_pixels=getattr(INPUT_ARG_OBJECT, MIN_NUM_PIXELS_ARG_NAME),
        fill_value_for_isotherm_stuff=getattr(
            INPUT_ARG_OBJECT, FILL_VALUE_ARG_NAME
        ),
        compress_output_files=bool(
            getattr(INPUT_ARG_OBJECT, COMPRESS_ARG_NAME)
        ),
        output_example_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
