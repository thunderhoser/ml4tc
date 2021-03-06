"""IO methods for learning examples."""

import os
import glob
import xarray
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tc.utils import satellite_utils

GZIP_FILE_EXTENSION = '.gz'
CYCLONE_ID_REGEX = '[0-9][0-9][0-9][0-9][A-Z][A-Z][0-9][0-9]'


def find_file(directory_name, cyclone_id_string, prefer_zipped=True,
              allow_other_format=True, raise_error_if_missing=True):
    """Finds NetCDF file with learning examples.

    :param directory_name: Name of directory with example files.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `satellite_utils.parse_cyclone_id`).
    :param prefer_zipped: Boolean flag.  If True, will look for zipped file
        first.  If False, will look for unzipped file first.
    :param allow_other_format: Boolean flag.  If True, will allow opposite of
        preferred file format (zipped or unzipped).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: example_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    satellite_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_boolean(prefer_zipped)
    error_checking.assert_is_boolean(allow_other_format)
    error_checking.assert_is_boolean(raise_error_if_missing)

    example_file_name = '{0:s}/learning_examples_{1:s}.nc{2:s}'.format(
        directory_name, cyclone_id_string,
        GZIP_FILE_EXTENSION if prefer_zipped else ''
    )

    if os.path.isfile(example_file_name):
        return example_file_name

    if allow_other_format:
        if prefer_zipped:
            example_file_name = example_file_name[:-len(GZIP_FILE_EXTENSION)]
        else:
            example_file_name += GZIP_FILE_EXTENSION

    if os.path.isfile(example_file_name) or not raise_error_if_missing:
        return example_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        example_file_name
    )
    raise ValueError(error_string)


def find_cyclones(directory_name, raise_error_if_all_missing=True):
    """Finds all cyclones.

    :param directory_name: Name of directory with example files.
    :param raise_error_if_all_missing: Boolean flag.  If no cyclones are found
        and `raise_error_if_all_missing == True`, will throw error.  If no
        cyclones are found and `raise_error_if_all_missing == False`, will
        return empty list.
    :return: cyclone_id_strings: List of cyclone IDs.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    file_pattern = '{0:s}/learning_examples_{1:s}.nc'.format(
        directory_name, CYCLONE_ID_REGEX
    )
    example_file_names = glob.glob(file_pattern)
    file_pattern = '{0:s}{1:s}'.format(file_pattern, GZIP_FILE_EXTENSION)
    example_file_names += glob.glob(file_pattern)

    cyclone_id_strings = []

    for this_file_name in example_file_names:
        try:
            cyclone_id_strings.append(
                file_name_to_cyclone_id(this_file_name)
            )
        except:
            pass

    cyclone_id_strings = list(set(cyclone_id_strings))
    cyclone_id_strings.sort()

    if raise_error_if_all_missing and len(cyclone_id_strings) == 0:
        error_string = (
            'Could not find any cyclone IDs from files with pattern: "{0:s}"'
        ).format(file_pattern)

        raise ValueError(error_string)

    return cyclone_id_strings


def file_name_to_cyclone_id(example_file_name):
    """Parses cyclone ID from name of file with learning examples.

    :param example_file_name: File path.
    :return: cyclone_id_string: Cyclone ID.
    """

    error_checking.assert_is_string(example_file_name)
    pathless_file_name = os.path.split(example_file_name)[1]

    cyclone_id_string = pathless_file_name.split('.')[0].split('_')[-1]
    satellite_utils.parse_cyclone_id(cyclone_id_string)

    return cyclone_id_string


def read_file(netcdf_file_name):
    """Reads learning examples from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: example_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(example_table_xarray, netcdf_file_name):
    """Writes learning examples to NetCDF file.

    :param example_table_xarray: xarray table in format returned by `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    example_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )
