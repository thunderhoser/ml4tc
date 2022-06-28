"""IO methods for raw SHIPS predictions."""

import os
import sys
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking

EXAMPLE_DIMENSION_KEY = 'example'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_character'

MAIN_PROBABILITIES_KEY = 'main_probabilities'
CONSENSUS_PROBABILITIES_KEY = 'consensus_probabilities'
CYCLONE_IDS_KEY = 'cyclone_id_strings'
INIT_TIMES_KEY = 'init_times_unix_sec'


def write_file(netcdf_file_name, ri_probability_matrix, cyclone_id_strings,
               init_times_unix_sec):
    """Writes SHIPS predictions to NetCDF file.

    E = number of examples

    :param netcdf_file_name: Path to output file.
    :param ri_probability_matrix: E-by-2 numpy array of rapid-intensification
        probabilities.
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param init_times_unix_sec: length-E numpy array of forecast-init times.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    error_checking.assert_is_numpy_array(
        ri_probability_matrix, num_dimensions=2
    )
    num_examples = ri_probability_matrix.shape[0]

    error_checking.assert_is_numpy_array(
        ri_probability_matrix,
        exact_dimensions=numpy.array([num_examples, 2], dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(
        ri_probability_matrix, 0, allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        ri_probability_matrix, 1, allow_nan=True
    )

    error_checking.assert_is_string_list(cyclone_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(cyclone_id_strings),
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )

    error_checking.assert_is_numpy_array(
        init_times_unix_sec,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(init_times_unix_sec)

    # Write file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)

    dataset_object.createVariable(
        MAIN_PROBABILITIES_KEY, datatype=numpy.float32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[MAIN_PROBABILITIES_KEY][:] = (
        ri_probability_matrix[:, 0]
    )

    dataset_object.createVariable(
        CONSENSUS_PROBABILITIES_KEY, datatype=numpy.float32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[CONSENSUS_PROBABILITIES_KEY][:] = (
        ri_probability_matrix[:, 1]
    )

    dataset_object.createVariable(
        INIT_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[INIT_TIMES_KEY][:] = init_times_unix_sec

    if num_examples == 0:
        num_id_characters = 1
    else:
        num_id_characters = numpy.max(numpy.array([
            len(id) for id in cyclone_id_strings
        ]))

    dataset_object.createDimension(CYCLONE_ID_CHAR_DIM_KEY, num_id_characters)

    string_format = 'S{0:d}'.format(num_id_characters)
    cyclone_ids_char_array = netCDF4.stringtochar(
        numpy.array(cyclone_id_strings, dtype=string_format)
    )

    dataset_object.createVariable(
        CYCLONE_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, CYCLONE_ID_CHAR_DIM_KEY)
    )
    dataset_object.variables[CYCLONE_IDS_KEY][:] = numpy.array(
        cyclone_ids_char_array
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads SHIPS predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: ri_probability_matrix: See doc for `write_file`.
    :return: cyclone_id_strings Same.
    :return: init_times_unix_sec Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    ri_probability_matrix = numpy.transpose(numpy.vstack((
        dataset_object.variables[MAIN_PROBABILITIES_KEY][:],
        dataset_object.variables[CONSENSUS_PROBABILITIES_KEY][:]
    )))
    cyclone_id_strings = [
        str(id) for id in
        netCDF4.chartostring(dataset_object.variables[CYCLONE_IDS_KEY][:])
    ]
    init_times_unix_sec = dataset_object.variables[INIT_TIMES_KEY][:]

    dataset_object.close()
    return ri_probability_matrix, cyclone_id_strings, init_times_unix_sec
