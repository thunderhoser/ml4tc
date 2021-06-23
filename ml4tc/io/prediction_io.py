"""Input/output methods for model predictions."""

import numpy
import netCDF4
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tc.utils import satellite_utils

EXAMPLE_DIMENSION_KEY = 'example'
CLASS_DIMENSION_KEY = 'class'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_character'

PROBABILITY_MATRIX_KEY = 'forecast_probability_matrix'
TARGET_CLASSES_KEY = 'target_classes'
CYCLONE_IDS_KEY = 'cyclone_id_strings'
VALID_TIMES_KEY = 'valid_times_unix_sec'
MODEL_FILE_KEY = 'model_file_name'


def write_file(
        netcdf_file_name, forecast_probability_matrix, target_classes,
        cyclone_id_strings, valid_times_unix_sec, model_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples
    K = number of classes

    :param netcdf_file_name: Path to output file.
    :param forecast_probability_matrix: E-by-K numpy array of forecast
        probabilities.
    :param target_classes: length-E numpy array of target classes, all integers
        in range [0, K - 1].
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param valid_times_unix_sec: length-E list of valid times.
    :param model_file_name: Path to file with trained model (readable by
        `neural_net.read_model`).
    """

    error_checking.assert_is_numpy_array(
        forecast_probability_matrix, num_dimensions=2
    )
    error_checking.assert_is_geq_numpy_array(forecast_probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(forecast_probability_matrix, 1.)

    num_examples = forecast_probability_matrix.shape[0]
    num_classes = forecast_probability_matrix.shape[1]
    expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_integer_numpy_array(target_classes)
    error_checking.assert_is_numpy_array(
        target_classes, exact_dimensions=expected_dim
    )
    error_checking.assert_is_geq_numpy_array(target_classes, 0)
    error_checking.assert_is_less_than_numpy_array(target_classes, num_classes)

    error_checking.assert_is_numpy_array(
        numpy.array(cyclone_id_strings), exact_dimensions=expected_dim
    )
    for this_id_string in cyclone_id_strings:
        _ = satellite_utils.parse_cyclone_id(this_id_string)

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec, exact_dimensions=expected_dim
    )
    error_checking.assert_is_string(model_file_name)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(CLASS_DIMENSION_KEY, num_classes)

    dataset_object.createVariable(
        PROBABILITY_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, CLASS_DIMENSION_KEY)
    )
    dataset_object.variables[PROBABILITY_MATRIX_KEY][:] = (
        forecast_probability_matrix
    )

    dataset_object.createVariable(
        TARGET_CLASSES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[TARGET_CLASSES_KEY][:] = target_classes

    dataset_object.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[VALID_TIMES_KEY][:] = valid_times_unix_sec

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
    """Reads predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['forecast_probability_matrix']: See doc for `write_file`.
    prediction_dict['target_classes']: Same.
    prediction_dict['cyclone_id_strings']: Same.
    prediction_dict['valid_times_unix_sec']: Same.
    prediction_dict['model_file_name']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        PROBABILITY_MATRIX_KEY:
            dataset_object.variables[PROBABILITY_MATRIX_KEY][:],
        TARGET_CLASSES_KEY: dataset_object.variables[TARGET_CLASSES_KEY][:],
        CYCLONE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[CYCLONE_IDS_KEY][:])
        ],
        VALID_TIMES_KEY: dataset_object.variables[VALID_TIMES_KEY][:],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY))
    }

    dataset_object.close()
    return prediction_dict
