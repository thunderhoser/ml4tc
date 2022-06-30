"""IO methods for raw SHIPS predictions."""

import numpy
import netCDF4
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

EXAMPLE_DIMENSION_KEY = 'example'
LEAD_TIME_DIMENSION_KEY = 'lead_time'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_character'

MAIN_PROBABILITIES_KEY = 'main_probabilities'
CONSENSUS_PROBABILITIES_KEY = 'consensus_probabilities'

RI_PROBABILITIES_KEY = 'ri_probability_matrix'
FORECAST_LABELS_LAND_KEY = 'forecast_label_matrix_land'
FORECAST_LABELS_LGE_KEY = 'forecast_label_matrix_lge'
LEAD_TIMES_KEY = 'lead_times_hours'
CYCLONE_IDS_KEY = 'cyclone_id_strings'
INIT_LATITUDES_KEY = 'init_latitudes_deg_n'
INIT_LONGITUDES_KEY = 'init_longitudes_deg_e'
INIT_TIMES_KEY = 'init_times_unix_sec'


def write_td_to_ts_file(
        netcdf_file_name, lead_times_hours, forecast_label_matrix_land,
        forecast_label_matrix_lge, init_latitudes_deg_n, init_longitudes_deg_e,
        cyclone_id_strings, init_times_unix_sec):
    """Writes TD-to-TS predictions to NetCDF file.

    E = number of examples
    L = number of lead times

    :param netcdf_file_name: Path to output file.
    :param lead_times_hours: length-L numpy array of lead times.
    :param forecast_label_matrix_land: E-by-L numpy array of forecast labels
        from "land" model (1 for TD that becomes a TS, 0 for TD that does not
        become a TS, -1 for unknown).
    :param forecast_label_matrix_lge: Same but for LGE (logistic growth
        equation) model.
    :param init_latitudes_deg_n: length-E numpy array of initial latitudes
        (deg north).
    :param init_longitudes_deg_e: length-E numpy array of initial latitudes
        (deg north).
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param init_times_unix_sec: length-E numpy array of forecast-init times.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    error_checking.assert_is_numpy_array(lead_times_hours, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(lead_times_hours)
    error_checking.assert_is_geq_numpy_array(lead_times_hours, 0)
    error_checking.assert_is_string_list(cyclone_id_strings)

    num_examples = len(cyclone_id_strings)
    num_lead_times = len(lead_times_hours)
    expected_dim = numpy.array([num_examples, num_lead_times], dtype=int)

    error_checking.assert_is_numpy_array(
        forecast_label_matrix_land, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(forecast_label_matrix_land)
    error_checking.assert_is_geq_numpy_array(forecast_label_matrix_land, -1)
    error_checking.assert_is_leq_numpy_array(forecast_label_matrix_land, 1)

    error_checking.assert_is_numpy_array(
        forecast_label_matrix_lge, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(forecast_label_matrix_lge)
    error_checking.assert_is_geq_numpy_array(forecast_label_matrix_lge, -1)
    error_checking.assert_is_leq_numpy_array(forecast_label_matrix_lge, 1)

    error_checking.assert_is_numpy_array(
        init_latitudes_deg_n,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )
    error_checking.assert_is_valid_lat_numpy_array(
        init_latitudes_deg_n, allow_nan=False
    )

    error_checking.assert_is_numpy_array(
        init_longitudes_deg_e,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )
    init_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        init_longitudes_deg_e, allow_nan=False
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
    dataset_object.createDimension(LEAD_TIME_DIMENSION_KEY, num_lead_times)

    dataset_object.createVariable(
        FORECAST_LABELS_LAND_KEY, datatype=numpy.int32,
        dimensions=(EXAMPLE_DIMENSION_KEY, LEAD_TIME_DIMENSION_KEY)
    )
    dataset_object.variables[FORECAST_LABELS_LAND_KEY][:] = (
        forecast_label_matrix_land
    )

    dataset_object.createVariable(
        FORECAST_LABELS_LGE_KEY, datatype=numpy.int32,
        dimensions=(EXAMPLE_DIMENSION_KEY, LEAD_TIME_DIMENSION_KEY)
    )
    dataset_object.variables[FORECAST_LABELS_LGE_KEY][:] = (
        forecast_label_matrix_land
    )

    dataset_object.createVariable(
        LEAD_TIMES_KEY, datatype=numpy.int32, dimensions=LEAD_TIME_DIMENSION_KEY
    )
    dataset_object.variables[LEAD_TIMES_KEY][:] = lead_times_hours

    dataset_object.createVariable(
        INIT_LATITUDES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[INIT_LATITUDES_KEY][:] = init_latitudes_deg_n

    dataset_object.createVariable(
        INIT_LONGITUDES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[INIT_LONGITUDES_KEY][:] = init_longitudes_deg_e

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


def read_td_to_ts_file(netcdf_file_name):
    """Reads TD-to-TS predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['lead_times_hours']: See doc for `write_td_to_ts_file`.
    prediction_dict['forecast_label_matrix_land']: Same.
    prediction_dict['forecast_label_matrix_lge']: Same.
    prediction_dict['init_latitudes_deg_n']: Same.
    prediction_dict['init_longitudes_deg_e']: Same.
    prediction_dict['cyclone_id_strings']: Cyclone ID.
    prediction_dict['init_times_unix_sec']: Forecast-initialization time.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        FORECAST_LABELS_LAND_KEY:
            dataset_object.variables[FORECAST_LABELS_LAND_KEY][:],
        FORECAST_LABELS_LGE_KEY:
            dataset_object.variables[FORECAST_LABELS_LGE_KEY][:],
        LEAD_TIMES_KEY: dataset_object.variables[LEAD_TIMES_KEY][:],
        INIT_LATITUDES_KEY: dataset_object.variables[INIT_LATITUDES_KEY][:],
        INIT_LONGITUDES_KEY: dataset_object.variables[INIT_LONGITUDES_KEY][:],
        CYCLONE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[CYCLONE_IDS_KEY][:])
        ],
        INIT_TIMES_KEY: dataset_object.variables[INIT_TIMES_KEY][:]
    }

    dataset_object.close()
    return prediction_dict


def write_ri_file(
        netcdf_file_name, ri_probability_matrix, cyclone_id_strings,
        init_latitudes_deg_n, init_longitudes_deg_e, init_times_unix_sec):
    """Writes rapid-intensification predictions to NetCDF file.

    E = number of examples

    :param netcdf_file_name: Path to output file.
    :param ri_probability_matrix: E-by-2 numpy array of rapid-intensification
        probabilities.
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param init_latitudes_deg_n: length-E numpy array of initial latitudes
        (deg north).
    :param init_longitudes_deg_e: length-E numpy array of initial latitudes
        (deg north).
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
        init_latitudes_deg_n,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )
    error_checking.assert_is_valid_lat_numpy_array(
        init_latitudes_deg_n, allow_nan=False
    )

    error_checking.assert_is_numpy_array(
        init_longitudes_deg_e,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )
    init_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        init_longitudes_deg_e, allow_nan=False
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
        INIT_LATITUDES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[INIT_LATITUDES_KEY][:] = init_latitudes_deg_n

    dataset_object.createVariable(
        INIT_LONGITUDES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[INIT_LONGITUDES_KEY][:] = init_longitudes_deg_e

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


def read_ri_file(netcdf_file_name):
    """Reads rapid-intensification predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['ri_probability_matrix']: See doc for `write_ri_file`.
    prediction_dict['init_latitudes_deg_n']: Same.
    prediction_dict['init_longitudes_deg_e']: Same.
    prediction_dict['cyclone_id_strings']: Same.
    prediction_dict['init_times_unix_sec']: Same.

    :return: ri_probability_matrix: See doc for `write_ri_file`.
    :return: cyclone_id_strings Same.
    :return: init_times_unix_sec Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        RI_PROBABILITIES_KEY: numpy.transpose(numpy.vstack((
            dataset_object.variables[MAIN_PROBABILITIES_KEY][:],
            dataset_object.variables[CONSENSUS_PROBABILITIES_KEY][:]
        ))),
        INIT_LATITUDES_KEY: dataset_object.variables[INIT_LATITUDES_KEY][:],
        INIT_LONGITUDES_KEY: dataset_object.variables[INIT_LONGITUDES_KEY][:],
        CYCLONE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[CYCLONE_IDS_KEY][:])
        ],
        INIT_TIMES_KEY: dataset_object.variables[INIT_TIMES_KEY][:]
    }

    dataset_object.close()
    return prediction_dict
