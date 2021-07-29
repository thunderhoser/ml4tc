"""Methods for computing, reading, and writing saliency maps."""

import numpy
import netCDF4
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps as saliency_utils

EXAMPLE_DIMENSION_KEY = 'example'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_char'
GRID_ROW_DIMENSION_KEY = 'grid_row'
GRID_COLUMN_DIMENSION_KEY = 'grid_column'
SATELLITE_LAG_TIME_KEY = 'satellite_lag_time'
GRIDDED_SATELLITE_CHANNEL_KEY = 'gridded_satellite_channel'
UNGRIDDED_SATELLITE_CHANNEL_KEY = 'ungridded_satellite_channel'
SHIPS_LAG_TIME_KEY = 'ships_lag_time'
SHIPS_CHANNEL_KEY = 'ships_channel'

CYCLONE_IDS_KEY = 'cyclone_id_strings'
INIT_TIMES_KEY = 'init_times_unix_sec'
GRIDDED_SATELLITE_SALIENCY_KEY = 'gridded_satellite_saliency_matrix'
UNGRIDDED_SATELLITE_SALIENCY_KEY = 'ungridded_satellite_saliency_matrix'
SHIPS_SALIENCY_KEY = 'ships_saliency_matrix'
SALIENCY_KEY = 'saliency_matrices'
GRIDDED_SATELLITE_INPUT_GRAD_KEY = 'gridded_satellite_input_grad_matrix'
UNGRIDDED_SATELLITE_INPUT_GRAD_KEY = 'ungridded_satellite_input_grad_matrix'
SHIPS_INPUT_GRAD_KEY = 'ships_input_grad_matrix'
INPUT_TIMES_GRAD_KEY = 'input_times_grad_matrices'

MODEL_FILE_KEY = 'model_file_name'
LAYER_NAME_KEY = 'layer_name'
NEURON_INDICES_KEY = 'neuron_indices'
IDEAL_ACTIVATION_KEY = 'ideal_activation'


def check_metadata(layer_name, neuron_indices, ideal_activation):
    """Checks metadata for errors.

    :param layer_name: See doc for `get_saliency_one_neuron`.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    """

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_integer_numpy_array(neuron_indices)
    error_checking.assert_is_geq_numpy_array(neuron_indices, 0)
    error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)
    error_checking.assert_is_not_nan(ideal_activation)


def get_saliency_one_neuron(
        model_object, predictor_matrices, layer_name, neuron_indices,
        ideal_activation):
    """Computes saliency maps with respect to activation of one neuron.

    T = number of input tensors to model

    The "relevant neuron" is that whose activation will be used in the numerator
    of the saliency equation.  In other words, if the relevant neuron is n,
    the saliency of each predictor x will be d(a_n) / dx, where a_n is the
    activation of n.

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrices: length-T list, where each element is a numpy
        array of predictors.  Predictors must be formatted in the same way as
        for training.
    :param layer_name: Name of layer with relevant neuron.
    :param neuron_indices: 1-D numpy array with indices of relevant neuron.
        Must have length D - 1, where D = number of dimensions in layer output.
        The first dimension is the batch dimension, which always has length
        `None` in Keras.
    :param ideal_activation: Ideal neuron activation, used to define loss
        function.  The loss function will be
        (neuron_activation - ideal_activation)**2.
    :return: saliency_matrices: length-T list, where each element is a numpy
        array of saliency values.  saliency_matrices[i] will have the same shape
        as predictor_matrices[i].
    """

    check_metadata(
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation
    )

    error_checking.assert_is_list(predictor_matrices)
    for this_predictor_matrix in predictor_matrices:
        error_checking.assert_is_numpy_array_without_nan(this_predictor_matrix)

    activation_tensor = None

    for k in neuron_indices[::-1]:
        if activation_tensor is None:
            activation_tensor = (
                model_object.get_layer(name=layer_name).output[..., k]
            )
        else:
            activation_tensor = activation_tensor[..., k]

    loss_tensor = (activation_tensor - ideal_activation) ** 2

    return saliency_utils.do_saliency_calculations(
        model_object=model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=predictor_matrices
    )


def write_file(
        netcdf_file_name, saliency_matrices, input_times_grad_matrices,
        cyclone_id_strings, init_times_unix_sec, model_file_name, layer_name,
        neuron_indices, ideal_activation):
    """Writes saliency maps to NetCDF file.

    E = number of examples
    T = number of input tensors to model

    :param netcdf_file_name: Path to output file.
    :param saliency_matrices: length-T list, where each element is a numpy
        array of saliency values.  saliency_matrices[i] should have the same
        shape as the [i]th input tensor to the model.  Also, the first axis of
        each numpy array must have length E.
    :param input_times_grad_matrices: Same as `saliency_matrices` but with
        input-times-gradient values instead.
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param init_times_unix_sec: length-E numpy array of forecast-init times.
    :param model_file_name: Path to file with neural net used to create saliency
        maps (readable by `neural_net.read_model`).
    :param layer_name: See doc for `get_saliency_one_neuron`.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    """

    # Check input args.
    check_metadata(
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation
    )

    error_checking.assert_is_list(saliency_matrices)
    error_checking.assert_is_list(input_times_grad_matrices)
    assert len(saliency_matrices) == len(input_times_grad_matrices)

    num_examples = -1

    for i in range(len(saliency_matrices)):
        error_checking.assert_is_numpy_array_without_nan(saliency_matrices[i])
        error_checking.assert_is_numpy_array_without_nan(
            input_times_grad_matrices[i]
        )
        if i == 0:
            num_examples = saliency_matrices[i].shape[0]

        expected_dim = numpy.array(
            (num_examples,) + saliency_matrices[i].shape[1:], dtype=int
        )
        error_checking.assert_is_numpy_array(
            saliency_matrices[i], exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array(
            input_times_grad_matrices[i], exact_dimensions=expected_dim
        )

    expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_string_list(cyclone_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(cyclone_id_strings), exact_dimensions=expected_dim
    )

    error_checking.assert_is_integer_numpy_array(init_times_unix_sec)
    error_checking.assert_is_numpy_array(
        init_times_unix_sec, exact_dimensions=expected_dim
    )

    error_checking.assert_is_string(model_file_name)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(LAYER_NAME_KEY, layer_name)
    dataset_object.setncattr(NEURON_INDICES_KEY, neuron_indices)
    dataset_object.setncattr(IDEAL_ACTIVATION_KEY, ideal_activation)

    num_grid_rows = saliency_matrices[0].shape[1]
    num_grid_columns = saliency_matrices[0].shape[2]
    num_satellite_lag_times = saliency_matrices[0].shape[3]
    num_gridded_satellite_channels = saliency_matrices[0].shape[4]
    num_ungridded_satellite_channels = saliency_matrices[1].shape[2]
    num_ships_lag_times = saliency_matrices[2].shape[1]
    num_ships_channels = saliency_matrices[2].shape[2]

    expected_dim = numpy.array([
        num_examples, num_satellite_lag_times, num_ungridded_satellite_channels
    ], dtype=int)
    error_checking.assert_is_numpy_array(
        saliency_matrices[1], exact_dimensions=expected_dim
    )

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(GRID_ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(GRID_COLUMN_DIMENSION_KEY, num_grid_columns)
    dataset_object.createDimension(
        SATELLITE_LAG_TIME_KEY, num_satellite_lag_times
    )
    dataset_object.createDimension(
        GRIDDED_SATELLITE_CHANNEL_KEY, num_gridded_satellite_channels
    )
    dataset_object.createDimension(
        UNGRIDDED_SATELLITE_CHANNEL_KEY, num_ungridded_satellite_channels
    )
    dataset_object.createDimension(SHIPS_LAG_TIME_KEY, num_ships_lag_times)
    dataset_object.createDimension(SHIPS_CHANNEL_KEY, num_ships_channels)

    if num_examples == 0:
        num_id_characters = 1
    else:
        num_id_characters = numpy.max(numpy.array([
            len(id) for id in cyclone_id_strings
        ]))

    dataset_object.createDimension(CYCLONE_ID_CHAR_DIM_KEY, num_id_characters)

    this_string_format = 'S{0:d}'.format(num_id_characters)
    cyclone_ids_char_array = netCDF4.stringtochar(numpy.array(
        cyclone_id_strings, dtype=this_string_format
    ))

    dataset_object.createVariable(
        CYCLONE_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, CYCLONE_ID_CHAR_DIM_KEY)
    )
    dataset_object.variables[CYCLONE_IDS_KEY][:] = numpy.array(
        cyclone_ids_char_array
    )

    dataset_object.createVariable(
        INIT_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[INIT_TIMES_KEY][:] = init_times_unix_sec

    these_dim = (
        EXAMPLE_DIMENSION_KEY, GRID_ROW_DIMENSION_KEY,
        GRID_COLUMN_DIMENSION_KEY, SATELLITE_LAG_TIME_KEY,
        GRIDDED_SATELLITE_CHANNEL_KEY
    )
    dataset_object.createVariable(
        GRIDDED_SATELLITE_SALIENCY_KEY,
        datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[GRIDDED_SATELLITE_SALIENCY_KEY][:] = (
        saliency_matrices[0]
    )

    dataset_object.createVariable(
        GRIDDED_SATELLITE_INPUT_GRAD_KEY,
        datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[GRIDDED_SATELLITE_INPUT_GRAD_KEY][:] = (
        input_times_grad_matrices[0]
    )

    these_dim = (
        EXAMPLE_DIMENSION_KEY, SATELLITE_LAG_TIME_KEY,
        UNGRIDDED_SATELLITE_CHANNEL_KEY
    )
    dataset_object.createVariable(
        UNGRIDDED_SATELLITE_SALIENCY_KEY,
        datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[UNGRIDDED_SATELLITE_SALIENCY_KEY][:] = (
        saliency_matrices[1]
    )

    dataset_object.createVariable(
        UNGRIDDED_SATELLITE_INPUT_GRAD_KEY,
        datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[UNGRIDDED_SATELLITE_INPUT_GRAD_KEY][:] = (
        input_times_grad_matrices[1]
    )

    these_dim = (
        EXAMPLE_DIMENSION_KEY, SHIPS_LAG_TIME_KEY, SHIPS_CHANNEL_KEY
    )
    dataset_object.createVariable(
        SHIPS_SALIENCY_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[SHIPS_SALIENCY_KEY][:] = saliency_matrices[2]

    dataset_object.createVariable(
        SHIPS_INPUT_GRAD_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[SHIPS_INPUT_GRAD_KEY][:] = (
        input_times_grad_matrices[2]
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads saliency maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: saliency_dict: Dictionary with the following keys.
    saliency_dict['saliency_matrices']: See doc for `write_file`.
    saliency_dict['input_times_grad_matrices']: Same.
    saliency_dict['cyclone_id_strings']: Same.
    saliency_dict['init_times_unix_sec']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['layer_name']: Same.
    saliency_dict['neuron_indices']: Same.
    saliency_dict['ideal_activation']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    saliency_matrices = [
        dataset_object.variables[GRIDDED_SATELLITE_SALIENCY_KEY][:],
        dataset_object.variables[UNGRIDDED_SATELLITE_SALIENCY_KEY][:],
        dataset_object.variables[SHIPS_SALIENCY_KEY][:]
    ]
    input_times_grad_matrices = [
        dataset_object.variables[GRIDDED_SATELLITE_INPUT_GRAD_KEY][:],
        dataset_object.variables[UNGRIDDED_SATELLITE_INPUT_GRAD_KEY][:],
        dataset_object.variables[SHIPS_INPUT_GRAD_KEY][:]
    ]

    saliency_dict = {
        SALIENCY_KEY: saliency_matrices,
        INPUT_TIMES_GRAD_KEY: input_times_grad_matrices,
        CYCLONE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[CYCLONE_IDS_KEY][:])
        ],
        INIT_TIMES_KEY: dataset_object.variables[INIT_TIMES_KEY][:],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        LAYER_NAME_KEY: str(getattr(dataset_object, LAYER_NAME_KEY)),
        NEURON_INDICES_KEY: numpy.array(
            getattr(dataset_object, NEURON_INDICES_KEY), dtype=int
        ),
        IDEAL_ACTIVATION_KEY: getattr(dataset_object, IDEAL_ACTIVATION_KEY)
    }

    dataset_object.close()
    return saliency_dict
