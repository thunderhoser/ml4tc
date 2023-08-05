"""Methods for computing, reading, and writing saliency maps."""

import os
import sys
import copy
import shutil
import numpy
import xarray
import netCDF4
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import gg_saliency_maps as saliency_utils

NUM_EXAMPLES_PER_BATCH = 8

EXAMPLE_DIMENSION_KEY = 'example'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_char'
GRID_ROW_DIMENSION_KEY = 'grid_row'
GRID_COLUMN_DIMENSION_KEY = 'grid_column'
SATELLITE_LAG_TIME_KEY = 'satellite_lag_time'
GRIDDED_SATELLITE_CHANNEL_KEY = 'gridded_satellite_channel'
UNGRIDDED_SATELLITE_CHANNEL_KEY = 'ungridded_satellite_channel'
SHIPS_CHANNEL_KEY = 'ships_channel'

CYCLONE_IDS_KEY = 'cyclone_id_strings'
INIT_TIMES_KEY = 'init_times_unix_sec'
GRIDDED_SATELLITE_SALIENCY_KEY = 'gridded_satellite_saliency_matrix'
UNGRIDDED_SATELLITE_SALIENCY_KEY = 'ungridded_satellite_saliency_matrix'
SHIPS_SALIENCY_KEY = 'ships_saliency_matrix'
THREE_SALIENCY_KEY = 'three_saliency_matrices'
GRIDDED_SATELLITE_INPUT_GRAD_KEY = 'gridded_satellite_input_grad_matrix'
UNGRIDDED_SATELLITE_INPUT_GRAD_KEY = 'ungridded_satellite_input_grad_matrix'
SHIPS_INPUT_GRAD_KEY = 'ships_input_grad_matrix'
THREE_INPUT_GRAD_KEY = 'three_input_grad_matrices'
GRIDDED_SATELLITE_PREDICTORS_KEY = 'gridded_satellite_predictor_matrix'
UNGRIDDED_SATELLITE_PREDICTORS_KEY = 'ungridded_satellite_predictor_matrix'
SHIPS_PREDICTORS_KEY = 'ships_predictor_matrix'
THREE_PREDICTORS_KEY = 'three_predictor_matrices'

MODEL_FILE_KEY = 'model_file_name'
LAYER_NAME_KEY = 'layer_name'
NEURON_INDICES_KEY = 'neuron_indices'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
USE_PMM_KEY = 'use_pmm'
PMM_MAX_PERCENTILE_KEY = 'pmm_max_percentile_level'


def _read_netcdf_file(netcdf_file_name):
    """Reads saliency results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: saliency_dict: See doc for `read_file`.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    three_saliency_matrices = []
    three_input_grad_matrices = []

    if GRIDDED_SATELLITE_SALIENCY_KEY in dataset_object.variables:
        three_saliency_matrices.append(
            dataset_object.variables[GRIDDED_SATELLITE_SALIENCY_KEY][:]
        )
        three_input_grad_matrices.append(
            dataset_object.variables[GRIDDED_SATELLITE_INPUT_GRAD_KEY][:]
        )
    else:
        three_saliency_matrices.append(None)
        three_input_grad_matrices.append(None)

    if UNGRIDDED_SATELLITE_SALIENCY_KEY in dataset_object.variables:
        three_saliency_matrices.append(
            dataset_object.variables[UNGRIDDED_SATELLITE_SALIENCY_KEY][:]
        )
        three_input_grad_matrices.append(
            dataset_object.variables[UNGRIDDED_SATELLITE_INPUT_GRAD_KEY][:]
        )
    else:
        three_saliency_matrices.append(None)
        three_input_grad_matrices.append(None)

    if SHIPS_SALIENCY_KEY in dataset_object.variables:
        three_saliency_matrices.append(
            dataset_object.variables[SHIPS_SALIENCY_KEY][:]
        )
        three_input_grad_matrices.append(
            dataset_object.variables[SHIPS_INPUT_GRAD_KEY][:]
        )
    else:
        three_saliency_matrices.append(None)
        three_input_grad_matrices.append(None)

    saliency_dict = {
        THREE_SALIENCY_KEY: three_saliency_matrices,
        THREE_INPUT_GRAD_KEY: three_input_grad_matrices,
        CYCLONE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[CYCLONE_IDS_KEY][:])
        ],
        INIT_TIMES_KEY: dataset_object.variables[INIT_TIMES_KEY][:],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        LAYER_NAME_KEY: str(getattr(dataset_object, LAYER_NAME_KEY)),
        NEURON_INDICES_KEY: numpy.array(
            getattr(dataset_object, NEURON_INDICES_KEY), dtype=float
        ),
        IDEAL_ACTIVATION_KEY: getattr(dataset_object, IDEAL_ACTIVATION_KEY)
    }

    if len(saliency_dict[NEURON_INDICES_KEY].shape) == 0:
        saliency_dict[NEURON_INDICES_KEY] = numpy.array(
            [saliency_dict[NEURON_INDICES_KEY]], dtype=float
        )

    dataset_object.close()
    return saliency_dict


def _read_zarr_file(zarr_file_name):
    """Reads saliency results from zarr file.

    :param zarr_file_name: Path to input file.
    :return: saliency_dict: See doc for `read_file`.
    """

    saliency_table_xarray = xarray.open_zarr(zarr_file_name)

    three_saliency_matrices = []
    three_input_grad_matrices = []

    if GRIDDED_SATELLITE_SALIENCY_KEY in list(saliency_table_xarray.data_vars):
        three_saliency_matrices.append(
            saliency_table_xarray[GRIDDED_SATELLITE_SALIENCY_KEY].values
        )
        three_input_grad_matrices.append(
            saliency_table_xarray[GRIDDED_SATELLITE_INPUT_GRAD_KEY].values
        )
    else:
        three_saliency_matrices.append(None)
        three_input_grad_matrices.append(None)

    if UNGRIDDED_SATELLITE_SALIENCY_KEY in list(
            saliency_table_xarray.data_vars
    ):
        three_saliency_matrices.append(
            saliency_table_xarray[UNGRIDDED_SATELLITE_SALIENCY_KEY].values
        )
        three_input_grad_matrices.append(
            saliency_table_xarray[UNGRIDDED_SATELLITE_INPUT_GRAD_KEY].values
        )
    else:
        three_saliency_matrices.append(None)
        three_input_grad_matrices.append(None)

    if SHIPS_SALIENCY_KEY in list(saliency_table_xarray.data_vars):
        three_saliency_matrices.append(
            saliency_table_xarray[SHIPS_SALIENCY_KEY].values
        )
        three_input_grad_matrices.append(
            saliency_table_xarray[SHIPS_INPUT_GRAD_KEY].values
        )
    else:
        three_saliency_matrices.append(None)
        three_input_grad_matrices.append(None)

    saliency_dict = {
        THREE_SALIENCY_KEY: three_saliency_matrices,
        THREE_INPUT_GRAD_KEY: three_input_grad_matrices,
        CYCLONE_IDS_KEY: saliency_table_xarray[CYCLONE_IDS_KEY].values.tolist(),
        INIT_TIMES_KEY: saliency_table_xarray[INIT_TIMES_KEY].values,
        MODEL_FILE_KEY: saliency_table_xarray.attrs[MODEL_FILE_KEY],
        LAYER_NAME_KEY: saliency_table_xarray.attrs[LAYER_NAME_KEY],
        NEURON_INDICES_KEY: numpy.array(
            saliency_table_xarray.attrs[NEURON_INDICES_KEY], dtype=float
        ),
        IDEAL_ACTIVATION_KEY: saliency_table_xarray.attrs[IDEAL_ACTIVATION_KEY]
    }

    if len(saliency_dict[NEURON_INDICES_KEY].shape) == 0:
        saliency_dict[NEURON_INDICES_KEY] = numpy.array(
            [saliency_dict[NEURON_INDICES_KEY]], dtype=float
        )

    return saliency_dict


def check_metadata(layer_name, neuron_indices, ideal_activation):
    """Checks metadata for errors.

    :param layer_name: See doc for `get_saliency_one_neuron`.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    """

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(neuron_indices, 0, allow_nan=True)
    error_checking.assert_is_not_nan(ideal_activation)


def get_saliency_one_neuron(
        model_object, three_predictor_matrices, layer_name, neuron_indices,
        ideal_activation):
    """Computes saliency maps with respect to activation of one neuron.

    The "relevant neuron" is that whose activation will be used in the numerator
    of the saliency equation.  In other words, if the relevant neuron is n,
    the saliency of each predictor x will be d(a_n) / dx, where a_n is the
    activation of n.

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param three_predictor_matrices: length-3 list, where each element is either
        None or a numpy array of predictors.  Predictors must be formatted in
        the same way as for training.
    :param layer_name: Name of layer with relevant neuron.
    :param neuron_indices: 1-D numpy array with indices of relevant neuron.  If
        this array contains NaN, activations will be averaged over many neurons.
        As one example, suppose that the layer has dimensions None x 28 x 101,
        while neuron_indices = [4, 0].  Then the relevant activation will be at
        location [:, 4, 0] in the layer.  As a second example, suppose that the
        layer has dimensions None x 28 x 101, while neuron_indices = [4, NaN].
        Then the relevant activation will be the average over locations
        [:, 4, :] in the layer.
    :param ideal_activation: Ideal neuron activation, used to define loss
        function.  The loss function will be
        (neuron_activation - ideal_activation)**2.
    :return: three_saliency_matrices: length-3 list, where each element is
        either None or a numpy array of saliency values.
        three_saliency_matrices[i] will have the same shape as
        three_predictor_matrices[i].
    """

    check_metadata(
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation
    )

    error_checking.assert_is_list(three_predictor_matrices)
    assert len(three_predictor_matrices) == 3

    for this_predictor_matrix in three_predictor_matrices:
        if this_predictor_matrix is None:
            continue
        error_checking.assert_is_numpy_array_without_nan(this_predictor_matrix)

    these_flags = numpy.array(
        [m is not None for m in three_predictor_matrices], dtype=bool
    )
    have_predictors_indices = numpy.where(these_flags)[0]

    activation_tensor = model_object.get_layer(name=layer_name).output

    for k in neuron_indices[::-1]:
        if numpy.isnan(k):
            activation_tensor = K.mean(activation_tensor, axis=-1)
        else:
            activation_tensor = activation_tensor[..., int(numpy.round(k))]

    loss_tensor = (activation_tensor - ideal_activation) ** 2

    num_examples = three_predictor_matrices[have_predictors_indices[0]].shape[0]
    saliency_matrices = [None] * len(have_predictors_indices)

    for i in range(0, num_examples, NUM_EXAMPLES_PER_BATCH):
        first_index = i
        last_index = min([
            i + NUM_EXAMPLES_PER_BATCH, num_examples
        ])

        these_matrices = saliency_utils.do_saliency_calculations(
            model_object=model_object, loss_tensor=loss_tensor,
            list_of_input_matrices=[
                three_predictor_matrices[k][first_index:last_index, ...]
                for k in have_predictors_indices
            ]
        )

        for j in range(len(have_predictors_indices)):
            if saliency_matrices[j] is None:
                these_dim = numpy.array(
                    (num_examples,) + these_matrices[j].shape[1:], dtype=int
                )
                saliency_matrices[j] = numpy.full(these_dim, numpy.nan)

            saliency_matrices[j][first_index:last_index, ...] = (
                these_matrices[j]
            )

    three_saliency_matrices = [None] * 3
    for i, j in enumerate(have_predictors_indices):
        three_saliency_matrices[j] = saliency_matrices[i]

    return three_saliency_matrices


def write_composite_file(
        netcdf_file_name, three_saliency_matrices, three_input_grad_matrices,
        three_predictor_matrices, model_file_name, use_pmm,
        pmm_max_percentile_level=None):
    """Writes composite saliency map to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param three_saliency_matrices: length-3 list, where each element is either
        None or a numpy array of saliency values.  three_saliency_matrices[i]
        should have the same shape as the [i]th input tensor to the model, but
        without the first axis, which is the example axis.
    :param three_input_grad_matrices: Same as `three_saliency_matrices` but with
        input-times-gradient values instead.
    :param three_predictor_matrices: Same as `three_saliency_matrices` but with
        predictor values instead.  Predictor values must be formatted the same
        way as for training, e.g., normalized here if they are normalized for
        training.
    :param model_file_name: Path to file with neural net used to create saliency
        maps (readable by `neural_net.read_model`).
    :param use_pmm: Boolean flag.  If True (False), maps were composited via
        probability-matched means (a simple average).
    :param pmm_max_percentile_level: Max percentile level for
        probability-matched means (PMM).  If PMM was not used, leave this alone.
    """

    # Check input args.
    error_checking.assert_is_list(three_saliency_matrices)
    error_checking.assert_is_list(three_input_grad_matrices)
    error_checking.assert_is_list(three_predictor_matrices)

    assert len(three_saliency_matrices) == 3
    assert len(three_input_grad_matrices) == 3
    assert len(three_predictor_matrices) == 3

    for i in range(len(three_saliency_matrices)):
        if three_saliency_matrices[i] is None:
            assert three_input_grad_matrices[i] is None
            assert three_predictor_matrices[i] is None
            continue

        error_checking.assert_is_numpy_array_without_nan(
            three_saliency_matrices[i]
        )
        error_checking.assert_is_numpy_array_without_nan(
            three_input_grad_matrices[i]
        )
        error_checking.assert_is_numpy_array_without_nan(
            three_predictor_matrices[i]
        )

        expected_dim = numpy.array(three_saliency_matrices[i].shape, dtype=int)
        error_checking.assert_is_numpy_array(
            three_saliency_matrices[i], exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array(
            three_input_grad_matrices[i], exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array(
            three_predictor_matrices[i], exact_dimensions=expected_dim
        )

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_boolean(use_pmm)

    if use_pmm:
        error_checking.assert_is_geq(pmm_max_percentile_level, 90.)
        error_checking.assert_is_leq(pmm_max_percentile_level, 100.)
    else:
        pmm_max_percentile_level = -1.

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(USE_PMM_KEY, int(use_pmm))
    dataset_object.setncattr(PMM_MAX_PERCENTILE_KEY, pmm_max_percentile_level)

    num_satellite_lag_times = None

    if three_saliency_matrices[0] is not None:
        num_grid_rows = three_saliency_matrices[0].shape[0]
        num_grid_columns = three_saliency_matrices[0].shape[1]
        num_satellite_lag_times = three_saliency_matrices[0].shape[2]
        num_gridded_satellite_channels = three_saliency_matrices[0].shape[3]

        dataset_object.createDimension(GRID_ROW_DIMENSION_KEY, num_grid_rows)
        dataset_object.createDimension(
            GRID_COLUMN_DIMENSION_KEY, num_grid_columns
        )
        dataset_object.createDimension(
            SATELLITE_LAG_TIME_KEY, num_satellite_lag_times
        )
        dataset_object.createDimension(
            GRIDDED_SATELLITE_CHANNEL_KEY, num_gridded_satellite_channels
        )

        these_dim = (
            GRID_ROW_DIMENSION_KEY, GRID_COLUMN_DIMENSION_KEY,
            SATELLITE_LAG_TIME_KEY, GRIDDED_SATELLITE_CHANNEL_KEY
        )
        dataset_object.createVariable(
            GRIDDED_SATELLITE_SALIENCY_KEY,
            datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[GRIDDED_SATELLITE_SALIENCY_KEY][:] = (
            three_saliency_matrices[0]
        )

        dataset_object.createVariable(
            GRIDDED_SATELLITE_INPUT_GRAD_KEY,
            datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[GRIDDED_SATELLITE_INPUT_GRAD_KEY][:] = (
            three_input_grad_matrices[0]
        )

        dataset_object.createVariable(
            GRIDDED_SATELLITE_PREDICTORS_KEY,
            datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[GRIDDED_SATELLITE_PREDICTORS_KEY][:] = (
            three_predictor_matrices[0]
        )

    if three_saliency_matrices[1] is not None:
        if num_satellite_lag_times is None:
            num_satellite_lag_times = three_saliency_matrices[1].shape[0]
            dataset_object.createDimension(
                SATELLITE_LAG_TIME_KEY, num_satellite_lag_times
            )
        else:
            assert (
                num_satellite_lag_times ==
                three_saliency_matrices[1].shape[0]
            )

        num_ungridded_satellite_channels = three_saliency_matrices[1].shape[1]
        dataset_object.createDimension(
            UNGRIDDED_SATELLITE_CHANNEL_KEY, num_ungridded_satellite_channels
        )

        these_dim = (SATELLITE_LAG_TIME_KEY, UNGRIDDED_SATELLITE_CHANNEL_KEY)
        dataset_object.createVariable(
            UNGRIDDED_SATELLITE_SALIENCY_KEY,
            datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[UNGRIDDED_SATELLITE_SALIENCY_KEY][:] = (
            three_saliency_matrices[1]
        )

        dataset_object.createVariable(
            UNGRIDDED_SATELLITE_INPUT_GRAD_KEY,
            datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[UNGRIDDED_SATELLITE_INPUT_GRAD_KEY][:] = (
            three_input_grad_matrices[1]
        )

        dataset_object.createVariable(
            UNGRIDDED_SATELLITE_PREDICTORS_KEY,
            datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[UNGRIDDED_SATELLITE_PREDICTORS_KEY][:] = (
            three_predictor_matrices[1]
        )

    if three_saliency_matrices[2] is not None:
        num_ships_channels = three_saliency_matrices[2].shape[0]
        dataset_object.createDimension(SHIPS_CHANNEL_KEY, num_ships_channels)

        these_dim = (SHIPS_CHANNEL_KEY,)
        dataset_object.createVariable(
            SHIPS_SALIENCY_KEY, datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[SHIPS_SALIENCY_KEY][:] = (
            three_saliency_matrices[2]
        )

        dataset_object.createVariable(
            SHIPS_INPUT_GRAD_KEY, datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[SHIPS_INPUT_GRAD_KEY][:] = (
            three_input_grad_matrices[2]
        )

        dataset_object.createVariable(
            SHIPS_PREDICTORS_KEY, datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[SHIPS_PREDICTORS_KEY][:] = (
            three_predictor_matrices[2]
        )

    dataset_object.close()


def read_composite_file(netcdf_file_name):
    """Reads composite saliency map from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: saliency_dict: Dictionary with the following keys.
    saliency_dict['three_saliency_matrices']: See doc for
        `write_composite_file`.
    saliency_dict['three_input_grad_matrices']: Same.
    saliency_dict['three_predictor_matrices']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['use_pmm']: Same.
    saliency_dict['pmm_max_percentile_level']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    three_saliency_matrices = []
    three_input_grad_matrices = []
    three_predictor_matrices = []

    if GRIDDED_SATELLITE_SALIENCY_KEY in dataset_object.variables:
        three_saliency_matrices.append(
            dataset_object.variables[GRIDDED_SATELLITE_SALIENCY_KEY][:]
        )
        three_input_grad_matrices.append(
            dataset_object.variables[GRIDDED_SATELLITE_INPUT_GRAD_KEY][:]
        )
        three_predictor_matrices.append(
            dataset_object.variables[GRIDDED_SATELLITE_PREDICTORS_KEY][:]
        )
    else:
        three_saliency_matrices.append(None)
        three_input_grad_matrices.append(None)
        three_predictor_matrices.append(None)

    if UNGRIDDED_SATELLITE_SALIENCY_KEY in dataset_object.variables:
        three_saliency_matrices.append(
            dataset_object.variables[UNGRIDDED_SATELLITE_SALIENCY_KEY][:]
        )
        three_input_grad_matrices.append(
            dataset_object.variables[UNGRIDDED_SATELLITE_INPUT_GRAD_KEY][:]
        )
        three_predictor_matrices.append(
            dataset_object.variables[UNGRIDDED_SATELLITE_PREDICTORS_KEY][:]
        )
    else:
        three_saliency_matrices.append(None)
        three_input_grad_matrices.append(None)
        three_predictor_matrices.append(None)

    if SHIPS_SALIENCY_KEY in dataset_object.variables:
        three_saliency_matrices.append(
            dataset_object.variables[SHIPS_SALIENCY_KEY][:]
        )
        three_input_grad_matrices.append(
            dataset_object.variables[SHIPS_INPUT_GRAD_KEY][:]
        )
        three_predictor_matrices.append(
            dataset_object.variables[SHIPS_PREDICTORS_KEY][:]
        )
    else:
        three_saliency_matrices.append(None)
        three_input_grad_matrices.append(None)
        three_predictor_matrices.append(None)

    saliency_dict = {
        THREE_SALIENCY_KEY: three_saliency_matrices,
        THREE_INPUT_GRAD_KEY: three_input_grad_matrices,
        THREE_PREDICTORS_KEY: three_predictor_matrices,
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        USE_PMM_KEY: bool(getattr(dataset_object, USE_PMM_KEY)),
        PMM_MAX_PERCENTILE_KEY:
            float(getattr(dataset_object, PMM_MAX_PERCENTILE_KEY))
    }

    dataset_object.close()
    return saliency_dict


def write_file(
        zarr_file_name, three_saliency_matrices, three_input_grad_matrices,
        cyclone_id_strings, init_times_unix_sec, model_file_name, layer_name,
        neuron_indices, ideal_activation):
    """Writes saliency maps to zarr file.

    E = number of examples

    :param zarr_file_name: Path to output file.
    :param three_saliency_matrices: length-3 list, where each element is either
        None or a numpy array of saliency values.  three_saliency_matrices[i]
        should have the same shape as the [i]th input tensor to the model.
        Also, the first axis of each numpy array must have length E.
    :param three_input_grad_matrices: Same as `three_saliency_matrices` but with
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

    error_checking.assert_is_list(three_saliency_matrices)
    error_checking.assert_is_list(three_input_grad_matrices)
    assert len(three_saliency_matrices) == 3
    assert len(three_input_grad_matrices) == 3

    num_examples = -1

    for i in range(len(three_saliency_matrices)):
        if three_saliency_matrices[i] is None:
            assert three_input_grad_matrices[i] is None
            continue

        error_checking.assert_is_numpy_array_without_nan(
            three_saliency_matrices[i]
        )
        error_checking.assert_is_numpy_array_without_nan(
            three_input_grad_matrices[i]
        )
        if i == 0:
            num_examples = three_saliency_matrices[i].shape[0]

        expected_dim = numpy.array(
            (num_examples,) + three_saliency_matrices[i].shape[1:], dtype=int
        )
        error_checking.assert_is_numpy_array(
            three_saliency_matrices[i], exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array(
            three_input_grad_matrices[i], exact_dimensions=expected_dim
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

    # Do actual stuff.
    metadata_dict = {
        EXAMPLE_DIMENSION_KEY: numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
    }

    main_data_dict = {}
    encoding_dict = {}
    num_satellite_lag_times = None

    if three_saliency_matrices[0] is not None:
        num_grid_rows = three_saliency_matrices[0].shape[1]
        num_grid_columns = three_saliency_matrices[0].shape[2]
        num_satellite_lag_times = three_saliency_matrices[0].shape[3]
        num_gridded_satellite_channels = three_saliency_matrices[0].shape[4]

        metadata_dict.update({
            GRID_ROW_DIMENSION_KEY: numpy.linspace(
                0, num_grid_rows - 1, num=num_grid_rows, dtype=int
            ),
            GRID_COLUMN_DIMENSION_KEY: numpy.linspace(
                0, num_grid_columns - 1, num=num_grid_columns, dtype=int
            ),
            SATELLITE_LAG_TIME_KEY: numpy.linspace(
                0, num_satellite_lag_times - 1,
                num=num_satellite_lag_times, dtype=int
            ),
            GRIDDED_SATELLITE_CHANNEL_KEY: numpy.linspace(
                0, num_gridded_satellite_channels - 1,
                num=num_gridded_satellite_channels, dtype=int
            )
        })

        these_dim = (
            EXAMPLE_DIMENSION_KEY, GRID_ROW_DIMENSION_KEY,
            GRID_COLUMN_DIMENSION_KEY, SATELLITE_LAG_TIME_KEY,
            GRIDDED_SATELLITE_CHANNEL_KEY
        )

        main_data_dict.update({
            GRIDDED_SATELLITE_SALIENCY_KEY: (
                these_dim, three_saliency_matrices[0]
            ),
            GRIDDED_SATELLITE_INPUT_GRAD_KEY: (
                these_dim, three_input_grad_matrices[0]
            )
        })

        encoding_dict.update({
            GRIDDED_SATELLITE_SALIENCY_KEY: {'dtype': 'float32'},
            GRIDDED_SATELLITE_INPUT_GRAD_KEY: {'dtype': 'float32'}
        })

    if three_saliency_matrices[1] is not None:
        if num_satellite_lag_times is None:
            num_satellite_lag_times = three_saliency_matrices[1].shape[1]

            metadata_dict.update({
                SATELLITE_LAG_TIME_KEY: numpy.linspace(
                    0, num_satellite_lag_times - 1,
                    num=num_satellite_lag_times, dtype=int
                )
            })
        else:
            assert (
                num_satellite_lag_times ==
                three_saliency_matrices[1].shape[1]
            )

        num_ungridded_satellite_channels = three_saliency_matrices[1].shape[2]
        metadata_dict.update({
            UNGRIDDED_SATELLITE_CHANNEL_KEY: numpy.linspace(
                0, num_ungridded_satellite_channels - 1,
                num=num_ungridded_satellite_channels, dtype=int
            )
        })

        these_dim = (
            EXAMPLE_DIMENSION_KEY, SATELLITE_LAG_TIME_KEY,
            UNGRIDDED_SATELLITE_CHANNEL_KEY
        )

        main_data_dict.update({
            UNGRIDDED_SATELLITE_SALIENCY_KEY: (
                these_dim, three_saliency_matrices[1]
            ),
            UNGRIDDED_SATELLITE_INPUT_GRAD_KEY: (
                these_dim, three_input_grad_matrices[1]
            )
        })

        encoding_dict.update({
            UNGRIDDED_SATELLITE_SALIENCY_KEY: {'dtype': 'float32'},
            UNGRIDDED_SATELLITE_INPUT_GRAD_KEY: {'dtype': 'float32'}
        })

    if three_saliency_matrices[2] is not None:
        num_ships_channels = three_saliency_matrices[2].shape[2]

        metadata_dict.update({
            SHIPS_CHANNEL_KEY: numpy.linspace(
                0, num_ships_channels - 1, num=num_ships_channels, dtype=int
            )
        })

        these_dim = (EXAMPLE_DIMENSION_KEY, SHIPS_CHANNEL_KEY)

        main_data_dict.update({
            SHIPS_SALIENCY_KEY: (these_dim, three_saliency_matrices[2]),
            SHIPS_INPUT_GRAD_KEY: (these_dim, three_input_grad_matrices[2])
        })
        encoding_dict.update({
            SHIPS_SALIENCY_KEY: {'dtype': 'float32'},
            SHIPS_INPUT_GRAD_KEY: {'dtype': 'float32'}
        })

    main_data_dict.update({
        CYCLONE_IDS_KEY: ((EXAMPLE_DIMENSION_KEY,), cyclone_id_strings),
        INIT_TIMES_KEY: ((EXAMPLE_DIMENSION_KEY,), init_times_unix_sec)
    })
    encoding_dict.update({
        INIT_TIMES_KEY: {'dtype': 'int32'}
    })

    attribute_dict = {
        MODEL_FILE_KEY: model_file_name,
        LAYER_NAME_KEY: layer_name,
        NEURON_INDICES_KEY: neuron_indices,
        IDEAL_ACTIVATION_KEY: ideal_activation
    }

    saliency_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict, attrs=attribute_dict
    )

    error_checking.assert_is_string(zarr_file_name)
    if os.path.isdir(zarr_file_name):
        shutil.rmtree(zarr_file_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=zarr_file_name
    )

    saliency_table_xarray.to_zarr(
        store=zarr_file_name, mode='w', encoding=encoding_dict
    )


def read_file(saliency_file_name):
    """Reads saliency maps from NetCDF or zarr file.

    :param saliency_file_name: Path to input file.
    :return: saliency_dict: Dictionary with the following keys.
    saliency_dict['three_saliency_matrices']: See doc for `write_file`.
    saliency_dict['three_input_grad_matrices']: Same.
    saliency_dict['cyclone_id_strings']: Same.
    saliency_dict['init_times_unix_sec']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['layer_name']: Same.
    saliency_dict['neuron_indices']: Same.
    saliency_dict['ideal_activation']: Same.
    """

    # TODO(thunderhoser): This is HACK to deal with change from NetCDF to zarr.
    if (
            saliency_file_name.endswith('.nc')
            and not os.path.isfile(saliency_file_name)
    ):
        saliency_file_name = '{0:s}.zarr'.format(saliency_file_name[:-3])

    if (
            saliency_file_name.endswith('.zarr')
            and not os.path.isdir(saliency_file_name)
    ):
        saliency_file_name = '{0:s}.nc'.format(saliency_file_name[:-5])

    if saliency_file_name.endswith('.zarr'):
        return _read_zarr_file(saliency_file_name)

    if not saliency_file_name.endswith('.nc'):
        return None

    saliency_dict = _read_netcdf_file(saliency_file_name)
    netcdf_file_name = copy.deepcopy(saliency_file_name)

    zarr_file_name = '{0:s}.zarr'.format(saliency_file_name[:-3])
    print('Writing saliency results to: "{0:s}"...'.format(zarr_file_name))

    write_file(
        zarr_file_name=zarr_file_name,
        three_saliency_matrices=saliency_dict[THREE_SALIENCY_KEY],
        three_input_grad_matrices=saliency_dict[THREE_INPUT_GRAD_KEY],
        cyclone_id_strings=saliency_dict[CYCLONE_IDS_KEY],
        init_times_unix_sec=saliency_dict[INIT_TIMES_KEY],
        model_file_name=saliency_dict[MODEL_FILE_KEY],
        layer_name=saliency_dict[LAYER_NAME_KEY],
        neuron_indices=saliency_dict[NEURON_INDICES_KEY],
        ideal_activation=saliency_dict[IDEAL_ACTIVATION_KEY]
    )

    os.remove(netcdf_file_name)

    return saliency_dict
