"""Methods for computing, reading, and writing Grad-CAM.

Grad-CAM = gradient-weighted class-activation maps
"""

import numpy
import netCDF4
from keras import backend as K
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator
)
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

EXAMPLE_DIMENSION_KEY = 'example'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_char'
GRID_ROW_DIMENSION_KEY = 'grid_row'
GRID_COLUMN_DIMENSION_KEY = 'grid_column'

MODEL_FILE_KEY = 'model_file_name'
TARGET_CLASS_KEY = 'target_class'
TARGET_LAYER_KEY = 'target_layer_name'

CYCLONE_IDS_KEY = 'cyclone_id_strings'
INIT_TIMES_KEY = 'init_times_unix_sec'
CLASS_ACTIVATION_KEY = 'class_activation_matrix'


def _normalize_tensor(input_tensor):
    """Divides tensor through by its Euclidean norm.

    :param input_tensor: Unnormalized tensor.
    :return: output_tensor: Normalized tensor.
    """

    rms_tensor = K.sqrt(K.mean(K.square(input_tensor)))
    return input_tensor / (rms_tensor + K.epsilon())


def _upsample_cam(class_activation_matrix, new_dimensions):
    """Upsamples class-activation map to higher spatial resolution.

    CAM may be 1-D, 2-D, or 3-D.

    :param class_activation_matrix: numpy array containing 1-D, 2-D, or 3-D
        class-activation map.
    :param new_dimensions: numpy array of new dimensions.  If matrix is
        {1D, 2D, 3D}, this must be a length-{1, 2, 3} array, respectively.
    :return: class_activation_matrix: Upsampled version of input.
    """

    num_rows_new = new_dimensions[0]
    row_indices_new = numpy.linspace(
        1, num_rows_new, num=num_rows_new, dtype=float
    )
    row_indices_orig = numpy.linspace(
        1, num_rows_new, num=class_activation_matrix.shape[0], dtype=float
    )

    if len(new_dimensions) == 1:
        # interp_object = UnivariateSpline(
        #     x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
        #     k=1, s=0
        # )

        interp_object = UnivariateSpline(
            x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
            k=3, s=0
        )

        return interp_object(row_indices_new)

    num_columns_new = new_dimensions[1]
    column_indices_new = numpy.linspace(
        1, num_columns_new, num=num_columns_new, dtype=float
    )
    column_indices_orig = numpy.linspace(
        1, num_columns_new, num=class_activation_matrix.shape[1], dtype=float
    )

    if len(new_dimensions) == 2:
        interp_object = RectBivariateSpline(
            x=row_indices_orig, y=column_indices_orig,
            z=class_activation_matrix, kx=3, ky=3, s=0
        )

        return interp_object(x=row_indices_new, y=column_indices_new, grid=True)

    num_heights_new = new_dimensions[2]
    height_indices_new = numpy.linspace(
        1, num_heights_new, num=num_heights_new, dtype=float
    )
    height_indices_orig = numpy.linspace(
        1, num_heights_new, num=class_activation_matrix.shape[2], dtype=float
    )

    interp_object = RegularGridInterpolator(
        points=(row_indices_orig, column_indices_orig, height_indices_orig),
        values=class_activation_matrix, method='linear'
    )

    column_index_matrix, row_index_matrix, height_index_matrix = (
        numpy.meshgrid(column_indices_new, row_indices_new, height_indices_new)
    )
    query_point_matrix = numpy.stack(
        (row_index_matrix, column_index_matrix, height_index_matrix), axis=-1
    )

    return interp_object(query_point_matrix)


def run_gradcam(model_object, predictor_matrices_one_example, target_class,
                target_layer_name):
    """Runs the Grad-CAM algorithm.

    T = number of input tensors to model
    M = number of rows in brightness-temperature grid
    N = number of columns in brightness-temperature grid

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrices_one_example: length-T list of numpy arrays,
        formatted in the same way as the training data.  The first axis (i.e.,
        the example axis) of each numpy array should have length 1.
    :param target_class: Class-activation maps will be created for this class.
        Must be an integer in 0...(K - 1), where K = number of classes.
    :param target_layer_name: Name of target layer.  Neuron-importance weights
        will be based on activations in this layer.  Layer must have spatial
        outputs.
    :return: class_activation_matrix: M-by-N numpy array of class activations.
    """

    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_geq(target_class, 0)
    error_checking.assert_is_string(target_layer_name)

    # Set up loss function.
    output_layer_object = model_object.layers[-1].output
    num_output_neurons = output_layer_object.get_shape().as_list()[-1]

    if num_output_neurons == 1:
        error_checking.assert_is_leq(target_class, 1)

        if target_class == 1:
            # loss_tensor = model_object.layers[-1].output[..., 0]
            loss_tensor = model_object.layers[-1].input[..., 0]
        else:
            # loss_tensor = -1 * model_object.layers[-1].output[..., 0]
            loss_tensor = -1 * model_object.layers[-1].input[..., 0]
    else:
        error_checking.assert_is_less_than(target_class, num_output_neurons)
        # loss_tensor = model_object.layers[-1].output[..., target_class]
        loss_tensor = model_object.layers[-1].input[..., target_class]

    # Set up gradient function.
    target_layer_activation_tensor = model_object.get_layer(
        name=target_layer_name
    ).output

    gradient_tensor = K.gradients(
        loss_tensor, [target_layer_activation_tensor]
    )[0]
    gradient_tensor = _normalize_tensor(gradient_tensor)
    gradient_function = K.function(
        model_object.input,
        [target_layer_activation_tensor, gradient_tensor]
    )

    # Evaluate gradient function.
    target_layer_activation_matrix, gradient_matrix = gradient_function(
        predictor_matrices_one_example
    )
    target_layer_activation_matrix = target_layer_activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    # Compute class-activation map.
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=(0, 1))
    class_activation_matrix = numpy.full(
        target_layer_activation_matrix.shape[:-1], 0.
    )
    num_filters = len(mean_weight_by_filter)

    for k in range(num_filters):
        class_activation_matrix += (
            mean_weight_by_filter[k] * target_layer_activation_matrix[..., k]
        )

    class_activation_matrix = _upsample_cam(
        class_activation_matrix=class_activation_matrix,
        new_dimensions=numpy.array(
            predictor_matrices_one_example[0].shape[1:3], dtype=int
        )
    )

    return numpy.maximum(class_activation_matrix, 0.)


def write_file(
        netcdf_file_name, class_activation_matrix, cyclone_id_strings,
        init_times_unix_sec, model_file_name, target_class, target_layer_name):
    """Writes class-activation maps to NetCDF file.

    E = number of examples
    M = number of rows in brightness-temperature grid
    N = number of columns in brightness-temperature grid

    :param netcdf_file_name: Path to output file.
    :param class_activation_matrix: E-by-M-by-N numpy array of class
        activations.
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param init_times_unix_sec: length-E numpy array of forecast-init times.
    :param model_file_name: Path to file with neural net used to create saliency
        maps (readable by `neural_net.read_model`).
    :param target_class: See doc for `run_gradcam`.
    :param target_layer_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq_numpy_array(class_activation_matrix, 0.)
    error_checking.assert_is_numpy_array(
        class_activation_matrix, num_dimensions=3
    )

    num_examples = class_activation_matrix.shape[0]
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
    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_geq(target_class, 0)
    error_checking.assert_is_string(target_layer_name)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(TARGET_CLASS_KEY, target_class)
    dataset_object.setncattr(TARGET_LAYER_KEY, target_layer_name)

    num_grid_rows = class_activation_matrix.shape[1]
    num_grid_columns = class_activation_matrix.shape[2]
    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(GRID_ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(GRID_COLUMN_DIMENSION_KEY, num_grid_columns)

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
        GRID_COLUMN_DIMENSION_KEY
    )
    dataset_object.createVariable(
        CLASS_ACTIVATION_KEY, datatype=numpy.float16, dimensions=these_dim
    )
    dataset_object.variables[CLASS_ACTIVATION_KEY][:] = (
        class_activation_matrix
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads class-activation maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: gradcam_dict: Dictionary with the following keys.
    gradcam_dict['class_activation_matrix']: See doc for `write_file`.
    gradcam_dict['cyclone_id_strings']: Same.
    gradcam_dict['init_times_unix_sec']: Same.
    gradcam_dict['model_file_name']: Same.
    gradcam_dict['target_class']: Same.
    gradcam_dict['target_layer_name']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    gradcam_dict = {
        CLASS_ACTIVATION_KEY: dataset_object.variables[CLASS_ACTIVATION_KEY][:],
        CYCLONE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[CYCLONE_IDS_KEY][:])
        ],
        INIT_TIMES_KEY: dataset_object.variables[INIT_TIMES_KEY][:],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        TARGET_CLASS_KEY: int(getattr(dataset_object, TARGET_CLASS_KEY)),
        TARGET_LAYER_KEY: str(getattr(dataset_object, TARGET_LAYER_KEY))
    }

    dataset_object.close()
    return gradcam_dict
