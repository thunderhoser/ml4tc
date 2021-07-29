"""Methods for computing, reading, and writing occlusion maps."""

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
import neural_net
import gradcam

NUM_EXAMPLES_PER_BATCH = 32

EXAMPLE_DIMENSION_KEY = 'example'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_char'
GRID_ROW_DIMENSION_KEY = 'grid_row'
GRID_COLUMN_DIMENSION_KEY = 'grid_column'

MODEL_FILE_KEY = 'model_file_name'
TARGET_CLASS_KEY = 'target_class'
HALF_WINDOW_SIZE_KEY = 'half_window_size_px'
STRIDE_LENGTH_KEY = 'stride_length_px'
FILL_VALUE_KEY = 'fill_value'

CYCLONE_IDS_KEY = 'cyclone_id_strings'
INIT_TIMES_KEY = 'init_times_unix_sec'
OCCLUSION_PROBS_KEY = 'occlusion_prob_matrix'
NORMALIZED_OCCLUSION_KEY = 'normalized_occlusion_matrix'


def get_occlusion_maps(
        model_object, predictor_matrices, target_class, half_window_size_px,
        stride_length_px, fill_value=0.):
    """Computes occlusion map for each example.

    E = number of examples
    T = number of input tensors to model
    M = number of rows in brightness-temperature grid
    N = number of columns in brightness-temperature grid

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrices: length-T list of numpy arrays, formatted in the
        same way as the training data.  The first axis (i.e., the example axis)
        of each numpy array should have length E.
    :param target_class: Occlusion maps will be created for this class.
        Must be an integer in 0...(K - 1), where K = number of classes.
    :param half_window_size_px: Half-size of occlusion window (pixels).  If
        half-size is P, the full window will (2 * P + 1) rows by (2 * P + 1)
        columns.
    :param stride_length_px: Stride length for occlusion window (pixels).
    :param fill_value: Fill value.  Inside the occlusion window, all brightness
        temperatures will be assigned this value, to simulate missing data.
    :return: occlusion_prob_matrix: E-by-M-by-N numpy array of predicted
        probabilities after occlusion.
    :return: original_probs: length-E numpy array of predicted probabilities
        before occlusion.
    """

    error_checking.assert_is_numpy_array(
        predictor_matrices[0], num_dimensions=5
    )
    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_geq(target_class, 0)
    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_integer(stride_length_px)
    error_checking.assert_is_geq(stride_length_px, 1)
    error_checking.assert_is_not_nan(fill_value)

    num_examples = predictor_matrices[0].shape[0]
    num_grid_rows_orig = predictor_matrices[0].shape[1]
    num_grid_columns_orig = predictor_matrices[0].shape[2]
    num_grid_rows_occluded = int(numpy.ceil(
        float(num_grid_rows_orig) / stride_length_px
    ))
    num_grid_columns_occluded = int(numpy.ceil(
        float(num_grid_columns_orig) / stride_length_px
    ))

    dimensions = (
        num_examples, num_grid_rows_occluded, num_grid_columns_occluded
    )
    occlusion_prob_matrix = numpy.full(dimensions, numpy.nan)

    for i in range(num_grid_rows_occluded):
        orig_row_index = min([
            i * stride_length_px,
            num_grid_rows_orig - 1
        ])

        print('Occluding windows centered on row {0:d} of {1:d}...'.format(
            orig_row_index, num_grid_rows_orig
        ))

        for j in range(num_grid_columns_occluded):
            orig_column_index = min([
                j * stride_length_px,
                num_grid_columns_orig - 1
            ])

            first_row = max([orig_row_index - half_window_size_px, 0])
            last_row = min([
                orig_row_index + half_window_size_px + 1,
                num_grid_rows_orig
            ])

            first_column = max([orig_column_index - half_window_size_px, 0])
            last_column = min([
                orig_column_index + half_window_size_px + 1,
                num_grid_columns_orig
            ])

            new_brightness_temp_matrix = predictor_matrices[0] + 0.
            new_brightness_temp_matrix[
                :, first_row:last_row, first_column:last_column, ...
            ] = fill_value

            this_prob_array = neural_net.apply_model(
                model_object=model_object,
                predictor_matrices=
                [new_brightness_temp_matrix] + predictor_matrices[1:],
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                verbose=True
            )
            this_prob_array = numpy.squeeze(this_prob_array)

            if len(this_prob_array.shape) == 1:
                error_checking.assert_is_leq(target_class, 1)

                if target_class == 1:
                    occlusion_prob_matrix[:, i, j] = this_prob_array
                else:
                    occlusion_prob_matrix[:, i, j] = 1. - this_prob_array
            else:
                num_classes = this_prob_array.shape[1]
                error_checking.assert_is_less_than(target_class, num_classes)

                occlusion_prob_matrix[:, i, j] = (
                    this_prob_array[:, target_class]
                )

    if stride_length_px > 1:
        occlusion_prob_matrix_coarse = occlusion_prob_matrix + 0.
        dimensions = (num_examples, num_grid_rows_orig, num_grid_columns_orig)
        occlusion_prob_matrix = numpy.full(dimensions, numpy.nan)

        for i in range(num_examples):
            if numpy.mod(i, 100) == 0:
                print((
                    'Have upsampled {0:d} of {1:d} occlusion maps to predictor '
                    'resolution...'
                ).format(
                    i, num_examples
                ))

            occlusion_prob_matrix[i, ...] = gradcam._upsample_cam(
                class_activation_matrix=occlusion_prob_matrix_coarse[i, ...],
                new_dimensions=numpy.array(
                    [num_grid_rows_orig, num_grid_columns_orig], dtype=int
                )
            )

        occlusion_prob_matrix = numpy.maximum(occlusion_prob_matrix, 0.)
        occlusion_prob_matrix = numpy.minimum(occlusion_prob_matrix, 1.)

        print((
            'Have upsampled all {0:d} occlusion maps to predictor resolution!'
        ).format(
            num_examples
        ))

        del occlusion_prob_matrix_coarse

    original_prob_array = neural_net.apply_model(
        model_object=model_object,
        predictor_matrices=predictor_matrices,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        verbose=True
    )

    if len(original_prob_array.shape) == 1:
        error_checking.assert_is_leq(target_class, 1)

        if target_class == 1:
            original_probs = original_prob_array
        else:
            original_probs = 1. - original_prob_array
    else:
        original_probs = original_prob_array[:, target_class]

    return occlusion_prob_matrix, original_probs


def normalize_occlusion_maps(occlusion_prob_matrix, original_probs):
    """Normalizes occlusion maps (scales to range -inf...1).

    :param occlusion_prob_matrix: See output doc for `get_occlusion_maps`.
    :param original_probs: Same.
    :return: normalized_occlusion_matrix: numpy array with same shape as input,
        except that each value is now a normalized *decrease* in probability.
        A value of 1 means that probability decreases all the way zero; a value
        of 0 means that probability does not decrease at all; a value of -1
        means that probability doubles; ...; etc.
    """

    error_checking.assert_is_numpy_array(
        occlusion_prob_matrix, num_dimensions=3
    )
    error_checking.assert_is_geq_numpy_array(occlusion_prob_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(occlusion_prob_matrix, 1.)

    num_examples = occlusion_prob_matrix.shape[0]
    expected_dim = numpy.array([num_examples], dtype=int)
    error_checking.assert_is_numpy_array(
        original_probs, exact_dimensions=expected_dim
    )

    error_checking.assert_is_geq_numpy_array(original_probs, 0.)
    error_checking.assert_is_leq_numpy_array(original_probs, 1.)

    normalized_occlusion_matrix = numpy.full(
        occlusion_prob_matrix.shape, numpy.nan
    )

    original_probs_with_nan = original_probs + 0.
    original_probs_with_nan[original_probs_with_nan == 0] = numpy.nan

    for i in range(num_examples):
        normalized_occlusion_matrix[i, ...] = (
            (original_probs_with_nan[i] - occlusion_prob_matrix[i, ...]) /
            original_probs_with_nan[i]
        )

    normalized_occlusion_matrix[numpy.isnan(normalized_occlusion_matrix)] = 0.
    return normalized_occlusion_matrix


def write_file(
        netcdf_file_name, occlusion_prob_matrix, normalized_occlusion_matrix,
        cyclone_id_strings, init_times_unix_sec, model_file_name, target_class,
        half_window_size_px, stride_length_px, fill_value):
    """Writes occlusion maps to NetCDF file.

    E = number of examples
    M = number of rows in brightness-temperature grid
    N = number of columns in brightness-temperature grid

    :param netcdf_file_name: Path to output file.
    :param occlusion_prob_matrix: E-by-M-by-N numpy array of predicted
        probabilities after occlusion.
    :param normalized_occlusion_matrix: E-by-M-by-N numpy array of normalized
        *decreases* in probability.  For more details, see output doc for
        `normalize_occlusion_maps`.
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param init_times_unix_sec: length-E numpy array of forecast-init times.
    :param model_file_name: Path to file with neural net used to create saliency
        maps (readable by `neural_net.read_model`).
    :param target_class: See doc for `get_occlusion_maps`.
    :param half_window_size_px: Same.
    :param stride_length_px: Same.
    :param fill_value: Same.
    """

    # Check input args.
    error_checking.assert_is_geq_numpy_array(occlusion_prob_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(occlusion_prob_matrix, 1.)
    error_checking.assert_is_numpy_array(
        occlusion_prob_matrix, num_dimensions=3
    )

    error_checking.assert_is_leq_numpy_array(normalized_occlusion_matrix, 1.)
    error_checking.assert_is_numpy_array(
        normalized_occlusion_matrix,
        exact_dimensions=numpy.array(occlusion_prob_matrix.shape, dtype=int)
    )

    num_examples = occlusion_prob_matrix.shape[0]
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
    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_integer(stride_length_px)
    error_checking.assert_is_geq(stride_length_px, 1)
    error_checking.assert_is_not_nan(fill_value)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(TARGET_CLASS_KEY, target_class)
    dataset_object.setncattr(HALF_WINDOW_SIZE_KEY, half_window_size_px)
    dataset_object.setncattr(STRIDE_LENGTH_KEY, stride_length_px)
    dataset_object.setncattr(FILL_VALUE_KEY, fill_value)

    num_grid_rows = occlusion_prob_matrix.shape[1]
    num_grid_columns = occlusion_prob_matrix.shape[2]
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
        OCCLUSION_PROBS_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[OCCLUSION_PROBS_KEY][:] = occlusion_prob_matrix

    dataset_object.createVariable(
        NORMALIZED_OCCLUSION_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[NORMALIZED_OCCLUSION_KEY][:] = (
        normalized_occlusion_matrix
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads occlusion maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: occlusion_dict: Dictionary with the following keys.
    occlusion_dict['occlusion_prob_matrix']: See doc for `write_file`.
    occlusion_dict['normalized_occlusion_matrix']: Same.
    occlusion_dict['cyclone_id_strings']: Same.
    occlusion_dict['init_times_unix_sec']: Same.
    occlusion_dict['model_file_name']: Same.
    occlusion_dict['target_class']: Same.
    occlusion_dict['half_window_size_px']: Same.
    occlusion_dict['stride_length_px']: Same.
    occlusion_dict['fill_value']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    occlusion_dict = {
        OCCLUSION_PROBS_KEY: dataset_object.variables[OCCLUSION_PROBS_KEY][:],
        NORMALIZED_OCCLUSION_KEY:
            dataset_object.variables[NORMALIZED_OCCLUSION_KEY][:],
        CYCLONE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[CYCLONE_IDS_KEY][:])
        ],
        INIT_TIMES_KEY: dataset_object.variables[INIT_TIMES_KEY][:],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        TARGET_CLASS_KEY: int(getattr(dataset_object, TARGET_CLASS_KEY)),
        HALF_WINDOW_SIZE_KEY:
            int(getattr(dataset_object, HALF_WINDOW_SIZE_KEY)),
        STRIDE_LENGTH_KEY: int(getattr(dataset_object, STRIDE_LENGTH_KEY)),
        FILL_VALUE_KEY: float(getattr(dataset_object, FILL_VALUE_KEY))
    }

    dataset_object.close()
    return occlusion_dict
