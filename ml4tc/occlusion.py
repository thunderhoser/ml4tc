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
SATELLITE_LAG_TIME_KEY = 'satellite_lag_time'
GRIDDED_SATELLITE_CHANNEL_KEY = 'gridded_satellite_channel'
UNGRIDDED_SATELLITE_CHANNEL_KEY = 'ungridded_satellite_channel'
SHIPS_LAG_TIME_KEY = 'ships_lag_time'
SHIPS_CHANNEL_KEY = 'ships_channel'

MODEL_FILE_KEY = 'model_file_name'
TARGET_CLASS_KEY = 'target_class'
HALF_WINDOW_SIZE_KEY = 'half_window_size_px'
STRIDE_LENGTH_KEY = 'stride_length_px'
FILL_VALUE_KEY = 'fill_value'

CYCLONE_IDS_KEY = 'cyclone_id_strings'
INIT_TIMES_KEY = 'init_times_unix_sec'
GRIDDED_SATELLITE_OCCLUSION_PROB_KEY = 'gridded_satellite_occlusion_prob_matrix'
UNGRIDDED_SATELLITE_OCCLUSION_PROB_KEY = (
    'ungridded_satellite_occlusion_prob_matrix'
)
SHIPS_OCCLUSION_PROB_KEY = 'ships_occlusion_prob_matrix'
THREE_OCCLUSION_PROB_KEY = 'three_occlusion_prob_matrices'
GRIDDED_SATELLITE_NORM_OCCLUSION_KEY = 'gridded_satellite_norm_occlusion_matrix'
UNGRIDDED_SATELLITE_NORM_OCCLUSION_KEY = (
    'ungridded_satellite_norm_occlusion_matrix'
)
SHIPS_NORM_OCCLUSION_KEY = 'ships_norm_occlusion_matrix'
THREE_NORM_OCCLUSION_KEY = 'three_norm_occlusion_matrices'


def get_occlusion_maps(
        model_object, predictor_matrices, target_class, half_window_size_px,
        stride_length_px, fill_value=0.):
    """Computes occlusion map for each example.

    E = number of examples
    T = number of input tensors to model

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrices: length-T list of numpy arrays, formatted in the
        same way as the training data.  The first axis (i.e., the example axis)
        of each numpy array should have length E.
    :param target_class: Occlusion maps will be created for this class.
        Must be an integer in 0...(K - 1), where K = number of classes.
    :param half_window_size_px: Half-size of occlusion window for gridded data
        (pixels).  If half-size is P, the full window will (2 * P + 1) rows by
        (2 * P + 1) columns.
    :param stride_length_px: Stride length for occlusion window for gridded data
        (pixels).
    :param fill_value: Fill value.  "Occlusion" will consist of replacing the
        original predictor value by this fill value, to simulate missing data.
    :return: occlusion_prob_matrices: length-T list of numpy arrays, containing
        predicted probabilities after occlusion.  occlusion_prob_matrices[i]
        has the same shape as predictor_matrices[i].
    :return: original_probs: length-E numpy array of predicted probabilities
        before occlusion.
    """

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
    num_lag_times = predictor_matrices[0].shape[3]

    num_grid_rows_occluded = int(numpy.ceil(
        float(num_grid_rows_orig) / stride_length_px
    ))
    num_grid_columns_occluded = int(numpy.ceil(
        float(num_grid_columns_orig) / stride_length_px
    ))

    dimensions = (
        num_examples, num_grid_rows_occluded, num_grid_columns_occluded,
        num_lag_times, 1
    )
    occlusion_prob_matrices = [numpy.full(dimensions, numpy.nan)]

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

            for k in range(num_lag_times):
                new_brightness_temp_matrix = predictor_matrices[0] + 0.
                new_brightness_temp_matrix[
                    :, first_row:last_row, first_column:last_column, k, 0
                ] = fill_value

                if len(predictor_matrices) > 1:
                    new_predictor_matrices = (
                        [new_brightness_temp_matrix] + predictor_matrices[1:]
                    )
                else:
                    new_predictor_matrices = [new_brightness_temp_matrix]

                this_prob_array = neural_net.apply_model(
                    model_object=model_object,
                    predictor_matrices=
                    [p for p in new_predictor_matrices if p is not None],
                    num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                    verbose=True
                )
                this_prob_array = numpy.squeeze(this_prob_array)

                if len(this_prob_array.shape) == 1:
                    error_checking.assert_is_leq(target_class, 1)

                    if target_class == 1:
                        occlusion_prob_matrices[0][:, i, j, k, 0] = (
                            this_prob_array
                        )
                    else:
                        occlusion_prob_matrices[0][:, i, j, k, 0] = (
                            1. - this_prob_array
                        )
                else:
                    num_classes = this_prob_array.shape[1]
                    error_checking.assert_is_less_than(
                        target_class, num_classes
                    )

                    occlusion_prob_matrices[0][:, i, j, k, 0] = (
                        this_prob_array[:, target_class]
                    )

    if stride_length_px > 1:
        first_prob_matrix_coarse = occlusion_prob_matrices[0] + 0.

        dimensions = (
            num_examples, num_grid_rows_orig, num_grid_columns_orig,
            num_lag_times, 1
        )
        occlusion_prob_matrices[0] = numpy.full(dimensions, numpy.nan)

        for i in range(num_examples):
            if numpy.mod(i, 100) == 0:
                print((
                    'Have upsampled {0:d} of {1:d} occlusion maps to predictor '
                    'resolution...'
                ).format(
                    i, num_examples
                ))

            for k in range(num_lag_times):
                occlusion_prob_matrices[0][i, ..., k, 0] = gradcam._upsample_cam(
                    class_activation_matrix=
                    first_prob_matrix_coarse[i, ..., k, 0],
                    new_dimensions=numpy.array(
                        [num_grid_rows_orig, num_grid_columns_orig], dtype=int
                    )
                )

        occlusion_prob_matrices[0] = numpy.maximum(
            occlusion_prob_matrices[0], 0.
        )
        occlusion_prob_matrices[0] = numpy.minimum(
            occlusion_prob_matrices[0], 1.
        )

        print((
            'Have upsampled all {0:d} occlusion maps to predictor resolution!'
        ).format(
            num_examples
        ))

        del first_prob_matrix_coarse

    for k in range(1, len(predictor_matrices)):
        if predictor_matrices[k] is None:
            occlusion_prob_matrices.append(None)
            continue

        num_scalar_predictors = numpy.prod(predictor_matrices[k].shape[1:])
        occlusion_prob_matrices.append(
            numpy.full((num_examples, num_scalar_predictors), numpy.nan)
        )

        for j in range(num_scalar_predictors):
            new_predictor_matrix = numpy.reshape(
                predictor_matrices[k] + 0.,
                (num_examples, num_scalar_predictors)
            )
            new_predictor_matrix[:, j] = fill_value
            new_predictor_matrix = numpy.reshape(
                new_predictor_matrix, predictor_matrices[k].shape
            )

            new_predictor_matrices = (
                predictor_matrices[:k] + [new_predictor_matrix] +
                predictor_matrices[(k + 1):]
            )

            this_prob_array = neural_net.apply_model(
                model_object=model_object,
                predictor_matrices=
                [p for p in new_predictor_matrices if p is not None],
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                verbose=True
            )
            this_prob_array = numpy.squeeze(this_prob_array)

            if len(this_prob_array.shape) == 1:
                error_checking.assert_is_leq(target_class, 1)

                if target_class == 1:
                    occlusion_prob_matrices[k][:, j] = this_prob_array
                else:
                    occlusion_prob_matrices[k][:, j] = 1. - this_prob_array
            else:
                num_classes = this_prob_array.shape[1]
                error_checking.assert_is_less_than(
                    target_class, num_classes
                )

                occlusion_prob_matrices[k][:, j] = (
                    this_prob_array[:, target_class]
                )

        occlusion_prob_matrices[k] = numpy.reshape(
            occlusion_prob_matrices[k], predictor_matrices[k].shape
        )

    original_prob_array = neural_net.apply_model(
        model_object=model_object,
        predictor_matrices=
        [p for p in predictor_matrices if p is not None],
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

    return occlusion_prob_matrices, original_probs


def normalize_occlusion_maps(occlusion_prob_matrices, original_probs):
    """Normalizes occlusion maps (scales to range -inf...1).

    :param occlusion_prob_matrices: See output doc for `get_occlusion_maps`.
    :param occlusion_prob_matrices: Same.
    :return: normalized_occlusion_matrices: List of numpy arrays with same shape
        as input, except that each value is now a normalized *decrease* in
        probability.  A value of 1 means that probability decreases all the way
        to zero; a value of 0 means that probability does not decrease at all; a
        value of -1 means that probability doubles; ...; etc.
    """

    error_checking.assert_is_list(occlusion_prob_matrices)
    num_examples = 0

    for this_matrix in occlusion_prob_matrices:
        if this_matrix is None:
            continue

    for this_matrix in occlusion_prob_matrices:
        if this_matrix is None:
            continue

        num_examples = this_matrix.shape[0]
        error_checking.assert_is_geq_numpy_array(this_matrix, 0.)
        error_checking.assert_is_leq_numpy_array(this_matrix, 1.)

    expected_dim = numpy.array([num_examples], dtype=int)
    error_checking.assert_is_numpy_array(
        original_probs, exact_dimensions=expected_dim
    )

    error_checking.assert_is_geq_numpy_array(original_probs, 0.)
    error_checking.assert_is_leq_numpy_array(original_probs, 1.)

    normalized_occlusion_matrices = [
        None if p is None else numpy.full(p.shape, numpy.nan)
        for p in occlusion_prob_matrices
    ]

    original_probs_with_nan = original_probs + 0.
    original_probs_with_nan[original_probs_with_nan == 0] = numpy.nan

    for k in range(len(occlusion_prob_matrices)):
        if occlusion_prob_matrices[k] is None:
            continue

        for i in range(num_examples):
            normalized_occlusion_matrices[k][i, ...] = (
                original_probs_with_nan[i] - occlusion_prob_matrices[k][i, ...]
            ) / original_probs_with_nan[i]

        normalized_occlusion_matrices[k][
            numpy.isnan(normalized_occlusion_matrices[k])
        ] = 0.

    return normalized_occlusion_matrices


def write_file(
        netcdf_file_name, three_occlusion_prob_matrices,
        three_norm_occlusion_matrices, cyclone_id_strings, init_times_unix_sec,
        model_file_name, target_class, half_window_size_px, stride_length_px,
        fill_value):
    """Writes occlusion maps to NetCDF file.

    E = number of examples

    :param netcdf_file_name: Path to output file.
    :param three_occlusion_prob_matrices: length-3 list, where each element is
        either None or a numpy array of post-occlusion probabilities.
        three_occlusion_prob_matrices[i] should have the same shape as the [i]th
        input tensor to the model.  Also, the first axis of each numpy array
        must have length E.
    :param three_norm_occlusion_matrices: Same as
        `three_occlusion_prob_matrices` but with normalized probabilities
        instead.
    :param cyclone_id_strings: length-E list of cyclone IDs.
    :param init_times_unix_sec: length-E numpy array of forecast-init times.
    :param model_file_name: Path to file with neural net used to create
        occlusion maps (readable by `neural_net.read_model`).
    :param target_class: See doc for `get_occlusion_maps`.
    :param half_window_size_px: Same.
    :param stride_length_px: Same.
    :param fill_value: Same.
    """

    # Check input args.
    error_checking.assert_is_list(three_occlusion_prob_matrices)
    error_checking.assert_is_list(three_norm_occlusion_matrices)
    assert len(three_occlusion_prob_matrices) == 3
    assert len(three_norm_occlusion_matrices) == 3

    num_examples = -1

    for i in range(len(three_occlusion_prob_matrices)):
        if three_occlusion_prob_matrices[i] is None:
            assert three_norm_occlusion_matrices[i] is None
            continue

        error_checking.assert_is_geq_numpy_array(
            three_occlusion_prob_matrices[i], 0.
        )
        error_checking.assert_is_leq_numpy_array(
            three_occlusion_prob_matrices[i], 1.
        )
        error_checking.assert_is_leq_numpy_array(
            three_norm_occlusion_matrices[i], 1.
        )

        if i == 0:
            num_examples = three_occlusion_prob_matrices[i].shape[0]

        expected_dim = numpy.array(
            (num_examples,) + three_occlusion_prob_matrices[i].shape[1:],
            dtype=int
        )
        error_checking.assert_is_numpy_array(
            three_occlusion_prob_matrices[i], exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array(
            three_norm_occlusion_matrices[i], exact_dimensions=expected_dim
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

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    num_satellite_lag_times = None

    if three_occlusion_prob_matrices[0] is not None:
        num_grid_rows = three_occlusion_prob_matrices[0].shape[1]
        num_grid_columns = three_occlusion_prob_matrices[0].shape[2]
        num_satellite_lag_times = three_occlusion_prob_matrices[0].shape[3]
        num_gridded_satellite_channels = (
            three_occlusion_prob_matrices[0].shape[4]
        )

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
            EXAMPLE_DIMENSION_KEY, GRID_ROW_DIMENSION_KEY,
            GRID_COLUMN_DIMENSION_KEY, SATELLITE_LAG_TIME_KEY,
            GRIDDED_SATELLITE_CHANNEL_KEY
        )
        dataset_object.createVariable(
            GRIDDED_SATELLITE_OCCLUSION_PROB_KEY,
            datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[GRIDDED_SATELLITE_OCCLUSION_PROB_KEY][:] = (
            three_occlusion_prob_matrices[0]
        )

        dataset_object.createVariable(
            GRIDDED_SATELLITE_NORM_OCCLUSION_KEY,
            datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[GRIDDED_SATELLITE_NORM_OCCLUSION_KEY][:] = (
            three_norm_occlusion_matrices[0]
        )

    if three_occlusion_prob_matrices[1] is not None:
        if num_satellite_lag_times is None:
            num_satellite_lag_times = three_occlusion_prob_matrices[1].shape[1]
            dataset_object.createDimension(
                SATELLITE_LAG_TIME_KEY, num_satellite_lag_times
            )
        else:
            assert (
                num_satellite_lag_times ==
                three_occlusion_prob_matrices[1].shape[1]
            )

        num_ungridded_satellite_channels = (
            three_occlusion_prob_matrices[1].shape[2]
        )
        dataset_object.createDimension(
            UNGRIDDED_SATELLITE_CHANNEL_KEY, num_ungridded_satellite_channels
        )

        these_dim = (
            EXAMPLE_DIMENSION_KEY, SATELLITE_LAG_TIME_KEY,
            UNGRIDDED_SATELLITE_CHANNEL_KEY
        )
        dataset_object.createVariable(
            UNGRIDDED_SATELLITE_OCCLUSION_PROB_KEY,
            datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[UNGRIDDED_SATELLITE_OCCLUSION_PROB_KEY][:] = (
            three_occlusion_prob_matrices[1]
        )

        dataset_object.createVariable(
            UNGRIDDED_SATELLITE_NORM_OCCLUSION_KEY,
            datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[UNGRIDDED_SATELLITE_NORM_OCCLUSION_KEY][:] = (
            three_norm_occlusion_matrices[1]
        )

    if three_occlusion_prob_matrices[2] is not None:
        num_ships_lag_times = three_occlusion_prob_matrices[2].shape[1]
        num_ships_channels = three_occlusion_prob_matrices[2].shape[2]
        dataset_object.createDimension(SHIPS_LAG_TIME_KEY, num_ships_lag_times)
        dataset_object.createDimension(SHIPS_CHANNEL_KEY, num_ships_channels)

        these_dim = (
            EXAMPLE_DIMENSION_KEY, SHIPS_LAG_TIME_KEY, SHIPS_CHANNEL_KEY
        )
        dataset_object.createVariable(
            SHIPS_OCCLUSION_PROB_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[SHIPS_OCCLUSION_PROB_KEY][:] = (
            three_occlusion_prob_matrices[2]
        )

        dataset_object.createVariable(
            SHIPS_NORM_OCCLUSION_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[SHIPS_NORM_OCCLUSION_KEY][:] = (
            three_norm_occlusion_matrices[2]
        )

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

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads occlusion maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: occlusion_dict: Dictionary with the following keys.
    occlusion_dict['three_occlusion_prob_matrices']: See doc for `write_file`.
    occlusion_dict['three_norm_occlusion_matrices']: Same.
    occlusion_dict['cyclone_id_strings']: Same.
    occlusion_dict['init_times_unix_sec']: Same.
    occlusion_dict['model_file_name']: Same.
    occlusion_dict['target_class']: Same.
    occlusion_dict['half_window_size_px']: Same.
    occlusion_dict['stride_length_px']: Same.
    occlusion_dict['fill_value']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    three_occlusion_prob_matrices = []
    three_norm_occlusion_matrices = []

    if GRIDDED_SATELLITE_OCCLUSION_PROB_KEY in dataset_object.variables:
        three_occlusion_prob_matrices.append(
            dataset_object.variables[GRIDDED_SATELLITE_OCCLUSION_PROB_KEY][:]
        )
        three_norm_occlusion_matrices.append(
            dataset_object.variables[GRIDDED_SATELLITE_NORM_OCCLUSION_KEY][:]
        )
    else:
        three_occlusion_prob_matrices.append(None)
        three_norm_occlusion_matrices.append(None)

    if UNGRIDDED_SATELLITE_OCCLUSION_PROB_KEY in dataset_object.variables:
        three_occlusion_prob_matrices.append(
            dataset_object.variables[UNGRIDDED_SATELLITE_OCCLUSION_PROB_KEY][:]
        )
        three_norm_occlusion_matrices.append(
            dataset_object.variables[UNGRIDDED_SATELLITE_NORM_OCCLUSION_KEY][:]
        )
    else:
        three_occlusion_prob_matrices.append(None)
        three_norm_occlusion_matrices.append(None)

    if SHIPS_OCCLUSION_PROB_KEY in dataset_object.variables:
        three_occlusion_prob_matrices.append(
            dataset_object.variables[SHIPS_OCCLUSION_PROB_KEY][:]
        )
        three_norm_occlusion_matrices.append(
            dataset_object.variables[SHIPS_NORM_OCCLUSION_KEY][:]
        )
    else:
        three_occlusion_prob_matrices.append(None)
        three_norm_occlusion_matrices.append(None)

    occlusion_dict = {
        THREE_OCCLUSION_PROB_KEY: three_occlusion_prob_matrices,
        THREE_NORM_OCCLUSION_KEY: three_norm_occlusion_matrices,
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
