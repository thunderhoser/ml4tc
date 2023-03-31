"""Splits predictions by cyclone intensity (max sustained surface wind)."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import model_activation as gg_model_activation
from ml4tc.io import ships_io
from ml4tc.io import prediction_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TOLERANCE = 1e-6
MAX_INTENSITY_CUTOFF_M_S01 = 100.

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
SHIPS_DIR_ARG_NAME = 'input_ships_dir_name'
INTENSITY_CUTOFFS_ARG_NAME = 'intensity_cutoffs_m_s01'
SUBSET_SIZE_ARG_NAME = 'num_examples_per_subset'
SUBSET_NAMES_ARG_NAME = 'subset_names'
UNIQUE_CYCLONES_ARG_NAME = 'enforce_unique_cyclones'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to input file, containing predictions and targets (correct answers) '
    'for many examples.  Will be read by `prediction_io.read_file`.'
)
SHIPS_DIR_HELP_STRING = (
    'Name of SHIPS directory, containing SHIPS files with (among other things) '
    'cyclone intensity.  Files therein will be found by `ships_io.find_file` '
    'and read by `ships_io.read_file`.'
)
INTENSITY_CUTOFFS_HELP_STRING = (
    'List of cutoffs for intensity categories.  The lowest cutoff will always '
    'be 0 m/s, and the highest will be infinity m/s, so don''t bother '
    'including these.'
)
SUBSET_SIZE_HELP_STRING = (
    'Number of examples to keep for each intensity category.'
)
SUBSET_NAMES_HELP_STRING = (
    'Space-separated list of subset names.  There should be N + 1 items in '
    'this list, where N = length of `{0:s}`.'
).format(INTENSITY_CUTOFFS_ARG_NAME)

UNIQUE_CYCLONES_HELP_STRING = (
    'Boolean flag.  If 1, will ensure that each subset contains only unique '
    'cyclones.  If 0, will allow non-unique cyclones (i.e., multiple time '
    'steps from the same cyclone) in the same subset.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  For each subset, one file will be written to '
    'this directory by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHIPS_DIR_ARG_NAME, type=str, required=True,
    help=SHIPS_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INTENSITY_CUTOFFS_ARG_NAME, type=float, nargs='+', required=True,
    help=INTENSITY_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SUBSET_SIZE_ARG_NAME, type=int, required=True,
    help=SUBSET_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SUBSET_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=SUBSET_NAMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + UNIQUE_CYCLONES_ARG_NAME, type=int, required=True,
    help=UNIQUE_CYCLONES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_prediction_file_name, ships_dir_name, intensity_cutoffs_m_s01,
         num_examples_per_subset, subset_names, enforce_unique_cyclones,
         output_dir_name):
    """Splits predictions by cyclone intensity (max sustained surface wind).

    This is effectively the same method.

    :param input_prediction_file_name: See documentation at top of file.
    :param ships_dir_name: Same.
    :param intensity_cutoffs_m_s01: Same.
    :param num_examples_per_subset: Same.
    :param subset_names: Same.
    :param enforce_unique_cyclones: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_less_than_numpy_array(
        intensity_cutoffs_m_s01, MAX_INTENSITY_CUTOFF_M_S01
    )
    error_checking.assert_is_greater_numpy_array(
        intensity_cutoffs_m_s01, 2 * TOLERANCE
    )
    intensity_cutoffs_m_s01 = number_rounding.round_to_nearest(
        intensity_cutoffs_m_s01, TOLERANCE
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(intensity_cutoffs_m_s01), TOLERANCE
    )

    intensity_cutoffs_m_s01 = numpy.concatenate((
        numpy.array([0.]),
        intensity_cutoffs_m_s01,
        numpy.array([numpy.inf])
    ))

    num_subsets = len(intensity_cutoffs_m_s01) - 1
    error_checking.assert_is_numpy_array(
        numpy.array(subset_names),
        exact_dimensions=numpy.array([num_subsets], dtype=int)
    )

    # Do actual stuff.
    print('Reading data from: "{0:s}"...'.format(input_prediction_file_name))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)

    num_examples = len(prediction_dict[prediction_io.CYCLONE_IDS_KEY])
    intensity_by_prediction_m_s01 = numpy.full(num_examples, numpy.nan)

    unique_cyclone_id_strings = list(set(
        prediction_dict[prediction_io.CYCLONE_IDS_KEY]
    ))
    unique_cyclone_id_strings.sort()

    for this_cyclone_id_string in unique_cyclone_id_strings:
        this_ships_file_name = ships_io.find_file(
            directory_name=ships_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(this_ships_file_name))
        this_ships_table_xarray = ships_io.read_file(this_ships_file_name)

        these_prediction_indices = numpy.where(
            numpy.array(prediction_dict[prediction_io.CYCLONE_IDS_KEY]) ==
            this_cyclone_id_string
        )[0]

        these_ships_indices = numpy.array([
            numpy.where(
                this_ships_table_xarray[ships_io.VALID_TIME_KEY].values == t
            )[0][0]
            for t in prediction_dict[prediction_io.INIT_TIMES_KEY][
                these_prediction_indices
            ]
        ], dtype=int)

        intensity_by_prediction_m_s01[these_prediction_indices] = (
            this_ships_table_xarray[ships_io.STORM_INTENSITY_KEY].values[
                these_ships_indices
            ]
        )

    assert numpy.all(intensity_by_prediction_m_s01 > 0)
    print(SEPARATOR_STRING)

    for i in range(num_subsets):
        subset_flags = numpy.logical_and(
            intensity_by_prediction_m_s01 >= intensity_cutoffs_m_s01[i],
            intensity_by_prediction_m_s01 < intensity_cutoffs_m_s01[i + 1]
        ).astype(float)

        _, subset_indices = gg_model_activation.get_hilo_activation_examples(
            storm_activations=subset_flags,
            num_high_activation_examples=num_examples_per_subset,
            num_low_activation_examples=num_examples_per_subset,
            unique_storm_cells=enforce_unique_cyclones,
            full_storm_id_strings=prediction_dict[prediction_io.CYCLONE_IDS_KEY]
        )

        subset_prediction_dict = prediction_io.subset_by_index(
            prediction_dict=copy.deepcopy(prediction_dict),
            desired_indices=subset_indices
        )
        subset_file_name = '{0:s}/predictions_{1:s}.nc'.format(
            output_dir_name, subset_names[i].replace('_', '-')
        )
        d = subset_prediction_dict

        print('Writing {0:d} examples to: "{1:s}"...'.format(
            len(subset_indices), subset_file_name
        ))
        prediction_io.write_file(
            netcdf_file_name=subset_file_name,
            forecast_probability_matrix=d[prediction_io.PROBABILITY_MATRIX_KEY],
            target_class_matrix=d[prediction_io.TARGET_MATRIX_KEY],
            cyclone_id_strings=d[prediction_io.CYCLONE_IDS_KEY],
            init_times_unix_sec=d[prediction_io.INIT_TIMES_KEY],
            storm_latitudes_deg_n=d[prediction_io.STORM_LATITUDES_KEY],
            storm_longitudes_deg_e=d[prediction_io.STORM_LONGITUDES_KEY],
            storm_intensity_changes_m_s01=
            d[prediction_io.STORM_INTENSITY_CHANGES_KEY],
            model_file_name=d[prediction_io.MODEL_FILE_KEY],
            lead_times_hours=d[prediction_io.LEAD_TIMES_KEY],
            quantile_levels=d[prediction_io.QUANTILE_LEVELS_KEY],
            uncertainty_calib_model_file_name=
            d[prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        ships_dir_name=getattr(INPUT_ARG_OBJECT, SHIPS_DIR_ARG_NAME),
        intensity_cutoffs_m_s01=numpy.array(
            getattr(INPUT_ARG_OBJECT, INTENSITY_CUTOFFS_ARG_NAME), dtype=float
        ),
        num_examples_per_subset=getattr(INPUT_ARG_OBJECT, SUBSET_SIZE_ARG_NAME),
        subset_names=getattr(INPUT_ARG_OBJECT, SUBSET_NAMES_ARG_NAME),
        enforce_unique_cyclones=bool(
            getattr(INPUT_ARG_OBJECT, UNIQUE_CYCLONES_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
