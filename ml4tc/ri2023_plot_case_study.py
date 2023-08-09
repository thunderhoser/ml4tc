"""Plots case study with predictors, NN predictions, and baseline predictions.

The three baselines are SHIPS-RII, SHIPS consensus, and DTOPS.
"""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import prediction_io

TIME_FORMAT = '%Y-%m-%d-%H'

BASELINE_DESCRIPTION_STRINGS = ['basic', 'consensus', 'dtops']
BASELINE_DESCRIPTION_STRINGS_FANCY = ['SHIPS-RII', 'SHIPS consensus', 'DTOPS']

BASELINE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
BASELINE_MARKER_TYPE = 'o'
BASELINE_MARKER_SIZE = 16

VIOLIN_LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
VIOLIN_LINE_WIDTH = 2.
VIOLIN_FACE_COLOUR = matplotlib.colors.to_rgba(c=VIOLIN_LINE_COLOUR, alpha=0.4)
VIOLIN_EDGE_COLOUR = matplotlib.colors.to_rgba(c=VIOLIN_LINE_COLOUR, alpha=0.4)
VIOLIN_EDGE_WIDTH = 0.

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

NN_MODEL_DIRS_ARG_NAME = 'input_nn_model_dir_names'
NN_MODEL_DESCRIPTIONS_ARG_NAME = 'nn_model_description_strings'
NORM_EXAMPLE_DIR_ARG_NAME = 'input_norm_example_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
INIT_TIME_ARG_NAME = 'init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

NN_MODEL_DIRS_HELP_STRING = (
    'List of input directories, one for each selected NN.  Each directory '
    'should be the top-level directory for the given NN.  This script will '
    'find results on the testing data, using isotonic regression.'
)
NN_MODEL_DESCRIPTIONS_HELP_STRING = (
    'List of NN descriptions, one per input directory.'
)
NORM_EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with normalized learning examples.  Predictors will be '
    'read by `example_io.read_file`, from files at locations determined by '
    '`example_io.find_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file.  This will be read by `example_io.read_file` '
    'and used to denormalize predictors.'
)
CYCLONE_ID_HELP_STRING = (
    'This script will plot the case study for one TC object, specified by this '
    '8-character cyclone ID and {0:s}'
).format(INIT_TIME_ARG_NAME)

INIT_TIME_HELP_STRING = (
    'This script will plot the case study for one TC object, specified by this '
    'forecast-init time (format "yyyy-mm-dd-HH") and {0:s}.'
).format(CYCLONE_ID_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NN_MODEL_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=NN_MODEL_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NN_MODEL_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=NN_MODEL_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORM_EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=NORM_EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_ARG_NAME, type=str, required=True,
    help=INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_nn_model_dir_names, nn_model_description_strings,
         norm_example_dir_name, normalization_file_name, cyclone_id_string,
         init_time_string, output_dir_name):
    """Plots case study with predictors, NN predictions, and baseline predictions.

    This is effectively the main method.

    :param top_nn_model_dir_names: See documentation at top of file.
    :param nn_model_description_strings: Same.
    :param norm_example_dir_name: Same.
    :param normalization_file_name: Same.
    :param cyclone_id_string: Same.
    :param init_time_string: Same.
    :param output_dir_name: Same.
    """

    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_nn_models = len(top_nn_model_dir_names)
    num_baseline_models = len(BASELINE_DESCRIPTION_STRINGS)
    num_models = num_nn_models + num_baseline_models

    assert len(nn_model_description_strings) == num_nn_models
    nn_model_description_strings = [
        s.replace('_', ' ') for s in nn_model_description_strings
    ]

    model_description_strings = (
        nn_model_description_strings + BASELINE_DESCRIPTION_STRINGS
    )
    model_description_strings_fancy = (
        nn_model_description_strings + BASELINE_DESCRIPTION_STRINGS_FANCY
    )
    top_model_dir_names = (
        top_nn_model_dir_names + [''] * len(BASELINE_DESCRIPTION_STRINGS)
    )

    nn_forecast_prob_matrix = numpy.array([])
    baseline_forecast_probs = []
    target_class = -1

    for i in range(num_models):
        if model_description_strings[i] in BASELINE_DESCRIPTION_STRINGS:
            prediction_file_name = (
                '{0:s}/real_time_testing_matched_with_ships/'
                'isotonic_regression/ships_predictions_{1:s}.nc'
            ).format(
                top_model_dir_names[0], model_description_strings[i]
            )
        else:
            prediction_file_name = (
                '{0:s}/real_time_testing_matched_with_ships/'
                'isotonic_regression/cnn_predictions_cf_dtops.nc'
            ).format(top_model_dir_names[i])

        print('Reading data from: "{0:s}"...'.format(prediction_file_name))
        prediction_dict = prediction_io.read_file(prediction_file_name)

        good_index = numpy.where(numpy.logical_and(
            numpy.array(prediction_dict[prediction_io.CYCLONE_IDS_KEY])
            == cyclone_id_string,
            prediction_dict[prediction_io.INIT_TIMES_KEY] == init_time_unix_sec
        ))[0][0]

        if model_description_strings[i] in BASELINE_DESCRIPTION_STRINGS:
            this_prob = prediction_io.get_mean_predictions(prediction_dict)[
                good_index, 0
            ]
            baseline_forecast_probs.append(this_prob)
        else:
            these_probs = prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][
                good_index, -1, 0, :
            ]

            if nn_forecast_prob_matrix.size == 0:
                nn_ensemble_size = len(these_probs)
                nn_forecast_prob_matrix = numpy.full(
                    (num_nn_models, nn_ensemble_size), numpy.nan
                )

            nn_forecast_prob_matrix[:, i] = these_probs

        this_target_class = prediction_dict[prediction_io.TARGET_MATRIX_KEY][
            good_index, 0
        ]
        if target_class == -1:
            target_class = this_target_class + 0

        assert target_class == this_target_class

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    x_tick_values = numpy.linspace(
        1, num_nn_models, num=num_nn_models, dtype=float
    )
    violin_handles = axes_object.violinplot(
        numpy.transpose(nn_forecast_prob_matrix),
        positions=x_tick_values,
        vert=True, widths=0.8, showmeans=False, showmedians=False,
        showextrema=True
    )

    for part_name in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        try:
            this_handle = violin_handles[part_name]
        except:
            continue

        this_handle.set_edgecolor(VIOLIN_LINE_COLOUR)
        this_handle.set_linewidth(VIOLIN_LINE_WIDTH)

    for this_handle in violin_handles['bodies']:
        this_handle.set_facecolor(VIOLIN_FACE_COLOUR)
        this_handle.set_edgecolor(VIOLIN_EDGE_COLOUR)
        this_handle.set_linewidth(VIOLIN_EDGE_WIDTH)
        this_handle.set_alpha(1.)

    for i in range(num_baseline_models):
        for this_x in x_tick_values:
            if BASELINE_DESCRIPTION_STRINGS[i] == 'basic':
                label_string = 'R'
            elif BASELINE_DESCRIPTION_STRINGS[i] == 'consensus':
                label_string = 'C'
            else:
                label_string = 'D'

            axes_object.plot(
                this_x, baseline_forecast_probs[i], linestyle='None',
                marker=BASELINE_MARKER_TYPE,
                markersize=BASELINE_MARKER_SIZE, markeredgewidth=0,
                markerfacecolor=BASELINE_COLOUR,
                markeredgecolor=BASELINE_COLOUR
            )
            axes_object.text(
                this_x + 0.05, baseline_forecast_probs[i], label_string,
                color=BASELINE_COLOUR, fontsize=DEFAULT_FONT_SIZE,
                horizontalalignment='left', verticalalignment='center'
            )

    axes_object.set_ylabel('RI probability')
    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(nn_model_description_strings, rotation=90)

    title_string = 'RI forecasts for {0:s}, init {1:s}'.format(
        cyclone_id_string, init_time_string
    )
    axes_object.set_title(title_string)

    output_file_name = '{0:s}/correlation_matrix.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_nn_model_dir_names=getattr(
            INPUT_ARG_OBJECT, NN_MODEL_DIRS_ARG_NAME
        ),
        nn_model_description_strings=getattr(
            INPUT_ARG_OBJECT, NN_MODEL_DESCRIPTIONS_ARG_NAME
        ),
        norm_example_dir_name=getattr(
            INPUT_ARG_OBJECT, NORM_EXAMPLE_DIR_ARG_NAME
        ),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        init_time_string=getattr(INPUT_ARG_OBJECT, INIT_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
