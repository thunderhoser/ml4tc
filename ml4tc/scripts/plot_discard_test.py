"""Plots results of discard test (error vs. discard fraction)."""

import argparse
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4tc.utils import uq_evaluation
from ml4tc.plotting import uq_evaluation_plotting as uq_eval_plotting

TITLE_FONT_SIZE = 30
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
MODEL_DESCRIPTION_ARG_NAME = 'model_description_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `uq_evaluation.read_discard_results`.'
)
MODEL_DESCRIPTION_HELP_STRING = (
    'Model description, for use in figure title.  If you want a plain figure '
    'title (just "Discard test"), leave this argument alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Figure will be saved as an image here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DESCRIPTION_ARG_NAME, type=str, required=False, default='',
    help=MODEL_DESCRIPTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, model_description_string, output_file_name):
    """Plots results of discard test (error vs. discard fraction).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param model_description_string: Same.
    :param output_file_name: Same.
    """

    if model_description_string == '':
        model_description_string = None

    model_description_string = (
        '' if model_description_string is None
        else ' for {0:s}'.format(model_description_string)
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    result_dict = uq_evaluation.read_discard_results(input_file_name)

    figure_object, axes_object = uq_eval_plotting.plot_discard_test(
        result_dict=result_dict
    )

    title_string = 'Discard test{0:s}\nMF = {1:.1f}%'.format(
        model_description_string,
        100 * result_dict[uq_evaluation.MONOTONICITY_FRACTION_KEY]
    )

    print(title_string)
    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        model_description_string=getattr(
            INPUT_ARG_OBJECT, MODEL_DESCRIPTION_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
