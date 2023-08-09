"""Creates 8-panel figure showing eval metrics for one model vs. baseline.

This includes evaluation metrics for both deterministic predictions and
uncertainty quantification (UQ).
"""

import os
import argparse
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils

# TODO(thunderhoser): Maaaaybe ignore_minor_false_alarms should be an input arg.

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

NUM_PANEL_ROWS = 4
NUM_PANEL_COLUMNS = 2
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

NN_MEAN_EVAL_DIR_ARG_NAME = 'input_nn_mean_evaluation_dir_name'
BASELINE_EVAL_DIR_ARG_NAME = 'input_baseline_evaluation_dir_name'
NN_UQ_EVAL_DIR_ARG_NAME = 'input_nn_uq_evaluation_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

NN_MEAN_EVAL_DIR_HELP_STRING = (
    'Name of directory with evaluation figures for NN mean predictions.'
)
BASELINE_EVAL_DIR_HELP_STRING = (
    'Name of directory with evaluation figures for baseline predictions.'
)
NN_UQ_EVAL_DIR_HELP_STRING = (
    'Name of directory with evaluation figures for NN uncertainty estimates.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  New images (paneled figures) will be saved '
    'here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NN_MEAN_EVAL_DIR_ARG_NAME, type=str, required=True,
    help=NN_MEAN_EVAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_EVAL_DIR_ARG_NAME, type=str, required=True,
    help=BASELINE_EVAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NN_UQ_EVAL_DIR_ARG_NAME, type=str, required=True,
    help=NN_UQ_EVAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _overlay_text(
        image_file_name, x_offset_from_left_px, y_offset_from_top_px,
        text_string):
    """Overlays text on pre-existing image file.

    :param image_file_name: Path to image file.
    :param x_offset_from_left_px: Left-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_left_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _run(nn_mean_evaluation_dir_name, baseline_evaluation_dir_name,
         nn_uq_evaluation_dir_name, output_dir_name):
    """Creates 8-panel figure showing eval metrics for one model vs. baseline.

    This is effectively the main method.

    :param nn_mean_evaluation_dir_name: See documentation at top of file.
    :param baseline_evaluation_dir_name: Same.
    :param nn_uq_evaluation_dir_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
    concat_figure_file_name = '{0:s}/overall_eval_matched.jpg'.format(
        output_dir_name
    )

    panel_file_names = [
        '{0:s}/roc_curve.jpg'.format(nn_mean_evaluation_dir_name),
        '{0:s}/roc_curve.jpg'.format(baseline_evaluation_dir_name),
        '{0:s}/performance_diagram.jpg'.format(nn_mean_evaluation_dir_name),
        '{0:s}/performance_diagram.jpg'.format(baseline_evaluation_dir_name),
        '{0:s}/attributes_diagram.jpg'.format(nn_mean_evaluation_dir_name),
        '{0:s}/attributes_diagram.jpg'.format(baseline_evaluation_dir_name),
        '{0:s}/spread_vs_skill.jpg'.format(nn_uq_evaluation_dir_name),
        '{0:s}/discard_test.jpg'.format(nn_uq_evaluation_dir_name)
    ]

    resized_panel_file_names = [
        '{0:s}/nn_roc_curve.jpg'.format(output_dir_name),
        '{0:s}/baseline_roc_curve.jpg'.format(output_dir_name),
        '{0:s}/nn_performance_diagram.jpg'.format(output_dir_name),
        '{0:s}/baseline_performance_diagram.jpg'.format(output_dir_name),
        '{0:s}/nn_attributes_diagram.jpg'.format(output_dir_name),
        '{0:s}/baseline_attributes_diagram.jpg'.format(output_dir_name),
        '{0:s}/nn_spread_vs_skill.jpg'.format(output_dir_name),
        '{0:s}/nn_discard_test.jpg'.format(output_dir_name)
    ]

    panel_letters = ['a', 'f', 'b', 'g', 'c', 'h', 'd', 'e']

    for i in range(len(panel_file_names)):
        print('Resizing panel and saving to: "{0:s}"...'.format(
            resized_panel_file_names[i]
        ))

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[i],
            output_file_name=resized_panel_file_names[i]
        )
        _overlay_text(
            image_file_name=resized_panel_file_names[i],
            x_offset_from_left_px=0, y_offset_from_top_px=TITLE_FONT_SIZE,
            text_string='({0:s})'.format(panel_letters[i])
        )
        imagemagick_utils.resize_image(
            input_file_name=resized_panel_file_names[i],
            output_file_name=resized_panel_file_names[i],
            output_size_pixels=PANEL_SIZE_PX
        )

    print('Concatenating panels to: "{0:s}"...'.format(
        concat_figure_file_name
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=resized_panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=NUM_PANEL_ROWS, num_panel_columns=NUM_PANEL_COLUMNS
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        nn_mean_evaluation_dir_name=getattr(
            INPUT_ARG_OBJECT, NN_MEAN_EVAL_DIR_ARG_NAME
        ),
        baseline_evaluation_dir_name=getattr(
            INPUT_ARG_OBJECT, BASELINE_EVAL_DIR_ARG_NAME
        ),
        nn_uq_evaluation_dir_name=getattr(
            INPUT_ARG_OBJECT, NN_UQ_EVAL_DIR_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
