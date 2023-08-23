"""Plots evaluation by month, then by ocean basin, separately."""

import os
import sys
import warnings
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
import matplotlib.patches

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import satellite_utils
import evaluation
import evaluation_plotting as eval_plotting

TOLERANCE = 1e-6
NUM_MONTHS = 12

MARKER_TYPE = 'o'
MARKER_SIZE = 16
LINE_WIDTH = 4
POLYGON_OPACITY = 0.5

AUC_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
AUPD_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
BSS_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

NUM_EXAMPLES_FACE_COLOUR = numpy.full(3, 152. / 255)
NUM_EXAMPLES_FACE_COLOUR = matplotlib.colors.to_rgba(
    NUM_EXAMPLES_FACE_COLOUR, 0.5
)

NUM_POS_EXAMPLES_FACE_COLOUR = numpy.full(3, 0.)
NUM_POS_EXAMPLES_FACE_COLOUR = matplotlib.colors.to_rgba(
    NUM_POS_EXAMPLES_FACE_COLOUR, 0.5
)

HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
HISTOGRAM_EDGE_WIDTH = 1.5

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
MODEL_DESCRIPTION_ARG_NAME = 'model_description_string'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Evaluation files therein will be found by '
    '`evaluation.find_file` and read by `evaluation.read_file`.'
)
MODEL_DESCRIPTION_HELP_STRING = (
    'Model description, for use in figure titles.  If you want plain figure '
    'titles (like just "Monthly results" and "By-basin results"), leave this '
    'argument alone.'
)
CONFIDENCE_LEVEL_HELP_STRING = 'Confidence level for error bars.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DESCRIPTION_ARG_NAME, type=str, required=False, default='',
    help=MODEL_DESCRIPTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_metrics(auc_matrix, aupd_matrix, bss_matrix, example_counts,
                  positive_example_counts, confidence_level, plot_legend):
    """Plots evaluation metrics either by month or by basin.

    N = number of subsets (months or basins)
    B = number of bootstrap replicates

    :param auc_matrix: B-by-N numpy array of AUC values.
    :param aupd_matrix: B-by-N numpy array of AUPD values.
    :param bss_matrix: B-by-N numpy array of BSS values.
    :param example_counts: length-N numpy array with number of examples for each
        subset.
    :param positive_example_counts: Same but for positive examples only.
    :param confidence_level: Confidence level for uncertainty envelope.
    :param plot_legend: Boolean flag.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Housekeeping.
    num_bootstrap_reps = auc_matrix.shape[0]

    figure_object, main_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    histogram_axes_object = main_axes_object.twinx()
    main_axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    main_axes_object.patch.set_visible(False)

    num_subsets = auc_matrix.shape[1]
    x_values = numpy.linspace(0, num_subsets - 1, num=num_subsets, dtype=float)

    # Plot mean AUC.
    this_handle = main_axes_object.plot(
        x_values, numpy.mean(auc_matrix, axis=0), color=AUC_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=AUC_COLOUR, markeredgecolor=AUC_COLOUR,
        markeredgewidth=0
    )[0]

    legend_handles = [this_handle]
    legend_strings = ['AUC']

    # Plot confidence interval for AUC.
    if num_bootstrap_reps > 1:
        x_matrix = numpy.expand_dims(x_values, axis=0)
        x_matrix = numpy.repeat(x_matrix, axis=0, repeats=num_bootstrap_reps)

        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=x_matrix, y_value_matrix=auc_matrix,
            confidence_level=confidence_level, same_order=False,
            for_reliability_curve=False
        )

        polygon_colour = matplotlib.colors.to_rgba(AUC_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        main_axes_object.add_patch(patch_object)

    # Plot mean AUPD.
    this_handle = main_axes_object.plot(
        x_values, numpy.mean(aupd_matrix, axis=0), color=AUPD_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=AUPD_COLOUR, markeredgecolor=AUPD_COLOUR,
        markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('AUPD')

    # Plot confidence interval for AUPD.
    if num_bootstrap_reps > 1:
        x_matrix = numpy.expand_dims(x_values, axis=0)
        x_matrix = numpy.repeat(x_matrix, axis=0, repeats=num_bootstrap_reps)

        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=x_matrix, y_value_matrix=aupd_matrix,
            confidence_level=confidence_level, same_order=False,
            for_reliability_curve=False
        )

        polygon_colour = matplotlib.colors.to_rgba(AUPD_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        main_axes_object.add_patch(patch_object)

    # Plot mean BSS.
    this_handle = main_axes_object.plot(
        x_values, numpy.mean(bss_matrix, axis=0), color=BSS_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=BSS_COLOUR, markeredgecolor=BSS_COLOUR,
        markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('BSS')

    # Plot confidence interval for BSS.
    if num_bootstrap_reps > 1:
        x_matrix = numpy.expand_dims(x_values, axis=0)
        x_matrix = numpy.repeat(x_matrix, axis=0, repeats=num_bootstrap_reps)

        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=x_matrix, y_value_matrix=bss_matrix,
            confidence_level=confidence_level, same_order=False,
            for_reliability_curve=False
        )

        polygon_colour = matplotlib.colors.to_rgba(BSS_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        main_axes_object.add_patch(patch_object)

    main_axes_object.set_ylabel('Evaluation metric (AUC/AUPD/BSS)')
    main_axes_object.set_xlim([
        numpy.min(x_values) - 0.5, numpy.max(x_values) + 0.5
    ])

    # Plot histogram of example counts.
    y_values = numpy.maximum(numpy.log10(example_counts), 0.)
    this_handle = histogram_axes_object.bar(
        x=x_values, height=y_values, width=1.,
        color=NUM_EXAMPLES_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('# of cases')
    histogram_axes_object.set_ylabel('Histogram count')

    # Plot histogram of positive-example counts.
    y_values = numpy.maximum(numpy.log10(positive_example_counts), 0.)
    this_handle = histogram_axes_object.bar(
        x=x_values, height=y_values, width=1.,
        color=NUM_POS_EXAMPLES_FACE_COLOUR,
        edgecolor=NUM_POS_EXAMPLES_FACE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('# of RI cases')
    histogram_axes_object.set_ylabel('Histogram count')

    tick_values = histogram_axes_object.get_yticks()
    tick_strings = [
        '{0:d}'.format(int(numpy.round(10 ** v))) for v in tick_values
    ]
    histogram_axes_object.set_yticklabels(tick_strings)

    print('Number of cases by subset: {0:s}'.format(
        str(example_counts)
    ))
    print('Number of RI cases by subset: {0:s}'.format(
        str(positive_example_counts)
    ))

    if plot_legend:
        main_axes_object.legend(
            legend_handles, legend_strings, loc='upper center',
            bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True,
            ncol=2
        )

    return figure_object, main_axes_object


def _plot_by_month(evaluation_dir_name, model_description_string,
                   confidence_level, output_dir_name):
    """Plots evaluation metrics by month.

    :param evaluation_dir_name: See documentation at top of file.
    :param model_description_string: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    :return: output_file_name: Path to figure saved by this method.
    """

    auc_matrix = None
    aupd_matrix = None
    bss_matrix = None
    num_bootstrap_reps = None
    example_counts = numpy.full(NUM_MONTHS, 0, dtype=int)
    positive_example_counts = numpy.full(NUM_MONTHS, 0, dtype=int)

    for k in range(NUM_MONTHS):
        evaluation_file_name = evaluation.find_file(
            directory_name=evaluation_dir_name, month=k + 1,
            raise_error_if_missing=False
        )

        if not os.path.isfile(evaluation_file_name):
            warning_string = (
                'POTENTIAL ERROR: Cannot find file expected at: "{0:s}"'
            ).format(evaluation_file_name)

            warnings.warn(warning_string)
            continue

        print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
        et = evaluation.read_file(evaluation_file_name)

        this_num_bootstrap_reps = len(
            et.coords[evaluation.BOOTSTRAP_REPLICATE_DIM].values
        )
        if num_bootstrap_reps is None:
            num_bootstrap_reps = this_num_bootstrap_reps
            auc_matrix = numpy.full((num_bootstrap_reps, NUM_MONTHS), numpy.nan)
            aupd_matrix = numpy.full((num_bootstrap_reps, NUM_MONTHS), numpy.nan)
            bss_matrix = numpy.full((num_bootstrap_reps, NUM_MONTHS), numpy.nan)

        assert num_bootstrap_reps == this_num_bootstrap_reps

        example_counts[k] = numpy.sum(et[evaluation.EXAMPLE_COUNT_NO_BS_KEY])
        num_true_positives = numpy.mean(
            et[evaluation.NUM_TRUE_POSITIVES_KEY].values[0, :]
        )
        num_false_negatives = numpy.mean(
            et[evaluation.NUM_FALSE_NEGATIVES_KEY].values[0, :]
        )
        positive_example_counts[k] = int(numpy.round(
            num_true_positives + num_false_negatives
        ))

        auc_matrix[:, k] = et[evaluation.AUC_KEY].values
        aupd_matrix[:, k] = et[evaluation.AUPD_KEY].values
        bss_matrix[:, k] = et[evaluation.BRIER_SKILL_SCORE_KEY].values

    x_tick_values = numpy.linspace(
        0, NUM_MONTHS - 1, num=NUM_MONTHS, dtype=float
    )
    x_tick_labels = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
        'Oct', 'Nov', 'Dec'
    ]

    figure_object, axes_object = _plot_metrics(
        auc_matrix=auc_matrix,
        aupd_matrix=aupd_matrix,
        bss_matrix=bss_matrix,
        example_counts=example_counts,
        positive_example_counts=positive_example_counts,
        confidence_level=confidence_level,
        plot_legend=True
    )

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)
    axes_object.set_title(
        'Monthly results{0:s}'.format(model_description_string)
    )

    output_file_name = '{0:s}/evaluation_by_month.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return output_file_name


def _plot_by_basin(evaluation_dir_name, model_description_string,
                   confidence_level, output_dir_name):
    """Plots evaluation metrics by basin.

    :param evaluation_dir_name: See documentation at top of file.
    :param model_description_string: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    :return: output_file_name: Path to figure saved by this method.
    """

    basin_id_strings = [
        satellite_utils.NORTH_ATLANTIC_ID_STRING,
        satellite_utils.NORTHEAST_PACIFIC_ID_STRING,
        satellite_utils.NORTH_CENTRAL_PACIFIC_ID_STRING,
        satellite_utils.NORTHWEST_PACIFIC_ID_STRING,
        satellite_utils.NORTH_INDIAN_ID_STRING,
        satellite_utils.SOUTHERN_HEMISPHERE_ID_STRING
    ]

    auc_matrix = None
    aupd_matrix = None
    bss_matrix = None
    num_bootstrap_reps = None

    num_basins = len(basin_id_strings)
    example_counts = numpy.full(num_basins, 0, dtype=int)
    positive_example_counts = numpy.full(num_basins, 0, dtype=int)

    for k in range(num_basins):
        evaluation_file_name = evaluation.find_file(
            directory_name=evaluation_dir_name,
            basin_id_string=basin_id_strings[k],
            raise_error_if_missing=False
        )

        if not os.path.isfile(evaluation_file_name):
            warning_string = (
                'POTENTIAL ERROR: Cannot find file expected at: "{0:s}"'
            ).format(evaluation_file_name)

            warnings.warn(warning_string)
            continue

        print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
        et = evaluation.read_file(evaluation_file_name)

        this_num_bootstrap_reps = len(
            et.coords[evaluation.BOOTSTRAP_REPLICATE_DIM].values
        )
        if num_bootstrap_reps is None:
            num_bootstrap_reps = this_num_bootstrap_reps
            auc_matrix = numpy.full((num_bootstrap_reps, num_basins), numpy.nan)
            aupd_matrix = numpy.full((num_bootstrap_reps, num_basins), numpy.nan)
            bss_matrix = numpy.full((num_bootstrap_reps, num_basins), numpy.nan)

        assert num_bootstrap_reps == this_num_bootstrap_reps

        example_counts[k] = numpy.sum(et[evaluation.EXAMPLE_COUNT_NO_BS_KEY])
        num_true_positives = numpy.mean(
            et[evaluation.NUM_TRUE_POSITIVES_KEY].values[0, :]
        )
        num_false_negatives = numpy.mean(
            et[evaluation.NUM_FALSE_NEGATIVES_KEY].values[0, :]
        )
        positive_example_counts[k] = int(numpy.round(
            num_true_positives + num_false_negatives
        ))

        auc_matrix[:, k] = et[evaluation.AUC_KEY].values
        aupd_matrix[:, k] = et[evaluation.AUPD_KEY].values
        bss_matrix[:, k] = et[evaluation.BRIER_SKILL_SCORE_KEY].values

    x_tick_values = numpy.linspace(
        0, num_basins - 1, num=num_basins, dtype=float
    )
    x_tick_labels = basin_id_strings

    figure_object, axes_object = _plot_metrics(
        auc_matrix=auc_matrix,
        bss_matrix=bss_matrix,
        aupd_matrix=aupd_matrix,
        example_counts=example_counts,
        positive_example_counts=positive_example_counts,
        confidence_level=confidence_level,
        plot_legend=True
    )

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)
    axes_object.set_title(
        'By-basin results{0:s}'.format(model_description_string)
    )

    output_file_name = '{0:s}/evaluation_by_basin.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return output_file_name


def _run(evaluation_dir_name, model_description_string, confidence_level,
         output_dir_name):
    """Plots evaluation by month, then by ocean basin, separately.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param model_description_string: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    if model_description_string == '':
        model_description_string = None

    model_description_string = (
        '' if model_description_string is None
        else ' for {0:s}'.format(model_description_string)
    )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    _plot_by_month(
        evaluation_dir_name=evaluation_dir_name,
        model_description_string=model_description_string,
        confidence_level=confidence_level,
        output_dir_name=output_dir_name
    )

    _plot_by_basin(
        evaluation_dir_name=evaluation_dir_name,
        model_description_string=model_description_string,
        confidence_level=confidence_level,
        output_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        model_description_string=getattr(
            INPUT_ARG_OBJECT, MODEL_DESCRIPTION_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
