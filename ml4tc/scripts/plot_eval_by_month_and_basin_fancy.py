"""Plots evaluation by month, then by ocean basin, separately."""

import os
import warnings
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
import matplotlib.patches
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4tc.utils import satellite_utils
from ml4tc.utils import evaluation
from ml4tc.plotting import evaluation_plotting as eval_plotting

TOLERANCE = 1e-6
NUM_MONTHS = 12

MARKER_TYPE = 'o'
MARKER_SIZE = 16
LINE_WIDTH = 4
HISTOGRAM_EDGE_WIDTH = 1.5

AUC_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
CSI_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
POD_COLOUR = AUC_COLOUR
FAR_COLOUR = CSI_COLOUR
POLYGON_OPACITY = 0.5

HISTOGRAM_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
HISTOGRAM_FACE_COLOUR = matplotlib.colors.to_rgba(HISTOGRAM_FACE_COLOUR, 0.5)
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
TOTAL_VALIDN_EVAL_FILE_ARG_NAME = 'input_total_validn_eval_file_name'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Evaluation files therein will be found by '
    '`evaluation.find_file` and read by `evaluation.read_file`.'
)
TOTAL_VALIDN_EVAL_FILE_HELP_STRING = (
    'Path to evaluation file for total validation set.  Will be read by '
    '`evaluation.read_file` and used to determine best probability threshold.'
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
    '--' + TOTAL_VALIDN_EVAL_FILE_ARG_NAME, type=str, required=True,
    help=TOTAL_VALIDN_EVAL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_auc_and_csi(auc_matrix, csi_matrix, example_counts, confidence_level,
                      plot_legend):
    """Plots AUC and CSI either by month or by basin.

    N = number of subsets (months or basins)
    B = number of bootstrap replicates

    :param auc_matrix: B-by-N numpy array of AUC values.
    :param csi_matrix: B-by-N numpy array of CSI values.
    :param example_counts: length-N numpy array with number of examples for each
        subset.
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

    # Plot mean CSI.
    this_handle = main_axes_object.plot(
        x_values, numpy.mean(csi_matrix, axis=0), color=CSI_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=CSI_COLOUR, markeredgecolor=CSI_COLOUR,
        markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('CSI')

    # Plot confidence interval for CSI.
    if num_bootstrap_reps > 1:
        x_matrix = numpy.expand_dims(x_values, axis=0)
        x_matrix = numpy.repeat(x_matrix, axis=0, repeats=num_bootstrap_reps)

        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=x_matrix, y_value_matrix=csi_matrix,
            confidence_level=confidence_level, same_order=False,
            for_reliability_curve=False
        )

        polygon_colour = matplotlib.colors.to_rgba(CSI_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        main_axes_object.add_patch(patch_object)

    main_axes_object.set_ylabel('AUC or CSI')
    main_axes_object.set_xlim([
        numpy.min(x_values) - 0.5, numpy.max(x_values) + 0.5
    ])

    # Plot histogram of example counts.
    y_values = numpy.maximum(numpy.log10(example_counts), 0.)

    this_handle = histogram_axes_object.bar(
        x=x_values, height=y_values, width=1., color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR, linewidth=HISTOGRAM_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append(r'Number of examples')
    histogram_axes_object.set_ylabel(r'Number of examples')

    tick_values = histogram_axes_object.get_yticks()
    tick_strings = [
        '{0:d}'.format(int(numpy.round(10 ** v))) for v in tick_values
    ]
    histogram_axes_object.set_yticklabels(tick_strings)

    print('Number of examples by subset: {0:s}'.format(
        str(example_counts)
    ))

    if plot_legend:
        main_axes_object.legend(
            legend_handles, legend_strings, loc='lower center',
            bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True,
            ncol=len(legend_handles)
        )

    return figure_object, main_axes_object


def _plot_pod_and_far(pod_matrix, far_matrix, positive_example_counts,
                      confidence_level, plot_legend):
    """Plots POD and FAR either by month or by basin.

    N = number of subsets (months or basins)
    B = number of bootstrap replicates

    :param pod_matrix: B-by-N numpy array of POD values.
    :param far_matrix: B-by-N numpy array of FAR values.
    :param positive_example_counts: length-N numpy array with number of positive
        examples for each subset.
    :param confidence_level: Confidence level for uncertainty envelope.
    :param plot_legend: Boolean flag.
    :return: figure_object: See doc for `_plot_auc_and_csi`.
    :return: axes_object: Same.
    """

    # Housekeeping.
    num_bootstrap_reps = pod_matrix.shape[0]

    figure_object, main_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    histogram_axes_object = main_axes_object.twinx()
    main_axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    main_axes_object.patch.set_visible(False)

    num_subsets = pod_matrix.shape[1]
    x_values = numpy.linspace(0, num_subsets - 1, num=num_subsets, dtype=float)

    # Plot mean POD.
    this_handle = main_axes_object.plot(
        x_values, numpy.mean(pod_matrix, axis=0), color=POD_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=POD_COLOUR, markeredgecolor=POD_COLOUR,
        markeredgewidth=0
    )[0]

    legend_handles = [this_handle]
    legend_strings = ['POD']

    # Plot confidence interval for POD.
    if num_bootstrap_reps > 1:
        x_matrix = numpy.expand_dims(x_values, axis=0)
        x_matrix = numpy.repeat(x_matrix, axis=0, repeats=num_bootstrap_reps)

        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=x_matrix, y_value_matrix=pod_matrix,
            confidence_level=confidence_level, same_order=False,
            for_reliability_curve=False
        )

        polygon_colour = matplotlib.colors.to_rgba(POD_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        main_axes_object.add_patch(patch_object)

    # Plot mean FAR.
    this_handle = main_axes_object.plot(
        x_values, numpy.mean(far_matrix, axis=0), color=FAR_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=FAR_COLOUR, markeredgecolor=FAR_COLOUR,
        markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('FAR')

    # Plot confidence interval for FAR.
    if num_bootstrap_reps > 1:
        x_matrix = numpy.expand_dims(x_values, axis=0)
        x_matrix = numpy.repeat(x_matrix, axis=0, repeats=num_bootstrap_reps)

        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=x_matrix, y_value_matrix=far_matrix,
            confidence_level=confidence_level, same_order=False,
            for_reliability_curve=False
        )

        polygon_colour = matplotlib.colors.to_rgba(FAR_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        main_axes_object.add_patch(patch_object)

    main_axes_object.set_ylabel('POD or FAR')
    main_axes_object.set_xlim([
        numpy.min(x_values) - 0.5, numpy.max(x_values) + 0.5
    ])

    # Plot histogram of positive-example counts.
    this_handle = histogram_axes_object.bar(
        x=x_values, height=positive_example_counts, width=1.,
        color=HISTOGRAM_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Number of RI examples')
    histogram_axes_object.set_ylabel('Number of RI examples')

    print('Number of RI examples by subset: {0:s}'.format(
        str(positive_example_counts)
    ))

    if plot_legend:
        main_axes_object.legend(
            legend_handles, legend_strings, loc='lower center',
            bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True,
            ncol=len(legend_handles)
        )

    return figure_object, main_axes_object


def _plot_by_month(evaluation_dir_name, probability_threshold, confidence_level,
                   output_dir_name):
    """Plots model evaluation by month.

    :param evaluation_dir_name: See documentation at top of file.
    :param probability_threshold: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    :return: output_file_names: Paths to figures saved by this method.
    """

    auc_matrix = None
    pod_matrix = None
    far_matrix = None
    csi_matrix = None
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
            pod_matrix = numpy.full((num_bootstrap_reps, NUM_MONTHS), numpy.nan)
            far_matrix = numpy.full((num_bootstrap_reps, NUM_MONTHS), numpy.nan)
            csi_matrix = numpy.full((num_bootstrap_reps, NUM_MONTHS), numpy.nan)

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

        all_prob_thresholds = (
            et.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values
        )
        threshold_index = numpy.argmin(numpy.absolute(
            all_prob_thresholds - probability_threshold
        ))
        threshold_diff = numpy.absolute(
            all_prob_thresholds[threshold_index] - probability_threshold
        )
        assert threshold_diff <= TOLERANCE

        auc_matrix[:, k] = et[evaluation.AUC_KEY].values
        pod_matrix[:, k] = et[evaluation.POD_KEY].values[threshold_index, :]
        far_matrix[:, k] = (
            1. - et[evaluation.SUCCESS_RATIO_KEY].values[threshold_index, :]
        )
        csi_matrix[:, k] = et[evaluation.CSI_KEY].values[threshold_index, :]

    x_tick_values = numpy.linspace(
        0, NUM_MONTHS - 1, num=NUM_MONTHS, dtype=float
    )
    x_tick_labels = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
        'Oct', 'Nov', 'Dec'
    ]

    figure_object, axes_object = _plot_auc_and_csi(
        auc_matrix=auc_matrix, csi_matrix=csi_matrix,
        example_counts=example_counts,
        confidence_level=confidence_level, plot_legend=True
    )
    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='(a)',
        x_coord_normalized=-0.075, y_coord_normalized=1.02
    )

    auc_csi_file_name = '{0:s}/auc_and_csi_by_month.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(auc_csi_file_name))
    figure_object.savefig(
        auc_csi_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _plot_pod_and_far(
        pod_matrix=pod_matrix, far_matrix=far_matrix,
        positive_example_counts=positive_example_counts,
        confidence_level=confidence_level, plot_legend=True
    )
    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='(c)',
        x_coord_normalized=-0.075, y_coord_normalized=1.02
    )

    pod_far_file_name = '{0:s}/pod_and_far_by_month.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(pod_far_file_name))
    figure_object.savefig(
        pod_far_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return [auc_csi_file_name, pod_far_file_name]


def _plot_by_basin(evaluation_dir_name, probability_threshold, confidence_level,
                   output_dir_name):
    """Plots model evaluation by basin.

    :param evaluation_dir_name: See documentation at top of file.
    :param probability_threshold: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    :return: output_file_names: Paths to figures saved by this method.
    """

    auc_matrix = None
    pod_matrix = None
    far_matrix = None
    csi_matrix = None
    num_bootstrap_reps = None

    basin_id_strings = satellite_utils.VALID_BASIN_ID_STRINGS
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
            pod_matrix = numpy.full((num_bootstrap_reps, num_basins), numpy.nan)
            far_matrix = numpy.full((num_bootstrap_reps, num_basins), numpy.nan)
            csi_matrix = numpy.full((num_bootstrap_reps, num_basins), numpy.nan)

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

        all_prob_thresholds = (
            et.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values
        )
        threshold_index = numpy.argmin(numpy.absolute(
            all_prob_thresholds - probability_threshold
        ))
        threshold_diff = numpy.absolute(
            all_prob_thresholds[threshold_index] - probability_threshold
        )
        assert threshold_diff <= TOLERANCE

        auc_matrix[:, k] = et[evaluation.AUC_KEY].values
        pod_matrix[:, k] = et[evaluation.POD_KEY].values[threshold_index, :]
        far_matrix[:, k] = (
            1. - et[evaluation.SUCCESS_RATIO_KEY].values[threshold_index, :]
        )
        csi_matrix[:, k] = et[evaluation.CSI_KEY].values[threshold_index, :]

    x_tick_values = numpy.linspace(
        0, num_basins - 1, num=num_basins, dtype=float
    )
    x_tick_labels = basin_id_strings

    figure_object, axes_object = _plot_auc_and_csi(
        auc_matrix=auc_matrix, csi_matrix=csi_matrix,
        example_counts=example_counts,
        confidence_level=confidence_level, plot_legend=False
    )
    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='(b)',
        x_coord_normalized=-0.075, y_coord_normalized=1.02
    )

    auc_csi_file_name = '{0:s}/auc_and_csi_by_basin.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(auc_csi_file_name))
    figure_object.savefig(
        auc_csi_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _plot_pod_and_far(
        pod_matrix=pod_matrix, far_matrix=far_matrix,
        positive_example_counts=positive_example_counts,
        confidence_level=confidence_level, plot_legend=False
    )
    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='(d)',
        x_coord_normalized=-0.075, y_coord_normalized=1.02
    )

    pod_far_file_name = '{0:s}/pod_and_far_by_basin.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(pod_far_file_name))
    figure_object.savefig(
        pod_far_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return [auc_csi_file_name, pod_far_file_name]


def _run(evaluation_dir_name, total_validn_eval_file_name, confidence_level,
         output_dir_name):
    """Plots evaluation by month, then by ocean basin, separately.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param total_validn_eval_file_name: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(total_validn_eval_file_name))
    this_table_xarray = evaluation.read_file(total_validn_eval_file_name)
    probability_threshold = evaluation.find_best_threshold(this_table_xarray)

    panel_file_names = _plot_by_month(
        evaluation_dir_name=evaluation_dir_name,
        probability_threshold=probability_threshold,
        confidence_level=confidence_level, output_dir_name=output_dir_name
    )

    panel_file_names += _plot_by_basin(
        evaluation_dir_name=evaluation_dir_name,
        probability_threshold=probability_threshold,
        confidence_level=confidence_level, output_dir_name=output_dir_name
    )

    concat_file_name = '{0:s}/eval_by_month_and_basin.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        total_validn_eval_file_name=getattr(
            INPUT_ARG_OBJECT, TOTAL_VALIDN_EVAL_FILE_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
