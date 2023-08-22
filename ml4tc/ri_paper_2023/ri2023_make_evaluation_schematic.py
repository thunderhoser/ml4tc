"""Makes schematic to explain evaluation methods."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.gg_utils import model_evaluation as gg_model_eval
from ml4tc.io import prediction_io
from ml4tc.utils import uq_evaluation
from ml4tc.plotting import evaluation_plotting as eval_plotting
from ml4tc.plotting import uq_evaluation_plotting as uq_eval_plotting

SAMPLE_SIZE = int(1e6)
ENSEMBLE_SIZE = 50

NUM_BINS_FOR_ATTR_DIAG = 20
NUM_SPREAD_BINS = 50
MIN_PLOT_COORD = -0.05
MAX_PLOT_COORD = 1.05

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

SCATTERPLOT_MEAN_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
SCATTERPLOT_MEMBER_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
REFERENCE_LINE_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

OUTPUT_DIR_ARG_NAME = 'output_dir_name'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _create_data_good_model():
    """Creates data (predictions and targets) for good model.

    E = number of examples
    S = ensemble size

    :return: target_classes: length-E numpy array of true classes (integers in
        range 0...1).
    :return: forecast_prob_matrix: E-by-S numpy array of forecast probabilities
        (floats in range 0...1).
    """

    mean_probabilities = numpy.random.gamma(
        shape=1.5, scale=5, size=SAMPLE_SIZE
    )
    mean_probabilities = numpy.minimum(mean_probabilities, 41.)
    mean_probabilities = mean_probabilities / numpy.max(mean_probabilities)

    bin_edges = numpy.linspace(
        0, 1, num=NUM_BINS_FOR_ATTR_DIAG + 1, dtype=float
    )
    target_classes = numpy.full(SAMPLE_SIZE, -1, dtype=int)

    for k in range(NUM_BINS_FOR_ATTR_DIAG):
        if k == NUM_BINS_FOR_ATTR_DIAG - 1:
            these_example_indices = numpy.where(
                mean_probabilities >= bin_edges[k]
            )[0]
        else:
            these_example_indices = numpy.where(numpy.logical_and(
                mean_probabilities >= bin_edges[k],
                mean_probabilities < bin_edges[k + 1]
            ))[0]

        this_mean_mean_prob = numpy.mean(
            mean_probabilities[these_example_indices]
        )
        this_event_freq = this_mean_mean_prob

        these_scores = numpy.random.uniform(
            low=0, high=1, size=len(these_example_indices)
        )
        target_classes[these_example_indices] = (
            these_scores >= (1. - this_event_freq)
        ).astype(int)

    forecast_prob_matrix = numpy.full(
        (SAMPLE_SIZE, ENSEMBLE_SIZE), numpy.nan
    )

    for i in range(SAMPLE_SIZE):
        if numpy.mod(i, 10000) == 0:
            print('Have created ensemble for {0:d} of {1:d} examples...'.format(
                i, SAMPLE_SIZE
            ))

        forecast_prob_matrix[i, :] = (
            mean_probabilities[i] +
            numpy.random.normal(loc=0, scale=0.1, size=ENSEMBLE_SIZE)
        )

    print('Have created ensemble for all {0:d} examples!'.format(SAMPLE_SIZE))
    forecast_prob_matrix = numpy.maximum(forecast_prob_matrix, 0.)
    forecast_prob_matrix = numpy.minimum(forecast_prob_matrix, 1.)

    # stdev_predicted_heating_rates_k_day01 = numpy.std(
    #     predicted_hr_matrix_k_day01, axis=1, ddof=1
    # )
    # good_indices = numpy.where(stdev_predicted_heating_rates_k_day01 <= 5)[0]
    #
    # predicted_hr_matrix_k_day01 = predicted_hr_matrix_k_day01[good_indices, :]
    # actual_heating_rates_k_day01 = actual_heating_rates_k_day01[good_indices]

    return target_classes, forecast_prob_matrix


def _create_data_poor_model():
    """Creates data (predictions and targets) for poor model.

    :return: target_classes: Same.
    :return: forecast_prob_matrix: Same.
    """

    mean_probabilities = numpy.random.gamma(
        shape=1.5, scale=5, size=SAMPLE_SIZE
    )
    mean_probabilities = numpy.minimum(mean_probabilities, 41.)
    mean_probabilities = mean_probabilities / numpy.max(mean_probabilities)

    bin_edges = numpy.linspace(
        0, 1, num=NUM_BINS_FOR_ATTR_DIAG + 1, dtype=float
    )
    target_classes = numpy.full(SAMPLE_SIZE, -1, dtype=int)

    for k in range(NUM_BINS_FOR_ATTR_DIAG):
        if k == NUM_BINS_FOR_ATTR_DIAG - 1:
            these_example_indices = numpy.where(
                mean_probabilities >= bin_edges[k]
            )[0]
        else:
            these_example_indices = numpy.where(numpy.logical_and(
                mean_probabilities >= bin_edges[k],
                mean_probabilities < bin_edges[k + 1]
            ))[0]

        this_mean_mean_prob = numpy.mean(
            mean_probabilities[these_example_indices]
        )
        this_event_freq = 1. / (
            1 + numpy.exp(-20 * (this_mean_mean_prob - 0.5))
        )

        these_scores = numpy.random.uniform(
            low=0, high=1, size=len(these_example_indices)
        )
        target_classes[these_example_indices] = (
            these_scores >= (1. - this_event_freq)
        ).astype(int)

    forecast_prob_matrix = numpy.vstack([
        mp + numpy.random.normal(
            loc=0., scale=max([2 * mp, 0.01]), size=ENSEMBLE_SIZE
        )
        for mp in mean_probabilities
    ])
    forecast_prob_matrix = numpy.maximum(forecast_prob_matrix, 0.)
    forecast_prob_matrix = numpy.minimum(forecast_prob_matrix, 1.)

    return target_classes, forecast_prob_matrix


def _overlay_text(
        image_file_name, x_offset_from_left_px, y_offset_from_top_px,
        text_string):
    """Creates two figures showing overall evaluation of uncertainty quant (UQ).

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


def _make_scatterplot_good_model(output_dir_name, panel_letter):
    """Creates scatterplot for good model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    target_classes, forecast_prob_matrix = _create_data_good_model()
    mean_forecast_probs = numpy.mean(forecast_prob_matrix, axis=1)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    legend_handles = [None] * 2

    for j in range(0, ENSEMBLE_SIZE, 5):
        legend_handles[1] = axes_object.plot(
            forecast_prob_matrix[::10, j], target_classes[::10],
            linestyle='None', marker='o', markersize=7.5, markeredgewidth=0,
            markerfacecolor=SCATTERPLOT_MEMBER_COLOUR,
            markeredgecolor=SCATTERPLOT_MEMBER_COLOUR
        )[0]

    dummy_target_classes = target_classes[::10] + 0.
    dummy_target_classes[dummy_target_classes < 0.5] = -0.03
    dummy_target_classes[dummy_target_classes > 0.5] = 1.03

    legend_handles[0] = axes_object.plot(
        mean_forecast_probs[::10], dummy_target_classes,
        linestyle='None', marker='o', markersize=5, markeredgewidth=0,
        markerfacecolor=SCATTERPLOT_MEAN_COLOUR,
        markeredgecolor=SCATTERPLOT_MEAN_COLOUR
    )[0]

    axes_object.plot(
        [MIN_PLOT_COORD, MAX_PLOT_COORD],
        [MIN_PLOT_COORD, MAX_PLOT_COORD],
        linestyle='dashed', color=REFERENCE_LINE_COLOUR, linewidth=4
    )

    legend_strings = ['Ensemble mean', 'Ensemble member']
    the_one_legend_handle = axes_object.legend(
        legend_handles, legend_strings, loc='center left',
        bbox_to_anchor=(0, 0.5), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=1., ncol=1,
        fontsize=36
    )
    the_one_legend_handle.legendHandles[0]._legmarker.set_markersize(12)
    the_one_legend_handle.legendHandles[1]._legmarker.set_markersize(12)

    axes_object.set_xlim(MIN_PLOT_COORD, MAX_PLOT_COORD)
    axes_object.set_ylim(MIN_PLOT_COORD, MAX_PLOT_COORD)

    axes_object.set_xlabel('RI probability')
    axes_object.set_ylabel('True RI label')
    axes_object.set_title('Scatterplot for Model A')

    output_file_name = '{0:s}/scatterplot_good_model.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _make_scatterplot_poor_model(output_dir_name, panel_letter):
    """Creates scatterplot for poor model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    target_classes, forecast_prob_matrix = _create_data_poor_model()
    mean_forecast_probs = numpy.mean(forecast_prob_matrix, axis=1)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    legend_handles = [None] * 2

    for j in range(0, ENSEMBLE_SIZE, 5):
        legend_handles[1] = axes_object.plot(
            forecast_prob_matrix[::10, j], target_classes[::10],
            linestyle='None', marker='o', markersize=7.5, markeredgewidth=0,
            markerfacecolor=SCATTERPLOT_MEMBER_COLOUR,
            markeredgecolor=SCATTERPLOT_MEMBER_COLOUR
        )[0]

    dummy_target_classes = target_classes[::10] + 0.
    dummy_target_classes[dummy_target_classes < 0.5] = -0.03
    dummy_target_classes[dummy_target_classes > 0.5] = 1.03

    legend_handles[0] = axes_object.plot(
        mean_forecast_probs[::10], dummy_target_classes,
        linestyle='None', marker='o', markersize=5, markeredgewidth=0,
        markerfacecolor=SCATTERPLOT_MEAN_COLOUR,
        markeredgecolor=SCATTERPLOT_MEAN_COLOUR
    )[0]

    axes_object.plot(
        [MIN_PLOT_COORD, MAX_PLOT_COORD],
        [MIN_PLOT_COORD, MAX_PLOT_COORD],
        linestyle='dashed', color=REFERENCE_LINE_COLOUR, linewidth=4
    )

    legend_strings = ['Ensemble mean', 'Ensemble member']
    the_one_legend_handle = axes_object.legend(
        legend_handles, legend_strings, loc='center left',
        bbox_to_anchor=(0, 0.5), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=1., ncol=1,
        fontsize=36
    )
    the_one_legend_handle.legendHandles[0]._legmarker.set_markersize(12)
    the_one_legend_handle.legendHandles[1]._legmarker.set_markersize(12)

    axes_object.set_xlim(MIN_PLOT_COORD, MAX_PLOT_COORD)
    axes_object.set_ylim(MIN_PLOT_COORD, MAX_PLOT_COORD)

    axes_object.set_xlabel('RI probability')
    axes_object.set_ylabel('True RI label')
    axes_object.set_title('Scatterplot for Model B')

    output_file_name = '{0:s}/scatterplot_poor_model.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _plot_attr_diag_poor_model(output_dir_name, panel_letter):
    """Plots attributes diagram for poor model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    target_classes, forecast_probs = _create_data_poor_model()
    forecast_probs = numpy.mean(forecast_probs, axis=1)

    (
        mean_predictions, mean_observations, example_counts
    ) = gg_model_eval.get_points_in_reliability_curve(
        forecast_probabilities=forecast_probs,
        observed_labels=target_classes,
        num_forecast_bins=NUM_BINS_FOR_ATTR_DIAG
    )

    bss_dict = gg_model_eval.get_brier_skill_score(
        mean_forecast_prob_by_bin=mean_predictions,
        mean_observed_label_by_bin=mean_observations,
        num_examples_by_bin=example_counts,
        climatology=numpy.mean(target_classes)
    )
    brier_score = bss_dict[gg_model_eval.BRIER_SCORE_KEY]
    brier_skill_score = bss_dict[gg_model_eval.BSS_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_prediction_matrix=numpy.expand_dims(mean_predictions, axis=0),
        mean_observation_matrix=numpy.expand_dims(mean_observations, axis=0),
        example_counts=example_counts,
        mean_value_in_training=numpy.mean(target_classes),
        min_value_to_plot=0., max_value_to_plot=1.
    )

    title_string = (
        'Attributes diagram for Model B\nBS = {0:.2f}; BSS = {1:.2f}'
    ).format(brier_score, brier_skill_score)

    axes_object.set_xlabel('Mean RI probability')
    axes_object.set_ylabel('Conditional RI frequency')
    axes_object.set_title(title_string)

    output_file_name = '{0:s}/attributes_diagram_poor_model.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _plot_attr_diag_good_model(output_dir_name, panel_letter):
    """Plots attributes diagram for good model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    target_classes, forecast_probs = _create_data_good_model()
    forecast_probs = numpy.mean(forecast_probs, axis=1)

    (
        mean_predictions, mean_observations, example_counts
    ) = gg_model_eval.get_points_in_reliability_curve(
        forecast_probabilities=forecast_probs,
        observed_labels=target_classes,
        num_forecast_bins=NUM_BINS_FOR_ATTR_DIAG
    )

    bss_dict = gg_model_eval.get_brier_skill_score(
        mean_forecast_prob_by_bin=mean_predictions,
        mean_observed_label_by_bin=mean_observations,
        num_examples_by_bin=example_counts,
        climatology=numpy.mean(target_classes)
    )
    brier_score = bss_dict[gg_model_eval.BRIER_SCORE_KEY]
    brier_skill_score = bss_dict[gg_model_eval.BSS_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_prediction_matrix=numpy.expand_dims(mean_predictions, axis=0),
        mean_observation_matrix=numpy.expand_dims(mean_observations, axis=0),
        example_counts=example_counts,
        mean_value_in_training=numpy.mean(target_classes),
        min_value_to_plot=0., max_value_to_plot=1.
    )

    title_string = (
        'Attributes diagram for Model B\nBS = {0:.2f}; BSS = {1:.2f}'
    ).format(brier_score, brier_skill_score)

    axes_object.set_xlabel('Mean RI probability')
    axes_object.set_ylabel('Conditional RI frequency')
    axes_object.set_title(title_string)

    output_file_name = '{0:s}/attributes_diagram_good_model.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _plot_spread_vs_skill_poor_model(output_dir_name, panel_letter):
    """Creates spread-skill plot for poor model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    target_classes, forecast_prob_matrix = _create_data_poor_model()
    forecast_prob_matrix = numpy.expand_dims(forecast_prob_matrix, axis=-2)
    forecast_prob_matrix = numpy.expand_dims(forecast_prob_matrix, axis=-2)
    forecast_prob_matrix = numpy.concatenate(
        (1. - forecast_prob_matrix, forecast_prob_matrix), axis=1
    )

    prediction_dict = {
        prediction_io.TARGET_MATRIX_KEY:
            numpy.expand_dims(target_classes, axis=-1),
        prediction_io.PROBABILITY_MATRIX_KEY: forecast_prob_matrix,
        prediction_io.QUANTILE_LEVELS_KEY: None
    }

    bin_edges = numpy.linspace(
        0, 0.5, num=NUM_SPREAD_BINS + 1, dtype=float
    )[1:-1]

    result_dict = uq_evaluation.get_spread_vs_skill(
        prediction_dict=prediction_dict,
        bin_edge_prediction_stdevs=bin_edges,
        use_median=False, use_fancy_quantile_method_for_stdev=False
    )
    figure_object, axes_object = uq_eval_plotting.plot_spread_vs_skill(
        result_dict
    )

    title_string = (
        'Spread vs. skill for Model B\nSSREL = {0:.3f}; SSRAT = {1:.3f}'
    ).format(
        result_dict[uq_evaluation.SPREAD_SKILL_RELIABILITY_KEY],
        result_dict[uq_evaluation.SPREAD_SKILL_RATIO_KEY]
    )

    axes_object.set_title(title_string)

    output_file_name = '{0:s}/spread_skill_plot_poor_model.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _plot_spread_vs_skill_good_model(output_dir_name, panel_letter):
    """Creates spread-skill plot for good model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    target_classes, forecast_prob_matrix = _create_data_good_model()
    forecast_prob_matrix = numpy.expand_dims(forecast_prob_matrix, axis=-2)
    forecast_prob_matrix = numpy.expand_dims(forecast_prob_matrix, axis=-2)
    forecast_prob_matrix = numpy.concatenate(
        (1. - forecast_prob_matrix, forecast_prob_matrix), axis=1
    )

    prediction_dict = {
        prediction_io.TARGET_MATRIX_KEY:
            numpy.expand_dims(target_classes, axis=-1),
        prediction_io.PROBABILITY_MATRIX_KEY: forecast_prob_matrix,
        prediction_io.QUANTILE_LEVELS_KEY: None
    }

    bin_edges = numpy.linspace(
        0, 0.5, num=NUM_SPREAD_BINS + 1, dtype=float
    )[1:-1]

    result_dict = uq_evaluation.get_spread_vs_skill(
        prediction_dict=prediction_dict,
        bin_edge_prediction_stdevs=bin_edges,
        use_median=False, use_fancy_quantile_method_for_stdev=False
    )
    figure_object, axes_object = uq_eval_plotting.plot_spread_vs_skill(
        result_dict
    )

    title_string = (
        'Spread vs. skill for Model A\nSSREL = {0:.3f}; SSRAT = {1:.3f}'
    ).format(
        result_dict[uq_evaluation.SPREAD_SKILL_RELIABILITY_KEY],
        result_dict[uq_evaluation.SPREAD_SKILL_RATIO_KEY]
    )

    axes_object.set_title(title_string)

    output_file_name = '{0:s}/spread_skill_plot_good_model.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _plot_discard_test_poor_model(output_dir_name, panel_letter):
    """Plots discard test for poor model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    target_classes, forecast_prob_matrix = _create_data_poor_model()
    forecast_prob_matrix = numpy.expand_dims(forecast_prob_matrix, axis=-2)
    forecast_prob_matrix = numpy.expand_dims(forecast_prob_matrix, axis=-2)
    forecast_prob_matrix = numpy.concatenate(
        (1. - forecast_prob_matrix, forecast_prob_matrix), axis=1
    )

    prediction_dict = {
        prediction_io.TARGET_MATRIX_KEY:
            numpy.expand_dims(target_classes, axis=-1),
        prediction_io.PROBABILITY_MATRIX_KEY: forecast_prob_matrix,
        prediction_io.QUANTILE_LEVELS_KEY: None
    }

    result_dict = uq_evaluation.run_discard_test(
        prediction_dict=prediction_dict,
        discard_fractions=numpy.linspace(0.05, 0.95, num=19),
        error_function=uq_evaluation.get_brier_score_error_function(
            use_median=False
        ),
        uncertainty_function=uq_evaluation.get_stdev_uncertainty_function(
            use_fancy_quantile_method=False
        ),
        use_median=False, is_error_pos_oriented=False
    )

    figure_object, axes_object = uq_eval_plotting.plot_discard_test(result_dict)

    title_string = 'Discard test for Model B\nMF = {0:.1f}%'.format(
        100 * result_dict[uq_evaluation.MONOTONICITY_FRACTION_KEY]
    )
    axes_object.set_title(title_string)

    output_file_name = '{0:s}/discard_test_poor_model.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _plot_discard_test_good_model(output_dir_name, panel_letter):
    """Plots discard test for good model.

    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param panel_letter: Letter used to label panel.
    :return: output_file_name: Full path to image file where figure was saved.
    """

    target_classes, forecast_prob_matrix = _create_data_good_model()
    forecast_prob_matrix = numpy.expand_dims(forecast_prob_matrix, axis=-2)
    forecast_prob_matrix = numpy.expand_dims(forecast_prob_matrix, axis=-2)
    forecast_prob_matrix = numpy.concatenate(
        (1. - forecast_prob_matrix, forecast_prob_matrix), axis=1
    )

    prediction_dict = {
        prediction_io.TARGET_MATRIX_KEY:
            numpy.expand_dims(target_classes, axis=-1),
        prediction_io.PROBABILITY_MATRIX_KEY: forecast_prob_matrix,
        prediction_io.QUANTILE_LEVELS_KEY: None
    }

    result_dict = uq_evaluation.run_discard_test(
        prediction_dict=prediction_dict,
        discard_fractions=numpy.linspace(0.05, 0.95, num=19),
        error_function=uq_evaluation.get_brier_score_error_function(
            use_median=False
        ),
        uncertainty_function=uq_evaluation.get_stdev_uncertainty_function(
            use_fancy_quantile_method=False
        ),
        use_median=False, is_error_pos_oriented=False
    )

    figure_object, axes_object = uq_eval_plotting.plot_discard_test(result_dict)

    title_string = 'Discard test for Model A\nMF = {0:.1f}%'.format(
        100 * result_dict[uq_evaluation.MONOTONICITY_FRACTION_KEY]
    )
    axes_object.set_title(title_string)

    output_file_name = '{0:s}/discard_test_good_model.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 75
    )
    _overlay_text(
        image_file_name=output_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + 50,
        y_offset_from_top_px=TITLE_FONT_SIZE + 200,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name
    )

    return output_file_name


def _run(output_dir_name):
    """Makes schematic to explain evaluation methods.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    panel_file_names = []
    panel_file_names.append(_make_scatterplot_good_model(
        output_dir_name=output_dir_name, panel_letter='a'
    ))
    panel_file_names.append(_make_scatterplot_poor_model(
        output_dir_name=output_dir_name, panel_letter='b'
    ))
    panel_file_names.append(_plot_attr_diag_good_model(
        output_dir_name=output_dir_name, panel_letter='c'
    ))
    panel_file_names.append(_plot_attr_diag_poor_model(
        output_dir_name=output_dir_name, panel_letter='d'
    ))
    panel_file_names.append(_plot_spread_vs_skill_good_model(
        output_dir_name=output_dir_name, panel_letter='e'
    ))
    panel_file_names.append(_plot_spread_vs_skill_poor_model(
        output_dir_name=output_dir_name, panel_letter='f'
    ))
    panel_file_names.append(_plot_discard_test_good_model(
        output_dir_name=output_dir_name, panel_letter='g'
    ))
    panel_file_names.append(_plot_discard_test_poor_model(
        output_dir_name=output_dir_name, panel_letter='h'
    ))

    for this_file_name in panel_file_names:
        imagemagick_utils.resize_image(
            input_file_name=this_file_name, output_file_name=this_file_name,
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/evaluation_schematic.jpg'.format(
        output_dir_name
    )

    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=3, num_panel_columns=3
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
