"""General utility methods."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

DEGREES_TO_RADIANS = numpy.pi / 180


def speed_and_heading_to_uv(storm_speeds_m_s01, storm_headings_deg):
    """Converts storm motion from speed-direction to u-v.

    N = number of storm objects

    :param storm_speeds_m_s01: length-N numpy array of storm speeds (metres per
        second).
    :param storm_headings_deg: length-N numpy array of storm headings (degrees).
    :return: u_motions_m_s01: length-N numpy array of eastward motions (metres
        per second).
    :return: v_motions_m_s01: length-N numpy array of northward motions (metres
        per second).
    """

    error_checking.assert_is_geq_numpy_array(storm_speeds_m_s01, 0.)
    error_checking.assert_is_numpy_array(storm_speeds_m_s01, num_dimensions=1)

    error_checking.assert_is_geq_numpy_array(storm_headings_deg, 0.)
    error_checking.assert_is_leq_numpy_array(storm_headings_deg, 360.)
    error_checking.assert_is_numpy_array(
        storm_headings_deg,
        exact_dimensions=numpy.array(storm_speeds_m_s01.shape, dtype=int)
    )

    u_motions_m_s01 = storm_speeds_m_s01 * numpy.sin(
        storm_headings_deg * DEGREES_TO_RADIANS
    )
    v_motions_m_s01 = storm_speeds_m_s01 * numpy.cos(
        storm_headings_deg * DEGREES_TO_RADIANS
    )

    return u_motions_m_s01, v_motions_m_s01
