"""Unit tests for raw_ships_io.py."""

import copy
import unittest
import numpy
from ml4tc.io import raw_ships_io

TOLERANCE = 1e-6

DISTANCES_METRES = numpy.array([-200, -100, -50, 0, 100, 150, 200], dtype=float)
DISTANCES_KM = numpy.array([-0.2, -0.1, -0.05, 0, 0.1, 0.15, 0.2])

TEMPERATURES_DECICELSIUS = numpy.array(
    [-200, -100, -50, 0, 100, 150, 200], dtype=float
)
TEMPERATURES_KELVINS = numpy.array([
    253.15, 263.15, 268.15, 273.15, 283.15, 288.15, 293.15
])

PRESSURES_1000MB_DEPARTURES_DECAPASCALS = numpy.array(
    [-200, -100, -50, 0, 100, 150, 200], dtype=float
)
PRESSURES_PASCALS = numpy.array(
    [98000, 99000, 99500, 100000, 101000, 101500, 102000], dtype=float
)

FORECAST_HOUR_LINE_5DAY = (
    '  -12   -6    0    6   12   18   24   30   36   42   48   54   60   66   '
    '72   78   84   90   96  102  108  114  120 TIME'
)
HOUR_INDEX_TO_CHAR_INDICES_5DAY = {
    0: numpy.array([1, 5], dtype=int),
    1: numpy.array([6, 10], dtype=int),
    2: numpy.array([11, 15], dtype=int),
    3: numpy.array([16, 20], dtype=int),
    4: numpy.array([21, 25], dtype=int),
    5: numpy.array([26, 30], dtype=int),
    6: numpy.array([31, 35], dtype=int),
    7: numpy.array([36, 40], dtype=int),
    8: numpy.array([41, 45], dtype=int),
    9: numpy.array([46, 50], dtype=int),
    10: numpy.array([51, 55], dtype=int),
    11: numpy.array([56, 60], dtype=int),
    12: numpy.array([61, 65], dtype=int),
    13: numpy.array([66, 70], dtype=int),
    14: numpy.array([71, 75], dtype=int),
    15: numpy.array([76, 80], dtype=int),
    16: numpy.array([81, 85], dtype=int),
    17: numpy.array([86, 90], dtype=int),
    18: numpy.array([91, 95], dtype=int),
    19: numpy.array([96, 100], dtype=int),
    20: numpy.array([101, 105], dtype=int),
    21: numpy.array([106, 110], dtype=int),
    22: numpy.array([111, 115], dtype=int)
}

FORECAST_HOUR_LINE_7DAY = (
    '  -12   -6    0    6   12   18   24   30   36   42   48   54   60   66   '
    '72   78   84   90   96  102  108  114  120  126  132  138  144  150  156  '
    '162  168 TIME'
)
HOUR_INDEX_TO_CHAR_INDICES_7DAY = copy.deepcopy(HOUR_INDEX_TO_CHAR_INDICES_5DAY)

HOUR_INDEX_TO_CHAR_INDICES_7DAY.update({
    23: numpy.array([116, 120], dtype=int),
    24: numpy.array([121, 125], dtype=int),
    25: numpy.array([126, 130], dtype=int),
    26: numpy.array([131, 135], dtype=int),
    27: numpy.array([136, 140], dtype=int),
    28: numpy.array([141, 145], dtype=int),
    29: numpy.array([146, 150], dtype=int),
    30: numpy.array([151, 155], dtype=int)
})

FORECAST_HOUR_LINE_BAD = (
    '   -6    0    6   12   18   24   30   36   42   48   54   60   66   '
    '72   78   84   90   96  102  108  114  120  126  132  138  144  150  156  '
    '162  168 TIME'
)

ORIG_CYCLONE_ID_STRING = 'WP011990'
CYCLONE_ID_STRING = '1990WP01'


class RawShipsIoTests(unittest.TestCase):
    """Each method is a unit test for raw_ships_io.py."""

    def test_get_multiply_function(self):
        """Ensures correct output from _get_multiply_function."""

        this_function = raw_ships_io._get_multiply_function(0.001)
        these_distances_km = this_function(DISTANCES_METRES)

        self.assertTrue(numpy.allclose(
            these_distances_km, DISTANCES_KM, atol=TOLERANCE
        ))

    def test_decicelsius_to_kelvins(self):
        """Ensures correct output from _decicelsius_to_kelvins."""

        these_temps_kelvins = raw_ships_io._decicelsius_to_kelvins(
            TEMPERATURES_DECICELSIUS
        )
        self.assertTrue(numpy.allclose(
            these_temps_kelvins, TEMPERATURES_KELVINS, atol=TOLERANCE
        ))

    def test_pressure_from_1000mb_departure_to_pa(self):
        """Ensures correct output from _pressure_from_1000mb_departure_to_pa."""

        these_pressures_pa = raw_ships_io._pressure_from_1000mb_departure_to_pa(
            PRESSURES_1000MB_DEPARTURES_DECAPASCALS
        )
        self.assertTrue(numpy.allclose(
            these_pressures_pa, PRESSURES_PASCALS, atol=TOLERANCE
        ))

    def test_forecast_hour_to_chars_5day(self):
        """Ensures correct output from _forecast_hour_to_chars.

        In this case, assuming 5-day file rather than 7-day file.
        """

        this_dict = raw_ships_io._forecast_hour_to_chars(
            forecast_hour_line=FORECAST_HOUR_LINE_5DAY, seven_day_flag=False
        )

        these_keys = list(this_dict.keys())
        expected_keys = list(HOUR_INDEX_TO_CHAR_INDICES_5DAY.keys())
        self.assertTrue(set(these_keys) == set(expected_keys))

        for this_key in these_keys:
            self.assertTrue(numpy.array_equal(
                this_dict[this_key], HOUR_INDEX_TO_CHAR_INDICES_5DAY[this_key]
            ))

    def test_forecast_hour_to_chars_7day(self):
        """Ensures correct output from _forecast_hour_to_chars.

        In this case, assuming 7-day file rather than 5-day file.
        """

        this_dict = raw_ships_io._forecast_hour_to_chars(
            forecast_hour_line=FORECAST_HOUR_LINE_7DAY, seven_day_flag=True
        )

        these_keys = list(this_dict.keys())
        expected_keys = list(HOUR_INDEX_TO_CHAR_INDICES_7DAY.keys())
        self.assertTrue(set(these_keys) == set(expected_keys))

        for this_key in these_keys:
            self.assertTrue(numpy.array_equal(
                this_dict[this_key], HOUR_INDEX_TO_CHAR_INDICES_7DAY[this_key]
            ))

    def test_forecast_hour_to_chars_bad(self):
        """Ensures correct output from _forecast_hour_to_chars.

        In this case, line is badly formatted.
        """

        with self.assertRaises(AssertionError):
            raw_ships_io._forecast_hour_to_chars(
                forecast_hour_line=FORECAST_HOUR_LINE_BAD, seven_day_flag=True
            )

    def test_reformat_cyclone_id(self):
        """Ensures correct output from _reformat_cyclone_id."""

        this_cyclone_id_string = raw_ships_io._reformat_cyclone_id(
            ORIG_CYCLONE_ID_STRING
        )
        self.assertTrue(this_cyclone_id_string == CYCLONE_ID_STRING)


if __name__ == '__main__':
    unittest.main()
