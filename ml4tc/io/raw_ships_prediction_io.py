"""IO methods for raw SHIPS predictions."""

import os.path
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4tc.utils import satellite_utils

TIME_FORMAT = '%y%m%d%H'
RI_CUTOFF_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2018-01-01', '%Y-%m-%d'
)
TD_TO_TS_CUTOFF_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2011-06-16-15', '%Y-%m-%d-%H'
)

RI_KEYWORD_OLD_FILE = 'Prob of RI for 30 kt RI threshold='
LGE_KEYWORDS_NEW_FILE = ['V (KT) LGE mod', 'V (KT) LGEM']

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
MAX_INTENSITY_M_S01 = 110.
MIN_TROP_STORM_INTENSITY_M_S01 = 34 * KT_TO_METRES_PER_SECOND
MAX_ABSOLUTE_TROPICAL_LATITUDE_DEG_N = 35.

LEAD_TIMES_KEY = 'lead_times_hours'
FORECAST_LABELS_LAND_KEY = 'forecast_labels_land'
FORECAST_LABELS_LGE_KEY = 'forecast_labels_lge'
CYCLONE_ID_KEY = 'cyclone_id_string'
INIT_TIME_KEY = 'init_time_unix_sec'


def _read_td_to_ts_new_file(ascii_file_name):
    """Reads TD-to-TS prediction from file with new type.

    L = number of lead times

    :param ascii_file_name: Path to input file.
    :return: lead_times_hours: length-L numpy array of lead times.
    :return: forecast_labels_land: length-L numpy array of labels from "land"
        forecast (1 if TD becomes TS, 0 otherwise).
    :return: forecast_labels_lge: length-L numpy array of labels from LGE
        (logistic growth equation) forecast (1 if TD becomes TS, 0 otherwise).
    """

    file_handle = open(ascii_file_name, 'r')

    lead_times_hours = None
    forecast_intensities_land_m_s01 = None
    forecast_intensities_lge_m_s01 = None
    forecast_tropical_flags = None

    for this_line in file_handle.readlines():
        if not (
                lead_times_hours is None
                or forecast_intensities_land_m_s01 is None
                or forecast_intensities_lge_m_s01 is None
                or forecast_tropical_flags is None
        ):
            break

        if lead_times_hours is None:
            if not this_line.strip().startswith('TIME (HR)'):
                continue

            this_line = this_line.strip().replace('TIME (HR)', '')
            lead_times_hours = numpy.array([
                int(word) for word in this_line.split()
            ], dtype=int)

            continue

        if this_line.strip().startswith('V (KT) LAND'):
            this_line = this_line.strip().replace('V (KT) LAND', '')

            these_intensity_strings = [
                word.replace('N/A', 'Inf').replace('DIS', 'Inf')
                for word in this_line.split()
            ]

            forecast_intensities_land_m_s01 = numpy.array(
                [float(s) for s in these_intensity_strings]
            ) * KT_TO_METRES_PER_SECOND

            assert (
                len(forecast_intensities_land_m_s01) == len(lead_times_hours)
            )

            these_flags = numpy.logical_or(
                forecast_intensities_land_m_s01 >= 0,
                forecast_intensities_land_m_s01 <= MAX_INTENSITY_M_S01
            )
            assert numpy.all(numpy.logical_or(
                numpy.isinf(forecast_intensities_land_m_s01),
                these_flags
            ))

            continue

        if any(
                [this_line.strip().startswith(w) for w in LGE_KEYWORDS_NEW_FILE]
        ):
            for w in LGE_KEYWORDS_NEW_FILE:
                this_line = this_line.strip().replace(w, '')

            these_intensity_strings = [
                word.replace('N/A', 'Inf').replace('DIS', 'Inf')
                for word in this_line.split()
            ]

            forecast_intensities_lge_m_s01 = numpy.array(
                [float(s) for s in these_intensity_strings]
            ) * KT_TO_METRES_PER_SECOND

            these_flags = numpy.logical_or(
                forecast_intensities_lge_m_s01 >= 0,
                forecast_intensities_lge_m_s01 <= MAX_INTENSITY_M_S01
            )
            assert numpy.all(numpy.logical_or(
                numpy.isinf(forecast_intensities_lge_m_s01),
                these_flags
            ))

            continue

        if this_line.strip().startswith('Storm Type'):
            this_line = this_line.strip().replace('Storm Type', '')
            forecast_tropical_flags = numpy.array([
                word.upper() == 'TROP' for word in this_line.split()
            ], dtype=bool)

            continue

    assert not (
        lead_times_hours is None
        or forecast_intensities_land_m_s01 is None
        or forecast_intensities_lge_m_s01 is None
        or forecast_tropical_flags is None
    )

    file_handle.close()

    forecast_intensities_land_m_s01[
        numpy.isinf(forecast_intensities_land_m_s01)
    ] = numpy.nan

    forecast_intensities_lge_m_s01[
        numpy.isinf(forecast_intensities_lge_m_s01)
    ] = numpy.nan

    forecast_labels_land = numpy.logical_and(
        forecast_intensities_land_m_s01 >= MIN_TROP_STORM_INTENSITY_M_S01,
        forecast_tropical_flags
    ).astype(int)

    forecast_labels_lge = numpy.logical_and(
        forecast_intensities_lge_m_s01 >= MIN_TROP_STORM_INTENSITY_M_S01,
        forecast_tropical_flags
    ).astype(int)

    return lead_times_hours, forecast_labels_land, forecast_labels_lge


def _read_td_to_ts_old_file(ascii_file_name):
    """Reads TD-to-TS prediction from file with old type.

    L = number of lead times

    :param ascii_file_name: See doc for `_read_td_to_ts_old_file`.
    :return: lead_times_hours: Same.
    :return: forecast_labels_land: Same.
    :return: forecast_labels_lge: Same.
    """

    file_handle = open(ascii_file_name, 'r')

    lead_times_hours = None
    forecast_intensities_land_m_s01 = None
    forecast_intensities_lge_m_s01 = None
    forecast_tropical_flags = None

    for this_line in file_handle.readlines():
        if not (
                lead_times_hours is None
                or forecast_intensities_land_m_s01 is None
                or forecast_intensities_lge_m_s01 is None
                or forecast_tropical_flags is None
        ):
            break

        if lead_times_hours is None:
            if not this_line.strip().startswith('TIME (HR)'):
                continue

            this_line = this_line.strip().replace('TIME (HR)', '')
            lead_times_hours = numpy.array([
                int(word) for word in this_line.split()
            ], dtype=int)

            continue

        if this_line.strip().startswith('V (KT) LAND'):
            this_line = this_line.strip().replace('V (KT) LAND', '')
            these_intensity_strings = [
                word.replace('DIS', 'Inf') for word in this_line.split()
            ]

            forecast_intensities_land_m_s01 = numpy.array(
                [float(s) for s in these_intensity_strings]
            ) * KT_TO_METRES_PER_SECOND

            assert (
                len(forecast_intensities_land_m_s01) == len(lead_times_hours)
            )

            these_flags = numpy.logical_or(
                forecast_intensities_land_m_s01 >= 0,
                forecast_intensities_land_m_s01 <= MAX_INTENSITY_M_S01
            )
            assert numpy.all(numpy.logical_or(
                numpy.isinf(forecast_intensities_land_m_s01),
                these_flags
            ))

            continue

        if this_line.strip().startswith('V (KT) LGE mod'):
            this_line = this_line.strip().replace('V (KT) LGE mod', '')
            these_intensity_strings = [
                word.replace('DIS', 'Inf') for word in this_line.split()
            ]

            forecast_intensities_lge_m_s01 = numpy.array(
                [float(s) for s in these_intensity_strings]
            ) * KT_TO_METRES_PER_SECOND

            assert (
                len(forecast_intensities_lge_m_s01) == len(lead_times_hours)
            )

            these_flags = numpy.logical_or(
                forecast_intensities_lge_m_s01 >= 0,
                forecast_intensities_lge_m_s01 <= MAX_INTENSITY_M_S01
            )
            assert numpy.all(numpy.logical_or(
                numpy.isinf(forecast_intensities_lge_m_s01),
                these_flags
            ))

            continue

        if this_line.strip().startswith('LAT (DEG N)'):
            this_line = this_line.strip().replace('LAT (DEG N)', '')
            these_latitude_strings = [
                word.replace('N/A', 'Inf').replace('xx.x', 'Inf')
                for word in this_line.split()
            ]
            these_latitudes_deg_n = numpy.array([
                float(s) for s in these_latitude_strings
            ])

            assert len(these_latitudes_deg_n) == len(lead_times_hours)
            assert numpy.all(numpy.logical_or(
                numpy.isinf(these_latitudes_deg_n),
                numpy.absolute(these_latitudes_deg_n) <= 90
            ))

            forecast_tropical_flags = (
                numpy.absolute(these_latitudes_deg_n) <=
                MAX_ABSOLUTE_TROPICAL_LATITUDE_DEG_N
            )

            continue

    assert not (
        lead_times_hours is None
        or forecast_intensities_land_m_s01 is None
        or forecast_intensities_lge_m_s01 is None
        or forecast_tropical_flags is None
    )

    file_handle.close()

    forecast_intensities_land_m_s01[
        numpy.isinf(forecast_intensities_land_m_s01)
    ] = numpy.nan

    forecast_intensities_lge_m_s01[
        numpy.isinf(forecast_intensities_lge_m_s01)
    ] = numpy.nan

    forecast_labels_land = numpy.logical_and(
        forecast_intensities_land_m_s01 >= MIN_TROP_STORM_INTENSITY_M_S01,
        forecast_tropical_flags
    ).astype(int)

    forecast_labels_lge = numpy.logical_and(
        forecast_intensities_lge_m_s01 >= MIN_TROP_STORM_INTENSITY_M_S01,
        forecast_tropical_flags
    ).astype(int)

    return lead_times_hours, forecast_labels_land, forecast_labels_lge


def _read_ri_new_file(ascii_file_name):
    """Reads rapid intensification (RI) prediction from file with new type.

    :param ascii_file_name: Path to input file.
    :return: ships_rii_probability: Probability from SHIPS-RII model.
    :return: consensus_probability: Probability from model consensus.
    """

    file_handle = open(ascii_file_name, 'r')

    found_ri_table = False
    ri_30hour_index = -1
    ships_rii_probability = numpy.nan
    consensus_probability = numpy.nan

    for this_line in file_handle.readlines():
        if not (
                numpy.isnan(ships_rii_probability) or
                numpy.isnan(consensus_probability)
        ):
            break

        if not found_ri_table:
            found_ri_table = this_line.strip().startswith(
                'Matrix of RI probabilities'
            )
            continue

        if ri_30hour_index < 0:
            if '|' not in this_line:
                continue

            ri_table_headers = this_line.split('|')
            ri_table_headers = [h.strip() for h in ri_table_headers]
            ri_30hour_index = ri_table_headers.index('30/24')
            continue

        if this_line.strip().startswith('SHIPS-RII:'):
            prob_string = this_line.split()[ri_30hour_index]
            assert prob_string.endswith('%')

            if prob_string.startswith('999'):
                ships_rii_probability = numpy.inf
                continue

            ships_rii_probability = 0.01 * float(prob_string[:-1])
            assert 0. <= ships_rii_probability <= 1.
            continue

        if this_line.strip().startswith('Consensus:'):
            prob_string = this_line.split()[ri_30hour_index]
            assert prob_string.endswith('%')

            if prob_string.startswith('999'):
                consensus_probability = numpy.inf
                continue

            consensus_probability = 0.01 * float(prob_string[:-1])
            assert 0. <= consensus_probability <= 1.

    assert not (
        numpy.isnan(ships_rii_probability) or
        numpy.isnan(consensus_probability)
    )

    file_handle.close()

    if numpy.isinf(ships_rii_probability):
        ships_rii_probability = numpy.nan
    if numpy.isinf(consensus_probability):
        consensus_probability = numpy.nan

    return ships_rii_probability, consensus_probability


def _read_ri_old_file(ascii_file_name):
    """Reads rapid intensification (RI) prediction from file with old type.

    :param ascii_file_name: Path to input file.
    :return: ri_probability: RI probability.
    """

    file_handle = open(ascii_file_name, 'r')
    ri_probability = numpy.nan

    for this_line in file_handle.readlines():
        if not numpy.isnan(ri_probability):
            break

        if not this_line.strip().startswith(RI_KEYWORD_OLD_FILE):
            continue

        prob_string = (
            this_line.strip().replace(RI_KEYWORD_OLD_FILE, '').split()[0]
        )
        assert prob_string.endswith('%')

        if prob_string == '999%':
            ri_probability = numpy.nan
            break

        ri_probability = 0.01 * float(prob_string[:-1])
        assert 0. <= ri_probability <= 1.

    file_handle.close()

    return ri_probability


def _file_name_to_metadata(ships_prediction_file_name):
    """Parses metadata from file name.

    :param ships_prediction_file_name: Path to raw file with SHIPS predictions.
    :return: cyclone_id_string: Cyclone ID.
    :return: init_time_unix_sec: Forecast-initialization time.
    """

    pathless_file_name = os.path.split(ships_prediction_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    words = extensionless_file_name.split('_')
    assert len(words) == 2
    assert words[1] == 'ships'

    metadata_string = words[0]
    assert len(metadata_string) == 14

    init_time_unix_sec = time_conversion.string_to_unix_sec(
        metadata_string[:8], TIME_FORMAT
    )

    cyclone_year = 1900 + int(metadata_string[-2:])
    if cyclone_year < 1970:
        cyclone_year += 100

    cyclone_id_string = satellite_utils.get_cyclone_id(
        year=cyclone_year, basin_id_string=metadata_string[-6:-4],
        cyclone_number=int(metadata_string[-4:-2])
    )

    return cyclone_id_string, init_time_unix_sec


def read_ri_predictions(ascii_file_name):
    """Reads rapid intensification (RI) predictions from file.

    :param ascii_file_name: Path to input file.
    :return: ri_probabilities: length-1 or length-2 numpy array of
        probabilities.  If length-2, the entries are
        [ships_rii_probability, consensus_probability].
    :return: cyclone_id_string: Cyclone ID.
    :return: init_time_unix_sec: Forecast-initialization time.
    """

    cyclone_id_string, init_time_unix_sec = _file_name_to_metadata(
        ascii_file_name
    )

    if init_time_unix_sec >= RI_CUTOFF_TIME_UNIX_SEC:
        ships_rii_probability, consensus_probability = _read_ri_new_file(
            ascii_file_name
        )
        ri_probabilities = numpy.array([
            ships_rii_probability, consensus_probability
        ])
    else:
        ri_probabilities = numpy.array([
            _read_ri_old_file(ascii_file_name)
        ])

    return ri_probabilities, cyclone_id_string, init_time_unix_sec


def read_td_to_ts_predictions(ascii_file_name):
    """Reads TD-to-TS predictions from file.

    :param ascii_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['lead_times_hours']: See doc for `_read_td_to_ts_new_file`.
    prediction_dict['forecast_labels_land']: Same.
    prediction_dict['forecast_labels_lge']: Same.
    prediction_dict['cyclone_id_string']: Cyclone ID.
    prediction_dict['init_time_unix_sec']: Forecast-initialization time.
    """

    cyclone_id_string, init_time_unix_sec = _file_name_to_metadata(
        ascii_file_name
    )

    if init_time_unix_sec >= TD_TO_TS_CUTOFF_TIME_UNIX_SEC:
        lead_times_hours, forecast_labels_land, forecast_labels_lge = (
            _read_td_to_ts_new_file(ascii_file_name)
        )
    else:
        lead_times_hours, forecast_labels_land, forecast_labels_lge = (
            _read_td_to_ts_old_file(ascii_file_name)
        )

    return {
        LEAD_TIMES_KEY: lead_times_hours,
        FORECAST_LABELS_LAND_KEY: forecast_labels_land,
        FORECAST_LABELS_LGE_KEY: forecast_labels_lge,
        CYCLONE_ID_KEY: cyclone_id_string,
        INIT_TIME_KEY: init_time_unix_sec,
    }
