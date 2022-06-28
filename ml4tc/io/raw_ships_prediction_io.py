"""IO methods for raw SHIPS predictions."""

import os.path
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4tc.utils import satellite_utils

TIME_FORMAT = '%y%m%d%H'
CUTOFF_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2018-01-01', '%Y-%m-%d'
)

OLD_FILE_KEYWORD = 'Prob of RI for 30 kt RI threshold='


def _read_new_file_type(ascii_file_name):
    """Reads rapid intensification (RI) predictions from file with new type.

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

    file_handle.close()

    if numpy.isinf(ships_rii_probability):
        ships_rii_probability = numpy.nan
    if numpy.isinf(consensus_probability):
        consensus_probability = numpy.nan

    return ships_rii_probability, consensus_probability


def _read_old_file_type(ascii_file_name):
    """Reads rapid intensification (RI) predictions from file with old type.

    :param ascii_file_name: Path to input file.
    :return: ri_probability: RI probability.
    """

    file_handle = open(ascii_file_name, 'r')
    ri_probability = numpy.nan

    for this_line in file_handle.readlines():
        if not numpy.isnan(ri_probability):
            break

        if not this_line.strip().startswith(OLD_FILE_KEYWORD):
            continue

        prob_string = this_line.strip().replace(OLD_FILE_KEYWORD, '').split()[0]
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


def read_file(ascii_file_name):
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

    if init_time_unix_sec >= CUTOFF_TIME_UNIX_SEC:
        ships_rii_probability, consensus_probability = _read_new_file_type(
            ascii_file_name
        )
        ri_probabilities = numpy.array([
            ships_rii_probability, consensus_probability
        ])
    else:
        ri_probabilities = numpy.array([
            _read_old_file_type(ascii_file_name)
        ])

    return ri_probabilities, cyclone_id_string, init_time_unix_sec
