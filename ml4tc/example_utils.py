"""Helper methods for learning examples."""

import os
import sys
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_periods
import error_checking
import ships_io
import satellite_utils

SATELLITE_GRID_ROW_DIM = satellite_utils.GRID_ROW_DIM
SATELLITE_GRID_COLUMN_DIM = satellite_utils.GRID_COLUMN_DIM
SATELLITE_TIME_DIM = satellite_utils.TIME_DIM
SATELLITE_PREDICTOR_UNGRIDDED_DIM = 'satellite_predictor_name_ungridded'
SATELLITE_PREDICTOR_GRIDDED_DIM = 'satellite_predictor_name_gridded'

SHIPS_FORECAST_HOUR_DIM = ships_io.FORECAST_HOUR_DIM
SHIPS_THRESHOLD_DIM = ships_io.THRESHOLD_DIM
SHIPS_LAG_TIME_DIM = ships_io.LAG_TIME_DIM
SHIPS_VALID_TIME_DIM = ships_io.VALID_TIME_DIM
SHIPS_PREDICTOR_LAGGED_DIM = 'ships_predictor_name_lagged'
SHIPS_PREDICTOR_FORECAST_DIM = 'ships_predictor_name_forecast'

SATELLITE_PREDICTORS_UNGRIDDED_KEY = 'satellite_predictors_ungridded'
SATELLITE_PREDICTORS_GRIDDED_KEY = 'satellite_predictors_gridded'

SHIPS_PREDICTORS_LAGGED_KEY = 'ships_predictors_lagged'
SHIPS_PREDICTORS_FORECAST_KEY = 'ships_predictors_forecast'

SATELLITE_METADATA_KEYS = [
    satellite_utils.SATELLITE_NUMBER_KEY,
    satellite_utils.BAND_NUMBER_KEY,
    satellite_utils.BAND_WAVELENGTH_KEY,
    satellite_utils.SATELLITE_LONGITUDE_KEY,
    satellite_utils.CYCLONE_ID_KEY,
    satellite_utils.STORM_TYPE_KEY,
    satellite_utils.STORM_NAME_KEY,
    satellite_utils.STORM_LATITUDE_KEY,
    satellite_utils.STORM_LONGITUDE_KEY,
    satellite_utils.STORM_INTENSITY_NUM_KEY,
    satellite_utils.GRID_LATITUDE_KEY,
    satellite_utils.GRID_LONGITUDE_KEY
]

SHIPS_METADATA_KEYS = [
    ships_io.CYCLONE_ID_KEY,
    ships_io.STORM_LATITUDE_KEY,
    ships_io.STORM_LONGITUDE_KEY,
    ships_io.STORM_TYPE_KEY,
    ships_io.FORECAST_LATITUDE_KEY,
    ships_io.FORECAST_LONGITUDE_KEY,
    ships_io.VORTEX_LATITUDE_KEY,
    ships_io.VORTEX_LONGITUDE_KEY,
    ships_io.THRESHOLD_EXCEEDANCE_KEY
]


def merge_data(satellite_table_xarray, ships_table_xarray):
    """Merges satellite and SHIPS data.

    :param satellite_table_xarray: Table returned by `satellite_io.read_file`.
    :param ships_table_xarray: Table returned by `ships_io.read_file`.
    :return: example_table_xarray: Table created by merging inputs.  Metadata in
        table should make fields self-explanatory.
    """

    # Ensure that both tables contain data for the same cyclone.
    satellite_cyclone_id_strings = numpy.array(
        satellite_table_xarray[satellite_utils.CYCLONE_ID_KEY].values
    )
    assert len(numpy.unique(satellite_cyclone_id_strings)) == 1

    ships_cyclone_id_strings = numpy.array(
        ships_table_xarray[ships_io.CYCLONE_ID_KEY].values
    )
    assert len(numpy.unique(ships_cyclone_id_strings)) == 1

    try:
        satellite_cyclone_id_string = (
            satellite_cyclone_id_strings[0].decode('UTF-8')
        )
    except AttributeError:
        satellite_cyclone_id_string = satellite_cyclone_id_strings[0]

    try:
        ships_cyclone_id_string = ships_cyclone_id_strings[0].decode('UTF-8')
    except AttributeError:
        ships_cyclone_id_string = ships_cyclone_id_strings[0]

    assert satellite_cyclone_id_string == ships_cyclone_id_string

    # Do actual stuff.
    satellite_metadata_dict = satellite_table_xarray.to_dict()['coords']
    example_metadata_dict = dict()

    for this_key in satellite_metadata_dict:
        example_metadata_dict[this_key] = (
            satellite_metadata_dict[this_key]['data']
        )

    ships_metadata_dict = ships_table_xarray.to_dict()['coords']
    del ships_metadata_dict[ships_io.STORM_OBJECT_DIM]

    for this_key in ships_metadata_dict:
        example_metadata_dict[this_key] = (
            ships_metadata_dict[this_key]['data']
        )

    satellite_dict = satellite_table_xarray.to_dict()['data_vars']
    example_dict = dict()

    satellite_predictor_names_ungridded = []
    satellite_predictor_matrix_ungridded = numpy.array([])

    for this_key in satellite_dict:
        if this_key in SATELLITE_METADATA_KEYS:
            example_dict[this_key] = (
                satellite_dict[this_key]['dims'],
                satellite_dict[this_key]['data']
            )

            continue

        if this_key == satellite_utils.BRIGHTNESS_TEMPERATURE_KEY:
            these_dim = (
                satellite_dict[this_key]['dims'] +
                (SATELLITE_PREDICTOR_GRIDDED_DIM,)
            )

            example_dict[SATELLITE_PREDICTORS_GRIDDED_KEY] = (
                these_dim,
                numpy.expand_dims(satellite_dict[this_key]['data'], axis=-1)
            )

            continue

        satellite_predictor_names_ungridded.append(this_key)
        new_matrix = numpy.expand_dims(
            satellite_dict[this_key]['data'], axis=-1
        )

        if satellite_predictor_matrix_ungridded.size == 0:
            satellite_predictor_matrix_ungridded = new_matrix + 0.
        else:
            satellite_predictor_matrix_ungridded = numpy.concatenate(
                (satellite_predictor_matrix_ungridded, new_matrix), axis=-1
            )

    example_metadata_dict.update({
        SATELLITE_PREDICTOR_GRIDDED_DIM:
            numpy.array([satellite_utils.BRIGHTNESS_TEMPERATURE_KEY]),
        SATELLITE_PREDICTOR_UNGRIDDED_DIM:
            numpy.array(satellite_predictor_names_ungridded)
    })

    these_dim = (SATELLITE_TIME_DIM, SATELLITE_PREDICTOR_UNGRIDDED_DIM)
    example_dict[SATELLITE_PREDICTORS_UNGRIDDED_KEY] = (
        these_dim, satellite_predictor_matrix_ungridded
    )

    ships_dict = ships_table_xarray.to_dict()['data_vars']
    example_metadata_dict[SHIPS_VALID_TIME_DIM] = (
        ships_dict[ships_io.VALID_TIME_KEY]['data']
    )
    del ships_dict[ships_io.VALID_TIME_KEY]

    ships_predictor_names_forecast = []
    ships_predictor_names_lagged = []
    ships_predictor_matrix_forecast = numpy.array([])
    ships_predictor_matrix_lagged = numpy.array([])

    for this_key in ships_dict:
        if this_key in SHIPS_METADATA_KEYS:
            example_dict[this_key] = (
                ships_dict[this_key]['dims'],
                ships_dict[this_key]['data']
            )

            continue

        new_matrix = numpy.expand_dims(ships_dict[this_key]['data'], axis=-1)

        if SHIPS_LAG_TIME_DIM in ships_dict[this_key]['dims']:
            ships_predictor_names_lagged.append(this_key)

            if ships_predictor_matrix_lagged.size == 0:
                ships_predictor_matrix_lagged = new_matrix + 0.
            else:
                ships_predictor_matrix_lagged = numpy.concatenate(
                    (ships_predictor_matrix_lagged, new_matrix), axis=-1
                )

            continue

        new_matrix = numpy.expand_dims(ships_dict[this_key]['data'], axis=-1)

        if SHIPS_FORECAST_HOUR_DIM in ships_dict[this_key]['dims']:
            ships_predictor_names_forecast.append(this_key)

            if ships_predictor_matrix_forecast.size == 0:
                ships_predictor_matrix_forecast = new_matrix + 0.
            else:
                ships_predictor_matrix_forecast = numpy.concatenate(
                    (ships_predictor_matrix_forecast, new_matrix), axis=-1
                )

    example_metadata_dict.update({
        SHIPS_PREDICTOR_LAGGED_DIM: numpy.array(ships_predictor_names_lagged),
        SHIPS_PREDICTOR_FORECAST_DIM:
            numpy.array(ships_predictor_names_forecast)
    })

    these_dim = (
        SHIPS_VALID_TIME_DIM, SHIPS_LAG_TIME_DIM, SHIPS_PREDICTOR_LAGGED_DIM
    )
    example_dict[SHIPS_PREDICTORS_LAGGED_KEY] = (
        these_dim, ships_predictor_matrix_lagged
    )

    these_dim = (
        SHIPS_VALID_TIME_DIM, SHIPS_FORECAST_HOUR_DIM,
        SHIPS_PREDICTOR_FORECAST_DIM
    )
    example_dict[SHIPS_PREDICTORS_FORECAST_KEY] = (
        these_dim, ships_predictor_matrix_forecast
    )

    return xarray.Dataset(data_vars=example_dict, coords=example_metadata_dict)


def subset_satellite_times(example_table_xarray, time_interval_sec):
    """Subsets valid times for satellite data.

    :param example_table_xarray: Table in format created by `merge_data`.
        Metadata in table should make fields self-explanatory.
    :param time_interval_sec: Desired time interval for satellite data
        (seconds).
    :return: example_table_xarray: Same as input but maybe with fewer satellite
        times.
    """

    error_checking.assert_is_integer(time_interval_sec)
    error_checking.assert_is_geq(time_interval_sec, 600)
    error_checking.assert_is_leq(time_interval_sec, 3600)

    all_times_unix_sec = example_table_xarray.coords[SATELLITE_TIME_DIM].values
    desired_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=numpy.min(all_times_unix_sec),
        end_time_unix_sec=numpy.max(all_times_unix_sec),
        time_interval_sec=time_interval_sec, include_endpoint=True
    )

    good_times_unix_sec = []

    for t in desired_times_unix_sec:
        this_index = numpy.argmin(numpy.absolute(all_times_unix_sec - t))
        good_times_unix_sec.append(all_times_unix_sec[this_index])

    good_times_unix_sec = numpy.unique(
        numpy.array(good_times_unix_sec, dtype=int)
    )

    return example_table_xarray.sel(
        indexers={SATELLITE_TIME_DIM: good_times_unix_sec}
    )
