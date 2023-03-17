"""Helper methods for learning examples."""

import numpy
import xarray
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import ships_io
from ml4tc.utils import satellite_utils
from ml4tc.utils import general_utils

NCODA_SST_CUTOFF_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2013-01-01', '%Y-%m-%d'
)

GRID_SPACING_METRES = satellite_utils.GRID_SPACING_METRES
STORM_INTENSITY_KEY = ships_io.STORM_INTENSITY_KEY

SATELLITE_GRID_ROW_DIM = satellite_utils.GRID_ROW_DIM
SATELLITE_GRID_COLUMN_DIM = satellite_utils.GRID_COLUMN_DIM
SATELLITE_TIME_DIM = satellite_utils.TIME_DIM
SATELLITE_METADATA_TIME_DIM = 'satellite_metadata_time_unix_sec'
SATELLITE_PREDICTOR_UNGRIDDED_DIM = 'satellite_predictor_name_ungridded'
SATELLITE_PREDICTOR_GRIDDED_DIM = 'satellite_predictor_name_gridded'

SHIPS_FORECAST_HOUR_DIM = ships_io.FORECAST_HOUR_DIM
SHIPS_THRESHOLD_DIM = ships_io.THRESHOLD_DIM
SHIPS_LAG_TIME_DIM = ships_io.LAG_TIME_DIM
SHIPS_VALID_TIME_DIM = ships_io.VALID_TIME_DIM
SHIPS_METADATA_TIME_DIM = 'ships_metadata_time_unix_sec'
SHIPS_PREDICTOR_LAGGED_DIM = 'ships_predictor_name_lagged'
SHIPS_PREDICTOR_FORECAST_DIM = 'ships_predictor_name_forecast'

SATELLITE_PREDICTORS_UNGRIDDED_KEY = 'satellite_predictors_ungridded'
SATELLITE_PREDICTORS_GRIDDED_KEY = 'satellite_predictors_gridded'

SHIPS_PREDICTORS_LAGGED_KEY = 'ships_predictors_lagged'
SHIPS_PREDICTORS_FORECAST_KEY = 'ships_predictors_forecast'

SATELLITE_NUMBER_KEY = satellite_utils.SATELLITE_NUMBER_KEY
BAND_NUMBER_KEY = satellite_utils.BAND_NUMBER_KEY
BAND_WAVELENGTH_KEY = satellite_utils.BAND_WAVELENGTH_KEY
SATELLITE_LONGITUDE_KEY = satellite_utils.SATELLITE_LONGITUDE_KEY
SATELLITE_CYCLONE_ID_KEY = satellite_utils.CYCLONE_ID_KEY
SATELLITE_STORM_TYPE_KEY = satellite_utils.STORM_TYPE_KEY
STORM_NAME_KEY = satellite_utils.STORM_NAME_KEY
SATELLITE_STORM_LATITUDE_KEY = satellite_utils.STORM_LATITUDE_KEY
SATELLITE_STORM_LONGITUDE_KEY = satellite_utils.STORM_LONGITUDE_KEY
SATELLITE_STORM_MOTION_U_KEY = satellite_utils.STORM_MOTION_U_KEY
SATELLITE_STORM_MOTION_V_KEY = satellite_utils.STORM_MOTION_V_KEY
STORM_INTENSITY_NUM_KEY = satellite_utils.STORM_INTENSITY_NUM_KEY
GRID_LATITUDE_KEY = satellite_utils.GRID_LATITUDE_KEY
GRID_LONGITUDE_KEY = satellite_utils.GRID_LONGITUDE_KEY

SATELLITE_METADATA_KEYS = [
    SATELLITE_NUMBER_KEY,
    BAND_NUMBER_KEY,
    BAND_WAVELENGTH_KEY,
    SATELLITE_LONGITUDE_KEY,
    SATELLITE_CYCLONE_ID_KEY,
    SATELLITE_STORM_TYPE_KEY,
    STORM_NAME_KEY,
    SATELLITE_STORM_LATITUDE_KEY,
    SATELLITE_STORM_LONGITUDE_KEY,
    SATELLITE_STORM_MOTION_U_KEY,
    SATELLITE_STORM_MOTION_V_KEY,
    STORM_INTENSITY_NUM_KEY,
    GRID_LATITUDE_KEY,
    GRID_LONGITUDE_KEY
]

SATELLITE_METADATA_AND_FORECAST_KEYS = [
    SATELLITE_STORM_MOTION_U_KEY,
    SATELLITE_STORM_MOTION_V_KEY
]

SHIPS_CYCLONE_ID_KEY = ships_io.CYCLONE_ID_KEY
SHIPS_STORM_LATITUDE_KEY = ships_io.STORM_LATITUDE_KEY
SHIPS_STORM_LONGITUDE_KEY = ships_io.STORM_LONGITUDE_KEY
SHIPS_STORM_TYPE_KEY = ships_io.STORM_TYPE_KEY
FORECAST_LATITUDE_KEY = ships_io.FORECAST_LATITUDE_KEY
FORECAST_LONGITUDE_KEY = ships_io.FORECAST_LONGITUDE_KEY
VORTEX_LATITUDE_KEY = ships_io.VORTEX_LATITUDE_KEY
VORTEX_LONGITUDE_KEY = ships_io.VORTEX_LONGITUDE_KEY
THRESHOLD_EXCEEDANCE_KEY = ships_io.THRESHOLD_EXCEEDANCE_KEY
SRH_1000TO700MB_OUTER_RING_KEY = ships_io.SRH_1000TO700MB_OUTER_RING_KEY
SRH_1000TO500MB_OUTER_RING_KEY = ships_io.SRH_1000TO500MB_OUTER_RING_KEY
V_WIND_200MB_INNER_RING_KEY = ships_io.V_WIND_200MB_INNER_RING_KEY
V_MOTION_KEY = ships_io.V_MOTION_KEY
V_MOTION_1000TO100MB_KEY = ships_io.V_MOTION_1000TO100MB_KEY
V_MOTION_OPTIMAL_KEY = ships_io.V_MOTION_OPTIMAL_KEY
MEAN_TAN_WIND_850MB_0TO600KM_KEY = ships_io.MEAN_TAN_WIND_850MB_0TO600KM_KEY
MAX_TAN_WIND_850MB_KEY = ships_io.MAX_TAN_WIND_850MB_KEY
MEAN_TAN_WIND_1000MB_500KM_KEY = ships_io.MEAN_TAN_WIND_1000MB_500KM_KEY
MEAN_TAN_WIND_850MB_500KM_KEY = ships_io.MEAN_TAN_WIND_850MB_500KM_KEY
MEAN_TAN_WIND_500MB_500KM_KEY = ships_io.MEAN_TAN_WIND_500MB_500KM_KEY
MEAN_TAN_WIND_300MB_500KM_KEY = ships_io.MEAN_TAN_WIND_300MB_500KM_KEY
VORTICITY_850MB_BIG_RING_KEY = ships_io.VORTICITY_850MB_BIG_RING_KEY

SHIPS_METADATA_KEYS = [
    SHIPS_CYCLONE_ID_KEY,
    SHIPS_STORM_LATITUDE_KEY,
    SHIPS_STORM_LONGITUDE_KEY,
    SHIPS_STORM_TYPE_KEY,
    FORECAST_LATITUDE_KEY,
    FORECAST_LONGITUDE_KEY,
    VORTEX_LATITUDE_KEY,
    VORTEX_LONGITUDE_KEY,
    THRESHOLD_EXCEEDANCE_KEY,
    SRH_1000TO700MB_OUTER_RING_KEY,
    SRH_1000TO500MB_OUTER_RING_KEY,
    V_WIND_200MB_INNER_RING_KEY,
    V_MOTION_KEY,
    V_MOTION_1000TO100MB_KEY,
    V_MOTION_OPTIMAL_KEY,
    MEAN_TAN_WIND_850MB_0TO600KM_KEY,
    MAX_TAN_WIND_850MB_KEY,
    MEAN_TAN_WIND_1000MB_500KM_KEY,
    MEAN_TAN_WIND_850MB_500KM_KEY,
    MEAN_TAN_WIND_500MB_500KM_KEY,
    MEAN_TAN_WIND_300MB_500KM_KEY,
    VORTICITY_850MB_BIG_RING_KEY
]

SHIPS_METADATA_AND_FORECAST_KEYS = [
    FORECAST_LATITUDE_KEY,
    SRH_1000TO700MB_OUTER_RING_KEY,
    SRH_1000TO500MB_OUTER_RING_KEY,
    V_WIND_200MB_INNER_RING_KEY,
    V_MOTION_KEY,
    V_MOTION_1000TO100MB_KEY,
    V_MOTION_OPTIMAL_KEY,
    MEAN_TAN_WIND_850MB_0TO600KM_KEY,
    MAX_TAN_WIND_850MB_KEY,
    MEAN_TAN_WIND_1000MB_500KM_KEY,
    MEAN_TAN_WIND_850MB_500KM_KEY,
    MEAN_TAN_WIND_500MB_500KM_KEY,
    MEAN_TAN_WIND_300MB_500KM_KEY,
    VORTICITY_850MB_BIG_RING_KEY
]


def _merge_lagged_ships_variables(example_table_xarray):
    """Merges lagged SHIPS variables.

    For each variable, the priority list is:

    - 0-hour lag time if available,
    - else 1.5-hour lag time if available,
    - else 3-hour lag time if available,
    - else climatology if available.

    :param example_table_xarray: xarray table in format created by `merge_data`.
    :return: example_table_xarray: Same but with additional SHIPS forecast
        variable.
    """

    predictor_matrix = example_table_xarray[SHIPS_PREDICTORS_LAGGED_KEY].values
    lag_times_hours = example_table_xarray.coords[SHIPS_LAG_TIME_DIM].values

    real_indices = numpy.where(
        numpy.invert(numpy.isnan(lag_times_hours))
    )[0]
    predictor_matrix = predictor_matrix[:, real_indices, :]
    lag_times_hours = lag_times_hours[real_indices]

    lag_times_hours[lag_times_hours < 0] = numpy.inf
    sort_indices = numpy.argsort(lag_times_hours)

    merged_predictor_matrix = numpy.full(
        (predictor_matrix.shape[0], predictor_matrix.shape[2]), numpy.nan
    )

    for j in sort_indices:
        nan_flag_matrix = numpy.isnan(merged_predictor_matrix)
        merged_predictor_matrix[nan_flag_matrix] = (
            predictor_matrix[:, j, :][nan_flag_matrix]
        )

    predictor_matrix = numpy.concatenate((
        predictor_matrix,
        numpy.expand_dims(merged_predictor_matrix, axis=-2)
    ), axis=-2)

    lag_times_hours = numpy.concatenate((
        lag_times_hours, numpy.full(1, numpy.nan)
    ))

    example_table_xarray = example_table_xarray.drop(
        labels=SHIPS_PREDICTORS_LAGGED_KEY
    )
    example_table_xarray = example_table_xarray.assign_coords({
        SHIPS_LAG_TIME_DIM: lag_times_hours
    })

    these_dim = (
        SHIPS_VALID_TIME_DIM, SHIPS_LAG_TIME_DIM,
        SHIPS_PREDICTOR_LAGGED_DIM
    )
    return example_table_xarray.assign({
        SHIPS_PREDICTORS_LAGGED_KEY: (these_dim, predictor_matrix)
    })


def _create_merged_sst_variable(example_table_xarray):
    """Creates merged SST (sea-surface temperature) variable.

    :param example_table_xarray: xarray table in format created by `merge_data`.
    :return: example_table_xarray: Same but with additional SHIPS forecast
        variable.
    """

    all_predictor_names = example_table_xarray.coords[
        SHIPS_PREDICTOR_FORECAST_DIM
    ].values.tolist()

    if ships_io.MERGED_SST_KEY in all_predictor_names:
        merged_sst_index = all_predictor_names.index(ships_io.MERGED_SST_KEY)
    else:
        merged_sst_index = -1

    original_var_names = [
        ships_io.NCODA_SST_KEY,
        ships_io.REYNOLDS_SST_DAILY_KEY,
        ships_io.REYNOLDS_SST_KEY,  # Real-time SHIPS has this.
        ships_io.CLIMO_SST_KEY  # Real-time SHIPS has this.
    ]
    original_var_indices = numpy.array([
        all_predictor_names.index(n) for n in original_var_names
    ], dtype=int)

    original_sst_matrix_kelvins = example_table_xarray[
        SHIPS_PREDICTORS_FORECAST_KEY
    ].values[..., original_var_indices]

    init_times_unix_sec = (
        example_table_xarray.coords[SHIPS_VALID_TIME_DIM].values
    )
    early_indices = numpy.where(
        init_times_unix_sec < NCODA_SST_CUTOFF_TIME_UNIX_SEC
    )[0]
    original_sst_matrix_kelvins[..., 0][early_indices, :] = numpy.nan

    merged_sst_matrix_kelvins = original_sst_matrix_kelvins[..., 0]

    for k in range(1, len(original_var_indices)):
        nan_flag_matrix = numpy.isnan(merged_sst_matrix_kelvins)
        merged_sst_matrix_kelvins[nan_flag_matrix] = (
            original_sst_matrix_kelvins[..., k][nan_flag_matrix]
        )

    if merged_sst_index == -1:
        new_matrix = numpy.concatenate((
            example_table_xarray[SHIPS_PREDICTORS_FORECAST_KEY].values,
            numpy.expand_dims(merged_sst_matrix_kelvins, axis=-1)
        ), axis=-1)

        example_table_xarray = example_table_xarray.drop(
            labels=SHIPS_PREDICTORS_FORECAST_KEY
        )
        example_table_xarray = example_table_xarray.assign_coords({
            SHIPS_PREDICTOR_FORECAST_DIM:
                all_predictor_names + [ships_io.MERGED_SST_KEY]
        })

        these_dim = (
            SHIPS_VALID_TIME_DIM, SHIPS_FORECAST_HOUR_DIM,
            SHIPS_PREDICTOR_FORECAST_DIM
        )
        example_table_xarray = example_table_xarray.assign({
            SHIPS_PREDICTORS_FORECAST_KEY: (these_dim, new_matrix)
        })
    else:
        example_table_xarray[SHIPS_PREDICTORS_FORECAST_KEY].values[
            ..., merged_sst_index
        ] = merged_sst_matrix_kelvins

    return example_table_xarray


def _create_merged_ohc_variable(example_table_xarray):
    """Creates merged OHC (ocean heat content) variable.

    :param example_table_xarray: xarray table in format created by `merge_data`.
    :return: example_table_xarray: Same but with additional SHIPS forecast
        variable.
    """

    all_predictor_names = example_table_xarray.coords[
        SHIPS_PREDICTOR_FORECAST_DIM
    ].values.tolist()

    if ships_io.MERGED_OHC_KEY in all_predictor_names:
        merged_ohc_index = all_predictor_names.index(ships_io.MERGED_OHC_KEY)
    else:
        merged_ohc_index = -1

    original_var_names = [
        ships_io.NCODA_OHC_26C_KEY,
        ships_io.SATELLITE_OHC_KEY,  # Real-time SHIPS has this.
        ships_io.OHC_FROM_SST_AND_CLIMO_KEY,  # Real-time SHIPS has this.
        ships_io.NCODA_OHC_26C_CLIMO_KEY,
        ships_io.CLIMO_OHC_KEY
    ]

    original_var_indices = numpy.array([
        all_predictor_names.index(n) for n in original_var_names
    ], dtype=int)

    original_ohc_matrix_j_m02 = example_table_xarray[
        SHIPS_PREDICTORS_FORECAST_KEY
    ].values[..., original_var_indices]

    merged_ohc_matrix_j_m02 = original_ohc_matrix_j_m02[..., 0]

    for k in range(1, len(original_var_indices)):
        nan_flag_matrix = numpy.isnan(merged_ohc_matrix_j_m02)
        merged_ohc_matrix_j_m02[nan_flag_matrix] = (
            original_ohc_matrix_j_m02[..., k][nan_flag_matrix]
        )

    if merged_ohc_index == -1:
        new_matrix = numpy.concatenate((
            example_table_xarray[SHIPS_PREDICTORS_FORECAST_KEY].values,
            numpy.expand_dims(merged_ohc_matrix_j_m02, axis=-1)
        ), axis=-1)

        example_table_xarray = example_table_xarray.drop(
            labels=SHIPS_PREDICTORS_FORECAST_KEY
        )
        example_table_xarray = example_table_xarray.assign_coords({
            SHIPS_PREDICTOR_FORECAST_DIM:
                all_predictor_names + [ships_io.MERGED_OHC_KEY]
        })

        these_dim = (
            SHIPS_VALID_TIME_DIM, SHIPS_FORECAST_HOUR_DIM,
            SHIPS_PREDICTOR_FORECAST_DIM
        )
        example_table_xarray = example_table_xarray.assign({
            SHIPS_PREDICTORS_FORECAST_KEY: (these_dim, new_matrix)
        })
    else:
        example_table_xarray[SHIPS_PREDICTORS_FORECAST_KEY].values[
            ..., merged_ohc_index
        ] = merged_ohc_matrix_j_m02

    return example_table_xarray


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

    satellite_northern_hemi_flags = (
        satellite_table_xarray[satellite_utils.STORM_LATITUDE_KEY].values >= 0
    ).astype(int)

    for this_key in satellite_dict:
        if this_key in SATELLITE_METADATA_KEYS:
            new_matrix = satellite_dict[this_key]['data']

            if this_key == satellite_utils.GRID_LATITUDE_KEY:
                for i in range(len(satellite_northern_hemi_flags)):
                    if satellite_northern_hemi_flags[i]:
                        continue

                    new_matrix = numpy.array(new_matrix)
                    new_matrix[i, ...] = numpy.flip(new_matrix[i, ...], axis=0)

            example_dict[this_key] = (
                satellite_dict[this_key]['dims'], new_matrix
            )

            if this_key not in SATELLITE_METADATA_AND_FORECAST_KEYS:
                continue

        if this_key == satellite_utils.BRIGHTNESS_TEMPERATURE_KEY:
            these_dim = (
                satellite_dict[this_key]['dims'] +
                (SATELLITE_PREDICTOR_GRIDDED_DIM,)
            )
            new_matrix = numpy.expand_dims(
                satellite_dict[this_key]['data'], axis=-1
            )

            for i in range(len(satellite_northern_hemi_flags)):
                if satellite_northern_hemi_flags[i]:
                    continue

                new_matrix[i, ...] = numpy.flip(new_matrix[i, ...], axis=0)

            example_dict[SATELLITE_PREDICTORS_GRIDDED_KEY] = (
                these_dim, new_matrix
            )

            continue

        satellite_predictor_names_ungridded.append(this_key)
        new_matrix = numpy.expand_dims(
            satellite_dict[this_key]['data'], axis=-1
        )

        if this_key == SATELLITE_STORM_MOTION_V_KEY:
            for i in range(len(satellite_northern_hemi_flags)):
                if satellite_northern_hemi_flags[i]:
                    continue

                new_matrix[i, ...] *= -1

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

    example_dict[STORM_INTENSITY_KEY] = (
        (SHIPS_VALID_TIME_DIM,),
        ships_table_xarray[ships_io.STORM_INTENSITY_KEY].values
    )

    ships_predictor_names_forecast = []
    ships_predictor_names_lagged = []
    ships_predictor_matrix_forecast = numpy.array([])
    ships_predictor_matrix_lagged = numpy.array([])

    ships_northern_hemi_flags = (
        ships_table_xarray[ships_io.STORM_LATITUDE_KEY].values >= 0
    ).astype(int)

    ships_northern_hemi_flags[ships_northern_hemi_flags == 0] = -1

    for this_key in ships_dict:
        if this_key in SHIPS_METADATA_KEYS:
            example_dict[this_key] = (
                ships_dict[this_key]['dims'],
                ships_dict[this_key]['data']
            )

            if this_key not in SHIPS_METADATA_AND_FORECAST_KEYS:
                continue

        new_matrix = numpy.expand_dims(ships_dict[this_key]['data'], axis=-1)
        for i in range(len(ships_northern_hemi_flags)):
            new_matrix[i, ...] = (
                new_matrix[i, ...] * ships_northern_hemi_flags[i]
            )

        if SHIPS_LAG_TIME_DIM in ships_dict[this_key]['dims']:
            ships_predictor_names_lagged.append(this_key)

            if ships_predictor_matrix_lagged.size == 0:
                ships_predictor_matrix_lagged = new_matrix + 0.
            else:
                ships_predictor_matrix_lagged = numpy.concatenate(
                    (ships_predictor_matrix_lagged, new_matrix), axis=-1
                )

            continue

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

    example_table_xarray = xarray.Dataset(
        data_vars=example_dict, coords=example_metadata_dict
    )
    example_table_xarray = _create_merged_sst_variable(example_table_xarray)
    example_table_xarray = _create_merged_ohc_variable(example_table_xarray)
    return _merge_lagged_ships_variables(example_table_xarray)


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

    good_indices = []

    for t in desired_times_unix_sec:
        this_index = numpy.argmin(numpy.absolute(all_times_unix_sec - t))
        good_indices.append(this_index)

    good_indices = numpy.unique(numpy.array(good_indices, dtype=int))

    return example_table_xarray.isel(
        indexers={SATELLITE_TIME_DIM: good_indices}
    )


def rotate_satellite_grid(
        center_latitude_deg_n, center_longitude_deg_e, east_velocity_m_s01,
        north_velocity_m_s01, num_rows, num_columns):
    """Rotates satellite grid to align with motion vector.

    The "motion vector" could be storm motion, wind, wind shear, etc.

    M = number of rows in new grid
    N = number of columns in new grid

    :param center_latitude_deg_n: Latitude (deg N) of tropical-cyclone center.
    :param center_longitude_deg_e: Longitude (deg E) of tropical-cyclone center.
    :param east_velocity_m_s01: Eastward component of motion vector (metres per
        second).
    :param north_velocity_m_s01: Northward component of motion vector (metres
        per second).
    :param num_rows: Number of rows in grid.
    :param num_columns: Number of columns in grid.
    :return: latitude_matrix_deg_n: M-by-N numpy array of latitudes (deg N).
    :return: longitude_matrix_deg_e: M-by-N numpy array of longitudes (deg E).
    """

    # Check input args.
    error_checking.assert_is_not_nan(center_latitude_deg_n)
    error_checking.assert_is_not_nan(center_longitude_deg_e)
    error_checking.assert_is_not_nan(east_velocity_m_s01)
    error_checking.assert_is_not_nan(north_velocity_m_s01)

    error_checking.assert_is_integer(num_rows)
    error_checking.assert_is_greater(num_rows, 0)
    assert numpy.mod(num_rows, 2) == 0

    error_checking.assert_is_integer(num_columns)
    error_checking.assert_is_greater(num_columns, 0)
    assert numpy.mod(num_columns, 2) == 0

    # Do actual stuff.
    bearing_deg = geodetic_utils.xy_to_scalar_displacements_and_bearings(
        x_displacements_metres=numpy.array([east_velocity_m_s01]),
        y_displacements_metres=numpy.array([north_velocity_m_s01])
    )[-1][0]

    x_max_metres = GRID_SPACING_METRES * (num_columns / 2 - 0.5)
    x_coords_metres = numpy.linspace(
        -x_max_metres, x_max_metres, num=num_columns
    )
    y_max_metres = GRID_SPACING_METRES * (num_rows / 2 - 0.5)
    y_coords_metres = numpy.linspace(
        -y_max_metres, y_max_metres, num=num_rows
    )
    x_coord_matrix_metres, y_coord_matrix_metres = grids.xy_vectors_to_matrices(
        x_unique_metres=x_coords_metres, y_unique_metres=y_coords_metres
    )

    x_coord_matrix_metres, y_coord_matrix_metres = (
        geodetic_utils.rotate_displacement_vectors(
            x_displacements_metres=x_coord_matrix_metres,
            y_displacements_metres=y_coord_matrix_metres,
            ccw_rotation_angle_deg=-(bearing_deg - 90)
        )
    )

    displacement_matrix_metres, bearing_matrix_deg = (
        geodetic_utils.xy_to_scalar_displacements_and_bearings(
            x_displacements_metres=x_coord_matrix_metres,
            y_displacements_metres=y_coord_matrix_metres
        )
    )

    start_latitude_matrix_deg = numpy.full(
        (num_rows, num_columns), center_latitude_deg_n
    )
    start_longitude_matrix_deg = numpy.full(
        (num_rows, num_columns), center_longitude_deg_e
    )

    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        geodetic_utils.start_points_and_displacements_to_endpoints(
            start_latitudes_deg=start_latitude_matrix_deg,
            start_longitudes_deg=start_longitude_matrix_deg,
            scalar_displacements_metres=displacement_matrix_metres,
            geodetic_bearings_deg=bearing_matrix_deg
        )
    )

    if center_latitude_deg_n < 0:
        latitude_matrix_deg_n = numpy.flip(latitude_matrix_deg_n, axis=0)
        longitude_matrix_deg_e = numpy.flip(longitude_matrix_deg_e, axis=0)

    return latitude_matrix_deg_n, longitude_matrix_deg_e


def rotate_satellite_image(
        brightness_temp_matrix_kelvins, orig_latitudes_deg_n,
        orig_longitudes_deg_e, new_latitude_matrix_deg_n,
        new_longitude_matrix_deg_e, fill_value):
    """Rotates satellite image.

    The original grid must be a regular lat/long grid; the new grid is assumed
    not to be.

    m = number of rows in original grid
    n = number of columns in original grid
    M = number of rows in new grid
    N = number of columns in new grid

    :param brightness_temp_matrix_kelvins: m-by-n numpy array.
    :param orig_latitudes_deg_n: length-m numpy array of latitudes (deg N).
    :param orig_longitudes_deg_e: length-n numpy array of longitudes (deg E).
    :param new_latitude_matrix_deg_n: M-by-N numpy array of latitudes (deg N).
    :param new_longitude_matrix_deg_e: M-by-N numpy array of longitudes (deg E).
    :param fill_value: Fill value, used instead of extrapolation.
    :return: brightness_temp_matrix_kelvins: M-by-N numpy array.
    """

    this_flag, increasing_latitudes_deg_n, increasing_longitudes_deg_e = (
        satellite_utils.is_regular_grid_valid(
            latitudes_deg_n=orig_latitudes_deg_n,
            longitudes_deg_e=orig_longitudes_deg_e
        )
    )

    assert this_flag

    orig_num_rows = len(increasing_latitudes_deg_n)
    orig_num_columns = len(increasing_longitudes_deg_e)
    expected_dim = numpy.array([orig_num_rows, orig_num_columns], dtype=int)

    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, exact_dimensions=expected_dim
    )
    brightness_temp_matrix_kelvins = general_utils.fill_nans(
        brightness_temp_matrix_kelvins
    )

    error_checking.assert_is_numpy_array_without_nan(new_latitude_matrix_deg_n)
    error_checking.assert_is_numpy_array(
        new_latitude_matrix_deg_n, num_dimensions=2
    )

    error_checking.assert_is_numpy_array_without_nan(new_longitude_matrix_deg_e)
    error_checking.assert_is_numpy_array(
        new_longitude_matrix_deg_e,
        exact_dimensions=numpy.array(new_latitude_matrix_deg_n.shape, dtype=int)
    )

    error_checking.assert_is_not_nan(fill_value)

    # Deal with grids that cross International Date Line or Prime Meridian.
    new_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        new_longitude_matrix_deg_e
    )
    longitude_range_deg = (
        numpy.max(new_longitude_matrix_deg_e) -
        numpy.min(new_longitude_matrix_deg_e)
    )

    # TODO(thunderhoser): This solution is a bit hacky (assumes grids small
    # enough that they never truly span 100 deg of longitude).
    if longitude_range_deg > 100:
        new_longitude_matrix_deg_e = (
            lng_conversion.convert_lng_negative_in_west(
                new_longitude_matrix_deg_e
            )
        )

    # Do actual stuff.
    central_latitude_deg_n = numpy.mean(new_latitude_matrix_deg_n)
    central_longitude_deg_e = numpy.mean(new_longitude_matrix_deg_e)

    projection_object = projections.init_cylindrical_equidistant_projection(
        central_latitude_deg=central_latitude_deg_n,
        central_longitude_deg=central_longitude_deg_e,
        true_scale_latitude_deg=central_latitude_deg_n
    )

    new_x_matrix_metres, new_y_matrix_metres = projections.project_latlng_to_xy(
        latitudes_deg=new_latitude_matrix_deg_n,
        longitudes_deg=new_longitude_matrix_deg_e,
        projection_object=projection_object
    )

    orig_x_coords_metres, _ = projections.project_latlng_to_xy(
        latitudes_deg=numpy.full(
            len(increasing_longitudes_deg_e), central_latitude_deg_n
        ),
        longitudes_deg=increasing_longitudes_deg_e,
        projection_object=projection_object
    )

    _, orig_y_coords_metres = projections.project_latlng_to_xy(
        latitudes_deg=increasing_latitudes_deg_n,
        longitudes_deg=numpy.full(
            len(increasing_latitudes_deg_n), central_longitude_deg_e
        ),
        projection_object=projection_object
    )

    brightness_temp_matrix_kelvins = interp.interp_from_xy_grid_to_points(
        input_matrix=brightness_temp_matrix_kelvins,
        sorted_grid_point_x_metres=orig_x_coords_metres,
        sorted_grid_point_y_metres=orig_y_coords_metres,
        query_x_coords_metres=numpy.ravel(new_x_matrix_metres),
        query_y_coords_metres=numpy.ravel(new_y_matrix_metres),
        method_string=interp.SPLINE_METHOD_STRING, spline_degree=1,
        extrapolate=True
    )
    brightness_temp_matrix_kelvins = numpy.reshape(
        brightness_temp_matrix_kelvins, new_x_matrix_metres.shape
    )

    invalid_x_flags = numpy.logical_or(
        new_x_matrix_metres < numpy.min(orig_x_coords_metres),
        new_x_matrix_metres > numpy.max(orig_x_coords_metres)
    )
    invalid_y_flags = numpy.logical_or(
        new_y_matrix_metres < numpy.min(orig_y_coords_metres),
        new_y_matrix_metres > numpy.max(orig_y_coords_metres)
    )
    invalid_indices = numpy.where(
        numpy.logical_or(invalid_x_flags, invalid_y_flags)
    )

    brightness_temp_matrix_kelvins[invalid_indices] = fill_value

    return brightness_temp_matrix_kelvins
