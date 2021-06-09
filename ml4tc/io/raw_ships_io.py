"""IO methods for raw SHIPS data.

SHIPS = Statistical Hurricane-intensity-prediction Scheme
"""

import numpy
import xarray
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import ships_io
from ml4tc.utils import general_utils
from ml4tc.utils import satellite_utils

SENTINEL_STRING = '9999'
TIME_FORMAT_IN_FILES = '%Y%m%d%H'

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
MB_TO_PASCALS = 100.
HOURS_TO_SECONDS = 3600.
DAYS_TO_SECONDS = 86400.
KM_TO_METRES = 1000.

BASIN_ID_TO_LNG_MULTIPLIER = {
    satellite_utils.NORTH_ATLANTIC_ID_STRING: -1.,
    satellite_utils.SOUTH_ATLANTIC_ID_STRING: -1.,
    satellite_utils.NORTHEAST_PACIFIC_ID_STRING: -1.,
    satellite_utils.NORTH_CENTRAL_PACIFIC_ID_STRING: -1.,
    satellite_utils.NORTHWEST_PACIFIC_ID_STRING: 1.,
    satellite_utils.NORTH_INDIAN_ID_STRING: 1.,
    satellite_utils.SOUTHERN_HEMISPHERE_ID_STRING: 1.
}

FORECAST_FIELD_NAMES_RAW = [
    'VMAX', 'MSLP', 'TYPE', 'DELV', 'INCV', 'LAT', 'LON', 'CSST', 'CD20',
    'CD26', 'COHC', 'DTL', 'OAGE', 'NAGE', 'RSST', 'DSST', 'DSTA', 'U200',
    'U20C', 'V20C', 'E000', 'EPOS', 'ENEG', 'EPSS', 'ENSS', 'RHLO', 'RHMD',
    'RHHI', 'Z850', 'D200', 'REFC', 'PEFC', 'T000', 'R000', 'Z000', 'TLAT',
    'TLON', 'TWAC', 'TWXC', 'G150', 'G200', 'G250', 'V000', 'V850', 'V500',
    'V300', 'TGRD', 'TADV', 'PENC', 'SHDC', 'SDDC', 'SHGC', 'DIVC', 'T150',
    'T200', 'T250', 'SHRD', 'SHTD', 'SHRS', 'SHTS', 'SHRG', 'PENV', 'VMPI',
    'VVAV', 'VMFX', 'VVAC', 'HE07', 'HE05', 'O500', 'O700', 'CFLX', 'PW01',
    'PW02', 'PW03', 'PW04', 'PW05', 'PW06', 'PW07', 'PW08', 'PW09', 'PW10',
    'PW11', 'PW12', 'PW13', 'PW14', 'PW15', 'PW16', 'PW17', 'PW18', 'PW19',
    'PW20', 'PW21', 'XDST', 'XNST', 'XOHC', 'XDFR', 'XTMX', 'XDTX', 'XDML',
    'XD30', 'XD28', 'XD26', 'XD24', 'XD22', 'XD20', 'XD18', 'XD16', 'XTFR',
    'XO20', 'NSST', 'NSTA', 'NTMX', 'NDTX', 'NDML', 'ND30', 'ND28', 'ND26',
    'ND24', 'ND22', 'ND20', 'ND18', 'ND16', 'NDFR', 'NTFR', 'NOHC', 'NO20',
    'RHCN', 'RD20', 'RD26', 'PHCN'
]
SATELLITE_FIELD_NAMES_RAW = ['IRXX', 'IR00', 'IRM1', 'IRM3']
MISC_FIELD_NAMES_RAW = ['PC00', 'PCM1', 'PCM3', 'LAST', 'MTPW']
HISTORY_FIELD_NAME_RAW = 'HIST'
MOTION_FIELD_NAME_RAW = 'PSLV'

RAW_FIELD_NAMES = (
    FORECAST_FIELD_NAMES_RAW + SATELLITE_FIELD_NAMES_RAW +
    MISC_FIELD_NAMES_RAW + [HISTORY_FIELD_NAME_RAW] +
    [MOTION_FIELD_NAME_RAW]
)

FORECAST_FIELD_RENAMING_DICT = {
    'VMAX': ships_io.INTENSITY_KEY,
    'MSLP': ships_io.MINIMUM_SLP_KEY,
    'TYPE': ships_io.STORM_TYPE_KEY,
    'DELV': ships_io.INTENSITY_CHANGE_M12HOURS_KEY,
    'INCV': ships_io.INTENSITY_CHANGE_6HOURS_KEY,
    'LAT': ships_io.FORECAST_LATITUDE_KEY,
    'LON': ships_io.FORECAST_LONGITUDE_KEY,
    'CSST': ships_io.CLIMO_SST_KEY,
    'CD20': ships_io.CLIMO_DEPTH_20C_KEY,
    'CD26': ships_io.CLIMO_DEPTH_26C_KEY,
    'COHC': ships_io.CLIMO_OHC_KEY,
    'DTL': ships_io.DISTANCE_TO_LAND_KEY,
    'OAGE': ships_io.OCEAN_AGE_KEY,
    'NAGE': ships_io.NORMALIZED_OCEAN_AGE_KEY,
    'RSST': ships_io.REYNOLDS_SST_KEY,
    'DSST': ships_io.REYNOLDS_SST_DAILY_KEY,
    'DSTA': ships_io.REYNOLDS_SST_DAILY_AREAL_KEY,
    'U200': ships_io.U_WIND_200MB_OUTER_RING_KEY,
    'U20C': ships_io.U_WIND_200MB_INNER_RING_KEY,
    'V20C': ships_io.V_WIND_200MB_INNER_RING_KEY,
    'E000': ships_io.THETA_E_1000MB_OUTER_RING_KEY,
    'EPOS': ships_io.THETA_E_SURPLUS_OUTER_RING_KEY,
    'ENEG': ships_io.THETA_E_DEFICIT_OUTER_RING_KEY,
    'EPSS': ships_io.THETA_E_SURPLUS_SATURATED_OUTER_RING_KEY,
    'ENSS': ships_io.THETA_E_DEFICIT_SATURATED_OUTER_RING_KEY,
    'RHLO': ships_io.RH_850TO700MB_OUTER_RING_KEY,
    'RHMD': ships_io.RH_700TO500MB_OUTER_RING_KEY,
    'RHHI': ships_io.RH_500TO300MB_OUTER_RING_KEY,
    'Z850': ships_io.VORTICITY_850MB_BIG_RING_KEY,
    'D200': ships_io.DIVERGENCE_200MB_BIG_RING_KEY,
    'REFC': ships_io.RELATIVE_EMFC_100TO600KM_KEY,
    'PEFC': ships_io.PLANETARY_EMFC_100TO600KM_KEY,
    'T000': ships_io.TEMPERATURE_1000MB_OUTER_RING_KEY,
    'R000': ships_io.RH_1000MB_OUTER_RING_KEY,
    'Z000': ships_io.HEIGHT_DEV_1000MB_OUTER_RING_KEY,
    'TLAT': ships_io.VORTEX_LATITUDE_KEY,
    'TLON': ships_io.VORTEX_LONGITUDE_KEY,
    'TWAC': ships_io.MEAN_TAN_WIND_850MB_0TO600KM_KEY,
    'TWXC': ships_io.MAX_TAN_WIND_850MB_KEY,
    'G150': ships_io.TEMP_PERTURB_150MB_OUTER_RING_KEY,
    'G200': ships_io.TEMP_PERTURB_200MB_OUTER_RING_KEY,
    'G250': ships_io.TEMP_PERTURB_250MB_OUTER_RING_KEY,
    'V000': ships_io.MEAN_TAN_WIND_1000MB_500KM_KEY,
    'V850': ships_io.MEAN_TAN_WIND_850MB_500KM_KEY,
    'V500': ships_io.MEAN_TAN_WIND_500MB_500KM_KEY,
    'V300': ships_io.MEAN_TAN_WIND_300MB_500KM_KEY,
    'TGRD': ships_io.TEMP_GRADIENT_850TO700MB_INNER_RING_KEY,
    'TADV': ships_io.TEMP_ADVECTION_850TO700MB_INNER_RING_KEY,
    'PENC': ships_io.SURFACE_PRESSURE_EDGE_KEY,
    # 'SHDC': ships_io.SHEAR_850TO200MB_INNER_RING_KEY,
    # 'SDDC': ships_io.SHEAR_850TO200MB_INNER_RING_HEADING_KEY,
    'SHGC': ships_io.SHEAR_850TO200MB_INNER_RING_GNRL_KEY,
    'DIVC': ships_io.DIVERGENCE_200MB_CENTERED_BIG_RING_KEY,
    'T150': ships_io.TEMP_150MB_OUTER_RING_KEY,
    'T200': ships_io.TEMP_200MB_OUTER_RING_KEY,
    'T250': ships_io.TEMP_250MB_OUTER_RING_KEY,
    # 'SHRD': ships_io.SHEAR_850TO200MB_OUTER_RING_KEY,
    # 'SHTD': ships_io.SHEAR_850TO200MB_OUTER_RING_HEADING_KEY,
    # 'SHRS': ships_io.SHEAR_850TO500MB_KEY,
    # 'SHTS': ships_io.SHEAR_850TO500MB_HEADING_KEY,
    'SHRG': ships_io.SHEAR_850TO200MB_GENERALIZED_KEY,
    'PENV': ships_io.SURFACE_PRESSURE_OUTER_RING_KEY,
    'VMPI': ships_io.MAX_PTTL_INTENSITY_KEY,
    'VVAV': ships_io.W_WIND_0TO15KM_KEY,
    'VMFX': ships_io.W_WIND_0TO15KM_WEIGHTED_KEY,
    'VVAC': ships_io.W_WIND_0TO15KM_INNER_RING_KEY,
    'HE07': ships_io.SRH_1000TO700MB_OUTER_RING_KEY,
    'HE05': ships_io.SRH_1000TO500MB_OUTER_RING_KEY,
    'O500': ships_io.W_WIND_500MB_BIG_RING_KEY,
    'O700': ships_io.W_WIND_700MB_BIG_RING_KEY,
    'CFLX': ships_io.DRY_AIR_PREDICTOR_KEY,  # units??
    'PW01': ships_io.PRECIP_WATER_0TO200KM_KEY,
    'PW02': ships_io.PRECIP_WATER_0TO200KM_STDEV_KEY,
    'PW03': ships_io.PRECIP_WATER_200TO400KM_KEY,
    'PW04': ships_io.PRECIP_WATER_200TO400KM_STDEV_KEY,
    'PW05': ships_io.PRECIP_WATER_400TO600KM_KEY,
    'PW06': ships_io.PRECIP_WATER_400TO600KM_STDEV_KEY,
    'PW07': ships_io.PRECIP_WATER_600TO800KM_KEY,
    'PW08': ships_io.PRECIP_WATER_600TO800KM_STDEV_KEY,
    'PW09': ships_io.PRECIP_WATER_800TO1000KM_KEY,
    'PW10': ships_io.PRECIP_WATER_800TO1000KM_STDEV_KEY,
    'PW11': ships_io.PRECIP_WATER_0TO400KM_KEY,
    'PW12': ships_io.PRECIP_WATER_0TO400KM_STDEV_KEY,
    'PW13': ships_io.PRECIP_WATER_0TO600KM_KEY,
    'PW14': ships_io.PRECIP_WATER_0TO600KM_STDEV_KEY,
    'PW15': ships_io.PRECIP_WATER_0TO800KM_KEY,
    'PW16': ships_io.PRECIP_WATER_0TO800KM_STDEV_KEY,
    'PW17': ships_io.PRECIP_WATER_0TO1000KM_KEY,
    'PW18': ships_io.PRECIP_WATER_0TO1000KM_STDEV_KEY,
    'PW19': ships_io.PW_INNER_UPSHEAR_FRACTION_UNDER45MM_KEY,
    'PW20': ships_io.PRECIP_WATER_INNER_RING_UPSHEAR_KEY,
    'PW21': ships_io.PRECIP_WATER_INNER_RING_KEY,
    'XDST': ships_io.REYNOLDS_SST_DAILY_CLIMO_KEY,
    'XNST': ships_io.NCODA_SST_CLIMO_KEY,
    'XOHC': ships_io.NCODA_OHC_26C_CLIMO_KEY,
    'XDFR': ships_io.NCODA_BOTTOM_CLIMO_KEY,
    'XTMX': ships_io.NCODA_MAX_TEMP_CLIMO_KEY,
    'XDTX': ships_io.DEPTH_MAX_TEMP_CLIMO_KEY,  # I think XDMX in documentation is a typo?
    'XDML': ships_io.MIXED_LAYER_DEPTH_CLIMO_KEY,
    'XD30': ships_io.DEPTH_30C_CLIMO_KEY,
    'XD28': ships_io.DEPTH_28C_CLIMO_KEY,
    'XD26': ships_io.DEPTH_26C_CLIMO_KEY,
    'XD24': ships_io.DEPTH_24C_CLIMO_KEY,
    'XD22': ships_io.DEPTH_22C_CLIMO_KEY,
    'XD20': ships_io.DEPTH_20C_CLIMO_KEY,
    'XD18': ships_io.DEPTH_18C_CLIMO_KEY,
    'XD16': ships_io.DEPTH_16C_CLIMO_KEY,
    'XTFR': ships_io.NCODA_BOTTOM_TEMP_CLIMO_KEY,
    'XO20': ships_io.NCODA_OHC_20C_CLIMO_KEY,
    'NSST': ships_io.NCODA_SST_KEY,
    'NSTA': ships_io.NCODA_SST_AREAL_KEY,  # Maybe?  Documentation doesn't say.
    'NTMX': ships_io.NCODA_MAX_TEMP_KEY,
    'NDTX': ships_io.DEPTH_MAX_TEMP_KEY,  # I think NDMX in documentation is a typo?
    'NDML': ships_io.MIXED_LAYER_DEPTH_KEY,
    'ND30': ships_io.DEPTH_30C_KEY,
    'ND28': ships_io.DEPTH_28C_KEY,
    'ND26': ships_io.DEPTH_26C_KEY,
    'ND24': ships_io.DEPTH_24C_KEY,
    'ND22': ships_io.DEPTH_22C_KEY,
    'ND20': ships_io.DEPTH_20C_KEY,
    'ND18': ships_io.DEPTH_18C_KEY,
    'ND16': ships_io.DEPTH_16C_KEY,
    'NDFR': ships_io.NCODA_BOTTOM_KEY,
    'NTFR': ships_io.NCODA_BOTTOM_TEMP_KEY,
    'NOHC': ships_io.NCODA_OHC_26C_KEY,  # Are units right?
    'NO20': ships_io.NCODA_OHC_20C_KEY,  # Are units right?
    'RHCN': ships_io.SATELLITE_OHC_KEY,
    'RD20': ships_io.DEPTH_20C_SATELLITE_KEY,
    'RD26': ships_io.DEPTH_26C_SATELLITE_KEY,
    'PHCN': ships_io.OHC_FROM_SST_AND_CLIMO_KEY
}

SATELLITE_FIELD_NAMES_PROCESSED = [
    ships_io.SATELLITE_LAG_TIME_KEY,
    ships_io.SATELLITE_TEMP_0TO200KM_KEY,
    ships_io.SATELLITE_TEMP_0TO200KM_STDEV_KEY,
    ships_io.SATELLITE_TEMP_100TO300KM_KEY,
    ships_io.SATELLITE_TEMP_100TO300KM_STDEV_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M10C_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M20C_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M30C_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M40C_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M50C_KEY,
    ships_io.SATELLITE_TEMP_FRACTION_BELOW_M60C_KEY,
    ships_io.SATELLITE_MAX_TEMP_0TO30KM_KEY,
    ships_io.SATELLITE_MEAN_TEMP_0TO30KM_KEY,
    ships_io.SATELLITE_MAX_TEMP_RADIUS_KEY,
    ships_io.SATELLITE_MIN_TEMP_20TO120KM_KEY,
    ships_io.SATELLITE_MEAN_TEMP_20TO120KM_KEY,
    ships_io.SATELLITE_MIN_TEMP_RADIUS_KEY
]

MOTION_FIELD_NAMES_PROCESSED = [
    ships_io.STEERING_LEVEL_PRESSURE_KEY,
    ships_io.U_MOTION_KEY,
    ships_io.V_MOTION_KEY,
    ships_io.U_MOTION_1000TO100MB_KEY,
    ships_io.V_MOTION_1000TO100MB_KEY,
    ships_io.U_MOTION_OPTIMAL_KEY,
    ships_io.V_MOTION_OPTIMAL_KEY
]


def _get_multiply_function(multiplier):
    """Returns function that multiplies by scalar.

    :param multiplier: Function will multiply all inputs by this scalar.
    :return: multiply_function: Function handle (see below).
    """

    def multiply_function(input_array):
        """Multiplies input by scalar.

        :param input_array: numpy array.
        :return: output_array: numpy array.
        """

        return input_array * multiplier

    return multiply_function


def _decicelsius_to_kelvins(temperatures_decicelsius):
    """Converts from temperatures from decidegrees Celsius to Kelvins.

    :param temperatures_decicelsius: numpy array of temperatures in decidegrees
        Celsius.
    :return: temperatures_kelvins: numpy array of temperatures in Kelvins, with
        same shape as input.
    """

    return temperatures_decicelsius * 0.1 + 273.15


def _pressure_from_1000mb_departure_to_pa(
        pressures_1000mb_departures_decapascals):
    """Converts pressures from 1000-mb departures

    :param pressures_1000mb_departures_decapascals: numpy array of pressure
        departures from 1000 mb, in deca-Pascals.
    :return: pressures_pascals: numpy array of pressures in Pascals, with same
        shape as input.
    """

    pressures_mb = 1000 + 0.1 * pressures_1000mb_departures_decapascals
    return pressures_mb * MB_TO_PASCALS


FORECAST_FIELD_TO_CONV_FUNCTION = {
    'VMAX': _get_multiply_function(KT_TO_METRES_PER_SECOND),
    'MSLP': _get_multiply_function(MB_TO_PASCALS),
    'TYPE': _get_multiply_function(1.),
    'DELV': _get_multiply_function(KT_TO_METRES_PER_SECOND),
    'INCV': _get_multiply_function(KT_TO_METRES_PER_SECOND),
    'LAT': _get_multiply_function(0.1),
    'LON': _get_multiply_function(0.1),
    'CSST': _decicelsius_to_kelvins,
    'CD20': _get_multiply_function(1.),
    'CD26': _get_multiply_function(1.),
    'COHC': _get_multiply_function(1e7),
    'DTL': _get_multiply_function(1000.),
    'OAGE': _get_multiply_function(0.1 * HOURS_TO_SECONDS),
    'NAGE': _get_multiply_function(0.1 * HOURS_TO_SECONDS),
    'RSST': _decicelsius_to_kelvins,
    'DSST': _decicelsius_to_kelvins,
    'DSTA': _decicelsius_to_kelvins,
    'U200': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'U20C': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'V20C': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'E000': _get_multiply_function(0.1),
    'EPOS': _get_multiply_function(0.1),
    'ENEG': _get_multiply_function(0.1),
    'EPSS': _get_multiply_function(0.1),
    'ENSS': _get_multiply_function(0.1),
    'RHLO': _get_multiply_function(0.01),
    'RHMD': _get_multiply_function(0.01),
    'RHHI': _get_multiply_function(0.01),
    'Z850': _get_multiply_function(1e-7),
    'D200': _get_multiply_function(1e-7),
    'REFC': _get_multiply_function(DAYS_TO_SECONDS ** -1),
    'PEFC': _get_multiply_function(DAYS_TO_SECONDS ** -1),
    'T000': _decicelsius_to_kelvins,
    'R000': _get_multiply_function(0.01),
    'Z000': _get_multiply_function(1.),
    'TLAT': _get_multiply_function(0.1),
    'TLON': _get_multiply_function(0.1),
    'TWAC': _get_multiply_function(0.1),
    'TWXC': _get_multiply_function(0.1),
    'G150': _get_multiply_function(0.1),
    'G200': _get_multiply_function(0.1),
    'G250': _get_multiply_function(0.1),
    'V000': _get_multiply_function(0.1),
    'V850': _get_multiply_function(0.1),
    'V500': _get_multiply_function(0.1),
    'V300': _get_multiply_function(0.1),
    'TGRD': _get_multiply_function(1e-7),
    'TADV': _get_multiply_function(1e-6),
    'PENC': _pressure_from_1000mb_departure_to_pa,
    'SHDC': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'SDDC': _get_multiply_function(1.),
    'SHGC': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'DIVC': _get_multiply_function(1e-7),
    'T150': _decicelsius_to_kelvins,
    'T200': _decicelsius_to_kelvins,
    'T250': _decicelsius_to_kelvins,
    'SHRD': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'SHTD': _get_multiply_function(1.),
    'SHRS': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'SHTS': _get_multiply_function(1.),
    'SHRG': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'PENV': _pressure_from_1000mb_departure_to_pa,
    'VMPI': _get_multiply_function(KT_TO_METRES_PER_SECOND),
    'VVAV': _get_multiply_function(0.01),
    'VMFX': _get_multiply_function(0.01),
    'VVAC': _get_multiply_function(0.01),
    'HE07': _get_multiply_function(0.1),
    'HE05': _get_multiply_function(0.1),
    'O500': _get_multiply_function(MB_TO_PASCALS / DAYS_TO_SECONDS),
    'O700': _get_multiply_function(MB_TO_PASCALS / DAYS_TO_SECONDS),
    'CFLX': _get_multiply_function(1.),
    'PW01': _get_multiply_function(0.1),
    'PW02': _get_multiply_function(0.1),
    'PW03': _get_multiply_function(0.1),
    'PW04': _get_multiply_function(0.1),
    'PW05': _get_multiply_function(0.1),
    'PW06': _get_multiply_function(0.1),
    'PW07': _get_multiply_function(0.1),
    'PW08': _get_multiply_function(0.1),
    'PW09': _get_multiply_function(0.1),
    'PW10': _get_multiply_function(0.1),
    'PW11': _get_multiply_function(0.1),
    'PW12': _get_multiply_function(0.1),
    'PW13': _get_multiply_function(0.1),
    'PW14': _get_multiply_function(0.1),
    'PW15': _get_multiply_function(0.1),
    'PW16': _get_multiply_function(0.1),
    'PW17': _get_multiply_function(0.1),
    'PW18': _get_multiply_function(0.1),
    'PW19': _get_multiply_function(0.001),
    'PW20': _get_multiply_function(0.1),
    'PW21': _get_multiply_function(0.1),
    'XDST': _decicelsius_to_kelvins,
    'XNST': _decicelsius_to_kelvins,
    'XOHC': _get_multiply_function(1.),
    'XDFR': _get_multiply_function(1.),
    'XTMX': _decicelsius_to_kelvins,
    'XDTX': _get_multiply_function(1.),
    'XDML': _get_multiply_function(1.),
    'XD30': _get_multiply_function(1.),
    'XD28': _get_multiply_function(1.),
    'XD26': _get_multiply_function(1.),
    'XD24': _get_multiply_function(1.),
    'XD22': _get_multiply_function(1.),
    'XD20': _get_multiply_function(1.),
    'XD18': _get_multiply_function(1.),
    'XD16': _get_multiply_function(1.),
    'XTFR': _decicelsius_to_kelvins,
    'XO20': _get_multiply_function(1.),
    'NSST': _decicelsius_to_kelvins,
    'NSTA': _decicelsius_to_kelvins,
    'NTMX': _decicelsius_to_kelvins,
    'NDTX': _get_multiply_function(1.),
    'NDML': _get_multiply_function(1.),
    'ND30': _get_multiply_function(1.),
    'ND28': _get_multiply_function(1.),
    'ND26': _get_multiply_function(1.),
    'ND24': _get_multiply_function(1.),
    'ND22': _get_multiply_function(1.),
    'ND20': _get_multiply_function(1.),
    'ND18': _get_multiply_function(1.),
    'ND16': _get_multiply_function(1.),
    'NDFR': _get_multiply_function(1.),
    'NTFR': _decicelsius_to_kelvins,
    'NOHC': _get_multiply_function(1.),
    'NO20': _get_multiply_function(1.),
    'RHCN': _get_multiply_function(1e7),
    'RD20': _get_multiply_function(1.),
    'RD26': _get_multiply_function(1.),
    'PHCN': _get_multiply_function(1e7)
}

SATELLITE_FIELD_CONV_FUNCTIONS = [
    _get_multiply_function(60.),
    _decicelsius_to_kelvins,
    _get_multiply_function(0.1),
    _decicelsius_to_kelvins,
    _get_multiply_function(0.1),
    _get_multiply_function(0.01),
    _get_multiply_function(0.01),
    _get_multiply_function(0.01),
    _get_multiply_function(0.01),
    _get_multiply_function(0.01),
    _get_multiply_function(0.01),
    _decicelsius_to_kelvins,
    _decicelsius_to_kelvins,
    _get_multiply_function(KM_TO_METRES),
    _decicelsius_to_kelvins,
    _decicelsius_to_kelvins,
    _get_multiply_function(KM_TO_METRES)
]

MOTION_FIELD_CONV_FUNCTIONS = [
    _get_multiply_function(MB_TO_PASCALS),
    _get_multiply_function(0.1),
    _get_multiply_function(0.1),
    _get_multiply_function(0.1),
    _get_multiply_function(0.1),
    _get_multiply_function(0.1),
    _get_multiply_function(0.1)
]


def _forecast_hour_to_chars(forecast_hour_line, seven_day):
    """Determines correspondence of forecast hour to character indices.

    :param forecast_hour_line: Line from SHIPS file with forecast hours.  This
        line should end with "TIME".
    :param seven_day: Boolean flag.  If True (False), line comes from 7-day
        (5-day) file.
    :return: hour_index_to_char_indices: Dictionary, where each key is the index
        of a forecast hour.  The corresponding value is a length-2 numpy array,
        containing the first and last character indices, for a given line in the
        SHIPS file (i.e., for one storm at one time), where data will be found
        for the forecast hour.
    """

    error_checking.assert_is_boolean(seven_day)
    assert forecast_hour_line.endswith('TIME')

    if seven_day:
        expected_forecast_hours = numpy.linspace(-12, 168, num=31, dtype=int)
    else:
        expected_forecast_hours = numpy.linspace(-12, 120, num=23, dtype=int)

    forecast_hours = numpy.array([
        int(word) for word in forecast_hour_line.split()[:-1]
    ], dtype=int)

    assert numpy.array_equal(forecast_hours, expected_forecast_hours)

    forecast_hour_indices = numpy.linspace(
        0, len(forecast_hours) - 1, num=len(forecast_hours), dtype=int
    )
    hour_index_to_char_indices = dict()

    for i in range(len(forecast_hours)):
        this_string = ' {0:d}'.format(forecast_hours[i])
        this_start_index = forecast_hour_line.find(this_string)
        if this_start_index == -1:
            raise ValueError

        this_end_index = this_start_index + len(this_string)
        this_start_index = this_end_index - 4

        hour_index_to_char_indices[forecast_hour_indices[i]] = numpy.array(
            [this_start_index, this_end_index], dtype=int
        )

    return hour_index_to_char_indices


def _reformat_cyclone_id(orig_cyclone_id_string):
    """Reformats cyclone ID.

    :param orig_cyclone_id_string: Original cyclone ID (from raw SHIPS file), in
        format bbnnyyyy, where bb is the basin; nn is the cyclone number; and
        yyyy is the year.
    :return: cyclone_id_string: Proper cyclone ID, in format yyyybbnn.
    """

    return satellite_utils.get_cyclone_id(
        year=int(orig_cyclone_id_string[-4:]),
        basin_id_string=orig_cyclone_id_string[:2],
        cyclone_number=int(orig_cyclone_id_string[2:4])
    )


def read_file(ascii_file_name, seven_day):
    """Reads SHIPS data from ASCII file.

    :param ascii_file_name: Path to input file.
    :param seven_day: Boolean flag.  If True (False), expecting 7-day (5-day)
        file.
    :return: ships_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    # TODO(thunderhoser): Make sure all fields are found.

    if seven_day:
        forecast_hours = numpy.linspace(-12, 168, num=31, dtype=int)
    else:
        forecast_hours = numpy.linspace(-12, 120, num=23, dtype=int)

    num_forecast_fields = len(FORECAST_FIELD_NAMES_RAW)
    num_forecast_hours = len(forecast_hours)
    num_satellite_fields_raw = len(SATELLITE_FIELD_NAMES_RAW)
    num_satellite_fields_processed = len(SATELLITE_FIELD_NAMES_PROCESSED)
    num_motion_fields_processed = len(MOTION_FIELD_NAMES_PROCESSED)

    intensity_thresholds_m_s01 = KT_TO_METRES_PER_SECOND * numpy.linspace(
        20, 120, num=21, dtype=float
    )
    num_intensity_thresholds = len(intensity_thresholds_m_s01)

    cyclone_id_strings = []
    storm_latitudes_deg_n = []
    storm_longitudes_deg_e = []
    storm_times_unix_sec = []

    forecast_field_matrix = numpy.full(
        (0, num_forecast_hours, num_forecast_fields), numpy.nan
    )
    satellite_field_matrix = numpy.full(
        (0, num_satellite_fields_raw, num_satellite_fields_processed), numpy.nan
    )
    motion_field_matrix = numpy.full(
        (0, num_motion_fields_processed), numpy.nan
    )
    threshold_exceedance_matrix = numpy.full(
        (0, num_intensity_thresholds), -1, dtype=int
    )

    ascii_file_handle = open(ascii_file_name, 'r')
    hour_index_to_char_indices = dict()

    for current_line in ascii_file_handle.readlines():
        current_line = current_line.rstrip()

        try:
            words = current_line.split()
            _ = int(words[-1])

            current_line = current_line[:-4].rstrip()
        except:
            pass

        if current_line.endswith('HEAD'):
            print(current_line)

            words = current_line.split()

            cyclone_id_strings.append(
                _reformat_cyclone_id(words[-2])
            )
            storm_longitudes_deg_e.append(float(words[-4]))
            storm_latitudes_deg_n.append(float(words[-5]))

            time_string = words[-8]
            year = int(time_string[:2])

            if year > 30:
                time_string = '19{0:s}'.format(time_string)
            else:
                time_string = '20{0:s}'.format(time_string)

            time_string += words[-7]

            storm_times_unix_sec.append(
                time_conversion.string_to_unix_sec(
                    time_string, TIME_FORMAT_IN_FILES
                )
            )

            forecast_field_matrix = numpy.concatenate((
                forecast_field_matrix,
                numpy.full((1,) + forecast_field_matrix.shape[1:], numpy.nan)
            ))
            satellite_field_matrix = numpy.concatenate((
                satellite_field_matrix,
                numpy.full((1,) + satellite_field_matrix.shape[1:], numpy.nan)
            ))
            motion_field_matrix = numpy.concatenate((
                motion_field_matrix,
                numpy.full((1,) + motion_field_matrix.shape[1:], numpy.nan)
            ))
            threshold_exceedance_matrix = numpy.concatenate((
                threshold_exceedance_matrix,
                numpy.full(
                    (1,) + threshold_exceedance_matrix.shape[1:], -1, dtype=int
                )
            ))

            hour_index_to_char_indices = dict()
            continue

        if current_line.endswith('TIME'):
            hour_index_to_char_indices = _forecast_hour_to_chars(
                forecast_hour_line=current_line, seven_day=seven_day
            )
            continue

        raw_field_name = current_line.split()[-1]

        if raw_field_name in MISC_FIELD_NAMES_RAW:
            continue

        if raw_field_name in FORECAST_FIELD_NAMES_RAW:
            field_index = FORECAST_FIELD_NAMES_RAW.index(raw_field_name)

            for hour_index in hour_index_to_char_indices:
                first_char_index = hour_index_to_char_indices[hour_index][0]
                last_char_index = hour_index_to_char_indices[hour_index][1]
                this_string = current_line[first_char_index:last_char_index]

                if this_string == SENTINEL_STRING:
                    continue
                if len(this_string.strip()) == 0:
                    continue

                forecast_field_matrix[-1, hour_index, field_index] = (
                    float(this_string)
                )

            continue

        if raw_field_name in SATELLITE_FIELD_NAMES_RAW:
            raw_field_index = SATELLITE_FIELD_NAMES_RAW.index(raw_field_name)

            for hour_index in hour_index_to_char_indices:
                processed_field_index = hour_index - 2
                if processed_field_index < 0:
                    continue
                if processed_field_index >= num_satellite_fields_processed:
                    continue

                first_char_index = hour_index_to_char_indices[hour_index][0]
                last_char_index = hour_index_to_char_indices[hour_index][1]
                this_string = current_line[first_char_index:last_char_index]

                if this_string == SENTINEL_STRING:
                    continue
                if len(this_string.strip()) == 0:
                    continue

                satellite_field_matrix[
                    -1, raw_field_index, processed_field_index
                ] = float(this_string)

            continue

        if raw_field_name == MOTION_FIELD_NAME_RAW:
            for hour_index in hour_index_to_char_indices:
                motion_field_index = hour_index - 2
                if motion_field_index < 0:
                    continue
                if motion_field_index >= num_motion_fields_processed:
                    continue

                first_char_index = hour_index_to_char_indices[hour_index][0]
                last_char_index = hour_index_to_char_indices[hour_index][1]
                this_string = current_line[first_char_index:last_char_index]

                if this_string == SENTINEL_STRING:
                    continue
                if len(this_string.strip()) == 0:
                    continue

                motion_field_matrix[-1, motion_field_index] = float(this_string)

            continue

        if raw_field_name == HISTORY_FIELD_NAME_RAW:
            for hour_index in hour_index_to_char_indices:
                threshold_index = hour_index - 2
                if threshold_index < 0:
                    continue
                if threshold_index >= num_intensity_thresholds:
                    continue

                first_char_index = hour_index_to_char_indices[hour_index][0]
                last_char_index = hour_index_to_char_indices[hour_index][1]
                this_string = current_line[first_char_index:last_char_index]

                if this_string == SENTINEL_STRING:
                    continue
                if len(this_string.strip()) == 0:
                    continue

                threshold_exceedance_matrix[-1, threshold_index] = (
                    int(this_string)
                )

            continue

        raise ValueError('Field "{0:s}" not recognized.'.format(raw_field_name))

    for k in range(num_forecast_fields):
        this_function = FORECAST_FIELD_TO_CONV_FUNCTION[
            FORECAST_FIELD_NAMES_RAW[k]
        ]
        forecast_field_matrix[..., k] = this_function(
            forecast_field_matrix[..., k]
        )

    for k in range(num_satellite_fields_processed):
        this_function = SATELLITE_FIELD_CONV_FUNCTIONS[k]
        satellite_field_matrix[..., k] = this_function(
            satellite_field_matrix[..., k]
        )

    for k in range(num_motion_fields_processed):
        this_function = MOTION_FIELD_CONV_FUNCTIONS[k]
        motion_field_matrix[..., k] = this_function(motion_field_matrix[..., k])

    num_storm_objects = forecast_field_matrix.shape[0]
    storm_object_indices = numpy.linspace(
        0, num_storm_objects - 1, num=num_storm_objects, dtype=int
    )

    metadata_dict = {
        ships_io.FORECAST_HOUR_DIM: forecast_hours,
        ships_io.THRESHOLD_DIM: intensity_thresholds_m_s01,
        ships_io.LAG_TIME_DIM: numpy.array([-1, 0, 1.5, 3]),
        ships_io.STORM_OBJECT_DIM: storm_object_indices
    }

    main_data_dict = dict()
    these_dim = (ships_io.STORM_OBJECT_DIM, ships_io.FORECAST_HOUR_DIM)

    speed_index = FORECAST_FIELD_NAMES_RAW.index('SHRS')
    heading_index = FORECAST_FIELD_NAMES_RAW.index('SHTS')
    u_components_m_s01, v_components_m_s01 = (
        general_utils.speed_and_heading_to_uv(
            storm_speeds_m_s01=forecast_field_matrix[..., speed_index],
            storm_headings_deg=forecast_field_matrix[..., heading_index]
        )
    )
    main_data_dict[ships_io.SHEAR_850TO500MB_U_KEY] = (
        these_dim, u_components_m_s01
    )
    main_data_dict[ships_io.SHEAR_850TO500MB_V_KEY] = (
        these_dim, v_components_m_s01
    )

    speed_index = FORECAST_FIELD_NAMES_RAW.index('SHDC')
    heading_index = FORECAST_FIELD_NAMES_RAW.index('SDDC')
    u_components_m_s01, v_components_m_s01 = (
        general_utils.speed_and_heading_to_uv(
            storm_speeds_m_s01=forecast_field_matrix[..., speed_index],
            storm_headings_deg=forecast_field_matrix[..., heading_index]
        )
    )
    main_data_dict[ships_io.SHEAR_850TO200MB_INNER_RING_U_KEY] = (
        these_dim, u_components_m_s01
    )
    main_data_dict[ships_io.SHEAR_850TO200MB_INNER_RING_V_KEY] = (
        these_dim, v_components_m_s01
    )

    speed_index = FORECAST_FIELD_NAMES_RAW.index('SHRD')
    heading_index = FORECAST_FIELD_NAMES_RAW.index('SHTD')
    u_components_m_s01, v_components_m_s01 = (
        general_utils.speed_and_heading_to_uv(
            storm_speeds_m_s01=forecast_field_matrix[..., speed_index],
            storm_headings_deg=forecast_field_matrix[..., heading_index]
        )
    )
    main_data_dict[ships_io.SHEAR_850TO200MB_OUTER_RING_U_KEY] = (
        these_dim, u_components_m_s01
    )
    main_data_dict[ships_io.SHEAR_850TO200MB_OUTER_RING_V_KEY] = (
        these_dim, v_components_m_s01
    )

    for k in range(num_forecast_fields):
        if FORECAST_FIELD_NAMES_RAW[k] not in FORECAST_FIELD_RENAMING_DICT:
            continue

        processed_field_name = FORECAST_FIELD_RENAMING_DICT[
            FORECAST_FIELD_NAMES_RAW[k]
        ]

        if processed_field_name == ships_io.STORM_TYPE_KEY:
            these_values = numpy.round(
                forecast_field_matrix[..., k]
            ).astype(int)

            these_values = numpy.maximum(these_values, 0)
            main_data_dict[processed_field_name] = (these_dim, these_values)
        else:
            main_data_dict[processed_field_name] = (
                these_dim, forecast_field_matrix[..., k]
            )

    these_dim = (ships_io.STORM_OBJECT_DIM, ships_io.LAG_TIME_DIM)
    for k in range(num_satellite_fields_processed):
        main_data_dict[SATELLITE_FIELD_NAMES_PROCESSED[k]] = (
            these_dim, satellite_field_matrix[..., k]
        )

    these_dim = (ships_io.STORM_OBJECT_DIM, ships_io.THRESHOLD_DIM)
    # threshold_exceedance_matrix = numpy.maximum(threshold_exceedance_matrix, -1)
    main_data_dict[ships_io.THRESHOLD_EXCEEDANCE_KEY] = (
        these_dim, threshold_exceedance_matrix
    )

    these_dim = (ships_io.STORM_OBJECT_DIM,)
    for k in range(num_motion_fields_processed):
        main_data_dict[MOTION_FIELD_NAMES_PROCESSED[k]] = (
            these_dim, motion_field_matrix[:, k]
        )

    main_data_dict[ships_io.CYCLONE_ID_KEY] = (
        these_dim, numpy.array(cyclone_id_strings)
    )
    main_data_dict[ships_io.STORM_LATITUDE_KEY] = (
        these_dim, storm_latitudes_deg_n
    )
    main_data_dict[ships_io.STORM_LONGITUDE_KEY] = (
        these_dim, storm_longitudes_deg_e
    )
    main_data_dict[ships_io.VALID_TIME_KEY] = (these_dim, storm_times_unix_sec)

    ships_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    basin_id_strings = [
        satellite_utils.parse_cyclone_id(c)[1] for c in cyclone_id_strings
    ]
    multipliers = numpy.array([
        BASIN_ID_TO_LNG_MULTIPLIER[b] for b in basin_id_strings
    ])

    storm_longitudes_deg_e = (
        multipliers * ships_table_xarray[ships_io.STORM_LONGITUDE_KEY].values
    )
    storm_longitudes_deg_e[storm_longitudes_deg_e < -180.] += 360.
    storm_longitudes_deg_e[storm_longitudes_deg_e > 360.] -= 360.
    ships_table_xarray[ships_io.STORM_LONGITUDE_KEY].values = (
        lng_conversion.convert_lng_positive_in_west(
            storm_longitudes_deg_e, allow_nan=True
        )
    )

    multiplier_matrix = numpy.expand_dims(multipliers, axis=-1)
    multiplier_matrix = numpy.repeat(
        multiplier_matrix, repeats=num_forecast_hours, axis=1
    )

    forecast_lng_matrix_deg_e = (
        multiplier_matrix *
        ships_table_xarray[ships_io.FORECAST_LONGITUDE_KEY].values
    )
    forecast_lng_matrix_deg_e[forecast_lng_matrix_deg_e < -180.] += 360.
    forecast_lng_matrix_deg_e[forecast_lng_matrix_deg_e > 360.] -= 360.
    ships_table_xarray[ships_io.FORECAST_LONGITUDE_KEY].values = (
        lng_conversion.convert_lng_positive_in_west(
            forecast_lng_matrix_deg_e, allow_nan=True
        )
    )

    vortex_lng_matrix_deg_e = (
        multiplier_matrix *
        ships_table_xarray[ships_io.VORTEX_LONGITUDE_KEY].values
    )
    vortex_lng_matrix_deg_e[vortex_lng_matrix_deg_e < -180.] += 360.
    vortex_lng_matrix_deg_e[vortex_lng_matrix_deg_e > 360.] -= 360.
    ships_table_xarray[ships_io.VORTEX_LONGITUDE_KEY].values = (
        lng_conversion.convert_lng_positive_in_west(
            vortex_lng_matrix_deg_e, allow_nan=True
        )
    )

    return ships_table_xarray
