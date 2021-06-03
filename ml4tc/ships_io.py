"""IO methods for processed SHIPS data.

SHIPS = Statistical Hurricane-intensity-prediction Scheme
"""

import os
import sys
import glob
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import satellite_utils

TOLERANCE = 1e-6
CYCLONE_ID_REGEX = '[0-9][0-9][0-9][0-9][A-Z][A-Z][0-9][0-9]'

FORECAST_HOUR_DIM = 'ships_forecast_hour'
THRESHOLD_DIM = 'ships_intensity_threshold_m_s01'
LAG_TIME_DIM = 'ships_lag_time_hours'
STORM_OBJECT_DIM = 'ships_storm_object_index'
VALID_TIME_DIM = 'ships_valid_time_unix_sec'

THRESHOLD_EXCEEDANCE_KEY = 'ships_threshold_exceedance_num_6hour_periods'
CYCLONE_ID_KEY = 'ships_cyclone_id_string'
STORM_LATITUDE_KEY = 'ships_storm_latitude_deg_n'
STORM_LONGITUDE_KEY = 'ships_storm_longitude_deg_e'
VALID_TIME_KEY = 'ships_valid_time_unix_sec'

INTENSITY_KEY = 'ships_intensity_m_s01'
MINIMUM_SLP_KEY = 'ships_minimum_slp_pascals'
STORM_TYPE_KEY = 'ships_storm_type_enum'
INTENSITY_CHANGE_M12HOURS_KEY = (
    'ships_intensity_change_since_init_minus_12hours_m_s01'
)
INTENSITY_CHANGE_6HOURS_KEY = 'ships_intensity_change_6hours_m_s01'
FORECAST_LATITUDE_KEY = 'ships_forecast_latitude_deg_n'
FORECAST_LONGITUDE_KEY = 'ships_forecast_longitude_deg_e'
CLIMO_SST_KEY = 'ships_climo_sst_kelvins'
CLIMO_DEPTH_20C_KEY = 'ships_climo_depth_20c_isotherm_metres'
CLIMO_DEPTH_26C_KEY = 'ships_climo_depth_26c_isotherm_metres'
CLIMO_OHC_KEY = 'ships_climo_ocean_heat_content_j_m02'
DISTANCE_TO_LAND_KEY = 'ships_distance_to_land_metres'
OCEAN_AGE_KEY = 'ships_ocean_age_seconds'
NORMALIZED_OCEAN_AGE_KEY = 'ships_normalized_ocean_age_seconds'
REYNOLDS_SST_KEY = 'ships_reynolds_sst_kelvins'
REYNOLDS_SST_DAILY_KEY = 'ships_reynolds_sst_daily_kelvins'
REYNOLDS_SST_DAILY_AREAL_KEY = 'ships_reynolds_sst_daily_50km_diamond_kelvins'
U_WIND_200MB_OUTER_RING_KEY = 'ships_u_wind_200mb_200to800km_m_s01'
U_WIND_200MB_INNER_RING_KEY = 'ships_u_wind_200mb_0to500km_m_s01'
V_WIND_200MB_INNER_RING_KEY = 'ships_v_wind_200mb_0to500km_m_s01'
THETA_E_1000MB_OUTER_RING_KEY = 'ships_theta_e_1000mb_200to800km_kelvins'
THETA_E_SURPLUS_OUTER_RING_KEY = (
    'ships_theta_e_parcel_surplus_200to800km_kelvins'
)
THETA_E_DEFICIT_OUTER_RING_KEY = (
    'ships_theta_e_parcel_deficit_200to800km_kelvins'
)
THETA_E_SURPLUS_SATURATED_OUTER_RING_KEY = (
    'ships_theta_e_parcel_surplus_saturated_200to800km_kelvins'
)
THETA_E_DEFICIT_SATURATED_OUTER_RING_KEY = (
    'ships_theta_e_parcel_deficit_saturated_200to800km_kelvins'
)
RH_850TO700MB_OUTER_RING_KEY = 'ships_relative_humidity_850to700mb_200to800km'
RH_700TO500MB_OUTER_RING_KEY = 'ships_relative_humidity_700to500mb_200to800km'
RH_500TO300MB_OUTER_RING_KEY = 'ships_relative_humidity_500to300mb_200to800km'
VORTICITY_850MB_BIG_RING_KEY = 'ships_vorticity_850mb_0to1000km_s01'
DIVERGENCE_200MB_BIG_RING_KEY = 'ships_divergence_200mb_0to1000km_s01'
RELATIVE_EMFC_100TO600KM_KEY = (
    'ships_relative_eddy_momentum_flux_conv_100to600km_m_s02'
)
PLANETARY_EMFC_100TO600KM_KEY = (
    'ships_planetary_eddy_momentum_flux_conv_100to600km_m_s02'
)
TEMPERATURE_1000MB_OUTER_RING_KEY = (
    'ships_temperature_1000mb_200to800km_kelvins'
)
RH_1000MB_OUTER_RING_KEY = 'ships_relative_humidity_1000mb_200to800km'
HEIGHT_DEV_1000MB_OUTER_RING_KEY = (
    'ships_height_deviation_1000mb_200to800km_metres'
)
VORTEX_LATITUDE_KEY = 'ships_vortex_latitude_deg_n'
VORTEX_LONGITUDE_KEY = 'ships_vortex_longitude_deg_e'
MEAN_TAN_WIND_850MB_0TO600KM_KEY = (
    'ships_mean_tangential_wind_850mb_0to600km_m_s01'
)
MAX_TAN_WIND_850MB_KEY = 'ships_max_tangential_wind_850mb_m_s01'
TEMP_PERTURB_150MB_OUTER_RING_KEY = (
    'ships_temp_perturb_150mb_200to800km_kelvins'
)
TEMP_PERTURB_200MB_OUTER_RING_KEY = (
    'ships_temp_perturb_200mb_200to800km_kelvins'
)
TEMP_PERTURB_250MB_OUTER_RING_KEY = (
    'ships_temp_perturb_250mb_200to800km_kelvins'
)
MEAN_TAN_WIND_1000MB_500KM_KEY = (
    'ships_mean_tangential_wind_1000mb_at500km_m_s01'
)
MEAN_TAN_WIND_850MB_500KM_KEY = 'ships_mean_tangential_wind_850mb_at500km_m_s01'
MEAN_TAN_WIND_500MB_500KM_KEY = 'ships_mean_tangential_wind_500mb_at500km_m_s01'
MEAN_TAN_WIND_300MB_500KM_KEY = 'ships_mean_tangential_wind_300mb_at500km_m_s01'
TEMP_GRADIENT_850TO700MB_INNER_RING_KEY = (
    'ships_temp_gradient_850to700mb_0to500km_k_m01'
)
TEMP_ADVECTION_850TO700MB_INNER_RING_KEY = (
    'ships_temp_advection_850to700mb_0to500km_k_s01'
)
SURFACE_PRESSURE_EDGE_KEY = 'ships_sfc_pressure_vortex_edge_pascals'
SHEAR_850TO200MB_INNER_RING_U_KEY = (
    'ships_shear_850to200mb_0to500km_no_vortex_eastward_m_s01'
)
SHEAR_850TO200MB_INNER_RING_V_KEY = (
    'ships_shear_850to200mb_0to500km_no_vortex_northward_m_s01'
)
SHEAR_850TO200MB_INNER_RING_GNRL_KEY = (
    'ships_shear_850to200mb_gnrl_0to500km_no_vortex_m_s01'
)
DIVERGENCE_200MB_CENTERED_BIG_RING_KEY = (
    'ships_divergence_200mb_0to1000km_vortex_centered_s01'
)
TEMP_150MB_OUTER_RING_KEY = 'ships_temp_150mb_200to800km_kelvins'
TEMP_200MB_OUTER_RING_KEY = 'ships_temp_200mb_200to800km_kelvins'
TEMP_250MB_OUTER_RING_KEY = 'ships_temp_250mb_200to800km_kelvins'
SHEAR_850TO200MB_OUTER_RING_U_KEY = (
    'ships_shear_850to200mb_200to800km_eastward_m_s01'
)
SHEAR_850TO200MB_OUTER_RING_V_KEY = (
    'ships_shear_850to200mb_200to800km_northward_m_s01'
)
SHEAR_850TO500MB_U_KEY = 'ships_shear_850to500mb_eastward_m_s01'
SHEAR_850TO500MB_V_KEY = 'ships_shear_850to500mb_northward_m_s01'
SHEAR_850TO200MB_GENERALIZED_KEY = 'ships_shear_850to200mb_gnrl_m_s01'
SURFACE_PRESSURE_OUTER_RING_KEY = 'ships_sfc_pressure_200to800km_pascals'
MAX_PTTL_INTENSITY_KEY = 'ships_max_pttl_intensity_m_s01'
W_WIND_0TO15KM_KEY = 'ships_w_wind_0to15km_agl_m_s01'
W_WIND_0TO15KM_WEIGHTED_KEY = 'ships_w_wind_0to15km_agl_density_weighted_m_s01'
W_WIND_0TO15KM_INNER_RING_KEY = (
    'ships_w_wind_0to15km_agl_0to500km_no_vortex_m_s01'
)
SRH_1000TO700MB_OUTER_RING_KEY = 'ships_srh_1000to700mb_200to800km_j_kg01'
SRH_1000TO500MB_OUTER_RING_KEY = 'ships_srh_1000to500mb_200to800km_j_kg01'
W_WIND_500MB_BIG_RING_KEY = 'ships_w_wind_500mb_0to1000km_pa_s01'
W_WIND_700MB_BIG_RING_KEY = 'ships_w_wind_700mb_0to1000km_pa_s01'
DRY_AIR_PREDICTOR_KEY = 'ships_dry_air_predictor'
PRECIP_WATER_0TO200KM_KEY = 'ships_precipitable_water_0to200km_mm'
PRECIP_WATER_0TO200KM_STDEV_KEY = 'ships_precipitable_water_stdev_0to200km_mm'
PRECIP_WATER_200TO400KM_KEY = 'ships_precipitable_water_200to400km_mm'
PRECIP_WATER_200TO400KM_STDEV_KEY = (
    'ships_precipitable_water_stdev_200to400km_mm'
)
PRECIP_WATER_400TO600KM_KEY = 'ships_precipitable_water_400to600km_mm'
PRECIP_WATER_400TO600KM_STDEV_KEY = (
    'ships_precipitable_water_stdev_400to600km_mm'
)
PRECIP_WATER_600TO800KM_KEY = 'ships_precipitable_water_600to800km_mm'
PRECIP_WATER_600TO800KM_STDEV_KEY = (
    'ships_precipitable_water_stdev_600to800km_mm'
)
PRECIP_WATER_800TO1000KM_KEY = 'ships_precipitable_water_800to1000km_mm'
PRECIP_WATER_800TO1000KM_STDEV_KEY = (
    'ships_precipitable_water_stdev_800to1000km_mm'
)
PRECIP_WATER_0TO400KM_KEY = 'ships_precipitable_water_0to400km_mm'
PRECIP_WATER_0TO400KM_STDEV_KEY = 'ships_precipitable_water_stdev_0to400km_mm'
PRECIP_WATER_0TO600KM_KEY = 'ships_precipitable_water_0to600km_mm'
PRECIP_WATER_0TO600KM_STDEV_KEY = 'ships_precipitable_water_stdev_0to600km_mm'
PRECIP_WATER_0TO800KM_KEY = 'ships_precipitable_water_0to800km_mm'
PRECIP_WATER_0TO800KM_STDEV_KEY = 'ships_precipitable_water_stdev_0to800km_mm'
PRECIP_WATER_0TO1000KM_KEY = 'ships_precipitable_water_0to1000km_mm'
PRECIP_WATER_0TO1000KM_STDEV_KEY = 'ships_precipitable_water_stdev_0to1000km_mm'
PW_INNER_UPSHEAR_FRACTION_UNDER45MM_KEY = (
    'ships_pw_fraction_under_45mm_0to500km_upshear_quadrant'
)
PRECIP_WATER_INNER_RING_UPSHEAR_KEY = (
    'ships_precipitable_water_0to500km_upshear_quad_mm'
)
PRECIP_WATER_INNER_RING_KEY = 'ships_precipitable_water_0to500km_mm'
REYNOLDS_SST_DAILY_CLIMO_KEY = 'ships_reynolds_sst_daily_climo_kelvins'
NCODA_SST_CLIMO_KEY = 'ships_sst_ncoda_climo_kelvins'
NCODA_OHC_26C_CLIMO_KEY = 'ships_ohc_ncoda_26c_isotherm_climo_j_kg01'
NCODA_BOTTOM_CLIMO_KEY = 'ships_depth_ncoda_bottom_climo_metres'
NCODA_MAX_TEMP_CLIMO_KEY = 'ships_ocean_temp_column_max_ncoda_climo_kelvins'
DEPTH_MAX_TEMP_CLIMO_KEY = 'ships_depth_max_temp_climo_metres'
MIXED_LAYER_DEPTH_CLIMO_KEY = 'ships_ocean_mixed_layer_depth_climo_metres'
DEPTH_30C_CLIMO_KEY = 'ships_depth_30c_isotherm_climo_metres'
DEPTH_28C_CLIMO_KEY = 'ships_depth_28c_isotherm_climo_metres'
DEPTH_26C_CLIMO_KEY = 'ships_depth_26c_isotherm_climo_metres'
DEPTH_24C_CLIMO_KEY = 'ships_depth_24c_isotherm_climo_metres'
DEPTH_22C_CLIMO_KEY = 'ships_depth_22c_isotherm_climo_metres'
DEPTH_20C_CLIMO_KEY = 'ships_depth_20c_isotherm_climo_metres'
DEPTH_18C_CLIMO_KEY = 'ships_depth_18c_isotherm_climo_metres'
DEPTH_16C_CLIMO_KEY = 'ships_depth_16c_isotherm_climo_metres'
NCODA_BOTTOM_TEMP_CLIMO_KEY = 'ships_ocean_temp_ncoda_bottom_climo_kelvins'
NCODA_OHC_20C_CLIMO_KEY = 'ships_ohc_ncoda_20c_isotherm_climo_j_kg01'
NCODA_SST_KEY = 'ships_sst_ncoda_kelvins'
NCODA_SST_AREAL_KEY = 'ships_sst_ncoda_daily_50km_diamond_kelvins'
NCODA_MAX_TEMP_KEY = 'ships_ocean_temp_column_max_ncoda_kelvins'
DEPTH_MAX_TEMP_KEY = 'ships_depth_max_temp_metres'
MIXED_LAYER_DEPTH_KEY = 'ships_ocean_mixed_layer_depth_metres'
DEPTH_30C_KEY = 'ships_depth_30c_isotherm_metres'
DEPTH_28C_KEY = 'ships_depth_28c_isotherm_metres'
DEPTH_26C_KEY = 'ships_depth_26c_isotherm_metres'
DEPTH_24C_KEY = 'ships_depth_24c_isotherm_metres'
DEPTH_22C_KEY = 'ships_depth_22c_isotherm_metres'
DEPTH_20C_KEY = 'ships_depth_20c_isotherm_metres'
DEPTH_18C_KEY = 'ships_depth_18c_isotherm_metres'
DEPTH_16C_KEY = 'ships_depth_16c_isotherm_metres'
NCODA_BOTTOM_KEY = 'ships_depth_ncoda_bottom_metres'
NCODA_BOTTOM_TEMP_KEY = 'ships_ocean_temp_ncoda_bottom_kelvins'
NCODA_OHC_26C_KEY = 'ships_ohc_ncoda_26c_isotherm_j_kg01'
NCODA_OHC_20C_KEY = 'ships_ohc_ncoda_20c_isotherm_j_kg01'
SATELLITE_OHC_KEY = 'ships_ohc_satellite_j_m02'
DEPTH_20C_SATELLITE_KEY = 'ships_depth_20c_isotherm_satellite_metres'
DEPTH_26C_SATELLITE_KEY = 'ships_depth_26c_isotherm_satellite_metres'
OHC_FROM_SST_AND_CLIMO_KEY = 'ships_ohc_climo_and_sst_j_m02'

FORECAST_FIELD_NAMES = [
    THRESHOLD_EXCEEDANCE_KEY,
    CYCLONE_ID_KEY,
    STORM_LATITUDE_KEY,
    STORM_LONGITUDE_KEY,
    VALID_TIME_KEY,
    INTENSITY_KEY,
    MINIMUM_SLP_KEY,
    STORM_TYPE_KEY,
    INTENSITY_CHANGE_M12HOURS_KEY,
    INTENSITY_CHANGE_6HOURS_KEY,
    FORECAST_LATITUDE_KEY,
    FORECAST_LONGITUDE_KEY,
    CLIMO_SST_KEY,
    CLIMO_DEPTH_20C_KEY,
    CLIMO_DEPTH_26C_KEY,
    CLIMO_OHC_KEY,
    DISTANCE_TO_LAND_KEY,
    OCEAN_AGE_KEY,
    NORMALIZED_OCEAN_AGE_KEY,
    REYNOLDS_SST_KEY,
    REYNOLDS_SST_DAILY_KEY,
    REYNOLDS_SST_DAILY_AREAL_KEY,
    U_WIND_200MB_OUTER_RING_KEY,
    U_WIND_200MB_INNER_RING_KEY,
    V_WIND_200MB_INNER_RING_KEY,
    THETA_E_1000MB_OUTER_RING_KEY,
    THETA_E_SURPLUS_OUTER_RING_KEY,
    THETA_E_DEFICIT_OUTER_RING_KEY,
    THETA_E_SURPLUS_SATURATED_OUTER_RING_KEY,
    THETA_E_DEFICIT_SATURATED_OUTER_RING_KEY,
    RH_850TO700MB_OUTER_RING_KEY,
    RH_700TO500MB_OUTER_RING_KEY,
    RH_500TO300MB_OUTER_RING_KEY,
    VORTICITY_850MB_BIG_RING_KEY,
    DIVERGENCE_200MB_BIG_RING_KEY,
    RELATIVE_EMFC_100TO600KM_KEY,
    PLANETARY_EMFC_100TO600KM_KEY,
    TEMPERATURE_1000MB_OUTER_RING_KEY,
    RH_1000MB_OUTER_RING_KEY,
    HEIGHT_DEV_1000MB_OUTER_RING_KEY,
    VORTEX_LATITUDE_KEY,
    VORTEX_LONGITUDE_KEY,
    MEAN_TAN_WIND_850MB_0TO600KM_KEY,
    MAX_TAN_WIND_850MB_KEY,
    TEMP_PERTURB_150MB_OUTER_RING_KEY,
    TEMP_PERTURB_200MB_OUTER_RING_KEY,
    TEMP_PERTURB_250MB_OUTER_RING_KEY,
    MEAN_TAN_WIND_1000MB_500KM_KEY,
    MEAN_TAN_WIND_850MB_500KM_KEY,
    MEAN_TAN_WIND_500MB_500KM_KEY,
    MEAN_TAN_WIND_300MB_500KM_KEY,
    TEMP_GRADIENT_850TO700MB_INNER_RING_KEY,
    TEMP_ADVECTION_850TO700MB_INNER_RING_KEY,
    SURFACE_PRESSURE_EDGE_KEY,
    SHEAR_850TO200MB_INNER_RING_U_KEY,
    SHEAR_850TO200MB_INNER_RING_V_KEY,
    SHEAR_850TO200MB_INNER_RING_GNRL_KEY,
    DIVERGENCE_200MB_CENTERED_BIG_RING_KEY,
    TEMP_150MB_OUTER_RING_KEY,
    TEMP_200MB_OUTER_RING_KEY,
    TEMP_250MB_OUTER_RING_KEY,
    SHEAR_850TO200MB_OUTER_RING_U_KEY,
    SHEAR_850TO200MB_OUTER_RING_V_KEY,
    SHEAR_850TO500MB_U_KEY,
    SHEAR_850TO500MB_V_KEY,
    SHEAR_850TO200MB_GENERALIZED_KEY,
    SURFACE_PRESSURE_OUTER_RING_KEY,
    MAX_PTTL_INTENSITY_KEY,
    W_WIND_0TO15KM_KEY,
    W_WIND_0TO15KM_WEIGHTED_KEY,
    W_WIND_0TO15KM_INNER_RING_KEY,
    SRH_1000TO700MB_OUTER_RING_KEY,
    SRH_1000TO500MB_OUTER_RING_KEY,
    W_WIND_500MB_BIG_RING_KEY,
    W_WIND_700MB_BIG_RING_KEY,
    DRY_AIR_PREDICTOR_KEY,
    PRECIP_WATER_0TO200KM_KEY,
    PRECIP_WATER_0TO200KM_STDEV_KEY,
    PRECIP_WATER_200TO400KM_KEY,
    PRECIP_WATER_200TO400KM_STDEV_KEY,
    PRECIP_WATER_400TO600KM_KEY,
    PRECIP_WATER_400TO600KM_STDEV_KEY,
    PRECIP_WATER_600TO800KM_KEY,
    PRECIP_WATER_600TO800KM_STDEV_KEY,
    PRECIP_WATER_800TO1000KM_KEY,
    PRECIP_WATER_800TO1000KM_STDEV_KEY,
    PRECIP_WATER_0TO400KM_KEY,
    PRECIP_WATER_0TO400KM_STDEV_KEY,
    PRECIP_WATER_0TO600KM_KEY,
    PRECIP_WATER_0TO600KM_STDEV_KEY,
    PRECIP_WATER_0TO800KM_KEY,
    PRECIP_WATER_0TO800KM_STDEV_KEY,
    PRECIP_WATER_0TO1000KM_KEY,
    PRECIP_WATER_0TO1000KM_STDEV_KEY,
    PW_INNER_UPSHEAR_FRACTION_UNDER45MM_KEY,
    PRECIP_WATER_INNER_RING_UPSHEAR_KEY,
    PRECIP_WATER_INNER_RING_KEY,
    REYNOLDS_SST_DAILY_CLIMO_KEY,
    NCODA_SST_CLIMO_KEY,
    NCODA_OHC_26C_CLIMO_KEY,
    NCODA_BOTTOM_CLIMO_KEY,
    NCODA_MAX_TEMP_CLIMO_KEY,
    DEPTH_MAX_TEMP_CLIMO_KEY,
    MIXED_LAYER_DEPTH_CLIMO_KEY,
    DEPTH_30C_CLIMO_KEY,
    DEPTH_28C_CLIMO_KEY,
    DEPTH_26C_CLIMO_KEY,
    DEPTH_24C_CLIMO_KEY,
    DEPTH_22C_CLIMO_KEY,
    DEPTH_20C_CLIMO_KEY,
    DEPTH_18C_CLIMO_KEY,
    DEPTH_16C_CLIMO_KEY,
    NCODA_BOTTOM_TEMP_CLIMO_KEY,
    NCODA_OHC_20C_CLIMO_KEY,
    NCODA_SST_KEY,
    NCODA_SST_AREAL_KEY,
    NCODA_MAX_TEMP_KEY,
    DEPTH_MAX_TEMP_KEY,
    MIXED_LAYER_DEPTH_KEY,
    DEPTH_30C_KEY,
    DEPTH_28C_KEY,
    DEPTH_26C_KEY,
    DEPTH_24C_KEY,
    DEPTH_22C_KEY,
    DEPTH_20C_KEY,
    DEPTH_18C_KEY,
    DEPTH_16C_KEY,
    NCODA_BOTTOM_KEY,
    NCODA_BOTTOM_TEMP_KEY,
    NCODA_OHC_26C_KEY,
    NCODA_OHC_20C_KEY,
    SATELLITE_OHC_KEY,
    DEPTH_20C_SATELLITE_KEY,
    DEPTH_26C_SATELLITE_KEY,
    OHC_FROM_SST_AND_CLIMO_KEY
]

SATELLITE_LAG_TIME_KEY = 'ships_goes_time_lag_seconds'
SATELLITE_TEMP_0TO200KM_KEY = 'ships_goes_ch4_temp_0to200km_kelvins'
SATELLITE_TEMP_0TO200KM_STDEV_KEY = 'ships_goes_ch4_temp_stdev_0to200km_kelvins'
SATELLITE_TEMP_100TO300KM_KEY = 'ships_goes_ch4_temp_100to300km_kelvins'
SATELLITE_TEMP_100TO300KM_STDEV_KEY = (
    'ships_goes_ch4_temp_stdev_100to300km_kelvins'
)
SATELLITE_TEMP_FRACTION_BELOW_M10C_KEY = (
    'ships_goes_ch4_fraction_temp_below_m10c_50to200km'
)
SATELLITE_TEMP_FRACTION_BELOW_M20C_KEY = (
    'ships_goes_ch4_fraction_temp_below_m20c_50to200km'
)
SATELLITE_TEMP_FRACTION_BELOW_M30C_KEY = (
    'ships_goes_ch4_fraction_temp_below_m30c_50to200km'
)
SATELLITE_TEMP_FRACTION_BELOW_M40C_KEY = (
    'ships_goes_ch4_fraction_temp_below_m40c_50to200km'
)
SATELLITE_TEMP_FRACTION_BELOW_M50C_KEY = (
    'ships_goes_ch4_fraction_temp_below_m50c_50to200km'
)
SATELLITE_TEMP_FRACTION_BELOW_M60C_KEY = (
    'ships_goes_ch4_fraction_temp_below_m60c_50to200km'
)
SATELLITE_MAX_TEMP_0TO30KM_KEY = 'ships_goes_ch4_max_temp_0to30km_kelvins'
SATELLITE_MEAN_TEMP_0TO30KM_KEY = 'ships_goes_ch4_mean_temp_0to30km_kelvins'
SATELLITE_MAX_TEMP_RADIUS_KEY = 'ships_goes_ch4_max_temp_radius_metres'
SATELLITE_MIN_TEMP_20TO120KM_KEY = 'ships_goes_ch4_min_temp_20to120km_kelvins'
SATELLITE_MEAN_TEMP_20TO120KM_KEY = 'ships_goes_ch4_mean_temp_20to120km_kelvins'
SATELLITE_MIN_TEMP_RADIUS_KEY = 'ships_goes_ch4_min_temp_radius_metres'

SATELLITE_FIELD_NAMES = [
    SATELLITE_LAG_TIME_KEY,
    SATELLITE_TEMP_0TO200KM_KEY,
    SATELLITE_TEMP_0TO200KM_STDEV_KEY,
    SATELLITE_TEMP_100TO300KM_KEY,
    SATELLITE_TEMP_100TO300KM_STDEV_KEY,
    SATELLITE_TEMP_FRACTION_BELOW_M10C_KEY,
    SATELLITE_TEMP_FRACTION_BELOW_M20C_KEY,
    SATELLITE_TEMP_FRACTION_BELOW_M30C_KEY,
    SATELLITE_TEMP_FRACTION_BELOW_M40C_KEY,
    SATELLITE_TEMP_FRACTION_BELOW_M50C_KEY,
    SATELLITE_TEMP_FRACTION_BELOW_M60C_KEY,
    SATELLITE_MAX_TEMP_0TO30KM_KEY,
    SATELLITE_MEAN_TEMP_0TO30KM_KEY,
    SATELLITE_MAX_TEMP_RADIUS_KEY,
    SATELLITE_MIN_TEMP_20TO120KM_KEY,
    SATELLITE_MEAN_TEMP_20TO120KM_KEY,
    SATELLITE_MIN_TEMP_RADIUS_KEY
]

STEERING_LEVEL_PRESSURE_KEY = 'ships_steering_level_pressure_pa'
U_MOTION_KEY = 'ships_u_motion_observed_m_s01'
V_MOTION_KEY = 'ships_v_motion_observed_m_s01'
U_MOTION_1000TO100MB_KEY = 'ships_u_motion_1000to100mb_flow_m_s01'
V_MOTION_1000TO100MB_KEY = 'ships_v_motion_1000to100mb_flow_m_s01'
U_MOTION_OPTIMAL_KEY = 'ships_u_motion_optimal_flow_m_s01'
V_MOTION_OPTIMAL_KEY = 'ships_v_motion_optimal_flow_m_s01'

MOTION_FIELD_NAMES_PROCESSED = [
    STEERING_LEVEL_PRESSURE_KEY,
    U_MOTION_KEY,
    V_MOTION_KEY,
    U_MOTION_1000TO100MB_KEY,
    V_MOTION_1000TO100MB_KEY,
    U_MOTION_OPTIMAL_KEY,
    V_MOTION_OPTIMAL_KEY
]


def find_file(directory_name, cyclone_id_string, raise_error_if_missing=True):
    """Finds NetCDF file with SHIPS data.

    :param directory_name: Name of directory with SHIPS data.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `satellite_utils.parse_cyclone_id`).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: ships_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    satellite_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    ships_file_name = '{0:s}/ships_{1:s}.nc'.format(
        directory_name, cyclone_id_string
    )

    if os.path.isfile(ships_file_name) or not raise_error_if_missing:
        return ships_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        ships_file_name
    )
    raise ValueError(error_string)


def find_cyclones(directory_name, raise_error_if_all_missing=True):
    """Finds all cyclones.

    :param directory_name: Name of directory with SHIPS data.
    :param raise_error_if_all_missing: Boolean flag.  If no cyclones are found
        and `raise_error_if_all_missing == True`, will throw error.  If no
        cyclones are found and `raise_error_if_all_missing == False`, will
        return empty list.
    :return: cyclone_id_strings: List of cyclone IDs.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    file_pattern = '{0:s}/ships_{1:s}.nc'.format(
        directory_name, CYCLONE_ID_REGEX
    )
    ships_file_names = glob.glob(file_pattern)
    cyclone_id_strings = []

    for this_file_name in ships_file_names:
        try:
            cyclone_id_strings.append(
                file_name_to_cyclone_id(this_file_name)
            )
        except:
            pass

    cyclone_id_strings.sort()

    if raise_error_if_all_missing and len(cyclone_id_strings) == 0:
        error_string = (
            'Could not find any cyclone IDs from files with pattern: "{0:s}"'
        ).format(file_pattern)

        raise ValueError(error_string)

    return cyclone_id_strings


def file_name_to_cyclone_id(ships_file_names):
    """Parses cyclone ID from name of file with SHIPS data.

    :param ships_file_names: File path.
    :return: cyclone_id_string: Cyclone ID.
    """

    error_checking.assert_is_string(ships_file_names)
    pathless_file_name = os.path.split(ships_file_names)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    cyclone_id_string = extensionless_file_name.split('_')[-1]
    satellite_utils.parse_cyclone_id(cyclone_id_string)

    return cyclone_id_string


def read_file(netcdf_file_name):
    """Reads SHIPS data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: ships_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(ships_table_xarray, netcdf_file_name):
    """Writes SHIPS data to NetCDF file.

    :param ships_table_xarray: xarray table in format returned by
        `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    ships_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def concat_tables_over_storm_object(ships_tables_xarray):
    """Concatenates tables with SHIPS data over the storm-object dimension.

    :param ships_tables_xarray: 1-D list of xarray tables in format returned by
        `read_file`.
    :return: ships_table_xarray: One xarray table, in format returned by
        `read_file`, created by concatenating inputs.
    """

    num_storm_objects_found = 0

    for i in range(len(ships_tables_xarray)):
        assert numpy.array_equal(
            ships_tables_xarray[0].coords[FORECAST_HOUR_DIM].values,
            ships_tables_xarray[i].coords[FORECAST_HOUR_DIM].values
        )
        assert numpy.allclose(
            ships_tables_xarray[0].coords[THRESHOLD_DIM].values,
            ships_tables_xarray[i].coords[THRESHOLD_DIM].values,
            atol=TOLERANCE
        )
        assert numpy.allclose(
            ships_tables_xarray[0].coords[LAG_TIME_DIM].values,
            ships_tables_xarray[i].coords[LAG_TIME_DIM].values,
            atol=TOLERANCE
        )

        this_num_storm_objects = len(ships_tables_xarray[i].index)
        these_indices = numpy.linspace(
            num_storm_objects_found,
            num_storm_objects_found + this_num_storm_objects - 1,
            num=this_num_storm_objects, dtype=int
        )
        num_storm_objects_found += this_num_storm_objects

        ships_tables_xarray[i].assign_coords({
            STORM_OBJECT_DIM: these_indices
        })

    return xarray.concat(objs=ships_tables_xarray, dim=STORM_OBJECT_DIM)
