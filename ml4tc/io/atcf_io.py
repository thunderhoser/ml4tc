"""IO methods for raw ATCF (Automated TC-forecasting System) data."""

import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT_IN_FILES = '%Y%m%d%H'

FORECAST_HOUR_FIELD_NAMES = [
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
    'ND24', 'ND22', 'ND20', 'ND18', 'ND16', 'NDFR', 'NTFR', 'NOHC', 'NO20'
]

# 'PC00', 'PCM1', 'PCM3'

RAW_TO_PROCESSED_FIELD_NAME = {
    'VMAX': 'intensity_m_s01',
    'MSLP': 'minimum_slp_pascals',
    'TYPE': 'storm_type_enum',
    'DELV': 'intensity_change_since_init_minus_12hours_m_s01',
    'INCV': 'intensity_change_6hours_m_s01',
    'LAT': 'forecast_latitude_deg_n',
    'LON': 'forecast_longitude_deg_e',
    'CSST': 'climo_sst_kelvins',
    'CD20': 'climo_depth_20c_isotherm_metres',
    'CD26': 'climo_depth_26c_isotherm_metres',
    'COHC': 'climo_ocean_heat_content_j_m02',
    'DTL': 'distance_to_land_metres',
    'OAGE': 'ocean_age_seconds',
    'NAGE': 'normalized_ocean_age_seconds',
    'RSST': 'reynolds_sst_kelvins',
    'DSST': 'reynolds_sst_daily_kelvins',
    'DSTA': 'reynolds_sst_daily_50km_diamond_kelvins',
    'U200': 'u_wind_200mb_200to800km_m_s01',
    'U20C': 'u_wind_200mb_0to500km_m_s01',
    'V20C': 'v_wind_200mb_0to500km_m_s01',
    'E000': 'theta_e_1000mb_200to800km_kelvins',
    'EPOS': 'theta_e_parcel_surplus_200to800km_kelvins',
    'ENEG': 'theta_e_parcel_deficit_200to800km_kelvins',
    'EPSS': 'theta_e_parcel_surplus_saturated_200to800km_kelvins',
    'ENSS': 'theta_e_parcel_deficit_saturated_200to800km_kelvins',
    'RHLO': 'relative_humidity_850to700mb_200to800km',
    'RHMD': 'relative_humidity_700to500mb_200to800km',
    'RHHI': 'relative_humidity_500to300mb_200to800km',
    'Z850': 'vorticity_850mb_0to1000km_s01',
    'D200': 'divergence_200mb_0to1000km_s01',
    'REFC': 'relative_eddy_momentum_flux_conv_100to600km_m_s02',
    'PEFC': 'planetary_eddy_momentum_flux_conv_100to600km_m_s02',
    'T000': 'temperature_1000mb_200to800km_kelvins',
    'R000': 'relative_humidity_1000mb_200to800km_kelvins',
    'Z000': 'height_deviation_1000mb_200to800km_metres',
    'TLAT': 'vortex_latitude_deg_n',
    'TLON': 'vortex_longitude_deg_e',
    'TWAC': 'mean_tangential_wind_850mb_0to600km_m_s01',
    'TWXC': 'max_tangential_wind_850mb_m_s01',
    'G150': 'temp_perturb_150mb_200to800km_kelvins',
    'G200': 'temp_perturb_200mb_200to800km_kelvins',
    'G250': 'temp_perturb_250mb_200to800km_kelvins',
    'V000': 'mean_tangential_wind_1000mb_at500km_m_s01',
    'V850': 'mean_tangential_wind_850mb_at500km_m_s01',
    'V500': 'mean_tangential_wind_500mb_at500km_m_s01',
    'V300': 'mean_tangential_wind_300mb_at500km_m_s01',
    'TGRD': 'temp_gradient_850to700mb_0to500km_k_m01',
    'TADV': 'temp_advection_850to700mb_0to500km_k_s01',
    'PENC': 'sfc_pressure_vortex_edge_pascals',
    'SHDC': 'shear_850to200mb_0to500km_no_vortex_m_s01',
    'SDDC': 'shear_850to200mb_0to500km_no_vortex_heading_deg',
    'SHGC': 'shear_850to200mb_gnrl_0to500km_no_vortex_m_s01',
    'DIVC': 'divergence_200mb_0to1000km_vortex_centered_s01',
    'T150': 'temp_150mb_200to800km_kelvins',
    'T200': 'temp_200mb_200to800km_kelvins',
    'T250': 'temp_250mb_200to800km_kelvins',
    'SHRD': 'shear_850to200mb_200to800km_m_s01',
    'SHTD': 'shear_850to200mb_200to800km_heading_deg',
    'SHRS': 'shear_850to500mb_m_s01',
    'SHTS': 'shear_850to500mb_heading_deg',
    'SHRG': 'shear_850to200mb_gnrl_m_s01',
    'PENV': 'sfc_pressure_200to800km_pascals',
    'VMPI': 'max_pttl_intensity_m_s01',
    'VVAV': 'w_wind_0to15km_agl_m_s01',
    'VMFX': 'w_wind_0to15km_agl_density_weighted_m_s01',
    'VVAC': 'w_wind_0to15km_agl_0to500km_no_vortex_m_s01',
    'HE07': 'srh_1000to700mb_200to800km_j_kg01',
    'HE05': 'srh_1000to500mb_200to800km_j_kg01',
    'O500': 'w_wind_500mb_0to1000km_pa_s01',
    'O700': 'w_wind_700mb_0to1000km_pa_s01',
    'CFLX': 'dry_air_predictor',  # units?
    'PW01': 'precipitable_water_0to200km_mm',
    'PW02': 'precipitable_water_stdev_0to200km_mm',
    'PW03': 'precipitable_water_200to400km_mm',
    'PW04': 'precipitable_water_stdev_200to400km_mm',
    'PW05': 'precipitable_water_400to600km_mm',
    'PW06': 'precipitable_water_stdev_400to600km_mm',
    'PW07': 'precipitable_water_600to800km_mm',
    'PW08': 'precipitable_water_stdev_600to800km_mm',
    'PW09': 'precipitable_water_800to1000km_mm',
    'PW10': 'precipitable_water_stdev_800to1000km_mm',
    'PW11': 'precipitable_water_0to400km_mm',
    'PW12': 'precipitable_water_stdev_0to400km_mm',
    'PW13': 'precipitable_water_0to600km_mm',
    'PW14': 'precipitable_water_stdev_0to600km_mm',
    'PW15': 'precipitable_water_0to800km_mm',
    'PW16': 'precipitable_water_stdev_0to800km_mm',
    'PW17': 'precipitable_water_0to1000km_mm',
    'PW18': 'precipitable_water_stdev_0to1000km_mm',
    'PW19': 'pw_fraction_under_45mm_0to500km_upshear_quadrant',
    'PW20': 'precipitable_water_0to500km_upshear_quad_mm',
    'PW21': 'precipitable_water_0to500km_mm',
    'XDST': 'reynolds_sst_daily_climo_kelvins',
    'XNST': 'sst_ncoda_climo_kelvins',
    'XOHC': 'ohc_ncoda_26c_isotherm_climo_j_kg01',
    'XDFR': 'depth_ncoda_bottom_climo_metres',
    'XTMX': 'ocean_temp_column_max_ncoda_climo_kelvins',
    'XDTX': 'foo',
    'XDML': 'ocean_mixed_layer_depth_climo_metres',
    'XD30': 'depth_30c_isotherm_climo_metres',
    'XD28': 'depth_28c_isotherm_climo_metres',
    'XD26': 'depth_26c_isotherm_climo_metres',
    'XD24': 'depth_24c_isotherm_climo_metres',
    'XD22': 'depth_22c_isotherm_climo_metres',
    'XD20': 'depth_20c_isotherm_climo_metres',
    'XD18': 'depth_18c_isotherm_climo_metres',
    'XD16': 'depth_16c_isotherm_climo_metres',
    'XTFR': 'ocean_temp_ncoda_bottom_climo_kelvins',
    'XO20': 'ohc_ncoda_20c_isotherm_climo_j_kg01',
    'NSST': 'sst_ncoda_kelvins',
    'NSTA': 'foo',
    'NTMX': 'ocean_temp_column_max_ncoda_kelvins',
    'NDTX': 'foo',
    'NDML': 'ocean_mixed_layer_depth_metres',
    'ND30': 'depth_30c_isotherm_metres',
    'ND28': 'depth_28c_isotherm_metres',
    'ND26': 'depth_26c_isotherm_metres',
    'ND24': 'depth_24c_isotherm_metres',
    'ND22': 'depth_22c_isotherm_metres',
    'ND20': 'depth_20c_isotherm_metres',
    'ND18': 'depth_18c_isotherm_metres',
    'ND16': 'depth_16c_isotherm_metres',
    'NDFR': 'depth_ncoda_bottom_metres',
    'NTFR': 'ocean_temp_ncoda_bottom_kelvins',
    'NOHC': 'ohc_ncoda_26c_isotherm_j_kg01',  # from (J kg^-1) - Celsius????
    'NO20': 'ohc_ncoda_20c_isotherm_j_kg01'  # relative to an isotherm????
}

FIELD_NAME_TO_CONV_FUNCTION = {
    'VMAX': _get_multiply_function(KT_TO_METRES_PER_SECOND),
    'MSLP': _get_multiply_function(MB_TO_PASCALS),
    'TYPE': _get_multiply_function(1.),  # integer
    'DELV': _get_multiply_function(KT_TO_METRES_PER_SECOND),
    'INCV': _get_multiply_function(KT_TO_METRES_PER_SECOND),
    'LAT': _get_multiply_function(0.1),
    'LON': _get_multiply_function(0.1),
    'CSST': _decicelsius_to_kelvins(),
    'CD20': _get_multiply_function(1.),
    'CD26': _get_multiply_function(1.),
    'COHC': _get_multiply_function(1e7),
    'DTL': _get_multiply_function(1000.),
    'OAGE': _get_multiply_function(1. / 360),
    'NAGE': _get_multiply_function(1. / 360),
    'RSST': _decicelsius_to_kelvins(),
    'DSST': _decicelsius_to_kelvins(),
    'DSTA': _decicelsius_to_kelvins(),
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
    'T000': _decicelsius_to_kelvins(),
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
    'PENC': _pressure_from_1000mb_departure_to_pa(),
    'SHDC': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'SDDC': _get_multiply_function(1.),
    'SHGC': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'DIVC': _get_multiply_function(1e-7),
    'T150': _decicelsius_to_kelvins(),
    'T200': _decicelsius_to_kelvins(),
    'T250': _decicelsius_to_kelvins(),
    'SHRD': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'SHTD': _get_multiply_function(1.),
    'SHRS': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'SHTS': _get_multiply_function(1.),
    'SHRG': _get_multiply_function(0.1 * KT_TO_METRES_PER_SECOND),
    'PENV': _pressure_from_1000mb_departure_to_pa(),
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
    'PW19': _get_multiply_function(0.01),
    'PW20': _get_multiply_function(0.1),
    'PW21': _get_multiply_function(0.1),
    'XDST': _decicelsius_to_kelvins(),
    'XNST': _decicelsius_to_kelvins(),
    'XOHC': 'ohc_ncoda_26c_isotherm_climo_j_kg01',
    'XDFR': _get_multiply_function(1.),
    'XTMX': _decicelsius_to_kelvins(),
    'XDTX': 'foo',
    'XDML': _get_multiply_function(1.),
    'XD30': _get_multiply_function(1.),
    'XD28': _get_multiply_function(1.),
    'XD26': _get_multiply_function(1.),
    'XD24': _get_multiply_function(1.),
    'XD22': _get_multiply_function(1.),
    'XD20': _get_multiply_function(1.),
    'XD18': _get_multiply_function(1.),
    'XD16': _get_multiply_function(1.),
    'XTFR': _decicelsius_to_kelvins(),
    'XO20': 'ohc_ncoda_20c_isotherm_climo_j_kg01',
    'NSST': _decicelsius_to_kelvins(),
    'NSTA': 'foo',
    'NTMX': _decicelsius_to_kelvins(),
    'NDTX': 'foo',
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
    'NTFR': _decicelsius_to_kelvins(),
    'NOHC': 'ohc_ncoda_26c_isotherm_j_kg01',  # from (J kg^-1) - Celsius????
    'NO20': 'ohc_ncoda_20c_isotherm_j_kg01'  # relative to an isotherm????
}


def _forecast_hour_to_chars(forecast_hour_line, seven_day):
    """Determines correspondence of forecast hour to character indices.

    :param forecast_hour_line: Line from ATCF file with forecast hours.  This
        line should end with "TIME".
    :param seven_day: Boolean flag.  If True (False), line comes from 7-day
        (5-day) file.
    :return: hour_index_to_first_char_index: Dictionary, where each key is the
        index of a forecast hour and the corresponding value is the first
        character index.  This dictionary tells the user, for the given entry in
        the ATCF file (i.e., one storm at one time), where data will be found
        for each forecast hour.
    :return: hour_index_to_last_char_index: Same as above but with last
        character index.
    """

    error_checking.assert_is_boolean(seven_day)
    assert forecast_hour_line.endswith('TIME')

    if seven_day:
        all_forecast_hours = numpy.linspace(-12, 168, num=31, dtype=int)
    else:
        all_forecast_hours = numpy.linspace(-12, 120, num=23, dtype=int)

    forecast_hours = numpy.array([
        int(word) for word in forecast_hour_line.split()[:-1]
    ], dtype=int)

    forecast_hour_indices = numpy.array([
        numpy.where(all_forecast_hours == h)[0][0] for h in forecast_hours
    ], dtype=int)

    hour_index_to_first_char_index = dict()
    hour_index_to_last_char_index = dict()

    for i in range(len(forecast_hours)):
        this_string = '{0:d}'.format(forecast_hours[i])
        this_start_index = forecast_hour_line.find(this_string)
        if this_start_index == -1:
            raise ValueError

        this_end_index = this_start_index + len(this_string)
        this_start_index = this_end_index - 4

        hour_index_to_first_char_index[forecast_hour_indices[i]] = (
            this_start_index
        )
        hour_index_to_last_char_index[forecast_hour_indices[i]] = this_end_index

    return hour_index_to_first_char_index, hour_index_to_last_char_index


def read_file(ascii_file_name, seven_day):
    """Reads ATCF data from ASCII file.

    :param ascii_file_name: Path to input file.
    :param seven_day: Boolean flag.  If True (False), expecting 7-day (5-day)
        file.
    :return: atcf_table_xarray: xarray table.  Documentation in the xarray table
        should make values self-explanatory.
    """

    assert not seven_day

    # TODO(thunderhoser): Deal with unit conversions.
    # TODO(thunderhoser): Make sure all fields are found.
    # TODO(thunderhoser): Make sure no unknown fields are found.

    num_forecast_hour_fields = len(FORECAST_HOUR_FIELD_NAMES)

    forecast_hour_field_matrix = numpy.full(
        (0, 23, num_forecast_hour_fields), numpy.nan
    )
    storm_id_strings = []
    storm_latitudes_deg_n = []
    storm_longitudes_deg_e = []
    storm_times_unix_sec = []

    hour_index_to_first_char_index = dict()
    hour_index_to_last_char_index = dict()

    ascii_file_handle = open(ascii_file_name, 'r')

    for this_line in ascii_file_handle.readlines():
        this_line = this_line.rstrip()

        try:
            these_words = this_line.split()
            _ = int(these_words[-1])

            this_line = this_line[:-4].rstrip()
        except:
            pass

        if this_line.endswith('HEAD'):
            print(this_line)

            these_words = this_line.split()
            storm_id_strings.append(these_words[-2])
            storm_longitudes_deg_e.append(float(these_words[-4]))
            storm_latitudes_deg_n.append(float(these_words[-5]))

            this_time_string = these_words[-8]
            this_year = int(this_time_string[:2])

            if this_year > 30:
                this_time_string = '19{0:s}'.format(this_time_string)
            else:
                this_time_string = '20{0:s}'.format(this_time_string)

            this_time_string += these_words[-7]

            storm_times_unix_sec.append(
                time_conversion.string_to_unix_sec(
                    this_time_string, TIME_FORMAT_IN_FILES
                )
            )

            forecast_hour_field_matrix = numpy.concatenate((
                forecast_hour_field_matrix,
                numpy.full((1, 23, num_forecast_hour_fields), numpy.nan)
            ))

            hour_index_to_first_char_index = dict()
            hour_index_to_last_char_index = dict()

            continue

        if this_line.endswith('TIME'):
            hour_index_to_first_char_index, hour_index_to_last_char_index = (
                _forecast_hour_to_chars(
                    forecast_hour_line=this_line, seven_day=seven_day
                )
            )

            continue

        try:
            this_field_index = FORECAST_HOUR_FIELD_NAMES.index(this_line.split()[-1])
        except:
            continue

        if not hour_index_to_first_char_index:
            raise ValueError

        for this_hour_index in hour_index_to_first_char_index:
            this_first_char_index = hour_index_to_first_char_index[this_hour_index]
            this_last_char_index = hour_index_to_last_char_index[
                this_hour_index]

            this_string = this_line[this_first_char_index:this_last_char_index]

            if this_string == '9999' or len(this_string.strip()) == 0:
                continue

            forecast_hour_field_matrix[
                -1, this_hour_index, this_field_index
            ] = float(this_string)

    print(forecast_hour_field_matrix.shape)
    print(forecast_hour_field_matrix)
