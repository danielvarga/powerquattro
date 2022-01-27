# adapted from
# https://nbviewer.org/github/pvlib/pvlib-python/blob/master/docs/tutorials/forecast_to_power.ipynb

# built-in python modules
import datetime
import inspect
import os
import sys

# scientific python add-ons
import numpy as np
import pandas as pd

# plotting stuff
# first line makes the plots appear in the notebook
import matplotlib.pyplot as plt
import matplotlib as mpl

# finally, we import the pvlib library
from pvlib import solarposition, irradiance, atmosphere, pvsystem, inverter, temperature
from pvlib.forecast import GFS, NAM, NDFD, RAP, HRRR



met_filename, pow_filename = sys.argv[1:]

met_data = pd.read_csv(met_filename, sep=';', skipinitialspace=True)

pow_data = pd.read_csv(pow_filename, sep=';', skipinitialspace=True)

met_data['Time'] = pd.to_datetime(met_data['Time'], format='%Y%m%d%H%M')
met_data = met_data.set_index('Time')

pow_data['Time'] = (pow_data['DATE'] + ' ' + pow_data['TIME']).str.strip()
pow_data['Time'] = pd.to_datetime(pow_data['Time'], format='%Y. %m. %d %H:%M:%S') # 2021. 04. 20;02:02:00
pow_data['Time'] = pow_data['Time'] - pd.DateOffset(hours=2)
pow_data = pow_data.set_index('Time')





# Choose a location.
# Tucson, AZ
latitude = 32.2
longitude = -110.9
tz = 'US/Mountain'

# Budapest
latitude = 47.4979
longitude = 19.0402
# Fót
#latitude = 47.6173
#longitude = 19.1892
tz = 'CET'

surface_tilt = 30
surface_azimuth = 180 # pvlib uses 0=North, 90=East, 180=South, 270=West convention
albedo = 0.2

start = pd.Timestamp(datetime.date.today(), tz=tz) # today's date
end = start + pd.Timedelta(days=7) # 7 days from today

# Define forecast model
fm = GFS()
#fm = NAM()
#fm = NDFD()
#fm = RAP()
#fm = HRRR()

# Retrieve data
forecast_data = fm.get_processed_data(latitude, longitude, start, end)

print(forecast_data.head())


forecast_data['temp_air'].plot()
plt.show()


ghi = forecast_data['ghi']
ghi.plot()
plt.ylabel('Irradiance ($W/m^{-2}$)')
plt.show()


# retrieve time and location parameters
time = forecast_data.index
a_point = fm.location
solpos = a_point.get_solarposition(time)
solpos.plot()
plt.show()

dni_extra = irradiance.get_extra_radiation(fm.time)
# dni_extra.plot()
# plt.ylabel('Extra terrestrial radiation ($W/m^{-2}$)')
# plt.show()

airmass = atmosphere.get_relative_airmass(solpos['apparent_zenith'])

# airmass.plot()
# plt.ylabel('Airmass')
# plt.show()

poa_sky_diffuse = irradiance.haydavies(surface_tilt, surface_azimuth,
                                       forecast_data['dhi'], forecast_data['dni'], dni_extra,
                                       solpos['apparent_zenith'], solpos['azimuth'])

poa_ground_diffuse = irradiance.get_ground_diffuse(surface_tilt, ghi, albedo=albedo)

aoi = irradiance.aoi(surface_tilt, surface_azimuth, solpos['apparent_zenith'], solpos['azimuth'])

poa_irrad = irradiance.poa_components(aoi, forecast_data['dni'], poa_sky_diffuse, poa_ground_diffuse)

poa_irrad.plot()
plt.ylabel('Irradiance ($W/m^{-2}$)')
plt.title('POA Irradiance')
plt.show()


ambient_temperature = forecast_data['temp_air']
wnd_spd = forecast_data['wind_speed']
thermal_params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
pvtemp = temperature.sapm_cell(poa_irrad['poa_global'], ambient_temperature, wnd_spd, **thermal_params)

pvtemp.plot()
plt.ylabel('Temperature (C)')
plt.show()


sandia_modules = pvsystem.retrieve_sam('SandiaMod')
sandia_module = sandia_modules.Canadian_Solar_CS5P_220M___2009_


effective_irradiance = pvsystem.sapm_effective_irradiance(poa_irrad.poa_direct, poa_irrad.poa_diffuse, 
                                                          airmass, aoi, sandia_module)

sapm_out = pvsystem.sapm(effective_irradiance, pvtemp, sandia_module)
#print(sapm_out.head())

sapm_out[['p_mp']].plot()
plt.ylabel('DC Power (W)')
plt.show()

sapm_inverters = pvsystem.retrieve_sam('sandiainverter')
sapm_inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

p_ac = inverter.sandia(sapm_out.v_mp, sapm_out.p_mp, sapm_inverter)

p_ac.plot()
plt.ylabel('AC Power (W)')
plt.ylim(0, None)
plt.show()

# integrate power to find energy yield over the forecast period
energy_yield = p_ac.sum() * 3
print("total energy yield", energy_yield)
