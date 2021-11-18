from fbprophet import Prophet
from pylab import rcParams
from scipy import stats, optimize
import matplotlib
import statsmodels.api as sm
import pandas as pd
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


#Membaca file excel
df = pd.read_excel("cars.xls")

#mendapatkan data dengan kategori mobil 4x2 dan mobil hemat energi
car_4x2 = df.loc[df['Category'] == '4x2']
car_4x2['Date'].min(), car_4x2['Date'].max()
energy_saving = df.loc[df['Category']=='Energy Saving']
energy_saving['Date'].min(), energy_saving['Date'].max()

#Data pre processing, membuang kolom yg tidak dipakai
cols = ['Row ID', 'Category']
car_4x2.drop(cols, axis=1, inplace=True)
car_4x2 = car_4x2.sort_values('Date')
car_4x2.isnull().sum()

#data dikelompokkan berdasarkan tanggal
car_4x2 = car_4x2.groupby('Date')['Sales'].sum().reset_index()


"""Indexing with Time Series Data"""
car_4x2 = car_4x2.set_index('Date')
car_4x2.index

#Menggunakan rerata harian penjualan mobil pada bulan yang sama
y = car_4x2['Sales']


y['2011':]


"""Visualisasi Penjualan Mobil 4x2"""
y.plot(figsize=(15, 6))
plt.title('4x2 Car Sales Data')
plt.show()

#Dekomposisi
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive',period=4)
fig = decomposition.plot()
plt.title('Time Series Decomposition 4x2 Car Sales')
plt.show()


#Menerapkan SARIMAX
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#grid search untuk menemukan parameter yang optimal
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

"""SARIMAX(1, 1, 1)x(1, 1, 0, 12) .


Fitting the ARIMA model"""
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

#Menjalankan model diagnostic
results.plot_diagnostics(figsize=(16, 8))
plt.show()


#Validasi Forecast dengan membandingkan data real vs data forecast
pred = results.get_prediction(start=pd.to_datetime('2016-01-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2011':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('4x2 Car Sales')
plt.legend()
plt.title('4x2 Car Sales One Step Ahead Forecast')
plt.show()


#Mean squared error
y_forecasted = pred.predicted_mean
y_truth = y['2016-01-31':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


"""Forecast Penjualan Mobil 4x2 vs Mobil Hemat Energy"""

car_4x2 = df.loc[df['Category'] == '4x2']
energy_saving = df.loc[df['Category'] == 'Energy Saving']
car_4x2.shape, energy_saving.shape

cols = ['Row ID','Category']
car_4x2.drop(cols, axis=1, inplace=True)
energy_saving.drop(cols, axis=1, inplace=True)

car_4x2 = car_4x2.sort_values('Date')
energy_saving = energy_saving.sort_values('Date')

car_4x2 = car_4x2.groupby('Date')['Sales'].sum().reset_index()
energy_saving = energy_saving.groupby('Date')['Sales'].sum().reset_index()

car_4x2 = car_4x2.set_index('Date')
energy_saving = energy_saving.set_index('Date')

y_4x2 = car_4x2['Sales'].resample('MS').mean()
y_energy_saving = energy_saving['Sales'].resample('MS').mean()

car_4x2 = pd.DataFrame({'Date': y_4x2.index, 'Sales': y_4x2.values})
energy_saving = pd.DataFrame({'Date': y_energy_saving.index, 'Sales': y_energy_saving.values})

store = car_4x2.merge(energy_saving, how='inner', on='Date')
store.rename(columns={'Sales_x': '4x2_sales','Sales_y': 'energy_saving_sales'}, inplace=True)
store.head()
plt.figure(figsize=(20, 8))
plt.plot(store['Date'], store['4x2_sales'],'b-', label='4x2 Car Supplies')
plt.plot(store['Date'], store['energy_saving_sales'],'r-', label='Energy Saving Car Supplies')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales of 4x2 Car & Energy Saving Car')
plt.legend()
plt.show()



#Time Series Modeling with Prophet

car_4x2 = car_4x2.rename(columns={'Date': 'ds', 'Sales': 'y'})
car_4x2_model = Prophet(interval_width=0.95)
car_4x2_model.fit(car_4x2)
energy_saving = energy_saving.rename(columns={'Date': 'ds', 'Sales': 'y'})
energy_saving_model = Prophet(interval_width=0.95)
energy_saving_model.fit(energy_saving)
car_4x2_forecast = car_4x2_model.make_future_dataframe(
    periods=72, freq='MS')
car_4x2_forecast = car_4x2_model.predict(car_4x2_forecast)
energy_saving_forecast =energy_saving_model.make_future_dataframe(periods=72, freq='MS')
energy_saving_forecast = energy_saving_model.predict(energy_saving_forecast)
plt.figure(figsize=(18, 6))
car_4x2_model.plot(car_4x2_forecast, xlabel='Date', ylabel='Sales')
plt.title('4x2 Car Sales')
plt.show()

plt.figure(figsize=(18, 6))
energy_saving_model.plot(energy_saving_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Energy Saving Car Supplies Sales');
plt.show()

#Compare Forecasts
car_4x2_names = ['4x2_%s' % column for column in car_4x2_forecast.columns]
energy_saving_names = ['energy_saving_%s' % column for column in energy_saving_forecast.columns]
merge_car_4x2_forecast = car_4x2_forecast.copy()
merge_energy_saving_forecast = energy_saving_forecast.copy()
merge_car_4x2_forecast.columns = car_4x2_names
merge_energy_saving_forecast.columns = energy_saving_names
forecast = pd.merge(merge_car_4x2_forecast, merge_energy_saving_forecast, how = 'inner', left_on = '4x2_ds', right_on = 'energy_saving_ds')
forecast = forecast.rename(columns={'4x2_ds': 'Date'}).drop('energy_saving_ds', axis=1)
forecast.head()

#Trend and Forecast Visualization
plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_trend'], 'b-')
plt.plot(forecast['Date'], forecast['energy_saving_trend'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('4x2 vs. Energy Saving Car Supplies Sales Trend');
plt.legend()
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_yhat'], 'b-')
plt.plot(forecast['Date'], forecast['energy_saving_yhat'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('4x2 vs. Energy Saving Car Supplies Estimate');
plt.legend()
plt.show()

car_4x2_model.plot_components(car_4x2_forecast);
plt.title('4x2 Car Sales Forecast')
plt.show()
energy_saving_model.plot_components(energy_saving_forecast);
plt.title('Energy Saving Car Sales Forecast')
plt.show()


"""FORECAST PENJUALAN MOBIL SEDAN DAN 4X2"""

#Membaca file excel
df = pd.read_excel("cars.xls")

#mendapatkan data dengan kategori mobil 4x2 dan mobil hemat energi
car_4x2 = df.loc[df['Category'] == '4x2']
car_4x2['Date'].min(), car_4x2['Date'].max()
sedan = df.loc[df['Category']=='Sedan']
sedan['Date'].min(), sedan['Date'].max()

#Data pre processing, membuang kolom yg tidak dipakai
cols = ['Row ID', 'Category']
sedan.drop(cols, axis=1, inplace=True)
sedan = sedan.sort_values('Date')
sedan.isnull().sum()

#data dikelompokkan berdasarkan tanggal
sedan = sedan.groupby('Date')['Sales'].sum().reset_index()


"""Indexing with Time Series Data"""
sedan = sedan.set_index('Date')
sedan.index

#Menggunaka rerata harian penjualan mobil pada bulan yang sama
y = sedan['Sales']


y['2011':]


"""Visualisasi Penjualan Mobil Sedan"""
y.plot(figsize=(15, 6))
plt.title('Sedan Car Sales Data')
plt.show()

#Dekomposisi
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive',period=4)
fig = decomposition.plot()
plt.title('Time Series Decomposition Sedan Car Sales')
plt.show()


#Menerapkan ARIMA
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#grid search untuk menemukan parameter yang optimal
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

"""SARIMAX(1, 1, 1)x(1, 1, 0, 12) .


Fitting the ARIMA model"""
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

#Menjalankan model diagnostic
results.plot_diagnostics(figsize=(16, 8))
plt.show()


#Validasi Forecast dengan membandingkan data real vs data forecast
pred = results.get_prediction(start=pd.to_datetime('2016-01-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2011':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Sedan Car Sales')
plt.legend()
plt.title('Sedan Car Sales One Step Ahead Forecast')
plt.show()


#Mean squared error
y_forecasted = pred.predicted_mean
y_truth = y['2016-01-31':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


"""Forecast Penjualan Mobil 4x2 vs Mobil Sedan"""

car_4x2 = df.loc[df['Category'] == '4x2']
sedan = df.loc[df['Category'] == 'Sedan']
car_4x2.shape, sedan.shape

cols = ['Row ID','Category']
car_4x2.drop(cols, axis=1, inplace=True)
sedan.drop(cols, axis=1, inplace=True)

car_4x2 = car_4x2.sort_values('Date')
sedan = sedan.sort_values('Date')

car_4x2 = car_4x2.groupby('Date')['Sales'].sum().reset_index()
sedan = sedan.groupby('Date')['Sales'].sum().reset_index()

car_4x2 = car_4x2.set_index('Date')
sedan = sedan.set_index('Date')

y_4x2 = car_4x2['Sales'].resample('MS').mean()
y_sedan = sedan['Sales'].resample('MS').mean()

car_4x2 = pd.DataFrame({'Date': y_4x2.index, 'Sales': y_4x2.values})
sedan = pd.DataFrame({'Date': y_sedan.index, 'Sales': y_sedan.values})

store = car_4x2.merge(sedan, how='inner', on='Date')
store.rename(columns={'Sales_x': '4x2_sales','Sales_y': 'sedan_sales'}, inplace=True)
store.head()
plt.figure(figsize=(20, 8))
plt.plot(store['Date'], store['4x2_sales'],'b-', label='4x2 Car Supplies')
plt.plot(store['Date'], store['sedan_sales'],'r-', label='Sedan Car Supplies')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales of 4x2 Car & Sedan Car')
plt.legend()
plt.show()



#Time Series Modeling with Prophet

car_4x2 = car_4x2.rename(columns={'Date': 'ds', 'Sales': 'y'})
car_4x2_model = Prophet(interval_width=0.95)
car_4x2_model.fit(car_4x2)
sedan = sedan.rename(columns={'Date': 'ds', 'Sales': 'y'})
sedan_model = Prophet(interval_width=0.95)
sedan_model.fit(sedan)
car_4x2_forecast = car_4x2_model.make_future_dataframe(
    periods=72, freq='MS')
car_4x2_forecast = car_4x2_model.predict(car_4x2_forecast)
sedan_forecast =sedan_model.make_future_dataframe(periods=72, freq='MS')
sedan_forecast = sedan_model.predict(sedan_forecast)
plt.figure(figsize=(18, 6))
car_4x2_model.plot(car_4x2_forecast, xlabel='Date', ylabel='Sales')
plt.title('4x2 Car Sales')
plt.show()

plt.figure(figsize=(18, 6))
sedan_model.plot(sedan_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Sedan Car Supplies Sales');
plt.show()

#Compare Forecasts
car_4x2_names = ['4x2_%s' % column for column in car_4x2_forecast.columns]
sedan_names = ['sedan_%s' % column for column in sedan_forecast.columns]
merge_car_4x2_forecast = car_4x2_forecast.copy()
merge_sedan_forecast = sedan_forecast.copy()
merge_car_4x2_forecast.columns = car_4x2_names
merge_sedan_forecast.columns = sedan_names
forecast = pd.merge(merge_car_4x2_forecast, merge_sedan_forecast, how = 'inner', left_on = '4x2_ds', right_on = 'sedan_ds')
forecast = forecast.rename(columns={'4x2_ds': 'Date'}).drop('sedan_ds', axis=1)
forecast.head()

#Trend and Forecast Visualization
plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_trend'], 'b-')
plt.plot(forecast['Date'], forecast['sedan_trend'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('4x2 vs. Sedan Car Supplies Sales Trend');
plt.legend()
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_yhat'], 'b-')
plt.plot(forecast['Date'], forecast['sedan_yhat'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('4x2 vs. Sedan Car Supplies Estimate');
plt.legend()
plt.show()

car_4x2_model.plot_components(car_4x2_forecast);
plt.title('4x2 Car Sales Forecast')
plt.show()
sedan_model.plot_components(sedan_forecast);
plt.title('Sedan Car Sales Forecast')
plt.show()




"""FORECAST PENJUALAN MOBIL 4X2 VS 4X4"""

#Membaca file excel
df = pd.read_excel("cars.xls")

#mendapatkan data dengan kategori mobil 4x2 dan mobil hemat energi
car_4x2 = df.loc[df['Category'] == '4x2']
car_4x2['Date'].min(), car_4x2['Date'].max()
car_4x4 = df.loc[df['Category']=='4x4']
car_4x4['Date'].min(), car_4x4['Date'].max()

#Data pre processing, membuang kolom yg tidak dipakai
cols = ['Row ID', 'Category']
car_4x4.drop(cols, axis=1, inplace=True)
car_4x4 = car_4x4.sort_values('Date')
car_4x4.isnull().sum()

#data dikelompokkan berdasarkan tanggal
car_4x4 = car_4x4.groupby('Date')['Sales'].sum().reset_index()


"""Indexing with Time Series Data"""
car_4x4 = car_4x4.set_index('Date')
car_4x4.index

#Menggunaka rerata harian penjualan mobil pada bulan yang sama
y = car_4x4['Sales']


y['2011':]


"""Visualisasi Penjualan Mobil 4x4"""
y.plot(figsize=(15, 6))
plt.title('4x4 Car Sales Data')
plt.show()

#Dekomposisi
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive',period=4)
fig = decomposition.plot()
plt.title('Time Series Decomposition 4x4 Car Sales')
plt.show()


#Menerapkan ARIMA
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#grid search untuk menemukan parameter yang optimal
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

"""SARIMAX(1, 1, 1)x(1, 1, 0, 12) .


Fitting the ARIMA model"""
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

#Menjalankan model diagnostic
results.plot_diagnostics(figsize=(16, 8))
plt.show()


#Validasi Forecast dengan membandingkan data real vs data forecast
pred = results.get_prediction(start=pd.to_datetime('2016-01-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2011':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('4x4 Car Sales')
plt.legend()
plt.title('4x4 Car Sales One Step Ahead Forecast')
plt.show()


#Mean squared error
y_forecasted = pred.predicted_mean
y_truth = y['2016-01-31':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


"""Forecast Penjualan Mobil 4x2 vs Mobil 4x4"""

car_4x2 = df.loc[df['Category'] == '4x2']
car_4x4 = df.loc[df['Category'] == '4x4']
car_4x2.shape, car_4x4.shape

cols = ['Row ID','Category']
car_4x2.drop(cols, axis=1, inplace=True)
car_4x4.drop(cols, axis=1, inplace=True)

car_4x2 = car_4x2.sort_values('Date')
car_4x4 = car_4x4.sort_values('Date')

car_4x2 = car_4x2.groupby('Date')['Sales'].sum().reset_index()
car_4x4 = car_4x4.groupby('Date')['Sales'].sum().reset_index()

car_4x2 = car_4x2.set_index('Date')
car_4x4 = car_4x4.set_index('Date')

y_4x2 = car_4x2['Sales'].resample('MS').mean()
y_car_4x4 = car_4x4['Sales'].resample('MS').mean()

car_4x2 = pd.DataFrame({'Date': y_4x2.index, 'Sales': y_4x2.values})
car_4x4 = pd.DataFrame({'Date': y_car_4x4.index, 'Sales': y_car_4x4.values})

store = car_4x2.merge(car_4x4, how='inner', on='Date')
store.rename(columns={'Sales_x': '4x2_sales','Sales_y': 'car_4x4_sales'}, inplace=True)
store.head()
plt.figure(figsize=(20, 8))
plt.plot(store['Date'], store['4x2_sales'],'b-', label='4x2 Car Supplies')
plt.plot(store['Date'], store['car_4x4_sales'],'r-', label='4x4 Car Supplies')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales of 4x2 Car & 4x4 Car')
plt.legend()
plt.show()



#Time Series Modeling with Prophet

car_4x2 = car_4x2.rename(columns={'Date': 'ds', 'Sales': 'y'})
car_4x2_model = Prophet(interval_width=0.95)
car_4x2_model.fit(car_4x2)
car_4x4 = car_4x4.rename(columns={'Date': 'ds', 'Sales': 'y'})
car_4x4_model = Prophet(interval_width=0.95)
car_4x4_model.fit(car_4x4)
car_4x2_forecast = car_4x2_model.make_future_dataframe(periods=72, freq='MS')
car_4x2_forecast = car_4x2_model.predict(car_4x2_forecast)
car_4x4_forecast =car_4x4_model.make_future_dataframe(periods=72, freq='MS')
car_4x4_forecast = car_4x4_model.predict(car_4x4_forecast)
plt.figure(figsize=(18, 6))
car_4x2_model.plot(car_4x2_forecast, xlabel='Date', ylabel='Sales')
plt.title('4x2 Car Sales')
plt.show()

plt.figure(figsize=(18, 6))
car_4x4_model.plot(car_4x4_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('4x4 Car Supplies Sales');
plt.show()

#Compare Forecasts
car_4x2_names = ['4x2_%s' % column for column in car_4x2_forecast.columns]
car_4x4_names = ['4x4_%s' % column for column in car_4x4_forecast.columns]
merge_car_4x2_forecast = car_4x2_forecast.copy()
merge_car_4x4_forecast = car_4x4_forecast.copy()
merge_car_4x2_forecast.columns = car_4x2_names
merge_car_4x4_forecast.columns = car_4x4_names
forecast = pd.merge(merge_car_4x2_forecast, merge_car_4x4_forecast, how = 'inner', left_on = '4x2_ds', right_on = '4x4_ds')
forecast = forecast.rename(columns={'4x2_ds': 'Date'}).drop('4x4_ds', axis=1)
forecast.head()

#Trend and Forecast Visualization
plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_trend'], 'b-')
plt.plot(forecast['Date'], forecast['4x4_trend'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('4x2 vs. car_4x4 Car Supplies Sales Trend');
plt.legend()
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_yhat'], 'b-')
plt.plot(forecast['Date'], forecast['4x4_yhat'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('4x2 vs. 4x4 Car Supplies Estimate');
plt.legend()
plt.show()

car_4x2_model.plot_components(car_4x2_forecast);
plt.title('4x2 Car Sales Forecast')
plt.show()
car_4x4_model.plot_components(car_4x4_forecast);
plt.title('4x4 Car Sales Forecast')
plt.show()


"""CAR PRODUCTION FORECASTING"""

df = pd.read_excel("cars.xls")
car_4x2 = df.loc[df['Category'] == '4x2']
car_4x2['Date'].min(), car_4x2['Date'].max()
energy_saving = df.loc[df['Category']=='Energy Saving']
energy_saving['Date'].min(), energy_saving['Date'].max()


"""Analisa produksi mobil 4x2"""

cols = ['Row ID', 'Category']
car_4x2.drop(cols, axis=1, inplace=True)
car_4x2 = car_4x2.sort_values('Date')
car_4x2.isnull().sum()

car_4x2 = car_4x2.groupby('Date')['Production'].sum().reset_index()


"""Indexing with Time Series Data"""
car_4x2 = car_4x2.set_index('Date')
car_4x2.index

y = car_4x2['Production']

"""Menilik data produksi mobil 4x2 tahun 2011."""
y['2011':]


"""Visualisasi Data Produksi Mobil 4x2"""
y.plot(figsize=(15, 6))
plt.title('4x2 Car Production Data')
plt.show()

#Dekomposisi / seasonal decomposition
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive',period=4)
fig = decomposition.plot()
plt.title('Time Series Decomposition 4x2 Car Production')
plt.show()


"""Menggunakan ARIMA (Autoregressive Integrated Moving Average) untuk melakukan forecasting. Model ARIMA dinotasikan dengan ARIMA(p,d,q). Ketiga parameter tersebut merujuk ke seasonality, trend, dan noise"""
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

"""Menggunakan grid search untuk menemukan parameter optimal"""
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

"""Fitting the ARIMA model"""
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

"""Menjalankan model diagnostic"""
results.plot_diagnostics(figsize=(16, 8))
plt.title('Model Diagnostic')
plt.show()

"""Validasi Forecast
Visualisasi Forecast dibandingkan dengan Data Real 
"""

pred = results.get_prediction(
    start=pd.to_datetime('2016-01-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2011':].plot(label='observed')
pred.predicted_mean.plot(
    ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('4x2 Car Production')
plt.legend()
plt.title('4x2 Car Production One Step Ahead Forecast')
plt.show()


#Mean Squarred error
y_forecasted = pred.predicted_mean
y_truth = y['2016-01-31':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('Mean Squared Error : {}'.format(round(mse, 2)))
print('Root Mean Squared :  {}'.format(round(np.sqrt(mse), 2)))



"""Time Series Forecast of 4x2 Cars vs. Energy Saving Cars Production"""

car_4x2 = df.loc[df['Category'] == '4x2']
energy_saving = df.loc[df['Category'] == 'Energy Saving']
car_4x2.shape, energy_saving.shape

"""Data Exploration
Membandingkan produksi mobile 4x2 dan mobil hemat energy"""

cols = ['Row ID','Category']
car_4x2.drop(cols, axis=1, inplace=True)
energy_saving.drop(cols, axis=1, inplace=True)

car_4x2 = car_4x2.sort_values('Date')
energy_saving = energy_saving.sort_values('Date')

car_4x2 = car_4x2.groupby('Date')['Production'].sum().reset_index()
energy_saving = energy_saving.groupby('Date')['Production'].sum().reset_index()

car_4x2 = car_4x2.set_index('Date')
energy_saving = energy_saving.set_index('Date')

y_4x2 = car_4x2['Production'].resample('MS').mean()
y_energy_saving = energy_saving['Production'].resample('MS').mean()

car_4x2 = pd.DataFrame({'Date': y_4x2.index, 'Production': y_4x2.values})
energy_saving = pd.DataFrame({'Date': y_energy_saving.index, 'Production': y_energy_saving.values})

store = car_4x2.merge(energy_saving, how='inner', on='Date')
store.rename(columns={'Production_x': '4x2_Production','Production_y': 'energy_saving_Production'}, inplace=True)
store.head()
plt.figure(figsize=(20, 8))
plt.plot(store['Date'], store['4x2_Production'],'b-', label='4x2 Car Production')
plt.plot(store['Date'], store['energy_saving_Production'],'r-', label='Energy Saving Car Production')
plt.xlabel('Date')
plt.ylabel('Production')
plt.title('Production of 4x2 Car & Energy Saving Car')
plt.legend()
plt.show()



"""Time Series Modeling with Prophet"""


car_4x2 = car_4x2.rename(columns={'Date': 'ds', 'Production': 'y'})
car_4x2_model = Prophet(interval_width=0.95)
car_4x2_model.fit(car_4x2)
energy_saving = energy_saving.rename(columns={'Date': 'ds', 'Production': 'y'})
energy_saving_model = Prophet(interval_width=0.95)
energy_saving_model.fit(energy_saving)
car_4x2_forecast = car_4x2_model.make_future_dataframe(
    periods=72, freq='MS')
car_4x2_forecast = car_4x2_model.predict(car_4x2_forecast)
energy_saving_forecast =energy_saving_model.make_future_dataframe(periods=72, freq='MS')
energy_saving_forecast = energy_saving_model.predict(energy_saving_forecast)
plt.figure(figsize=(18, 6))
car_4x2_model.plot(car_4x2_forecast, xlabel='Date', ylabel='Production')
plt.title('4x2 Car Production')
plt.show()

plt.figure(figsize=(18, 6))
energy_saving_model.plot(energy_saving_forecast, xlabel = 'Date', ylabel = 'Production')
plt.title('Energy Saving Car Production');
plt.show()

"""Compare Forecasts
"""

car_4x2_names = ['4x2_%s' % column for column in car_4x2_forecast.columns]
energy_saving_names = ['energy_saving_%s' % column for column in energy_saving_forecast.columns]
merge_car_4x2_forecast = car_4x2_forecast.copy()
merge_energy_saving_forecast = energy_saving_forecast.copy()
merge_car_4x2_forecast.columns = car_4x2_names
merge_energy_saving_forecast.columns = energy_saving_names
forecast = pd.merge(merge_car_4x2_forecast, merge_energy_saving_forecast, how = 'inner', left_on = '4x2_ds', right_on = 'energy_saving_ds')
forecast = forecast.rename(columns={'4x2_ds': 'Date'}).drop('energy_saving_ds', axis=1)
forecast.head()

#Trend and Forecast Visualization
plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_trend'], 'b-')
plt.plot(forecast['Date'], forecast['energy_saving_trend'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Production')
plt.title('4x2 vs. Energy Saving Car Supplies Production Trend');
plt.legend()
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_yhat'], 'b-')
plt.plot(forecast['Date'], forecast['energy_saving_yhat'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Production')
plt.title('4x2 vs. Energy Saving Car Production Estimate');
plt.legend()
plt.show()

car_4x2_model.plot_components(car_4x2_forecast);
plt.title('4x2 Car Production Forecast')
plt.show()
energy_saving_model.plot_components(energy_saving_forecast);
plt.title('Energy Saving Car Production Forecast')
plt.show()


"""FORECAST PRODUKSI MOBIL SEDAN VS 4X2"""

df = pd.read_excel("cars.xls")
car_4x2 = df.loc[df['Category'] == '4x2']
car_4x2['Date'].min(), car_4x2['Date'].max()
sedan = df.loc[df['Category']=='Sedan']
sedan['Date'].min(), sedan['Date'].max()


"""Analisa produksi mobil Sedan"""

cols = ['Row ID', 'Category']
sedan.drop(cols, axis=1, inplace=True)
sedan = sedan.sort_values('Date')
sedan.isnull().sum()

sedan = sedan.groupby('Date')['Production'].sum().reset_index()


"""Indexing with Time Series Data"""
sedan = sedan.set_index('Date')
sedan.index

y = sedan['Production']

"""Menilik data produksi mobil sedan tahun 2011."""
y['2011':]


"""Visualisasi Data Produksi Mobil sedan"""
y.plot(figsize=(15, 6))
plt.title('Sedan Car Production Data')
plt.show()

#Dekomposisi / seasonal decomposition
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive',period=4)
fig = decomposition.plot()
plt.title('Time Series Decomposition Sedan Car Production')
plt.show()


"""Menggunakan ARIMA (Autoregressive Integrated Moving Average) untuk melakukan forecasting. Model ARIMA dinotasikan dengan ARIMA(p,d,q). Ketiga parameter tersebut merujuk ke seasonality, trend, dan noise"""
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

"""Menggunakan grid search untuk menemukan parameter optimal"""
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

"""Fitting the ARIMA model"""
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

"""Menjalankan model diagnostic"""
results.plot_diagnostics(figsize=(16, 8))
plt.title('Model Diagnostic - Sedan Car Production')
plt.show()

"""Validasi Forecast
Visualisasi Forecast dibandingkan dengan Data Real 
"""

pred = results.get_prediction(
    start=pd.to_datetime('2016-01-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2011':].plot(label='observed')
pred.predicted_mean.plot(
    ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Sedan Car Production')
plt.legend()
plt.title('Sedan Car Production One Step Ahead Forecast')
plt.show()


#Mean Squarred error
y_forecasted = pred.predicted_mean
y_truth = y['2016-01-31':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('Mean Squared Error : {}'.format(round(mse, 2)))
print('Root Mean Squared :  {}'.format(round(np.sqrt(mse), 2)))



"""Time Series Forecast of 4x2 Cars vs. Sedan Cars Production"""

car_4x2 = df.loc[df['Category'] == '4x2']
sedan = df.loc[df['Category'] == 'Sedan']
car_4x2.shape, sedan.shape

"""Data Exploration
Membandingkan produksi mobile 4x2 dan mobil hemat energy"""

cols = ['Row ID','Category']
car_4x2.drop(cols, axis=1, inplace=True)
sedan.drop(cols, axis=1, inplace=True)

car_4x2 = car_4x2.sort_values('Date')
sedan = sedan.sort_values('Date')

car_4x2 = car_4x2.groupby('Date')['Production'].sum().reset_index()
sedan = sedan.groupby('Date')['Production'].sum().reset_index()

car_4x2 = car_4x2.set_index('Date')
sedan = sedan.set_index('Date')

y_4x2 = car_4x2['Production'].resample('MS').mean()
y_sedan = sedan['Production'].resample('MS').mean()

car_4x2 = pd.DataFrame({'Date': y_4x2.index, 'Production': y_4x2.values})
sedan = pd.DataFrame({'Date': y_sedan.index, 'Production': y_sedan.values})

store = car_4x2.merge(sedan, how='inner', on='Date')
store.rename(columns={'Production_x': '4x2_Production','Production_y': 'sedan_Production'}, inplace=True)
store.head()
plt.figure(figsize=(20, 8))
plt.plot(store['Date'], store['4x2_Production'],'b-', label='4x2 Car Production')
plt.plot(store['Date'], store['sedan_Production'],'r-', label='Sedan Car Production')
plt.xlabel('Date')
plt.ylabel('Production')
plt.title('Production of 4x2 Car & Sedan Car')
plt.legend()
plt.show()



"""Time Series Modeling with Prophet"""


car_4x2 = car_4x2.rename(columns={'Date': 'ds', 'Production': 'y'})
car_4x2_model = Prophet(interval_width=0.95)
car_4x2_model.fit(car_4x2)
sedan = sedan.rename(columns={'Date': 'ds', 'Production': 'y'})
sedan_model = Prophet(interval_width=0.95)
sedan_model.fit(sedan)
car_4x2_forecast = car_4x2_model.make_future_dataframe(
    periods=72, freq='MS')
car_4x2_forecast = car_4x2_model.predict(car_4x2_forecast)
sedan_forecast =sedan_model.make_future_dataframe(periods=72, freq='MS')
sedan_forecast = sedan_model.predict(sedan_forecast)
plt.figure(figsize=(18, 6))

car_4x2_model.plot(car_4x2_forecast, xlabel='Date', ylabel='Production')
plt.title('4x2 Car Production')
plt.show()

plt.figure(figsize=(18, 6))
sedan_model.plot(sedan_forecast, xlabel = 'Date', ylabel = 'Production')
plt.title('Sedan Car Production');
plt.show()

"""Compare Forecasts
"""

car_4x2_names = ['4x2_%s' % column for column in car_4x2_forecast.columns]
sedan_names = ['sedan_%s' % column for column in sedan_forecast.columns]
merge_car_4x2_forecast = car_4x2_forecast.copy()
merge_sedan_forecast = sedan_forecast.copy()
merge_car_4x2_forecast.columns = car_4x2_names
merge_sedan_forecast.columns = sedan_names
forecast = pd.merge(merge_car_4x2_forecast, merge_sedan_forecast, how = 'inner', left_on = '4x2_ds', right_on = 'sedan_ds')
forecast = forecast.rename(columns={'4x2_ds': 'Date'}).drop('sedan_ds', axis=1)
forecast.head()

#Trend and Forecast Visualization
plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_trend'], 'b-')
plt.plot(forecast['Date'], forecast['sedan_trend'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Production')
plt.title('4x2 vs. Sedan Car Production Trend');
plt.legend()
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_yhat'], 'b-')
plt.plot(forecast['Date'], forecast['sedan_yhat'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Production')
plt.title('4x2 vs. Sedan Car Production Estimate');
plt.legend()
plt.show()

car_4x2_model.plot_components(car_4x2_forecast);
plt.title('4x2 Car Production Forecast')
plt.show()
sedan_model.plot_components(sedan_forecast);
plt.title('Sedan Car Production Forecast')
plt.show()




"""FORECAST PRODUKSI MOBIL 4X4 VS 4X2"""
df = pd.read_excel("cars.xls")
car_4x2 = df.loc[df['Category'] == '4x2']
car_4x2['Date'].min(), car_4x2['Date'].max()
car_4x4 = df.loc[df['Category']=='4x4']
car_4x4['Date'].min(), car_4x4['Date'].max()


"""Analisa produksi mobil 4xd"""

cols = ['Row ID', 'Category']
car_4x4.drop(cols, axis=1, inplace=True)
car_4x4 = car_4x4.sort_values('Date')
car_4x4.isnull().sum()

car_4x4 = car_4x4.groupby('Date')['Production'].sum().reset_index()


"""Indexing with Time Series Data"""
car_4x4 = car_4x4.set_index('Date')
car_4x4.index

y = car_4x4['Production']

"""Menilik data produksi mobil 4x2 tahun 2011."""
y['2011':]


"""Visualisasi Data Produksi Mobil 4x2"""
y.plot(figsize=(15, 6))
plt.title('4x4 Car Production Data')
plt.show()

#Dekomposisi / seasonal decomposition
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive',period=4)
fig = decomposition.plot()
plt.title('Time Series Decomposition 4x4 Car Production')
plt.show()


"""Menggunakan ARIMA (Autoregressive Integrated Moving Average) untuk melakukan forecasting. Model ARIMA dinotasikan dengan ARIMA(p,d,q). Ketiga parameter tersebut merujuk ke seasonality, trend, dan noise"""
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

"""Menggunakan grid search untuk menemukan parameter optimal"""
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

"""Fitting the ARIMA model"""
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

"""Menjalankan model diagnostic"""
results.plot_diagnostics(figsize=(16, 8))
plt.title('Model Diagnostic')
plt.show()

"""Validasi Forecast
Visualisasi Forecast dibandingkan dengan Data Real 
"""

pred = results.get_prediction(
    start=pd.to_datetime('2016-01-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2011':].plot(label='observed')
pred.predicted_mean.plot(
    ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('4x4 Car Production')
plt.legend()
plt.title('4x4 Car Production One Step Ahead Forecast')
plt.show()


#Mean Squarred error
y_forecasted = pred.predicted_mean
y_truth = y['2016-01-31':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('Mean Squared Error : {}'.format(round(mse, 2)))
print('Root Mean Squared :  {}'.format(round(np.sqrt(mse), 2)))



"""Time Series Forecast of 4x2 Cars vs. Energy Saving Cars Production"""

car_4x2 = df.loc[df['Category'] == '4x2']
car_4x4 = df.loc[df['Category'] == 'Energy Saving']
car_4x2.shape, car_4x4.shape

"""Data Exploration
Membandingkan produksi mobile 4x2 dan mobil hemat energy"""

cols = ['Row ID','Category']
car_4x2.drop(cols, axis=1, inplace=True)
car_4x4.drop(cols, axis=1, inplace=True)

car_4x2 = car_4x2.sort_values('Date')
car_4x4 = car_4x4.sort_values('Date')

car_4x2 = car_4x2.groupby('Date')['Production'].sum().reset_index()
car_4x4 = car_4x4.groupby('Date')['Production'].sum().reset_index()

car_4x2 = car_4x2.set_index('Date')
car_4x4 = car_4x4.set_index('Date')

y_4x2 = car_4x2['Production'].resample('MS').mean()
y_car_4x4 = car_4x4['Production'].resample('MS').mean()

car_4x2 = pd.DataFrame({'Date': y_4x2.index, 'Production': y_4x2.values})
car_4x4 = pd.DataFrame({'Date': y_car_4x4.index, 'Production': y_car_4x4.values})

store = car_4x2.merge(car_4x4, how='inner', on='Date')
store.rename(columns={'Production_x': '4x2_Production','Production_y': '4x4_Production'}, inplace=True)
store.head()
plt.figure(figsize=(20, 8))
plt.plot(store['Date'], store['4x2_Production'],'b-', label='4x4 Car Production')
plt.plot(store['Date'], store['4x4_Production'],'r-', label='4x4 Car Production')
plt.xlabel('Date')
plt.ylabel('Production')
plt.title('Production of 4x2 Car & 4x4 Car')
plt.legend()
plt.show()



"""Time Series Modeling with Prophet"""


car_4x2 = car_4x2.rename(columns={'Date': 'ds', 'Production': 'y'})
car_4x2_model = Prophet(interval_width=0.95)
car_4x2_model.fit(car_4x2)
car_4x4 = car_4x4.rename(columns={'Date': 'ds', 'Production': 'y'})
car_4x4_model = Prophet(interval_width=0.95)
car_4x4_model.fit(car_4x4)
car_4x2_forecast = car_4x2_model.make_future_dataframe(
    periods=72, freq='MS')
car_4x2_forecast = car_4x2_model.predict(car_4x2_forecast)
car_4x4_forecast =car_4x4_model.make_future_dataframe(periods=72, freq='MS')
car_4x4_forecast = car_4x4_model.predict(car_4x4_forecast)
plt.figure(figsize=(18, 6))
car_4x2_model.plot(car_4x2_forecast, xlabel='Date', ylabel='Production')
plt.title('4x2 Car Production')
plt.show()

plt.figure(figsize=(18, 6))
car_4x4_model.plot(car_4x4_forecast, xlabel = 'Date', ylabel = 'Production')
plt.title('4x4 Car Production');
plt.show()

"""Compare Forecasts
"""

car_4x2_names = ['4x2_%s' % column for column in car_4x2_forecast.columns]
car_4x4_names = ['4x4_%s' % column for column in car_4x4_forecast.columns]
merge_car_4x2_forecast = car_4x2_forecast.copy()
merge_car_4x4_forecast = car_4x4_forecast.copy()
merge_car_4x2_forecast.columns = car_4x2_names
merge_car_4x4_forecast.columns = car_4x4_names
forecast = pd.merge(merge_car_4x2_forecast, merge_car_4x4_forecast, how = 'inner', left_on = '4x2_ds', right_on = '4x4_ds')
forecast = forecast.rename(columns={'4x2_ds': 'Date'}).drop('4x4_ds', axis=1)
forecast.head()

#Trend and Forecast Visualization
plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_trend'], 'b-')
plt.plot(forecast['Date'], forecast['4x4_trend'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Production')
plt.title('4x2 vs. 4x4 Car Supplies Production Trend');
plt.legend()
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['4x2_yhat'], 'b-')
plt.plot(forecast['Date'], forecast['4x4_yhat'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Production')
plt.title('4x2 vs. 4x4 Car Production Estimate');
plt.legend()
plt.show()

car_4x2_model.plot_components(car_4x2_forecast);
plt.title('4x2 Car Production Forecast')
plt.show()
car_4x4_model.plot_components(car_4x4_forecast);
plt.title('4x4 Car Production Forecast')
plt.show()