from flask import Flask, request, jsonify
import numpy as np
import requests
import time
import pymysql
import pandas as pd
import threading
from datetime import datetime, timedelta
from neuralprophet import NeuralProphet, set_random_seed
from sklearn.metrics import r2_score, mean_squared_error
import logging
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# MySQL Configuration
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'db': 'SmartWeather'
}

# URL API
url = "https://3uojc35gb6.execute-api.ap-southeast-2.amazonaws.com/SmartWeather/SmartWeatherAgriculture"

# Global variable to store data
data_cache = []
forecast_hourly_cache = []
forecast_daily_cache = []

def fetch_and_sort_data_from_api():
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    # Tambahkan 7 jam pada setiap Timestamp
    for item in data:
        timestamp_str = item['TimeStamp']
        timestamp = datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S %Y\n')
        timestamp += timedelta(hours=7)  # Menambah 7 jam
        item['TimeStamp'] = timestamp.strftime('%a %b %d %H:%M:%S %Y\n')
        
    # Urutkan data berdasarkan TimeStamp yang sudah diubah
    sorted_data = sorted(data, key=lambda x: pd.to_datetime(x['TimeStamp'], format='%a %b %d %H:%M:%S %Y\n'))
    
    return sorted_data

# Route to fetch and sort data
@app.route('/fetch_and_sort_data', methods=['GET'])
def fetch_and_sort_data_route():
    try:
        sorted_data = fetch_and_sort_data_from_api()
        return jsonify(sorted_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(fetch_and_sort_data_from_api())

# Transform WindDirection
def transform_wind_direction(WindDirection):
    if WindDirection == "{\"":
        return "U"
    try:
        WindDirection = int(WindDirection)
        if WindDirection > 1 and WindDirection < 90:
            return "TL"
        elif WindDirection == 90:
            return "T"
        elif WindDirection > 90 and WindDirection < 180:
            return "TG"
        elif WindDirection == 180:
            return "S"
        elif WindDirection > 180 and WindDirection < 270:
            return "BD"
        elif WindDirection == 270:
            return "B"
        elif WindDirection > 270 and WindDirection < 360:
            return "BL"
        else:
            return "U"
    except ValueError:
        return "U"
    

# Check if a record with the given timestamp exists in the specified table
def record_exists(cursor, table_name, timestamp):
    query = f"SELECT COUNT(1) FROM {table_name} WHERE timestamp = %s"
    cursor.execute(query, (timestamp,))
    return cursor.fetchone()[0] > 0

# Insert data to MySQL
def insert_data_to_mysql(data, table_name, convert_winddir=True):
    connection = pymysql.connect(**mysql_config)
    cursor = connection.cursor()

    if table_name == 'dataperjam' or table_name == 'dataperhari':
        query = f"""
        INSERT INTO {table_name} (timestamp, temp, hum, press, uv, rainfall, windspeed, winddir)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        for item in data:
            try:
                timestamp = item.get('DATA TIMESTAMP', None)
                if record_exists(cursor, table_name, timestamp):
                    continue
                temp = item.get('Temperature', None)
                hum = item.get('Humidity', None)
                press = item.get('Pressure', None)
                uv = item.get('UV', None)
                rainfall = item.get('Rainfall', None)
                windspeed = item.get('WindSpeed', None)
                winddir = transform_wind_direction(item.get('WindDirection', 0)) if convert_winddir else item.get('WindDirection', 0)
                
                cursor.execute(query, (timestamp, temp, hum, press, uv, rainfall, windspeed, winddir))
            except KeyError as e:
                print(f"Missing key in data: {e}")
    else:
        query = f"""
        INSERT INTO {table_name} (timestamp, temp, hum, press, uv, rainfall, windspeed, winddir, latitude, longitude)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for item in data:
            try:
                timestamp = item.get('DATA TIMESTAMP', None)
                if record_exists(cursor, table_name, timestamp):
                    continue
                temp = item.get('Temperature', None)
                hum = item.get('Humidity', None)
                press = item.get('Pressure', None)
                uv = item.get('UV', None)
                rainfall = item.get('Rainfall', None)
                windspeed = item.get('WindSpeed', None)
                winddir = transform_wind_direction(item.get('WindDirection', 0)) if convert_winddir else item.get('WindDirection', 0)
                latitude = item.get('Latitude', None)
                longitude = item.get('Longitude', None)

                cursor.execute(query, (timestamp, temp, hum, press, uv, rainfall, windspeed, winddir, latitude, longitude))
            except KeyError as e:
                print(f"Missing key in data: {e}")
    
    connection.commit()
    cursor.close()
    connection.close()

@app.route('/update_weather_data', methods=['GET'])
def update_weather_data():
    data = fetch_and_sort_data_from_api()
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)    
    
    # Ensure all required columns are present
    required_columns = ['TS', 'Temperature', 'Humidity', 'Pressure', 'UV', 'Rainfall', 'WindSpeed', 'WindDirection']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column in data: {col}")
            return jsonify({'status': 'error', 'message': f'Missing column in data: {col}'})

    df['DATA TIMESTAMP'] = pd.to_datetime(df['TimeStamp'])
    df.set_index('DATA TIMESTAMP', inplace=True)
    
    # # Fill NaN values with 0 or an appropriate value
    # df.fillna(0, inplace=True)
    
    # Reset index to prepare for database insertion
    df.reset_index(inplace=True)
    
    # Convert DataFrame to dictionary for database insertion
    data_to_insert = df.to_dict(orient='records')
    
    insert_data_to_mysql(data_to_insert, 'datasensor')
    return jsonify({'status': 'success', 'message': 'Data inserted successfully'})

@app.route('/update_weather_hourly', methods=['GET'])
def update_weather_hourly():
    data = fetch_and_sort_data_from_api()
    
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)    
    
    # Ensure all required columns are present
    required_columns = ['TS', 'Temperature', 'Humidity', 'Pressure', 'UV', 'Rainfall', 'WindSpeed', 'WindDirection']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column in data: {col}")
            return jsonify({'status': 'error', 'message': f'Missing column in data: {col}'})

    df['DATA TIMESTAMP'] = pd.to_datetime(df['TimeStamp'])
    df.set_index('DATA TIMESTAMP', inplace=True)
    
    # Keep rainfall data from the last timestamp of each hour
    hourly_rainfall_data = df.groupby(pd.Grouper(freq='H'))['Rainfall'].last().reset_index()
    
    # Drop rainfall column before averaging
    df = df.drop(columns=['Rainfall'])
    
    # Rata-ratakan data per 1 jam
    hourly_data = df.resample('1H').mean(numeric_only=True).reset_index()
    
    # Handle WindDirection separately
    wind_direction_data = df[['WindDirection']].copy()
    hourly_wind_direction = wind_direction_data.resample('1H').apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 0).reset_index()
    
    hourly_data['Rainfall'] = hourly_rainfall_data['Rainfall'] # Set to last rainfall value
    hourly_data['WindDirection'] = hourly_wind_direction['WindDirection']
    
    # Fill NaN values with 0 or an appropriate value
    hourly_data = hourly_data.dropna()
    
    # # Transform WindDirection values if necessary
    # hourly_data['WindDirection'] = hourly_data['WindDirection'].apply(transform_wind_direction)
    
    insert_data_to_mysql(hourly_data.to_dict(orient='records'), 'dataperjam', convert_winddir=False)

    return jsonify({'status': 'success', 'message': 'Hourly data inserted successfully'})


@app.route('/update_weather_daily', methods=['GET'])
def update_weather_daily():
    data = fetch_and_sort_data_from_api()
    
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)    
    
    # Ensure all required columns are present
    required_columns = ['TS', 'Temperature', 'Humidity', 'Pressure', 'UV', 'Rainfall', 'WindSpeed', 'WindDirection']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column in data: {col}")
            return jsonify({'status': 'error', 'message': f'Missing column in data: {col}'})

    df['DATA TIMESTAMP'] = pd.to_datetime(df['TimeStamp'])
    df.set_index('DATA TIMESTAMP', inplace=True)
    
    # Keep rainfall data from the last timestamp of each day
    daily_rainfall_data = df.groupby(pd.Grouper(freq='D'))['Rainfall'].last().reset_index()
    
    # Drop rainfall column before averaging
    df = df.drop(columns=['Rainfall'])
    
    # Rata-ratakan data per 1 hari
    daily_data = df.resample('24H').mean(numeric_only=True).reset_index()
    
    # Handle WindDirection separately
    wind_direction_data = df[['WindDirection']].copy()
    daily_wind_direction = wind_direction_data.resample('24H').apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 0).reset_index()
    
    daily_data['Rainfall'] = daily_rainfall_data['Rainfall'] # Set to last rainfall value  # Set to last rainfall value
    daily_data['WindDirection'] = daily_wind_direction['WindDirection']
    
    # Fill NaN values with 0 or an appropriate value
    daily_data = daily_data.dropna()
    
    
    # # Transform WindDirection values if necessary
    # daily_data['WindDirection'] = daily_data['WindDirection'].apply(transform_wind_direction)
    insert_data_to_mysql(daily_data.to_dict(orient='records'), 'dataperhari', convert_winddir=False)
    
    return jsonify({'status': 'success', 'message': 'Daily data inserted successfully'})


# Fetch data from MySQL
def fetch_data_from_mysql(table_name):
    connection = pymysql.connect(**mysql_config)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

def fetch_data_from_mysql_latestdata(table_name):
    connection = pymysql.connect(**mysql_config)
    query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 1"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

uploaded_timestamps = set()

@app.route('/upload_weather_data', methods=['GET'])
def upload_weather_data():
    url = 'https://tanamap.drik.my.id/api/weather-data'
    token_bearer = 'GE70VpDZTMeVPDJbbJ1UsuFHvMcXX6JIwbtJ8cDGccc1170b'
    headers = {'Authorization': f'Bearer {token_bearer}'}

    data = fetch_data_from_mysql_latestdata('datasensor')
    data.drop(columns=['id'], inplace=True, errors='ignore')  # Drop 'id' column if it exists
    print(f'Weather data: {data}')  # Debug print to check forecast data
    
    success_count = 0
    failure_count = 0
    failures = []

    for index, item in data.iterrows():
        timestamp = pd.to_datetime(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        
        # Skip if the timestamp has already been uploaded
        if timestamp in uploaded_timestamps:
            continue

        print(f'Uploading item: {item}')  # Debug print to check each item
        payload = {
            "timestamp": timestamp,
            "temp": item['temp'],
            "hum": item['hum'],
            "press": item['press'],
            "uv": item['uv'],
            "rainfall": item['rainfall'],
            "windspeed": item['windspeed'],
            "winddir": item['winddir'],
            "latitude": item['latitude'],
            "longitude": item['longitude'],
        }
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            uploaded_timestamps.add(timestamp)
            success_count += 1
        else:
            failure_count += 1
            failures.append({"data": payload, "error": response.text})
    
    result = {
        "success_count": success_count,
        "failure_count": failure_count,
        "failures": failures
    }

    return jsonify(result)


@app.route('/predict_hourly', methods=['GET'])
def predict_hourly():
    global forecast_hourly_cache

    # Fetch data from MySQL
    data = fetch_data_from_mysql('dataperjam')
    if data.empty:
        return jsonify({"error": "No data available in the table."}), 400
    data.drop(columns=['id'], inplace=True)  # Drop 'id' column if it exists

    # Preprocessing
    data['DATA TIMESTAMP'] = pd.to_datetime(data['timestamp'])
    data.set_index('DATA TIMESTAMP', inplace=True)
    data.rename(columns={
        'rainfall': 'RAINFALL 24H RRRR',
        'temp': 'TEMP DRYBULB C TTTTTT',
        'winddir': 'WIND DIR DEG DD',
        'windspeed': 'WIND SPEED FF',
        'hum': 'RELATIVE HUMIDITY PC',
        'press': 'PRESSURE QFF MB DERIVED',
        'uv': 'INTENSITAS MATAHARI'
    }, inplace=True)
    cols_to_int = ['RAINFALL 24H RRRR', 'TEMP DRYBULB C TTTTTT', 'WIND DIR DEG DD', 'WIND SPEED FF', 'RELATIVE HUMIDITY PC', 'PRESSURE QFF MB DERIVED', 'INTENSITAS MATAHARI']
    for col in cols_to_int:
        data[col] = data[col].astype(float)

    def RAINFALL24HRRRR (val):
        if val < 0 or val > 1000:
            val = np.NaN
        return(val)

    def TEMPDRYBULBCTTTTTT (val):
        if val < 22 or val > 50:
            val = np.NaN
        return(val)

    def WINDDIRDEGDD (val):
        if val < 0 or val > 360:
            val = np.NaN
        return(val)

    def WINDSPEEDFF (val):
        if val > 100:
            val = np.NaN
        return(val)

    def RELATIVEHUMIDITYPC (val):
        if val < 0 or val> 100:
            val = np.NaN
        return(val)

    def PRESSUREQFFMBDERIVED (val):
        if val < 1000 or val> 1018:
            val = np.NaN
        return(val)

    def INTENSITASMATAHARI (val):
        if val < 0 or val> 1000:
            val = np.NaN
        return(val)

    data['RAINFALL 24H RRRR'] = data.apply(lambda row : RAINFALL24HRRRR(row['RAINFALL 24H RRRR']), axis = 1)
    data['TEMP DRYBULB C TTTTTT'] = data.apply(lambda row : TEMPDRYBULBCTTTTTT(row['TEMP DRYBULB C TTTTTT']), axis = 1)
    data['WIND DIR DEG DD'] = data.apply(lambda row : WINDDIRDEGDD(row['WIND DIR DEG DD']), axis = 1)
    data['WIND SPEED FF'] = data.apply(lambda row : WINDSPEEDFF(row['WIND SPEED FF']), axis = 1)
    data['RELATIVE HUMIDITY PC'] = data.apply(lambda row : RELATIVEHUMIDITYPC(row['RELATIVE HUMIDITY PC']), axis = 1)
    data['PRESSURE QFF MB DERIVED'] = data.apply(lambda row : PRESSUREQFFMBDERIVED(row['PRESSURE QFF MB DERIVED']), axis = 1)
    data['INTENSITAS MATAHARI'] = data.apply(lambda row : INTENSITASMATAHARI(row['INTENSITAS MATAHARI']), axis = 1)

    #resample rata-rata di hari dan bulan yg sama pada semua tahun
    data1 = data.groupby([data.index.month, data.index.day, data.index.hour], as_index=True).mean()
    #mengisikan rata2 pada tanggal dan bulan yg sama pada nilai NaN
    for kolom in list(data):
        index = data.index[data[kolom].apply(np.isnan)]
        for num, val in enumerate(index):
            data.loc[val, kolom] = data1.loc[index.month[num], index.day[num], index.hour[num]][kolom]
            

    # Prepare data for prediction
    data['ds'] = data.index
    data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)
    data['y'] = data['RAINFALL 24H RRRR'].shift(-1)
    data.dropna(axis=0, inplace=True)

    # NeuralProphet Model
    set_random_seed(42)
    model = NeuralProphet()
    model.add_future_regressor('RAINFALL 24H RRRR')
    model.add_future_regressor('TEMP DRYBULB C TTTTTT')
    model.add_future_regressor('WIND DIR DEG DD')
    model.add_future_regressor('WIND SPEED FF')
    model.add_future_regressor('RELATIVE HUMIDITY PC')
    model.add_future_regressor('PRESSURE QFF MB DERIVED')
    model.add_future_regressor('INTENSITAS MATAHARI')

    features = ['ds', 'y', 'RAINFALL 24H RRRR', 'TEMP DRYBULB C TTTTTT', 'WIND DIR DEG DD', 'WIND SPEED FF', 'RELATIVE HUMIDITY PC', 'PRESSURE QFF MB DERIVED', 'INTENSITAS MATAHARI']
    data = data[features]

    # Train model
    metrics = model.fit(data, freq='H')

    # Predict future
    last_row = data.iloc[-1]
    future_regressors = pd.DataFrame({
        'RAINFALL 24H RRRR': [last_row['RAINFALL 24H RRRR']] * 6,
        'TEMP DRYBULB C TTTTTT': [last_row['TEMP DRYBULB C TTTTTT']] * 6,
        'WIND DIR DEG DD': [last_row['WIND DIR DEG DD']] * 6,
        'WIND SPEED FF': [last_row['WIND SPEED FF']] * 6,
        'RELATIVE HUMIDITY PC': [last_row['RELATIVE HUMIDITY PC']] * 6,
        'PRESSURE QFF MB DERIVED': [last_row['PRESSURE QFF MB DERIVED']] * 6,
        'INTENSITAS MATAHARI': [last_row['INTENSITAS MATAHARI']] * 6,
    }, index=pd.date_range(start=data['ds'].iloc[-1] + pd.Timedelta(days=1), periods=6, freq='H'))

    future = model.make_future_dataframe(data, periods=6, regressors_df=future_regressors, n_historic_predictions=12)
    forecast = model.predict(future)

    # Add weather classification based on rainfall
    forecast['weather'] = 'Cerah/berawan'
    forecast.loc[forecast['yhat1'] > 100, 'weather'] = 'Hujan Sangat Lebat'
    forecast.loc[(forecast['yhat1'] > 50) & (forecast['yhat1'] <= 100), 'weather'] = 'Hujan Lebat'
    forecast.loc[(forecast['yhat1'] > 20) & (forecast['yhat1'] <= 50), 'weather'] = 'Hujan Sedang'
    forecast.loc[(forecast['yhat1'] > 5) & (forecast['yhat1'] <= 20), 'weather'] = 'Hujan Ringan'
    forecast.loc[(forecast['yhat1'] > 0) & (forecast['yhat1'] <= 5), 'weather'] = 'Hujan Sangat Ringan'
    
    print(forecast.tail(6))
    
    forecast_hourly_cache = forecast[['ds', 'weather']].tail(6).to_dict('records')
    

    # Calculate R-squared, MSE, and RMSE
    y_true = forecast['y'].values[:12]
    y_pred = forecast['yhat1'].values[:12]

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    result = {
        "forecast": forecast_hourly_cache,
        "r2": r2,
        "mse": mse,
        "rmse": rmse
    }
    return jsonify(result)



@app.route('/predict_daily', methods=['GET'])
def predict_daily():
    global forecast_daily_cache

    # Fetch data from MySQL
    data = fetch_data_from_mysql('dataperhari')
    if data.empty:
        return jsonify({"error": "No data available in the table."}), 400
    data.drop(columns=['id'], inplace=True)  # Drop 'id' column if it exists

    # Preprocessing
    data['DATA TIMESTAMP'] = pd.to_datetime(data['timestamp'])
    data.set_index('DATA TIMESTAMP', inplace=True)
    data.rename(columns={
        'rainfall': 'RAINFALL 24H RRRR',
        'temp': 'TEMP DRYBULB C TTTTTT',
        'winddir': 'WIND DIR DEG DD',
        'windspeed': 'WIND SPEED FF',
        'hum': 'RELATIVE HUMIDITY PC',
        'press': 'PRESSURE QFF MB DERIVED',
        'uv': 'INTENSITAS MATAHARI'
    }, inplace=True)
    cols_to_int = ['RAINFALL 24H RRRR', 'TEMP DRYBULB C TTTTTT', 'WIND DIR DEG DD', 'WIND SPEED FF', 'RELATIVE HUMIDITY PC', 'PRESSURE QFF MB DERIVED', 'INTENSITAS MATAHARI']
    for col in cols_to_int:
        data[col] = data[col].astype(float)
        
    print(data)

    def RAINFALL24HRRRR (val):
        if val < 0 or val > 1000:
            val = np.NaN
        return(val)

    def TEMPDRYBULBCTTTTTT (val):
        if val < 22 or val > 50:
            val = np.NaN
        return(val)

    def WINDDIRDEGDD (val):
        if val < 0 or val > 360:
            val = np.NaN
        return(val)

    def WINDSPEEDFF (val):
        if val > 100:
            val = np.NaN
        return(val)

    def RELATIVEHUMIDITYPC (val):
        if val < 0 or val> 100:
            val = np.NaN
        return(val)

    def PRESSUREQFFMBDERIVED (val):
        if val < 1000 or val> 1018:
            val = np.NaN
        return(val)

    def INTENSITASMATAHARI (val):
        if val < 0 or val> 1000:
            val = np.NaN
        return(val)

    data['RAINFALL 24H RRRR'] = data.apply(lambda row : RAINFALL24HRRRR(row['RAINFALL 24H RRRR']), axis = 1)
    data['TEMP DRYBULB C TTTTTT'] = data.apply(lambda row : TEMPDRYBULBCTTTTTT(row['TEMP DRYBULB C TTTTTT']), axis = 1)
    data['WIND DIR DEG DD'] = data.apply(lambda row : WINDDIRDEGDD(row['WIND DIR DEG DD']), axis = 1)
    data['WIND SPEED FF'] = data.apply(lambda row : WINDSPEEDFF(row['WIND SPEED FF']), axis = 1)
    data['RELATIVE HUMIDITY PC'] = data.apply(lambda row : RELATIVEHUMIDITYPC(row['RELATIVE HUMIDITY PC']), axis = 1)
    data['PRESSURE QFF MB DERIVED'] = data.apply(lambda row : PRESSUREQFFMBDERIVED(row['PRESSURE QFF MB DERIVED']), axis = 1)
    data['INTENSITAS MATAHARI'] = data.apply(lambda row : INTENSITASMATAHARI(row['INTENSITAS MATAHARI']), axis = 1)

    #resample rata-rata di hari dan bulan yg sama pada semua tahun
    data1 = data.groupby([data.index.month, data.index.day, data.index.hour], as_index=True).mean()
    #mengisikan rata2 pada tanggal dan bulan yg sama pada nilai NaN
    for kolom in list(data):
        index = data.index[data[kolom].apply(np.isnan)]
        for num, val in enumerate(index):
            data.loc[val, kolom] = data1.loc[index.month[num], index.day[num], index.hour[num]][kolom]
            

    # Prepare data for prediction
    data['ds'] = data.index
    data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)
    data['y'] = data['RAINFALL 24H RRRR'].shift(-1)
    data.dropna(axis=0, inplace=True)

    # NeuralProphet Model
    set_random_seed(42)
    model = NeuralProphet()
    model.add_future_regressor('RAINFALL 24H RRRR')
    model.add_future_regressor('TEMP DRYBULB C TTTTTT')
    model.add_future_regressor('WIND DIR DEG DD')
    model.add_future_regressor('WIND SPEED FF')
    model.add_future_regressor('RELATIVE HUMIDITY PC')
    model.add_future_regressor('PRESSURE QFF MB DERIVED')
    model.add_future_regressor('INTENSITAS MATAHARI')

    features = ['ds', 'y', 'RAINFALL 24H RRRR', 'TEMP DRYBULB C TTTTTT', 'WIND DIR DEG DD', 'WIND SPEED FF', 'RELATIVE HUMIDITY PC', 'PRESSURE QFF MB DERIVED', 'INTENSITAS MATAHARI']
    data = data[features]

    # Train model
    metrics = model.fit(data, freq='D')

    # Predict future
    last_row = data.iloc[-1]
    future_regressors = pd.DataFrame({
        'RAINFALL 24H RRRR': [last_row['RAINFALL 24H RRRR']] * 6,
        'TEMP DRYBULB C TTTTTT': [last_row['TEMP DRYBULB C TTTTTT']] * 6,
        'WIND DIR DEG DD': [last_row['WIND DIR DEG DD']] * 6,
        'WIND SPEED FF': [last_row['WIND SPEED FF']] * 6,
        'RELATIVE HUMIDITY PC': [last_row['RELATIVE HUMIDITY PC']] * 6,
        'PRESSURE QFF MB DERIVED': [last_row['PRESSURE QFF MB DERIVED']] * 6,
        'INTENSITAS MATAHARI': [last_row['INTENSITAS MATAHARI']] * 6,
    }, index=pd.date_range(start=data['ds'].iloc[-1] + pd.Timedelta(days=1), periods=6, freq='D'))

    future = model.make_future_dataframe(data, periods=6, regressors_df=future_regressors, n_historic_predictions=2)
    forecast = model.predict(future)

    # Add weather classification based on rainfall
    forecast['weather'] = 'Cerah/berawan'
    forecast.loc[forecast['yhat1'] > 100, 'weather'] = 'Hujan Sangat Lebat'
    forecast.loc[(forecast['yhat1'] > 50) & (forecast['yhat1'] <= 100), 'weather'] = 'Hujan Lebat'
    forecast.loc[(forecast['yhat1'] > 20) & (forecast['yhat1'] <= 50), 'weather'] = 'Hujan Sedang'
    forecast.loc[(forecast['yhat1'] > 5) & (forecast['yhat1'] <= 20), 'weather'] = 'Hujan Ringan'
    forecast.loc[(forecast['yhat1'] > 0) & (forecast['yhat1'] <= 5), 'weather'] = 'Hujan Sangat Ringan'
    
    forecast_daily_cache = forecast[['ds', 'weather']].tail(6).to_dict('records')

    # Calculate R-squared, MSE, and RMSE
    y_true = forecast['y'].values[:2]
    y_pred = forecast['yhat1'].values[:2]

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    result = {
        "forecast": forecast_daily_cache,
        "r2": r2,
        "mse": mse,
        "rmse": rmse
    }
    return jsonify(result)

# Insert forecast data to MySQL
def insert_data_forecast(forecast, table_name):
    connection = pymysql.connect(**mysql_config)
    cursor = connection.cursor()
    query = f"""
    INSERT INTO {table_name} (timestamp, weather) VALUES (%s, %s)
    """
    for item in forecast:
        cursor.execute(query, (item['ds'], item['weather']))
    connection.commit()
    cursor.close()
    connection.close()

@app.route('/update_forecast_hourly', methods=['GET'])
def update_forecast_hourly():
    insert_data_forecast(forecast_hourly_cache, 'dataprediksi')
    return jsonify({'status': 'success', 'message': 'Data inserted successfully'})

@app.route('/update_forecast_daily', methods=['GET'])
def update_forecast_daily():
    insert_data_forecast(forecast_daily_cache, 'dataprediksiperhari')
    return jsonify({'status': 'success', 'message': 'Data inserted successfully'})

@app.route('/upload_forecast_hourly', methods=['GET'])
def upload_data_hourly():
    url = 'https://tanamap.drik.my.id/api/weather-predict'
    token_bearer = 'GE70VpDZTMeVPDJbbJ1UsuFHvMcXX6JIwbtJ8cDGccc1170b'
    headers = {'Authorization': f'Bearer {token_bearer}'}

    forecast = forecast_hourly_cache
    print(f'Forecast data: {forecast}')  # Debug print to check forecast data
    
    success_count = 0
    failure_count = 0
    failures = []

    for item in forecast:
        print(f'Uploading item: {item}')  # Debug print to check each item
        payload = {
            "type": "jam",
            "time": item['ds'].strftime('%Y-%m-%d %H:%M:%S'),  # Convert Timestamp to string
            "description": item['weather'],
        }
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            success_count += 1
        else:
            failure_count += 1
            failures.append({"data": payload, "error": response.text})

    result = {
        "success_count": success_count,
        "failure_count": failure_count,
        "failures": failures
    }

    return jsonify(result)

@app.route('/upload_forecast_daily', methods=['GET'])
def upload_data_daily():
    url = 'https://tanamap.drik.my.id/api/weather-predict'
    token_bearer = 'GE70VpDZTMeVPDJbbJ1UsuFHvMcXX6JIwbtJ8cDGccc1170b'
    headers = {'Authorization': f'Bearer {token_bearer}'}

    forecast = forecast_daily_cache
    print(f'Forecast data: {forecast}')  # Debug print to check forecast data
    
    success_count = 0
    failure_count = 0
    failures = []

    for item in forecast:
        print(f'Uploading item: {item}')  # Debug print to check each item
        payload = {
            "type": "hari",
            "time": item['ds'].strftime('%Y-%m-%d'),  # Convert Timestamp to string
            "description": item['weather'],
        }
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            success_count += 1
        else:
            failure_count += 1
            failures.append({"data": payload, "error": response.text})

    result = {
        "success_count": success_count,
        "failure_count": failure_count,
        "failures": failures
    }

    return jsonify(result)

# Scheduler for automatic tasks
# @app.route('/schedule_tasks', methods=['GET'])
# def schedule_tasks():
#     scheduler = BackgroundScheduler()
#     # scheduler.add_job(fetch_and_sort_data_from_api, 'interval', seconds=60)
#     # # scheduler.add_job(fetch_data, 'interval', seconds=60)
#     # scheduler.add_job(update_weather_data, 'interval', seconds=60)
#     # scheduler.add_job(update_weather_hourly, 'interval', seconds=3600)
#     # scheduler.add_job(update_weather_daily, 'interval', seconds=86400)
#     scheduler.add_job(predict_hourly, 'interval', seconds=3600)
#     scheduler.add_job(predict_daily, 'interval', seconds=86400)
#     scheduler.add_job(update_forecast_hourly, 'interval', seconds=3600)
#     scheduler.add_job(update_forecast_daily, 'interval', seconds=86400)
#     # scheduler.add_job(upload_data_hourly, 'interval', seconds=3600)
#     # scheduler.add_job(upload_data_daily, 'interval', seconds=86400)
#     scheduler.start()


logging.basicConfig(level=logging.DEBUG)

@app.route('/schedule_tasks', methods=['GET'])
def schedule_tasks():
    scheduler = BackgroundScheduler()
    scheduler.add_job(weather_data, 'interval', seconds=60)
    # scheduler.add_job(upload_weather_data, 'interval', seconds=70)
    scheduler.add_job(predict_hourly_wrapper, 'interval', seconds=3620)
    scheduler.add_job(predict_daily_wrapper, 'interval', seconds=86440)
    # scheduler.add_job(fetch_and_sort_data_from_api, 'interval', seconds=60)
    # # # scheduler.add_job(fetch_data, 'interval', seconds=60)
    # scheduler.add_job(update_weather_data, 'interval', seconds=60)
    # scheduler.add_job(update_weather_hourly, 'interval', seconds=3620)
    # scheduler.add_job(predict_hourly, 'interval', seconds=3625)
    scheduler.add_job(upload_data_hourly, 'interval', seconds=3630)
    scheduler.add_job(update_forecast_hourly, 'interval', seconds=3635)
    # scheduler.add_job(update_weather_daily, 'interval', seconds=86415)
    # scheduler.add_job(predict_daily, 'interval', seconds=86425)
    scheduler.add_job(upload_data_daily, 'interval', seconds=86450)
    scheduler.add_job(update_forecast_daily, 'interval', seconds=86455)

    scheduler.start()
    logging.debug("Scheduler started")
    return "Tasks scheduled successfully", 200

def weather_data():
    fetch_and_sort_data_from_api()
    update_weather_data()
    # upload_weather_data()

def predict_hourly_wrapper():
    # update_weather_hourly()
    predict_hourly()
    upload_data_hourly()
    update_forecast_hourly()

def predict_daily_wrapper():
    # update_weather_daily()
    predict_daily()
    upload_data_daily()
    update_forecast_daily()

if __name__ == '__main__':
    fetch_thread = threading.Thread(target=fetch_and_sort_data_from_api, args=(60,))
    fetch_thread = threading.Thread(target=update_weather_data, args=(60,))
    fetch_thread = threading.Thread(target=update_weather_hourly, args=(3600,))
    fetch_thread = threading.Thread(target=update_weather_daily, args=(86400,))
    fetch_thread.daemon = True
    fetch_thread.start()

    schedule_tasks()

    app.run(debug=True)
