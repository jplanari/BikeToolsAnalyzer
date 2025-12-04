import requests
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ride_weather(df):
    if df.empty or 'time' not in df.columns:
        return df

    start_time = df['time'].min()
    end_time = df['time'].max()

    lat = df['lat'].mean()
    lon = df['lon'].mean()

    start_str = start_time.strftime('%Y-%m-%d')
    end_str = end_time.strftime('%Y-%m-%d')

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": ["temperature_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"],
            "windspeed_unit": "ms",
            }
    try:
        response = requests.get(url, params=params)
        data = response.json()

        if 'hourly' not in data:
            st.warning("Weather data not available for this location and time.")
            return df

        hourly = data['hourly']
        weather_df = pd.DataFrame({
            'time': pd.to_datetime(hourly['time']),
            'wind_speed': hourly['wind_speed_10m'],
            'wind_deg': hourly['wind_direction_10m'],
            'temp_c': hourly['temperature_2m'],
            'pressure': hourly['surface_pressure']
        })

        weather_df = weather_df.set_index('time')
        weather_df.index = weather_df.index.tz_localize(None)

        df = df.sort_values('time')

        combined_index = df['time']

        weather_interp = weather_df.reindex(weather_df.index.union(combined_index)).interpolate(method='time')
        weather_interp = weather_interp.reindex(combined_index)

        df['wind_speed'] = weather_interp['wind_speed'].values
        df['wind_deg'] = weather_interp['wind_deg'].values
        df['temp_c'] = weather_interp['temp_c'].values
        df['pressure'] = weather_interp['pressure'].values

        return df

    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return df


