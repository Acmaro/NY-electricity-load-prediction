"""Hourly weather observations.

Two sources, both returning a UTC-indexed frame with the same columns
(temp °F, dewpt °F, rh %, wspd mph, pressure inHg, precip inch/hour):

- weather_com: station observations from api.weather.com (needs WEATHER_COM_API_KEY).
- open_meteo: ERA5 reanalysis from archive-api.open-meteo.com (no key, gridded).
"""
import calendar
import os
import time

import pandas as pd
import requests

COLUMNS = ["temp", "dewpt", "rh", "wspd", "pressure", "precip"]

# Default station for each zone (ICAO code, latitude, longitude).
STATIONS = {
    "KALB": (42.7483, -73.8017),  # Albany International, inside the CAPITL zone
    "KLGA": (40.7769, -73.8740),  # LaGuardia, New York City (used by the original crawler)
}

WEATHER_COM_URL = (
    "https://api.weather.com/v1/location/{station}:9:US/observations/historical.json"
    "?apiKey={key}&units=e&startDate={start}&endDate={end}"
)
WEATHER_COM_FIELDS = {"temp": "temp", "dewPt": "dewpt", "rh": "rh", "wspd": "wspd",
                      "pressure": "pressure", "precip_hrly": "precip"}

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FIELDS = {"temperature_2m": "temp", "dew_point_2m": "dewpt", "relative_humidity_2m": "rh",
                     "wind_speed_10m": "wspd", "surface_pressure": "pressure", "precipitation": "precip"}


def fetch_weather_com(station: str, year: int, api_key: str | None = None, pause: float = 0.5) -> pd.DataFrame:
    api_key = api_key or os.environ.get("WEATHER_COM_API_KEY")
    if not api_key:
        raise RuntimeError("WEATHER_COM_API_KEY is not set (put it in .env), or use --weather-source open_meteo")
    records = []
    for month in range(1, 13):
        last_day = calendar.monthrange(year, month)[1]
        url = WEATHER_COM_URL.format(station=station, key=api_key,
                                     start=f"{year}{month:02d}01", end=f"{year}{month:02d}{last_day}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        records.extend(resp.json().get("observations", []))
        time.sleep(pause)
    df = pd.DataFrame(records)
    index = pd.to_datetime(df["valid_time_gmt"], unit="s", utc=True)
    out = df[list(WEATHER_COM_FIELDS)].rename(columns=WEATHER_COM_FIELDS).set_index(index)
    return out.apply(pd.to_numeric, errors="coerce").sort_index()


def fetch_open_meteo(station: str, year: int) -> pd.DataFrame:
    lat, lon = STATIONS[station]
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "hourly": ",".join(OPEN_METEO_FIELDS), "timezone": "UTC",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    index = pd.to_datetime(hourly.pop("time"), utc=True)
    out = pd.DataFrame(hourly, index=index).rename(columns=OPEN_METEO_FIELDS)
    out["pressure"] = out["pressure"] / 33.8639  # hPa -> inHg
    return out[COLUMNS]


def fetch(source: str, station: str, year: int) -> pd.DataFrame:
    if source == "weather_com":
        return fetch_weather_com(station, year)
    if source == "open_meteo":
        return fetch_open_meteo(station, year)
    raise ValueError(f"unknown weather source: {source}")
