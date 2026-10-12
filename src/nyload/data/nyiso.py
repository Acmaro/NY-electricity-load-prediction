"""NYISO public data: real-time actual load (pal) and ISO load forecast (isolf).

Archives are monthly zips of daily CSVs: http://mis.nyiso.com/public/csv/<kind>/YYYYMM01<kind>_csv.zip
"""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "http://mis.nyiso.com/public/csv/{kind}/{year}{month:02d}01{kind}_csv.zip"

# isolf uses abbreviated column names
ISOLF_ZONE_COLUMNS = {
    "CAPITL": "Capitl", "CENTRL": "Centrl", "DUNWOD": "Dunwod", "GENESE": "Genese",
    "HUD VL": "Hud Vl", "LONGIL": "Longil", "MHK VL": "Mhk Vl", "MILLWD": "Millwd",
    "N.Y.C.": "N.Y.C.", "NORTH": "North", "WEST": "West", "NYISO": "NYISO",
}


def download_month(kind: str, year: int, month: int, dest_dir: Path, overwrite: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL.format(kind=kind, year=year, month=month)
    path = dest_dir / url.rsplit("/", 1)[-1]
    if path.exists() and not overwrite:
        return path
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    return path


def _iter_csvs(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith(".csv"):
                yield name, zf.read(name)


def read_pal(zip_path: Path, zone: str) -> pd.Series:
    """Actual load for one zone as a UTC-indexed series (MW).

    Raw timestamps are local wall-clock time with an EST/EDT column, so the repeated
    hour on the fall-back day is disambiguated here.
    """
    frames = []
    for _, raw in _iter_csvs(zip_path):
        df = pd.read_csv(io.BytesIO(raw))
        df = df[df["Name"] == zone]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    local = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M:%S")
    offset = df["Time Zone"].map({"EST": 5, "EDT": 4})
    if offset.isna().any():
        raise ValueError(f"unexpected time zone labels in {zip_path}: {df['Time Zone'].unique()}")
    utc = (local + pd.to_timedelta(offset, unit="h")).dt.tz_localize("UTC")
    return pd.Series(df["Load"].to_numpy(dtype=float), index=utc, name="load")


def read_isolf(zip_path: Path, zone: str) -> pd.DataFrame:
    """Day-ahead ISO forecast for one zone.

    Each daily file (issued on day D) holds hourly forecasts for D..D+6 in local time.
    The day-ahead forecast for day d is taken from the file issued on d-1.
    Returns columns: timestamp (local, naive), forecast.
    """
    col = ISOLF_ZONE_COLUMNS[zone]
    rows = []
    for name, raw in _iter_csvs(zip_path):
        issued = pd.Timestamp(name[:8])
        df = pd.read_csv(io.BytesIO(raw))
        ts = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M")
        target_day = issued + pd.Timedelta(days=1)
        mask = ts.dt.normalize() == target_day
        rows.append(pd.DataFrame({"timestamp": ts[mask], "forecast": df.loc[mask, col].astype(float)}))
    return pd.concat(rows, ignore_index=True)
