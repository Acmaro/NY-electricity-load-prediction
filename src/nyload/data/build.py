"""Turn raw downloads into one clean parquet per (zone, year).

Output grid: local New York wall-clock time, 5-minute steps, exactly 288 rows per day.
- Off-grid NYISO readings are used only to interpolate missing grid points.
- Fall-back day: the repeated hour is averaged. Spring-forward day: the missing hour is interpolated.
"""
import json
from pathlib import Path

import pandas as pd

from nyload import paths
from nyload.data import nyiso, weather

TZ = "America/New_York"
FREQ = "5min"
STEPS_PER_DAY = 288
MAX_INTERP_STEPS = 12  # 1 hour


def _utc_to_local_grid(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """UTC-indexed frame on a 5-min UTC grid -> local naive 5-min grid covering `years`."""
    local = df.copy()
    local.index = local.index.tz_convert(TZ).tz_localize(None)
    local = local.groupby(level=0).mean()
    grid = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31 23:55", freq=FREQ)
    local = local.reindex(local.index.union(grid))
    local = local.interpolate(method="time", limit_direction="both")
    return local.reindex(grid)


def _utc_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.date_range(index.min().ceil(FREQ), index.max().floor(FREQ), freq=FREQ, tz="UTC")


def regrid_load(load_utc: pd.Series, years: list[int]) -> pd.DataFrame:
    s = load_utc.groupby(level=0).mean().sort_index()
    grid = _utc_grid(s.index)
    on_grid = s.reindex(s.index.union(grid)).interpolate(method="time", limit_area="inside").reindex(grid)
    df = pd.DataFrame({"load": on_grid, "load_imputed": (~grid.isin(s.index)).astype(float)}, index=grid)
    out = _utc_to_local_grid(df, years)
    imputed = out["load_imputed"] > 0
    out["load_imputed"] = imputed.astype("int8")
    # Runs of imputed points longer than MAX_INTERP_STEPS are source outages (e.g. 2021-04-01..07);
    # their interpolated values are placeholders and get excluded from training/evaluation windows.
    run_id = (imputed != imputed.shift()).cumsum()
    run_len = imputed.groupby(run_id).transform("size")
    out["load_gap"] = (imputed & (run_len > MAX_INTERP_STEPS)).astype("int8")
    return out


def regrid_weather(wx_utc: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    wx = wx_utc.groupby(level=0).mean().sort_index()
    grid = _utc_grid(wx.index)
    union = wx.reindex(wx.index.union(grid))
    smooth = [c for c in wx.columns if c != "precip"]
    union[smooth] = union[smooth].interpolate(method="time", limit_direction="both")
    if "precip" in union:
        # hourly accumulation reported at the end of the hour -> applies to the preceding steps
        union["precip"] = union["precip"].bfill(limit=12).fillna(0.0)
    return _utc_to_local_grid(union.reindex(grid), years)


# ---------------------------------------------------------------- raw downloads

def pal_dir() -> Path:
    return paths.raw_dir() / "nyiso" / "pal"


def isolf_dir() -> Path:
    return paths.raw_dir() / "nyiso" / "isolf"


def weather_path(source: str, station: str, year: int) -> Path:
    return paths.raw_dir() / "weather" / f"{source}_{station}_{year}.parquet"


def download(years: list[int], weather_source: str, station: str, forecast: bool = True, log=print) -> None:
    for year in years:
        for month in range(1, 13):
            log(f"pal {year}-{month:02d}")
            nyiso.download_month("pal", year, month, pal_dir())
        if forecast:
            # Jan 1 day-ahead forecast is issued on Dec 31 of the previous year
            months = [(year - 1, 12)] + [(year, m) for m in range(1, 13)]
            for y, m in months:
                log(f"isolf {y}-{m:02d}")
                nyiso.download_month("isolf", y, m, isolf_dir())
        if weather_source:
            # local Dec 31 evening falls into the next UTC year, so fetch one year ahead too
            for y in (year, year + 1):
                path = weather_path(weather_source, station, y)
                if path.exists() or y > pd.Timestamp.now().year:
                    continue
                log(f"weather {weather_source}/{station} {y}")
                path.parent.mkdir(parents=True, exist_ok=True)
                weather.fetch(weather_source, station, y).to_parquet(path)


# ---------------------------------------------------------------- processed files

def processed_path(zone: str, year: int) -> Path:
    return paths.processed_dir() / f"{zone.lower()}_{year}.parquet"


def forecast_path(zone: str, year: int) -> Path:
    return paths.processed_dir() / f"{zone.lower()}_forecast_{year}.parquet"


def build(years: list[int], zone: str, weather_source: str, station: str, forecast: bool = True, log=print) -> None:
    out_dir = paths.processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        load = pd.concat([nyiso.read_pal(nyiso.download_month("pal", year, m, pal_dir()), zone)
                          for m in range(1, 13)])
        df = regrid_load(load, [year])

        wx_frames = [pd.read_parquet(p) for y in (year - 1, year, year + 1)
                     if (p := weather_path(weather_source, station, y)).exists()]
        if not wx_frames:
            raise FileNotFoundError(f"no weather for {year}; run `nyload-data download` first")
        df = df.join(regrid_weather(pd.concat(wx_frames), [year]))

        df.index.name = "timestamp"
        df = df.reset_index()
        assert len(df) % STEPS_PER_DAY == 0 and df.isna().sum().sum() == 0, f"bad grid for {year}"
        df.to_parquet(processed_path(zone, year), index=False)
        log(f"{processed_path(zone, year).name}: {len(df)} rows, "
            f"{int(df['load_imputed'].sum())} imputed load points, "
            f"{int(df['load_gap'].sum())} in long gaps")

        if forecast:
            months = [(year - 1, 12)] + [(year, m) for m in range(1, 13)]
            fc = pd.concat([nyiso.read_isolf(nyiso.download_month("isolf", y, m, isolf_dir()), zone)
                            for y, m in months])
            fc = fc[fc["timestamp"].dt.year == year].groupby("timestamp", as_index=False).mean()
            fc.to_parquet(forecast_path(zone, year), index=False)
            log(f"{forecast_path(zone, year).name}: {len(fc)} hourly rows")

    meta = {"zone": zone, "weather_source": weather_source, "station": station}
    (out_dir / f"{zone.lower()}_meta.json").write_text(json.dumps(meta, indent=2))
