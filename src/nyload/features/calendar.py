"""Calendar features computed from the local timestamp."""
import holidays
import numpy as np
import pandas as pd


def workday_flag(ts: pd.Series, country: str = "US", subdiv: str | None = None) -> pd.Series:
    """1.0 on weekdays that are not public holidays, else 0.0."""
    years = sorted(ts.dt.year.unique().tolist())
    hol = holidays.country_holidays(country, subdiv=subdiv, years=years)
    is_holiday = ts.dt.normalize().isin(pd.to_datetime(list(hol.keys())))
    return ((ts.dt.dayofweek < 5) & ~is_holiday).astype(float)


def _cyclic(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def add_calendar_features(df: pd.DataFrame, names: list[str], country: str = "US",
                          subdiv: str | None = None) -> pd.DataFrame:
    """Add any of the supported calendar features listed in `names` (others are ignored)."""
    ts = df["timestamp"]
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    builders = {
        "is_workday": lambda: workday_flag(ts, country, subdiv),
        "year": lambda: ts.dt.year,
        "month": lambda: ts.dt.month,
        "day": lambda: ts.dt.day,
        "hour": lambda: ts.dt.hour,
        "dow": lambda: ts.dt.dayofweek,
        "doy": lambda: ts.dt.dayofyear,
        "minute_of_day": lambda: minute_of_day,
        "hour_sin": lambda: _cyclic(minute_of_day, 1440)[0],
        "hour_cos": lambda: _cyclic(minute_of_day, 1440)[1],
        "dow_sin": lambda: _cyclic(ts.dt.dayofweek, 7)[0],
        "dow_cos": lambda: _cyclic(ts.dt.dayofweek, 7)[1],
        "doy_sin": lambda: _cyclic(ts.dt.dayofyear, 365.25)[0],
        "doy_cos": lambda: _cyclic(ts.dt.dayofyear, 365.25)[1],
    }
    df = df.copy()
    for name in names:
        if name in builders and name not in df:
            df[name] = builders[name]().astype(float)
    return df
