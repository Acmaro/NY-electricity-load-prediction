import numpy as np
import pandas as pd
import pytest

from nyload.data.build import processed_path


def synthetic_year(year: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + year)
    ts = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:55", freq="5min")
    minutes = ts.hour * 60 + ts.minute
    load = 1200 + 200 * np.sin(2 * np.pi * minutes / 1440) + rng.normal(0, 10, len(ts))
    return pd.DataFrame({
        "timestamp": ts, "load": load, "load_imputed": 0, "load_gap": 0,
        "temp": 50 + 20 * np.sin(2 * np.pi * ts.dayofyear / 365), "dewpt": 40.0, "rh": 60.0,
        "wspd": 5.0, "pressure": 30.0, "precip": 0.0,
    })


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NYLOAD_DATA_DIR", str(tmp_path))
    (tmp_path / "processed").mkdir()
    for year in (2019, 2020, 2021):
        synthetic_year(year).to_parquet(processed_path("CAPITL", year), index=False)
    return tmp_path
