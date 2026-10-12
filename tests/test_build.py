import numpy as np
import pandas as pd

from nyload.data.build import STEPS_PER_DAY, regrid_load


def _utc_series(start_local: str, end_local: str) -> pd.Series:
    idx = pd.date_range(pd.Timestamp(start_local, tz="America/New_York"),
                        pd.Timestamp(end_local, tz="America/New_York"), freq="5min").tz_convert("UTC")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)


def test_every_local_day_has_288_steps_across_dst():
    s = _utc_series("2020-01-01 00:00", "2020-12-31 23:55")
    out = regrid_load(s, [2020])
    days = out.groupby(out.index.date).size()
    assert len(out) == 366 * STEPS_PER_DAY
    assert (days == STEPS_PER_DAY).all()
    assert not out["load"].isna().any()


def test_off_grid_readings_only_fill_missing_points():
    s = _utc_series("2020-06-01 00:00", "2020-06-01 23:55")
    extra = pd.Series([999.0], index=[s.index[10] + pd.Timedelta("90s")])
    out = regrid_load(pd.concat([s, extra]).sort_index(), [2020])
    june1 = out.loc["2020-06-01"]
    assert np.allclose(june1["load"].to_numpy(), s.to_numpy())  # 999 never lands on the grid
    assert june1["load_imputed"].sum() == 0


def test_long_outage_is_flagged_short_one_is_not():
    s = _utc_series("2020-06-01 00:00", "2020-06-03 23:55")
    short_hole = s.index[100:105]       # 25 minutes
    long_hole = s.index[400:500]        # > 8 hours
    out = regrid_load(s.drop(short_hole.union(long_hole)), [2020]).loc["2020-06-01":"2020-06-03"]
    assert out["load_imputed"].sum() == 105
    assert out["load_gap"].sum() == 100
