"""Feature engineering for daily risk-signal classification.

All features are constructed strictly from information available at the
close of day t to predict an event on day t+1 (or over t+1..t+5), so the
design is leakage-free by construction. Monthly macro variables are
merged as-of with a one-month publication lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOL_WINDOWS = (5, 21, 63)
MOM_WINDOWS = (5, 21, 63)


def build_features(rets: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a feature matrix indexed by date t (information set at t)."""
    f = {}
    for c in rets.columns:
        r = rets[c]
        f[f"{c}_ret_lag1"] = r
        f[f"{c}_ret_lag2"] = r.shift(1)
        f[f"{c}_ret_lag5_mean"] = r.rolling(5).mean()
        for w in VOL_WINDOWS:
            f[f"{c}_vol_{w}d"] = r.rolling(w).std()
        for w in MOM_WINDOWS:
            f[f"{c}_mom_{w}d"] = r.rolling(w).sum()
        f[f"{c}_absret_lag1"] = r.abs()
        f[f"{c}_drawdown_63d"] = (
            r.rolling(63).sum() - r.rolling(63).sum().rolling(63).max()
        )
        # vol-of-vol and vol ratio (short vs long) — regime indicators
        f[f"{c}_volratio"] = r.rolling(5).std() / r.rolling(63).std()
    # cross-market: rolling correlation of cocoa with the others
    if "cocoa" in rets.columns:
        for c in rets.columns:
            if c != "cocoa":
                f[f"corr_cocoa_{c}_21d"] = rets["cocoa"].rolling(21).corr(rets[c])
    # joint-stress feature: count of series below their rolling 5% quantile
    q05 = rets.rolling(250).quantile(0.05)
    f["joint_tail_count"] = (rets < q05).sum(axis=1).astype(float)

    X = pd.DataFrame(f, index=rets.index)

    if macro is not None:
        m = macro.copy()
        m.index = m.index + pd.offsets.MonthEnd(1)  # 1-month publication lag
        # Forward-fill each macro column's OWN gaps first, before reindexing
        # onto the daily calendar. Without this, a series that has simply
        # stopped updating (e.g. exports_usd_m's last real BoG value is
        # 2023-04) sits in the source as explicit NaN rows extending to the
        # present, and reindex(method="ffill") finds the nearest prior ROW
        # rather than the nearest prior VALID value -- if that row is
        # already NaN, ffill propagates NaN forever rather than carrying
        # the last real reading forward, silently truncating the entire
        # downstream feature matrix (X.dropna() below strips every date
        # after the stalest macro series' last observation). Carrying the
        # last known macro reading forward through an unavoidable
        # publication gap is standard, disclosed practice here -- the same
        # "hold the last known state through a gap" logic applied to the
        # cedi's own volatility elsewhere in this project -- not silently
        # inventing new information.
        m = m.ffill()
        m = m.reindex(X.index, method="ffill")
        m.columns = [f"macro_{c}" for c in m.columns]
        X = X.join(m)

    return X.dropna()


def make_labels(rets: pd.DataFrame) -> pd.DataFrame:
    """Binary risk labels aligned to the feature date t (event at t+1..).

    cocoa_var_breach : cocoa return on day t+1 below its rolling 250-day
                       5% quantile (VaR exceedance signal).
    highvol_day      : cocoa |return| on t+1 above its rolling 95% quantile.
    cedi_depreciation: cumulative return over t+1..t+5 above its
                       rolling 90% quantile (depreciation episode).

    Column name and orientation note: the canonical real-data panel
    (built by the companion tail-dependence project's BoG integration
    script) uses a column named 'cedi', already sign-flipped so that
    the LOWER tail denotes depreciation (consistent with the other four
    series' lower tail denoting a bad-for-Ghana outcome). This label
    function needs the RAW orientation (depreciation = large positive
    cumulative return) for its threshold logic, so it negates the
    column internally rather than assuming the caller has done so.
    """
    lbl = pd.DataFrame(index=rets.index)
    var5 = rets["cocoa"].rolling(250).quantile(0.05)
    lbl["cocoa_var_breach"] = (rets["cocoa"].shift(-1) < var5).astype(int)

    q95_abs = rets["cocoa"].abs().rolling(250).quantile(0.95)
    lbl["highvol_day"] = (rets["cocoa"].abs().shift(-1) > q95_abs).astype(int)

    cedi_col = "cedi" if "cedi" in rets.columns else "ghs_usd"
    raw_orientation = -rets[cedi_col] if cedi_col == "cedi" else rets[cedi_col]
    fx5 = raw_orientation.rolling(5).sum().shift(-5)
    thr = raw_orientation.rolling(5).sum().rolling(250).quantile(0.90)
    lbl["cedi_depreciation"] = (fx5 > thr).astype(int)
    return lbl
