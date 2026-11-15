"""Turn backtest forecasts into per-origin metrics and across-origin summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from demandcast.features import WEEK
from demandcast.metrics import coverage, mae, mase, mean_width, rmse, wape
from demandcast.pipeline import BacktestResult

POINT_METRICS = ("mae", "rmse", "wape", "mase")


def _mean_std(values: list[float] | np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "median": float(np.median(v)),
    }


def per_origin_point_metrics(result: BacktestResult) -> pd.DataFrame:
    """One row per (setting, model, origin) with MAE, RMSE, WAPE and MASE on recorded hours."""
    rows = []
    for (setting, model, origin), g in result.points.groupby(
        ["setting", "model", "origin"], sort=False
    ):
        y, yhat, rec = g["y"].to_numpy(), g["yhat"].to_numpy(), g["recorded"].to_numpy()
        rows.append(
            {
                "setting": setting,
                "model": model,
                "origin": origin,
                "mae": mae(y, yhat, rec),
                "rmse": rmse(y, yhat, rec),
                "wape": wape(y, yhat, rec),
                "mase": mase(y, yhat, result.scales[origin], rec),
            }
        )
    return pd.DataFrame(rows)


def point_summary(result: BacktestResult, baseline: str = "seasonal_naive") -> dict:
    """setting -> model -> {metric: {mean, std, median}, pooled_wape, per_origin, wins}.

    ``pooled_wape`` is total absolute error / total demand over all origins together, so a
    low-demand week (e.g. Christmas) carries its share of demand rather than 1/12 of the vote.
    """
    per = per_origin_point_metrics(result)
    pts = result.points[result.points["recorded"].astype(bool)]
    abs_err = (pts["y"] - pts["yhat"]).abs()
    pooled = abs_err.groupby([pts["setting"], pts["model"]]).sum() / pts.groupby(
        ["setting", "model"]
    )["y"].sum()
    out: dict = {}
    for (setting, model), g in per.groupby(["setting", "model"], sort=False):
        base = per[(per["setting"] == setting) & (per["model"] == baseline)]
        base_mae = base.set_index("origin")["mae"].reindex(g["origin"]).to_numpy()
        wins = int((g["mae"].to_numpy() < base_mae).sum())
        out.setdefault(setting, {})[model] = {
            **{m: _mean_std(g[m]) for m in POINT_METRICS},
            "pooled_wape": float(pooled.loc[(setting, model)]),
            "origins_beating_seasonal_naive_mae": wins,
            "per_origin": {m: [round(float(x), 6) for x in g[m]] for m in POINT_METRICS},
        }
    return out


def weather_oracle_gain(result: BacktestResult) -> dict:
    """Per model that uses weather: paired per-origin MAE reduction of setting B vs A."""
    per = per_origin_point_metrics(result)
    out = {}
    for model in per["model"].unique():
        sel = per[per["model"] == model].pivot(index="origin", columns="setting", values="mae")
        if not {"A", "B"} <= set(sel.columns):
            continue
        a, b = sel["A"].to_numpy(), sel["B"].to_numpy()
        if not np.allclose(a, b):
            out[model] = {
                "relative_mae_reduction": _mean_std(1.0 - b / a),
                "origins_improved": int((b < a).sum()),
            }
    return out


def error_profiles(result: BacktestResult) -> dict:
    """Pooled WAPE by forecast day (1-7) and pooled MAE by hour of day, per setting and model."""
    pts = result.points[result.points["recorded"].astype(bool)].copy()
    pts["abs_err"] = (pts["y"] - pts["yhat"]).abs()
    pts["day"] = (pts["step"] - 1) // 24 + 1
    pts["hour"] = pd.DatetimeIndex(pts["timestamp"]).hour
    out: dict = {}
    for (setting, model), g in pts.groupby(["setting", "model"], sort=False):
        by_day = g.groupby("day")[["abs_err", "y"]].sum()
        by_hour = g.groupby("hour")["abs_err"].mean()
        out.setdefault(setting, {})[model] = {
            "wape_by_horizon_day": [round(float(x), 6) for x in by_day["abs_err"] / by_day["y"]],
            "mae_by_hour_of_day": [round(float(x), 4) for x in by_hour],
        }
    return out


def interval_summary(result: BacktestResult) -> dict:
    """setting -> method -> level -> coverage / width summaries across origins."""
    out: dict = {}
    for (setting, method, level), g in result.intervals.groupby(
        ["setting", "method", "level"], sort=False
    ):
        covs, widths = [], []
        for _, o in g.groupby("origin", sort=False):
            rec = o["recorded"].to_numpy()
            covs.append(coverage(o["y"], o["lo"], o["hi"], rec))
            widths.append(mean_width(o["lo"], o["hi"], rec))
        rec = g["recorded"].to_numpy()
        key = f"{round(level * 100)}"
        out.setdefault(setting, {}).setdefault(method, {})[key] = {
            "coverage": _mean_std(covs),
            "coverage_min": float(np.min(covs)),
            "coverage_max": float(np.max(covs)),
            "pooled_coverage": coverage(g["y"], g["lo"], g["hi"], rec),
            "width": _mean_std(widths),
            "per_origin_coverage": [round(float(c), 6) for c in covs],
        }
    return out


def residual_spread_by_level(
    df: pd.DataFrame, end: str | pd.Timestamp, n_bins: int = 8, q: float = 0.9
) -> dict:
    """Pre-backtest check of which scale makes forecast errors homogeneous across levels.

    Uses only hours before ``end`` (before the first origin). Bins the one-week seasonal naive
    forecast into ``n_bins`` quantile bins and reports, per transform, the ``q``-quantile of
    absolute residuals per bin and the max/min ratio across bins (1 = perfectly homogeneous).
    Conformal scores are only as good as this homogeneity.
    """
    d = df[df.index < pd.Timestamp(end)]
    y, yhat = d["count"], d["count"].shift(WEEK)
    ok = (d["recorded"] & d["recorded"].shift(WEEK, fill_value=False)).to_numpy(dtype=bool)
    y, yhat = y.to_numpy(dtype=float)[ok], yhat.to_numpy(dtype=float)[ok]
    bins = pd.qcut(yhat, n_bins, labels=False, duplicates="drop")
    out = {}
    for name, f in (("raw", lambda x: x), ("sqrt", np.sqrt), ("log1p", np.log1p)):
        spread = pd.Series(np.abs(f(y) - f(yhat))).groupby(bins).quantile(q).to_numpy()
        out[name] = {
            "q90_abs_residual_by_level_bin": [round(float(s), 4) for s in spread],
            "max_min_ratio": round(float(spread.max() / spread.min()), 3),
        }
    return out
