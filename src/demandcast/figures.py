"""Report figures (static PNGs for the README)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from demandcast.backtest import Split  # noqa: E402
from demandcast.pipeline import POINT_MODELS, BacktestResult  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
VIOLET = "#6250d6"
BLUE_LIGHT, BLUE_PALE = "#9ec5f4", "#cde2fb"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "lines.linewidth": 2,
    "lines.markersize": 6.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
}

LABELS = {
    "seasonal_naive": "Seasonal naive\n(last week)",
    "seasonal_naive_4w": "Seasonal naive\n(4-week mean)",
    "daily_ets": "Daily ETS +\nhourly profile",
    "ridge_calendar": "Ridge\n(calendar)",
    "gbm_lag168": "GBM, one model\n(lags >= 1 week)",
    "gbm_direct": "GBM, direct\n(one model per day)",
}
METHOD_LABELS = {
    "seasonal_naive_conformal": "Seasonal naive\n+ split conformal",
    "gbm_quantile": "GBM quantile\n(uncalibrated)",
    "gbm_cqr": "GBM quantile\n+ conformal (CQR)",
}
MODEL_COLORS = {
    "gbm_direct": BLUE,
    "gbm_lag168": VIOLET,
    "seasonal_naive": ORANGE,
    "seasonal_naive_4w": AQUA,
    "daily_ets": YELLOW,
    "ridge_calendar": MAGENTA,
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_backtest_origins(df: pd.DataFrame, splits: list[Split], path: Path) -> None:
    """Daily demand on top; expanding training, calibration and test windows per origin below."""
    daily = df["count"].groupby(df.index.normalize()).sum()
    with plt.rc_context(STYLE):
        fig, (ax0, ax) = plt.subplots(
            2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 2.3]}
        )
        ax0.plot(daily.index, daily.to_numpy() / 1000, color=INK2, lw=0.9)
        ax0.set_ylabel("rentals per day\n(thousands)")
        ax0.set_title(f"Rolling-origin backtest: {len(splits)} origins, refit on an expanding "
                      "window, forecast the next 168 hours")
        start = df.index[0]
        for i, s in enumerate(splits):
            y = len(splits) - i
            t_cal, t_origin = df.index[s.calibration[0]], s.origin
            t_end = df.index[s.test[-1]] + pd.Timedelta(hours=1)
            ax.barh(y, t_cal - start, left=start, height=0.62, color=BLUE_PALE, lw=0)
            ax.barh(y, t_origin - t_cal, left=t_cal, height=0.62, color=BLUE, lw=0)
            ax.barh(y, t_end - t_origin, left=t_origin, height=0.62, color=ORANGE, lw=0)
            for a in (ax0, ax):
                a.axvspan(t_origin, t_end, color=ORANGE, alpha=0.10, lw=0)
        ax.set_yticks(range(len(splits), 0, -1),
                      [f"{s.origin:%d %b %Y}" for s in splits], fontsize=8.5)
        ax.set_ylabel("forecast origin")
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=BLUE_PALE),
            plt.Rectangle((0, 0), 1, 1, color=BLUE),
            plt.Rectangle((0, 0), 1, 1, color=ORANGE),
        ]
        ax.legend(handles, ["training (expanding)", "calibration (last 4 weeks, for intervals)",
                            "test: the next 168 hours"],
                  loc="lower left", ncol=1, fontsize=8.5)
        _save(fig, path)


def fig_forecast_week(result: BacktestResult, origin: pd.Timestamp, path: Path) -> None:
    """One test week: actuals, direct GBM (setting A) with CQR bands, seasonal naive."""
    pts = result.points[(result.points["origin"] == origin) & (result.points["setting"] == "A")]
    iv = result.intervals[
        (result.intervals["origin"] == origin)
        & (result.intervals["setting"] == "A")
        & (result.intervals["method"] == "gbm_cqr")
    ]
    gbm = pts[pts["model"] == "gbm_direct"]
    naive = pts[pts["model"] == "seasonal_naive"]
    t = pd.DatetimeIndex(gbm["timestamp"])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.5, 4.4))
        for level, color in ((0.95, BLUE_PALE), (0.8, BLUE_LIGHT)):
            band = iv[np.isclose(iv["level"], level)]
            ax.fill_between(t, band["lo"], band["hi"], color=color, lw=0, step=None,
                            label=f"{level:.0%} interval (CQR)")
        ax.plot(t, naive["yhat"], color=ORANGE, lw=1.3, ls=(0, (4, 2)),
                label="Seasonal naive (same hour last week)")
        ax.plot(t, gbm["yhat"], color=BLUE, lw=1.8, label="GBM, direct per day")
        y = gbm["y"].to_numpy(dtype=float).copy()
        y[~gbm["recorded"].to_numpy(dtype=bool)] = np.nan
        ax.plot(t, y, color=INK, lw=1.4, label="Actual rentals")
        end = t[-1] + pd.Timedelta(hours=1)
        ax.set_title(f"Forecast made at {origin:%a %d %b %Y} 00:00 for the next 168 hours "
                     "(setting A: no weather information)")
        ax.set_ylabel("rentals per hour")
        ax.set_xlim(t[0], end)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
        ax.legend(loc="upper left", ncol=3, fontsize=8.5)
        _save(fig, path)


def fig_model_comparison(summary: dict, path: Path) -> None:
    """MASE and WAPE per model (mean +/- std across origins), settings A and B."""
    models = [m for m in POINT_MODELS if m in summary["A"]]
    x = np.arange(len(models))
    w = 0.38
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, metric, title in (
            (axes[0], "mase", "MASE (lower is better; 1 = in-sample weekly naive)"),
            (axes[1], "wape", "WAPE: total absolute error / total demand"),
        ):
            for k, m in enumerate(models):
                a, b = summary["A"][m][metric], summary["B"][m][metric]
                uses_weather = not np.isclose(a["mean"], b["mean"])
                xa = x[k] - (w / 2 if uses_weather else 0)
                ax.bar(xa, a["mean"], w, yerr=a["std"], color=BLUE, capsize=3,
                       error_kw={"ecolor": INK2, "lw": 1})
                ax.text(xa, a["mean"] + a["std"] + 0.01 * (1 if metric == "wape" else 3),
                        f"{a['mean']:.2f}" if metric == "mase" else f"{a['mean']:.0%}",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
                if uses_weather:
                    ax.bar(x[k] + w / 2, b["mean"], w, yerr=b["std"], color=ORANGE, capsize=3,
                           error_kw={"ecolor": INK2, "lw": 1})
                    ax.text(x[k] + w / 2,
                            b["mean"] + b["std"] + 0.01 * (1 if metric == "wape" else 3),
                            f"{b['mean']:.2f}" if metric == "mase" else f"{b['mean']:.0%}",
                            ha="center", va="bottom", fontsize=8.5, color=INK)
            base = summary["A"]["seasonal_naive"][metric]["mean"]
            ax.axhline(base, color=INK2, ls="--", lw=1)
            ax.set_xticks(x, [LABELS[m] for m in models], fontsize=8.5)
            ax.set_title(title, fontsize=10.5)
            ax.set_ylim(bottom=0)
            ax.grid(axis="x", visible=False)
            if metric == "wape":
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)]
        handles.append(plt.Line2D([], [], color=INK2, ls="--", lw=1))
        fig.legend(handles, ["A: demand history + calendar (operational)",
                             "B: + observed weather of the forecast week (oracle upper bound)",
                             "seasonal naive, setting A"],
                   loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=9)
        n = len(summary["A"][models[0]]["per_origin"]["mase"])
        fig.text(0.5, -0.03, f"Bars: mean over {n} origins; whiskers: +/- 1 std across origins. "
                 "Naive and ETS models use no weather, so they have one bar.",
                 ha="center", color=INK2, fontsize=8.5)
        _save(fig, path)


def fig_intervals_and_horizon(intervals: dict, profiles: dict, path: Path) -> None:
    """Left: per-origin interval coverage (setting A). Right: WAPE by forecast day."""
    methods = list(METHOD_LABELS)
    levels = ["80", "95"]
    with plt.rc_context(STYLE):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                      gridspec_kw={"width_ratios": [1.25, 1]})
        rng = np.random.default_rng(0)
        for j, level in enumerate(levels):
            nominal = int(level) / 100
            for k, m in enumerate(methods):
                pos = k + (j - 0.5) * 0.36
                s = intervals["A"][m][level]
                covs = np.asarray(s["per_origin_coverage"])
                color = (BLUE_LIGHT, BLUE)[j]
                ax.scatter(pos + rng.uniform(-0.06, 0.06, len(covs)), covs, s=14, color=color,
                           alpha=0.9, lw=0, zorder=3)
                ax.plot([pos - 0.13, pos + 0.13], [s["coverage"]["mean"]] * 2, color=INK, lw=2.2,
                        zorder=4)
                ax.hlines(nominal, pos - 0.17, pos + 0.17, color=ORANGE, lw=1.6, zorder=2)
        ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], fontsize=8.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_ylabel("share of actual hours inside the interval")
        ax.set_title("Interval coverage per origin (setting A)", fontsize=10.5)
        ax.grid(axis="x", visible=False)
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=BLUE_LIGHT,
                       label="80% interval, one origin"),
            plt.Line2D([], [], marker="o", ls="", color=BLUE, label="95% interval, one origin"),
            plt.Line2D([], [], color=INK, lw=2.2, label=f"mean over {len(covs)} origins"),
            plt.Line2D([], [], color=ORANGE, lw=1.6, label="nominal level"),
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=8)

        days = np.arange(1, 8)
        for m in POINT_MODELS:
            vals = profiles["A"][m]["wape_by_horizon_day"]
            ax2.plot(days, vals, marker="o", color=MODEL_COLORS[m], lw=1.8, ms=5,
                     label=LABELS[m].replace("\n", " "))
        ax2.set_xticks(days, [f"day {d}" for d in days])
        ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel(f"WAPE (pooled over {len(covs)} origins)")
        ax2.set_title("Error by forecast day (setting A)", fontsize=10.5)
        ax2.legend(loc="lower right", fontsize=8)
        _save(fig, path)
