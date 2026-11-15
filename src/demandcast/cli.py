"""Command line: ``demandcast download`` and ``demandcast run`` (full reproduction)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from demandcast import evaluate, figures
from demandcast.data import OPENML_ID, UCI_URL, load_bike_hourly
from demandcast.pipeline import (
    INTERVAL_METHODS,
    POINT_MODELS,
    BacktestConfig,
    make_splitter,
    run_backtest,
)


def _versions() -> dict[str, str]:
    import matplotlib
    import sklearn
    import statsmodels

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _median_origin(per_origin_wape: list[float], origins: list[pd.Timestamp]) -> pd.Timestamp:
    """Origin with the (lower) median WAPE: a typical week, not a cherry-picked one."""
    order = np.argsort(per_origin_wape, kind="stable")
    return origins[int(order[(len(order) - 1) // 2])]


def build_metrics(df: pd.DataFrame, result, config: BacktestConfig) -> dict:
    """Assemble everything the README reports into one JSON-serialisable dict."""
    splits = list(make_splitter(config).split(df.index))
    points = evaluate.point_summary(result)
    origins = [s.origin for s in splits]
    hero = _median_origin(points["A"]["gbm_direct"]["per_origin"]["wape"], origins)
    years = df.index.year
    return {
        "dataset": {
            "name": "UCI Bike Sharing (hourly), Capital Bikeshare, Washington D.C.",
            "source": f"OpenML data_id={OPENML_ID} (fallback: {UCI_URL})",
            "period": f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}",
            "hours": int(len(df)),
            "recorded_hours": int(df["recorded"].sum()),
            "unrecorded_hours": int((~df["recorded"]).sum()),
            "mean_rentals_per_hour": {
                str(y): round(float(df.loc[years == y, "count"].mean()), 2)
                for y in sorted(set(years))
            },
        },
        "design": {
            "origins": [f"{o:%Y-%m-%d %H:%M}" for o in origins],
            "horizon_hours": config.horizon,
            "gap_hours": config.gap,
            "min_lag_hours": config.min_lag,
            "calibration_hours": config.calibration_hours,
            "step_days": config.step_days,
            "interval_levels": list(config.levels),
            "training_hours_first_origin": int(len(splits[0].train)),
            "training_hours_last_origin": int(len(splits[-1].train)),
            "gbm": {"max_iter": config.gbm_max_iter, "learning_rate": config.gbm_learning_rate,
                    "max_leaf_nodes": 31, "min_samples_leaf": 20, "early_stopping": False},
            "seed": config.seed,
        },
        "settings": {
            "A": "demand history + calendar only (information available at the origin)",
            "B": "A + observed weather of the forecast week (perfect weather forecast, "
            "an upper bound)",
        },
        "scale_diagnostic_before_first_origin": evaluate.residual_spread_by_level(
            df, origins[0]
        ),
        "point": points,
        "weather_oracle_gain": evaluate.weather_oracle_gain(result),
        "error_profiles": evaluate.error_profiles(result),
        "intervals": evaluate.interval_summary(result),
        "forecast_week_figure_origin": f"{hero:%Y-%m-%d}",
        "environment": _versions(),
    }


def _print_summary(m: dict) -> None:
    print("\nPoint forecasts (mean +/- std over origins):")
    for s in ("A", "B"):
        for model in POINT_MODELS:
            v = m["point"][s][model]
            print(f"  {s} {model:<20} MASE {v['mase']['mean']:.3f} +/- {v['mase']['std']:.3f}  "
                  f"WAPE {v['wape']['mean']:.1%} +/- {v['wape']['std']:.1%} "
                  f"(median {v['wape']['median']:.1%}, pooled {v['pooled_wape']:.1%})  "
                  f"MAE {v['mae']['mean']:.1f}  "
                  f"beats naive {v['origins_beating_seasonal_naive_mae']}"
                  f"/{len(m['design']['origins'])}")
    print("\nIntervals (coverage mean +/- std, mean width):")
    for s in ("A", "B"):
        for method in INTERVAL_METHODS:
            for level, v in m["intervals"][s][method].items():
                print(f"  {s} {method:<26} {level}%: coverage {v['coverage']['mean']:.1%} +/- "
                      f"{v['coverage']['std']:.1%}  width {v['width']['mean']:.0f}")


def run(df: pd.DataFrame, config: BacktestConfig, out: str | Path) -> dict:
    """Backtest ``df``, write ``metrics.json`` and the four figures to ``out``; return metrics."""
    result = run_backtest(df, config, log=lambda msg: print(msg, flush=True))
    metrics = build_metrics(df, result, config)
    out = Path(out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    splits = list(make_splitter(config).split(df.index))
    figures.fig_backtest_origins(df, splits, fig_dir / "backtest_origins.png")
    figures.fig_forecast_week(
        result, pd.Timestamp(metrics["forecast_week_figure_origin"]), fig_dir / "forecast_week.png"
    )
    figures.fig_model_comparison(metrics["point"], fig_dir / "model_comparison.png")
    figures.fig_intervals_and_horizon(
        metrics["intervals"], metrics["error_profiles"], fig_dir / "intervals_and_horizon.png"
    )
    return metrics


def cmd_run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    metrics = run(load_bike_hourly(args.data_home), BacktestConfig(), args.out)
    _print_summary(metrics)
    print(f"\nWrote {Path(args.out) / 'metrics.json'} and 4 figures "
          f"in {time.perf_counter() - t0:.0f} s.")


def cmd_download(args: argparse.Namespace) -> None:
    df = load_bike_hourly(args.data_home)
    print(f"Bike Sharing hourly data ready in {args.data_home}/: {len(df)} hours, "
          f"{int(df['recorded'].sum())} recorded.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="demandcast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_dl = sub.add_parser("download", help="download and cache the dataset")
    p_dl.add_argument("--data-home", default="data")
    p_dl.set_defaults(func=cmd_download)
    p_run = sub.add_parser("run", help="full backtest -> metrics.json + figures")
    p_run.add_argument("--data-home", default="data")
    p_run.add_argument("--out", default="reports")
    p_run.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
