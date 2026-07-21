from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "model" / "pattern_model.json"
LANDMARKS = np.arange(-52, 1, 4, dtype=int)
STRUCTURE_COLUMNS = [
    "pre_52w_return_pct",
    "pre_52w_volatility_ann_pct",
    "pre_52w_max_drawdown_pct",
    "ath_safety_margin_pct",
    "ma_150_200_tightness_pct",
    "ma_20_slope_4w_pct",
    "ma_50_slope_4w_pct",
    "ma_150_slope_4w_pct",
    "ma_200_slope_4w_pct",
    "candle_body_pct_of_range",
]


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _series(section: pd.DataFrame, column: str, expected: np.ndarray) -> pd.Series:
    if column not in section:
        return pd.Series(np.nan, index=expected, dtype=float)
    values = pd.to_numeric(section[column], errors="coerce").reindex(expected)
    return values.interpolate(limit_direction="both")


def _add_path(output: dict[str, float], family: str, name: str, values: pd.Series) -> None:
    smooth = values.rolling(3, center=True, min_periods=1).median()
    for week in LANDMARKS:
        output[f"{family}::{name}::{int(week):+03d}"] = _number(smooth.get(int(week)))


def build_pre_features(window: pd.DataFrame, event: Mapping[str, Any]) -> dict[str, float]:
    """Build the exact 220 pre-candle features used by the fixed MMRM model."""
    relative = pd.to_numeric(window["relative_week"], errors="coerce")
    section = window.loc[relative.between(-52, 0)].copy()
    section["relative_week"] = pd.to_numeric(section["relative_week"], errors="coerce")
    section = section.dropna(subset=["relative_week"]).drop_duplicates("relative_week").set_index("relative_week")
    expected = np.arange(-52, 1)
    close = pd.to_numeric(section.get("close"), errors="coerce").reindex(expected)
    if close.notna().sum() < 49 or not np.isfinite(close.get(0, np.nan)):
        raise ValueError("relative_week -52..0 구간에 유효한 close가 최소 49주 필요합니다.")
    close = close.interpolate(limit_direction="both")
    close0 = float(close.loc[0])
    if close0 <= 0:
        raise ValueError("기준봉 close는 0보다 커야 합니다.")

    output: dict[str, float] = {}
    price = (close / close0 - 1.0) * 100.0
    _add_path(output, "price", "close_path_pct", price)
    output["price::close_path_pct::+00"] = 0.0
    drawdown = (close / close.cummax() - 1.0) * 100.0
    _add_path(output, "price", "drawdown_path_pct", drawdown)

    for source in (
        "close_to_ma_5_pct",
        "close_to_ma_20_pct",
        "close_to_ma_50_pct",
        "close_to_ma_150_pct",
        "close_to_ma_200_pct",
    ):
        _add_path(output, "moving_average", source, _series(section, source, expected).clip(-100, 200))

    for source in ("macd", "signal", "histogram"):
        values = (_series(section, source, expected) / close * 100.0).clip(-25, 25)
        _add_path(output, "macd", f"{source}_pct_close", values)

    momentum = (_series(section, "momentum_14", expected) / close * 100.0).clip(-100, 100)
    _add_path(output, "oscillator", "momentum_pct_close", momentum)
    for source in ("rsi_14", "mfi_14"):
        values = ((_series(section, source, expected) - 50.0) / 20.0).clip(-2.5, 2.5)
        _add_path(output, "oscillator", source, values)

    volume = np.log1p(_series(section, "volume_ratio_50", expected).clip(lower=0, upper=10))
    _add_path(output, "activity", "log_volume_ratio", volume)
    weekly = _series(section, "weekly_return_pct", expected).clip(-40, 40)
    _add_path(output, "activity", "weekly_return_pct", weekly)

    for column in STRUCTURE_COLUMNS:
        output[f"structure::{column}"] = _number(event.get(column))
    return output


def _distance_percentile(distance: float, quantiles: Mapping[str, float]) -> tuple[str, str]:
    if distance <= float(quantiles["p50"]):
        return "central", "패턴 중심부"
    if distance <= float(quantiles["p75"]):
        return "typical", "일반적인 유사도"
    if distance <= float(quantiles["p90"]):
        return "outer", "패턴 바깥쪽"
    if distance <= float(quantiles["p95"]):
        return "borderline", "경계 사례"
    return "outlier", "저유사도·수동검토 필요"


def classify_event(
    engine: str,
    window: pd.DataFrame,
    event: Mapping[str, Any],
    model_path: Path | str = DEFAULT_MODEL,
) -> dict[str, Any]:
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if engine not in model["engines"]:
        raise ValueError(f"지원하지 않는 엔진: {engine}")
    spec = model["engines"][engine]
    features = build_pre_features(window, event)
    columns = spec["feature_columns"]
    values = np.array([features.get(column, np.nan) for column in columns], dtype=float)
    medians = np.array(spec["imputation_median"], dtype=float)
    values = np.where(np.isfinite(values), values, medians)
    center = np.array(spec["robust_center"], dtype=float)
    scale = np.array(spec["robust_scale"], dtype=float)
    weight = np.array(spec["feature_weight"], dtype=float)
    vector = np.clip((values - center) / scale, -6, 6) * weight

    codes = list(spec["clusters"])
    centers = np.array([spec["clusters"][code]["center"] for code in codes], dtype=float)
    distances = np.sqrt(((centers - vector) ** 2).sum(axis=1))
    index = int(np.argmin(distances))
    code = codes[index]
    cluster = spec["clusters"][code]
    distance = float(distances[index])
    distance_status, distance_label = _distance_percentile(
        distance, cluster["training_distance_quantiles"]
    )
    return {
        "engine": engine,
        "cluster_code": code,
        "pattern_id": cluster["pattern_id"],
        "pattern_name": cluster["pattern_name"],
        "pattern_distance": distance,
        "distance_status": distance_status,
        "distance_label": distance_label,
        "training_cluster_n": int(cluster["training_n"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MMRM 신규 기준봉의 고정 패턴을 분류합니다.")
    parser.add_argument("--engine", choices=["investing_proxy", "kis_compatible"], required=True)
    parser.add_argument("--window", type=Path, required=True, help="relative_week -52..0을 포함한 주봉 CSV")
    parser.add_argument("--event", type=Path, required=True, help="10개 구조 특징을 담은 JSON")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    window = pd.read_csv(args.window)
    event = json.loads(args.event.read_text(encoding="utf-8"))
    result = classify_event(args.engine, window, event, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
