from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REQUIRED = [
    "README.md",
    "MMRM_KNOWLEDGE_BASE.md",
    "MMRM_PATTERN_REPORT.html",
    "NEXT_SESSION_PROMPT.txt",
    "RESEARCH_MANIFEST.json",
    "data/events_investing_proxy.csv",
    "data/events_kis_compatible.csv",
    "data/pattern_definitions.csv",
    "data/pattern_metrics.csv",
    "data/provider_comparison.csv",
    "data/walk_forward_predictions.csv",
    "data/walk_forward_period_metrics.csv",
    "data/condition_metrics.csv",
    "data/outcome_flow_metrics.csv",
    "data/pattern_to_flow_metrics.csv",
    "model/pattern_model.json",
    "tools/pattern_classifier.py",
    "tools/rebuild_final_package.py",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    for relative in REQUIRED:
        require((ROOT / relative).is_file(), f"missing: {relative}")

    manifest = json.loads((ROOT / "RESEARCH_MANIFEST.json").read_text(encoding="utf-8"))
    model = json.loads((ROOT / "model" / "pattern_model.json").read_text(encoding="utf-8"))
    definitions = pd.read_csv(DATA / "pattern_definitions.csv")
    metrics = pd.read_csv(DATA / "pattern_metrics.csv")
    predictions = pd.read_csv(DATA / "walk_forward_predictions.csv", low_memory=False)
    conditions = pd.read_csv(DATA / "condition_metrics.csv")
    flows = pd.read_csv(DATA / "outcome_flow_metrics.csv")
    pattern_flows = pd.read_csv(DATA / "pattern_to_flow_metrics.csv")
    investing = pd.read_csv(DATA / "events_investing_proxy.csv", low_memory=False)
    kis = pd.read_csv(DATA / "events_kis_compatible.csv", low_memory=False)

    require(manifest["authority"] == "single_canonical_folder", "manifest authority mismatch")
    require(manifest["package_schema_version"] == "2.0", "package schema mismatch")
    require(manifest["taxonomy_training_end"] == "2015-10-05", "training end mismatch")
    require(manifest["walk_forward_evaluation_start"] == "2015-10-12", "evaluation start mismatch")
    require(len(investing) == manifest["engines"]["investing_proxy"]["event_count"], "investing count mismatch")
    require(len(kis) == manifest["engines"]["kis_compatible"]["event_count"], "KIS count mismatch")
    require(manifest["decision_policy"]["default_probability_threshold"] == 0.4, "decision policy mismatch")
    require(definitions["pattern_id"].nunique() == 5, "pattern definition count must be five")
    require(set(metrics["engine"]) == {"investing_proxy", "kis_compatible"}, "engine set mismatch")
    require(len(metrics) == 10, "expected five metric rows per engine")
    require(set(conditions["condition"]) == {"ma_order", "ma_tightness", "macd_zero", "ath_margin"}, "condition set mismatch")
    require(set(flows["engine"]) == {"investing_proxy", "kis_compatible"}, "outcome flow engine mismatch")
    require(flows[flows["engine"] == "investing_proxy"]["outcome_flow_name"].nunique() == 5, "investing outcome flow count mismatch")
    require(set(pattern_flows["engine"]) == {"investing_proxy", "kis_compatible"}, "pattern-to-flow engine mismatch")
    require(metrics["pattern_id"].isin(definitions["pattern_id"]).all(), "unknown pattern in metrics")
    require(predictions["buy_point_date"].min() >= "2015-10-12", "pre-evaluation prediction found")
    require(
        predictions[["engine", "event_id"]].duplicated().sum() == 0,
        "duplicate prediction engine/event_id",
    )
    require(
        int(((metrics["engine"] == "investing_proxy") & metrics["official_overall_rank"].notna()).sum()) == 4,
        "official ranked count mismatch",
    )

    required_event_columns = {
        "open", "high", "low", "close", "volume", "volume_ma_50", "volume_ratio_50",
        "candle_return_pct", "candle_body_pct_of_range", "histogram", "macd_gap",
        "pre_26w_return_pct", "pre_13w_return_pct", "pre_52w_volatility_ann_pct",
        "ma_20_slope_4w_pct", "ma_50_slope_4w_pct", "ma_150_slope_4w_pct",
        "ma_200_slope_4w_pct", "prior_ath_price", "prior_ath_date_last",
        "return_to_prior_ath_pct",
    }
    require(required_event_columns.issubset(investing.columns), "investing event detail columns missing")
    require(required_event_columns.issubset(kis.columns), "KIS event detail columns missing")

    require(model["schema_version"] == "1.0", "pattern model schema mismatch")
    require(set(model["engines"]) == {"investing_proxy", "kis_compatible"}, "model engine set mismatch")
    for engine, spec in model["engines"].items():
        require(spec["feature_count"] == 220, f"model feature count mismatch: {engine}")
        require(len(spec["feature_columns"]) == 220, f"model feature list mismatch: {engine}")
        require(len(spec["clusters"]) == 5, f"model cluster count mismatch: {engine}")
        require(spec["verification"]["assignment_mismatches"] == 0, f"model assignment mismatch: {engine}")
        for cluster in spec["clusters"].values():
            require(len(cluster["center"]) == 220, f"model center width mismatch: {engine}")
            require(set(cluster["training_distance_quantiles"]) == {"p25", "p50", "p75", "p90", "p95", "max"}, f"distance quantiles missing: {engine}")

    msft = investing[investing["event_id"] == "MSFT_2026-05-25"]
    require(len(msft) == 1, "MSFT regression event missing")
    require(msft.iloc[0]["pattern_id"] == "normal_pullback_recovery", "MSFT pattern regression mismatch")
    require(abs(float(msft.iloc[0]["pattern_distance"]) - 0.6266004818755909) < 1e-12, "MSFT distance regression mismatch")
    msft_prediction = predictions[
        (predictions["engine"] == "investing_proxy") & (predictions["event_id"] == "MSFT_2026-05-25")
    ]
    require(len(msft_prediction) == 1, "MSFT walk-forward regression row missing")
    require(abs(float(msft_prediction.iloc[0]["close_20_at_52w_walk_forward_probability"]) - 0.32737715379706445) < 1e-12, "MSFT probability regression mismatch")

    report = (ROOT / "MMRM_PATTERN_REPORT.html").read_text(encoding="utf-8")
    require("<meta charset=\"utf-8\">" in report, "HTML charset missing")
    require("워크포워드" in report and "2015-10-12" in report, "HTML method statement missing")
    require('id="conditions"' in report and 'id="stability"' in report and 'id="flows"' in report, "HTML evidence sections missing")
    for pattern_id in definitions["pattern_id"]:
        require(f'id="pattern-{pattern_id}"' in report, f"pattern section missing: {pattern_id}")
    for source in re.findall(r'<img src="([^"]+)"', report):
        require((ROOT / source).is_file(), f"HTML image missing: {source}")

    knowledge = (ROOT / "MMRM_KNOWLEDGE_BASE.md").read_text(encoding="utf-8")
    for phrase in (
        "신규 또는 과거 기준봉 분석 절차",
        "프로그램 확장 규격",
        "투자 논리가 성립하는 기대",
        "매수하지 않음",
        "pattern_model.json",
        "전체적인 방향",
    ):
        require(phrase in knowledge, f"knowledge section missing: {phrase}")

    prompt = (ROOT / "NEXT_SESSION_PROMPT.txt").read_text(encoding="utf-8")
    for phrase in ("쉬운 결론", "매수하지 않음", "사후검산", "pattern_classifier.py"):
        require(phrase in prompt, f"session prompt rule missing: {phrase}")

    print("MMRM final package validation: PASS")
    print(f"files={sum(1 for p in ROOT.rglob('*') if p.is_file())}")
    print(f"investing_events={len(investing)} kis_events={len(kis)} predictions={len(predictions)}")
    print("official_ranked_patterns=4 reference_only_patterns=1 model_engines=2 features=220 clusters=5")


if __name__ == "__main__":
    main()
