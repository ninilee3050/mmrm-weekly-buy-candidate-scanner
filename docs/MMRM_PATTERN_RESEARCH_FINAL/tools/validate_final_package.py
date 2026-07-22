from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REVISED_LONG_MA_PATTERN_NAME = "장기선 밀집 저항·회복 시도형"
REQUIRED = [
    "README.md",
    "MMRM_KNOWLEDGE_BASE.md",
    "MMRM_PATTERN_REPORT.html",
    "NEXT_SESSION_PROMPT.txt",
    "RESEARCH_MANIFEST.json",
    "data/events_investing_proxy.csv",
    "data/events_kis_compatible.csv",
    "data/pattern_definitions.csv",
    "data/representative_chart_windows.csv",
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
    require(
        manifest["visualization"]["type"] == "dual_familiar_actual_and_fixed_cluster_center",
        "visualization type mismatch",
    )
    require(manifest["visualization"]["future_data"] == "excluded", "visualization future-data rule mismatch")
    require(definitions["pattern_id"].nunique() == 5, "pattern definition count must be five")
    require({"representative_event", "familiar_chart", "representative_chart"}.issubset(definitions.columns), "pattern chart columns missing")
    revised_definition = definitions[
        definitions["pattern_id"] == "long_ma_cluster_discount_recovery"
    ]
    require(len(revised_definition) == 1, "revised long-MA pattern definition missing")
    require(
        revised_definition.iloc[0]["pattern_name"] == REVISED_LONG_MA_PATTERN_NAME,
        "revised long-MA pattern name mismatch",
    )
    require(
        "상단 저항" in revised_definition.iloc[0]["meaning"]
        and "지지 전환" in revised_definition.iloc[0]["classification_checklist"],
        "revised long-MA resistance interpretation missing",
    )
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
    for frame_name, frame in (("investing", investing), ("KIS", kis), ("metrics", metrics), ("predictions", predictions)):
        revised_names = frame.loc[
            frame["pattern_id"] == "long_ma_cluster_discount_recovery", "pattern_name"
        ].dropna().unique()
        require(
            set(revised_names) == {REVISED_LONG_MA_PATTERN_NAME},
            f"stale long-MA pattern name: {frame_name}",
        )

    representative_windows = pd.read_csv(DATA / "representative_chart_windows.csv", low_memory=False)
    require(len(representative_windows) == 265, "representative chart row count mismatch")
    require(representative_windows["event_id"].nunique() == 5, "representative chart event count mismatch")
    relative_week = pd.to_numeric(representative_windows["relative_week"], errors="raise")
    require(relative_week.min() == -52 and relative_week.max() == 0, "representative chart window mismatch")
    require((relative_week > 0).sum() == 0, "future row leaked into representative chart data")
    require(
        (representative_windows.groupby("event_id").size() == 53).all(),
        "representative chart must have 53 rows per event",
    )

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
            if cluster["pattern_id"] == "long_ma_cluster_discount_recovery":
                require(
                    cluster["pattern_name"] == REVISED_LONG_MA_PATTERN_NAME,
                    f"stale long-MA model name: {engine}",
                )

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
    require(REVISED_LONG_MA_PATTERN_NAME in report, "revised long-MA report name missing")
    require("장기선이 받쳐주는 할인 회복" not in report, "stale support interpretation in report")
    require("가격 위의 150·200주선은 저항 후보" in report, "report MA direction rule missing")
    require('id="conditions"' in report and 'id="stability"' in report and 'id="flows"' in report, "HTML evidence sections missing")
    for pattern_id in definitions["pattern_id"]:
        require(f'id="pattern-{pattern_id}"' in report, f"pattern section missing: {pattern_id}")
    image_sources = re.findall(r'<img src="([^"]+)"', report)
    require(len(image_sources) == 10, "report must contain two images for each pattern")
    for source in image_sources:
        asset = ROOT / source
        require(asset.is_file(), f"HTML image missing: {source}")
        svg = asset.read_text(encoding="utf-8")
        ET.fromstring(svg)
        if Path(source).name.startswith("actual_"):
            require("실제 대표사례" in svg, f"actual representative label missing: {source}")
            require("기준봉 이후 성과는 숨김" in svg, f"actual future exclusion label missing: {source}")
            require('viewBox="0 0 1200 1000"' in svg, f"actual chart viewport mismatch: {source}")
            for label in ("주봉 가격", "거래량", "Momentum 14", "MACD 12·26", "RSI 14", "MFI 14"):
                require(label in svg, f"actual chart panel missing ({label}): {source}")
        else:
            require("고정 패턴 중심 프로토타입" in svg, f"pattern prototype label missing: {source}")
            require("T-52~T0" in svg, f"pre-candle window label missing: {source}")
            require('viewBox="0 0 1200 710"' in svg, f"pattern prototype viewport mismatch: {source}")
        require("T+52" not in svg, f"future window leaked into chart: {source}")
        require(not re.search(r"(?<![A-Za-z])(nan|inf)(?![A-Za-z])", svg, re.I), f"non-finite SVG value: {source}")

    knowledge = (ROOT / "MMRM_KNOWLEDGE_BASE.md").read_text(encoding="utf-8")
    for phrase in (
        "신규 또는 과거 기준봉 분석 절차",
        "프로그램 확장 규격",
        "투자 논리가 성립하는 기대",
        "매수하지 않음",
        "pattern_model.json",
        "전체적인 방향",
        "보고서 대표 이미지 읽는 법",
        "실제 대표사례 차트",
        "이동평균선의 지지·저항 방향 규칙",
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
