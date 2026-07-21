from __future__ import annotations

import html
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVALUATION_START = pd.Timestamp("2015-10-12")
TAXONOMY_TRAIN_END = pd.Timestamp("2015-10-05")
MIN_OFFICIAL_N = 30

ENGINES = {
    "investing_proxy": {
        "label": "원형 MMRM 근사판",
        "file": "events_investing_proxy.csv",
        "role": "공식 주 기준",
    },
    "kis_compatible": {
        "label": "KIS 호환판",
        "file": "events_kis_compatible.csv",
        "role": "민감도 비교",
    },
}

TARGETS = {
    "rise_8w": (8, "return_8w_pct", lambda s: s > 0),
    "rise_26w": (26, "return_26w_pct", lambda s: s > 0),
    "rise_52w": (52, "return_52w_pct", lambda s: s > 0),
    "close_20_at_52w": (52, "return_52w_pct", lambda s: s >= 20),
    "touch_20_within_52w": (52, "hit_20_within_52w", None),
}

PATTERN_ADVICE = {
    "long_ma_cluster_discount_recovery": {
        "rank_note": "즉각성·방어력 우선형",
        "action": "가장 먼저 확인한다. 150·200주선 지지가 유지되고 MACD 회복이 이어지는지 본다.",
        "risk": "장기선 지지를 주봉 음봉 종가로 이탈하면 가설을 낮추고 재진입 신호를 기다린다.",
    },
    "long_ma_near_initial_rebound": {
        "rank_note": "1년 +20% 목표 우선형",
        "action": "52주 목표 성과가 강하지만 즉각성 편차가 있다. V/U 저점과 재시험 여부를 확인한다.",
        "risk": "첫 반등 뒤 지지점 재이탈과 이중바닥 가능성이 있어 분할·확인 접근이 적합하다.",
    },
    "normal_pullback_recovery": {
        "rank_note": "기준선·선별형",
        "action": "표본이 가장 많아 기준선으로 사용한다. 장기선 배열과 MACD 0선 근접 여부로 우선순위를 높인다.",
        "risk": "특별한 구조적 우위가 없으면 평균적인 결과에 머물 수 있다.",
    },
    "long_uptrend_extended_pullback": {
        "rank_note": "돌파 잠재력·고위험형",
        "action": "ATH 돌파 확장 가능성은 있지만 장기 지지선이 멀다. 추세 훼손 여부를 먼저 본다.",
        "risk": "즉각성과 최대하락폭이 불리할 수 있어 전고점 근처라는 이유만으로 우선매수하지 않는다.",
    },
    "deep_crash_below_long_ma": {
        "rank_note": "표본부족·관찰형",
        "action": "반등 폭은 클 수 있지만 공식 순위에서 제외한다. 장기구조 회복이나 두 번째 확인 신호를 기다린다.",
        "risk": "표본이 적고 장기선 아래 붕괴 상태라 재하락 위험이 크다.",
    },
}


def wilson(hits: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = hits / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / den
    return max(0.0, center - margin), min(1.0, center + margin)


def bool_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric.where(numeric.notna())
    text = series.astype(str).str.strip().str.lower()
    result = result.where(result.notna(), text.map({"true": 1.0, "false": 0.0}))
    return pd.to_numeric(result, errors="coerce")


def load_events(engine: str) -> pd.DataFrame:
    meta = ENGINES[engine]
    frame = pd.read_csv(DATA / meta["file"], low_memory=False)
    frame["buy_point_date"] = pd.to_datetime(frame["buy_point_date"], errors="coerce")
    frame = frame.dropna(subset=["buy_point_date", "pattern_id"]).copy()
    frame["engine"] = engine
    frame["engine_label"] = meta["label"]
    frame["evaluation_role"] = np.where(
        frame["buy_point_date"] <= TAXONOMY_TRAIN_END,
        "pattern_taxonomy_training",
        "walk_forward_evaluation",
    )
    return frame.sort_values(["buy_point_date", "event_id"]).reset_index(drop=True)


def target_values(frame: pd.DataFrame, target: str) -> pd.Series:
    _, column, transform = TARGETS[target]
    if target == "touch_20_within_52w":
        # Every event must have the same full 52-week observation opportunity.
        # A recent event may already have touched +20% while its 52-week window is
        # still incomplete; it is not eligible for the official 52-week rate yet.
        eligible = pd.to_numeric(frame["return_52w_pct"], errors="coerce").notna()
        return bool_series(frame[column]).where(eligible)
    values = pd.to_numeric(frame[column], errors="coerce")
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = values.notna()
    output.loc[valid] = transform(values.loc[valid]).astype(float)
    return output


def walk_forward_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pattern_id, group in frame.groupby("pattern_id", sort=False):
        if pattern_id == "unclassified":
            continue
        group = group.sort_values(["buy_point_date", "event_id"]).reset_index(drop=True)
        dates = group["buy_point_date"].to_numpy(dtype="datetime64[ns]")
        arrays = {target: target_values(group, target).to_numpy(dtype=float) for target in TARGETS}
        prefix: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for target, values in arrays.items():
            valid = np.isfinite(values)
            prefix[target] = (
                np.concatenate([[0], np.cumsum(valid.astype(int))]),
                np.concatenate([[0.0], np.cumsum(np.where(valid, values, 0.0))]),
            )
        for index, event in group.iterrows():
            if event["buy_point_date"] < EVALUATION_START:
                continue
            row: dict[str, object] = {
                "engine": event["engine"],
                "engine_label": event["engine_label"],
                "event_id": event["event_id"],
                "ticker": event["ticker"],
                "buy_point_date": event["buy_point_date"].strftime("%Y-%m-%d"),
                "pattern_id": pattern_id,
                "pattern_name": event["pattern_name"],
            }
            for target, (weeks, _, _) in TARGETS.items():
                maturity_cutoff = np.datetime64(event["buy_point_date"] - pd.Timedelta(weeks=weeks))
                eligible = int(np.searchsorted(dates, maturity_cutoff, side="right"))
                counts, hits = prefix[target]
                n = int(counts[eligible])
                hit = float(hits[eligible])
                raw = hit / n if n else np.nan
                smoothed = (hit + 1.0) / (n + 2.0) if n else np.nan
                row[f"{target}_history_n"] = n
                row[f"{target}_history_probability"] = raw
                row[f"{target}_walk_forward_probability"] = smoothed if n >= MIN_OFFICIAL_N else np.nan
                row[f"{target}_actual"] = arrays[target][index]
            rows.append(row)
    return pd.DataFrame(rows)


def metric_row(group: pd.DataFrame, engine: str, pattern_id: str, pattern_name: str) -> dict[str, object]:
    row: dict[str, object] = {
        "engine": engine,
        "engine_label": ENGINES[engine]["label"],
        "pattern_id": pattern_id,
        "pattern_name": pattern_name,
        "evaluation_event_n": int(len(group)),
        "evaluation_ticker_n": int(group["ticker"].nunique()),
    }
    for horizon in (1, 4, 8, 26, 52):
        values = pd.to_numeric(group[f"return_{horizon}w_pct"], errors="coerce").dropna()
        hits = int((values > 0).sum())
        low, high = wilson(hits, len(values))
        row[f"rise_{horizon}w_valid_n"] = int(len(values))
        row[f"rise_{horizon}w_probability"] = hits / len(values) if len(values) else np.nan
        row[f"rise_{horizon}w_ci_low"] = low
        row[f"rise_{horizon}w_ci_high"] = high
        row[f"return_{horizon}w_median_pct"] = values.median()
    for horizon in (8, 52):
        row[f"mae_{horizon}w_median_pct"] = pd.to_numeric(
            group[f"mae_{horizon}w_pct"], errors="coerce"
        ).median()
    ret52 = pd.to_numeric(group["return_52w_pct"], errors="coerce").dropna()
    close_hits = int((ret52 >= 20).sum())
    close_low, close_high = wilson(close_hits, len(ret52))
    full_52w = pd.to_numeric(group["return_52w_pct"], errors="coerce").notna()
    touch_all = bool_series(group["hit_20_within_52w"]).where(full_52w)
    touch = touch_all.dropna()
    touch_hits = int((touch > 0).sum())
    touch_low, touch_high = wilson(touch_hits, len(touch))
    row.update(
        {
            "close_20_at_52w_valid_n": int(len(ret52)),
            "close_20_at_52w_probability": close_hits / len(ret52) if len(ret52) else np.nan,
            "close_20_at_52w_ci_low": close_low,
            "close_20_at_52w_ci_high": close_high,
            "touch_20_within_52w_valid_n": int(len(touch)),
            "touch_20_within_52w_probability": touch_hits / len(touch) if len(touch) else np.nan,
            "touch_20_within_52w_ci_low": touch_low,
            "touch_20_within_52w_ci_high": touch_high,
            "weeks_to_gain_20_median_among_hits": pd.to_numeric(
                group.loc[touch_all > 0, "weeks_to_gain_20"], errors="coerce"
            ).median(),
        }
    )
    return row


def official_metrics(events_by_engine: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for engine, frame in events_by_engine.items():
        evaluation = frame[frame["buy_point_date"] >= EVALUATION_START]
        for pattern_id, group in evaluation.groupby("pattern_id", sort=False):
            if pattern_id == "unclassified":
                continue
            rows.append(metric_row(group, engine, pattern_id, str(group.iloc[0]["pattern_name"])))
    metrics = pd.DataFrame(rows)
    metrics["rank_status"] = np.where(
        metrics["evaluation_event_n"] >= MIN_OFFICIAL_N, "official", "insufficient_sample"
    )
    metrics["goal_52w_rank"] = np.nan
    metrics["immediate_rank"] = np.nan
    metrics["safety_rank"] = np.nan
    metrics["overall_score"] = np.nan
    metrics["official_overall_rank"] = np.nan
    for engine in ENGINES:
        eligible = metrics[(metrics["engine"] == engine) & (metrics["rank_status"] == "official")].copy()
        if eligible.empty:
            continue
        goal = eligible["close_20_at_52w_probability"].rank(ascending=False, method="min")
        touch = eligible["touch_20_within_52w_probability"].rank(ascending=False, method="min")
        immediate = eligible["rise_8w_probability"].rank(ascending=False, method="min")
        safety = eligible["mae_52w_median_pct"].rank(ascending=False, method="min")
        score = goal * 0.40 + touch * 0.10 + immediate * 0.20 + safety * 0.30
        overall = score.rank(ascending=True, method="min")
        metrics.loc[eligible.index, "goal_52w_rank"] = goal
        metrics.loc[eligible.index, "immediate_rank"] = immediate
        metrics.loc[eligible.index, "safety_rank"] = safety
        metrics.loc[eligible.index, "overall_score"] = score
        metrics.loc[eligible.index, "official_overall_rank"] = overall
    return metrics.sort_values(["engine", "official_overall_rank", "pattern_name"], na_position="last")


def attach_calibration(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    for target in TARGETS:
        pcol = f"{target}_walk_forward_probability"
        acol = f"{target}_actual"
        for (engine, pattern_id), group in predictions.groupby(["engine", "pattern_id"]):
            valid = group[pcol].notna() & group[acol].notna()
            if not valid.any():
                continue
            index = result.index[(result["engine"] == engine) & (result["pattern_id"] == pattern_id)]
            result.loc[index, f"{target}_wf_prediction_n"] = int(valid.sum())
            result.loc[index, f"{target}_wf_brier"] = float(
                np.mean((group.loc[valid, pcol] - group.loc[valid, acol]) ** 2)
            )
            result.loc[index, f"{target}_wf_mean_predicted_probability"] = float(group.loc[valid, pcol].mean())
    return result


def period_metrics(events_by_engine: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for engine, frame in events_by_engine.items():
        evaluation = frame[frame["buy_point_date"] >= EVALUATION_START].copy()
        year = evaluation["buy_point_date"].dt.year
        evaluation["period"] = np.select(
            [year <= 2019, year <= 2022],
            ["2015-2019", "2020-2022"],
            default="2023+",
        )
        for (period, pattern_id), group in evaluation.groupby(["period", "pattern_id"], sort=True):
            if pattern_id == "unclassified":
                continue
            base = metric_row(group, engine, pattern_id, str(group.iloc[0]["pattern_name"]))
            base["period"] = period
            rows.append(base)
    return pd.DataFrame(rows)


def condition_metrics(events_by_engine: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for engine, frame in events_by_engine.items():
        data = frame[frame["buy_point_date"] >= EVALUATION_START].copy()
        ma150 = pd.to_numeric(data["ma_150"], errors="coerce")
        ma200 = pd.to_numeric(data["ma_200"], errors="coerce")
        tight = pd.to_numeric(data["ma_150_200_tightness_pct"], errors="coerce")
        macd_pct = pd.to_numeric(data["macd"], errors="coerce") / pd.to_numeric(data["close"], errors="coerce") * 100
        ath = pd.to_numeric(data["ath_safety_margin_pct"], errors="coerce")
        groups = {
            "ma_order": pd.Series(
                np.where(ma150.isna() | ma200.isna(), None, np.where(ma150 > ma200, "150>200", "150<=200")),
                index=data.index,
            ),
            "ma_tightness": pd.cut(
                tight, [-np.inf, 2, 5, 10, np.inf], labels=["0~2%", "2~5%", "5~10%", "10% 초과"]
            ).astype(object),
            "macd_zero": pd.cut(
                macd_pct,
                [-np.inf, -2.5, -1, 0, np.inf],
                labels=["-2.5% 미만", "-2.5~-1%", "-1~0%", "0% 이상"],
            ).astype(object),
            "ath_margin": pd.cut(
                ath, [-np.inf, 5, 15, 30, 50, np.inf], labels=["0~5%", "5~15%", "15~30%", "30~50%", "50% 초과"]
            ).astype(object),
        }
        for condition, labels in groups.items():
            for label in pd.Series(labels, index=data.index).dropna().unique():
                group = data[pd.Series(labels, index=data.index) == label]
                ret8 = pd.to_numeric(group["return_8w_pct"], errors="coerce").dropna()
                ret52 = pd.to_numeric(group["return_52w_pct"], errors="coerce").dropna()
                touch = bool_series(group["hit_20_within_52w"]).where(
                    pd.to_numeric(group["return_52w_pct"], errors="coerce").notna()
                ).dropna()
                rows.append(
                    {
                        "engine": engine,
                        "engine_label": ENGINES[engine]["label"],
                        "condition": condition,
                        "group": str(label),
                        "event_n": int(len(group)),
                        "rise_8w_valid_n": int(len(ret8)),
                        "rise_8w_probability": float((ret8 > 0).mean()) if len(ret8) else np.nan,
                        "return_8w_median_pct": ret8.median(),
                        "close_20_at_52w_valid_n": int(len(ret52)),
                        "close_20_at_52w_probability": float((ret52 >= 20).mean()) if len(ret52) else np.nan,
                        "touch_20_within_52w_valid_n": int(len(touch)),
                        "touch_20_within_52w_probability": float((touch > 0).mean()) if len(touch) else np.nan,
                        "mae_52w_median_pct": pd.to_numeric(group["mae_52w_pct"], errors="coerce").median(),
                    }
                )
    return pd.DataFrame(rows)


def pct(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(number) else f"{number * 100:.{digits}f}%"


def num(value: object, suffix: str = "%", digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(number) else f"{number:.{digits}f}{suffix}"


def rank_text(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "표본부족"
    return "표본부족" if not np.isfinite(number) else f"{int(number)}위"


def definitions() -> pd.DataFrame:
    return pd.read_csv(DATA / "pattern_definitions.csv")


def build_readme(metrics: pd.DataFrame) -> str:
    inv = metrics[metrics["engine"] == "investing_proxy"]
    best = inv.sort_values("official_overall_rank").iloc[0]
    return f"""# MMRM 패턴 연구 — 단일 권위 폴더

이 폴더는 앞으로 MMRM 신규·과거 기준봉을 판단할 때 사용하는 **유일한 최종 연구 패키지**입니다. 별도의 구버전 폴더를 참조하지 않습니다.

## 가장 먼저 볼 파일

1. `MMRM_PATTERN_REPORT.html` — 사람이 읽는 완성 보고서와 전체 사건 목록
2. `MMRM_KNOWLEDGE_BASE.md` — 다음 Codex 세션과 프로그램 확장을 위한 상세 지식
3. `data/pattern_metrics.csv` — 워크포워드 공식 확률·순위
4. `data/events_investing_proxy.csv` — 원형 MMRM 근사 기준 전체 사건
5. `RESEARCH_MANIFEST.json` — 데이터 권위·검증법·한계

## 현재 한 줄 결론

사용자 목표(52주 후 +20% 40%, 52주 내 +20% 도달 10%, 8주 즉각성 20%, 52주 하락방어 30%)를 반영한 공식 종합 1위는 **{best['pattern_name']}**입니다. 단, 종합순위와 52주 +20% 단일목표 순위는 다를 수 있으므로 보고서의 두 순위를 함께 봅니다.

## 공식 연구 방식

- 패턴 분류체계 학습 종료: **2015-10-05**
- 워크포워드 평가 시작: **2015-10-12**
- 패턴 이름은 고정하고, 이후 각 기준봉에는 당시 이미 결과가 확정된 과거 사건만 사용해 확률을 계산합니다.
- 52주 결과는 기준봉 후 52주가 지난 뒤에만 다음 사건의 과거 자료로 편입됩니다.
- 기본 권위는 프로그램 출발점과 맞춘 `investing_proxy`이며 `kis_compatible`은 계산법 민감도 비교입니다.

## 다음 세션 복사문

```text
C:\\Users\\user\\Documents\\Codex\\mmrm\\docs\\MMRM_PATTERN_RESEARCH_FINAL\\README.md를 먼저 읽고,
MMRM_KNOWLEDGE_BASE.md의 규칙과 data/pattern_metrics.csv의 워크포워드 결과를 사용해
신규 또는 과거 MMRM 기준봉을 분석해줘.
```

## 주의

이 결과는 현재 시가총액 상위 100개 생존 종목의 과거 자료를 사용한 연구 통계입니다. 미래 수익을 보장하지 않으며 거래비용·환율·세금·실제 다음 주 체결가는 반영하지 않았습니다.
"""


def build_knowledge(metrics: pd.DataFrame, periods: pd.DataFrame, conditions: pd.DataFrame) -> str:
    inv = metrics[metrics["engine"] == "investing_proxy"].sort_values(
        "official_overall_rank", na_position="last"
    )
    ranking_lines = []
    for _, row in inv.iterrows():
        advice = PATTERN_ADVICE[row["pattern_id"]]
        ranking_lines.append(
            f"- **{rank_text(row['official_overall_rank'])} — {row['pattern_name']}**: "
            f"8주 상승 {pct(row['rise_8w_probability'])}, 52주 후 +20% {pct(row['close_20_at_52w_probability'])}, "
            f"52주 내 +20% 접촉 {pct(row['touch_20_within_52w_probability'])}, 52주 MAE 중앙값 {num(row['mae_52w_median_pct'])}. "
            f"{advice['rank_note']}. {advice['action']}"
        )
    goal = inv[inv["rank_status"] == "official"].sort_values("goal_52w_rank")
    goal_lines = [
        f"{int(row['goal_52w_rank'])}위 {row['pattern_name']} ({pct(row['close_20_at_52w_probability'])})"
        for _, row in goal.iterrows()
    ]
    ma_order = conditions[
        (conditions["engine"] == "investing_proxy") & (conditions["condition"] == "ma_order")
    ].set_index("group")
    ma_bull = ma_order.loc["150>200"]
    ma_other = ma_order.loc["150<=200"]
    return f"""# MMRM 기준봉 판단 종합 지식베이스

## 1. 이 파일의 역할

이 문서는 세션이 초기화돼도 신규·과거 MMRM 기준봉을 같은 방법으로 판정하기 위한 권위 문서다. 요약문이 아니라 **분류 절차, 확률 해석, 사례 검색, 위험판정, 프로그램 확장 규격**을 함께 제공한다.

## 2. 연구 목표

사용자의 핵심 목표는 최소 1년 보유를 전제로 연복리 20% 이상을 노리는 것이다. 따라서 핵심 질문은 다음과 같다.

- 기준봉 매수 후 8주 안에 바로 올라갈 가능성은 얼마인가?
- 52주 뒤 종가수익률이 +20% 이상일 가능성은 얼마인가?
- 52주 안에 한 번이라도 +20%에 도달할 가능성과 소요기간은 얼마인가?
- 기대와 다르게 지지를 깨고 내려갈 때 최대 하락폭은 어느 정도였는가?
- ATH 안전마진이 작을 때 신고가 확장으로 이어졌는가, 안전마진이 클 때 회복에 얼마나 걸렸는가?

## 3. 데이터 권위

공식 주 기준은 `investing_proxy`다. MMRM 프로그램이 Investing.com 주봉 화면 설정에서 출발했기 때문이다. 다만 현재 자료는 Investing.com 공식 원천을 직접 내려받은 완전 복제본이 아니라 Yahoo 일봉을 주봉으로 재구성한 근사판이다. `kis_compatible`은 사용자 한국투자증권 화면에 맞춘 계산법 민감도 비교이며, 서로 다른 엔진의 사건을 섞어 하나의 확률로 만들지 않는다.

## 4. 워크포워드 검증의 정확한 뜻

1. 2015-10-05까지의 사건에서 기준봉 이전 52주 차트 특징만 사용해 다섯 패턴의 분류체계를 만들었다.
2. 2015-10-12 이후 사건은 기준봉 이후 수익을 보지 않고 고정된 패턴에 배정한다.
3. 각 평가 사건 시점의 확률은 그때 이미 결과가 성숙한 과거 사건만 사용한다.
4. 예를 들어 52주 후 +20% 결과는 기준봉 발생 후 52주가 지난 다음에만 학습 이력에 들어간다.
5. 역사 사례가 30건 미만이면 실시간 공식확률을 숨기고 `표본부족`으로 표시한다.
6. 보고서의 공식 실현확률은 2015-10-12 이후 평가구간의 실제 빈도이고, Brier 점수는 당시 계산 가능했던 확률의 보정 정도를 나타낸다.

이 방식은 패턴명을 매년 재정의하지 않으면서 미래정보 누수를 막기 위한 **고정 분류체계 + 확장형 확률 갱신**이다. 가장 오래된 구간은 분류체계 학습에 필요하므로 독립 검증으로 주장하지 않는다.

## 5. 다섯 패턴과 판정 순서

패턴은 보조지표 숫자가 소수점까지 같은 사건을 찾는 방식이 아니다. 기준봉 이전 52주의 가격경로, 이동평균 이격, MACD, Momentum·RSI·MFI, 거래활동, ATH·장기선 구조를 4주 간격의 완만한 형태로 압축해 **전체적인 방향·굴곡·지지구조가 비슷한 사건**을 묶는다. 연구 가중치는 가격경로 35%, 이동평균 20%, MACD 15%, 오실레이터 10%, 거래활동 5%, ATH·장기선 등 구조 15%였다. 따라서 신규 사건에서는 한 지표의 정확한 숫자보다 전체 흐름을 먼저 보고, `pattern_distance`로 중심 사례와의 거리를 함께 표시한다.

### 장기선 밀집 할인 회복형

- 이전 1년에 의미 있는 조정이 있다.
- 가격이 150·200주선 근처 또는 약간 아래이고 두 장기선 간격이 좁다.
- 단기선과 MACD가 회복 방향으로 돌아선다.
- 즉각성과 하락방어를 함께 중시할 때 우선 확인한다.
- 장기선 지지를 음봉 주봉 종가로 이탈하면 가설을 낮춘다.

### 급락 후 장기선 부근 초기 반등형

- 이전 1년 낙폭이 크고 V형 또는 U형 저점을 만든다.
- 가격이 장기선 부근까지 돌아왔지만 중장기 구조가 완전히 회복되지는 않았다.
- 52주 목표 성과는 강할 수 있으나 즉시 상승하지 않고 저점을 재시험할 수 있다.
- 지지선 유지, 이중바닥, 20·50주선 안정 여부를 함께 본다.

### 완만한 조정·정상 회복형

- 장기 붕괴가 아닌 정상적인 눌림 또는 횡보 후 회복이다.
- 극단적인 장기선 이격이나 폭락이 없다.
- 가장 흔한 기준선 패턴이므로 추가 우위가 없으면 평균적인 기대값으로 본다.

### 장기 상승추세 고이격 조정형

- 장기 우상향이 명확하고 가격이 150·200주선보다 크게 위에 있다.
- ATH와 가깝고 돌파 후 큰 상승영역 흐름으로 확장할 가능성이 있다.
- 반대로 장기 방어선이 멀어 최대하락폭이 커질 수 있다.
- ATH 근처라는 이유만으로 좋은 기준봉이라고 단정하지 않는다.

### 대폭락·장기선 크게 아래 급반등 준비형

- 가격이 장기선 아래로 크게 붕괴했고 이전 낙폭과 변동성이 극단적이다.
- 반등 폭은 클 수 있지만 장기 구조 회복이 멀고 표본이 적다.
- 공식순위에서 제외하며 추가 확인 신호가 있을 때만 별도 관찰한다.

## 6. 공식 종합 우선순위

종합점수는 사용자의 목표를 반영해 52주 후 +20% 순위 40%, 52주 내 +20% 접촉 10%, 8주 즉각성 20%, 52주 MAE 방어력 30%를 사용한다. 확률값 자체를 임의로 더한 것이 아니라 각 항목의 패턴 순위를 가중 결합한다.

{chr(10).join(ranking_lines)}

표본부족 패턴은 수익률이 높아 보여도 공식 종합순위에서 제외한다.

### 52주 후 +20% 단일목표 순위

{chr(10).join(f'- {line}' for line in goal_lines)}

종합 1위와 단일목표 1위가 다르면 오류가 아니다. 종합순위는 즉각성과 하락방어까지 포함한다.

### 150주선과 200주선 정배열

워크포워드 평가구간의 단일조건 기술통계에서 `150>200`은 {int(ma_bull['event_n'])}건, 8주 상승 {pct(ma_bull['rise_8w_probability'])}, 52주 후 +20% {pct(ma_bull['close_20_at_52w_probability'])}, 52주 MAE 중앙값 {num(ma_bull['mae_52w_median_pct'])}였다. `150<=200`은 {int(ma_other['event_n'])}건, 8주 상승 {pct(ma_other['rise_8w_probability'])}, 52주 후 +20% {pct(ma_other['close_20_at_52w_probability'])}, 52주 MAE 중앙값 {num(ma_other['mae_52w_median_pct'])}였다. 이 값은 다른 조건을 통제한 인과효과가 아니므로 `150>200` 하나만으로 매수하지 않고 패턴·MACD·ATH·지지구조와 결합한다. 전체 단일조건 표는 `data/condition_metrics.csv`에 있다.

## 7. 신규 기준봉 분석 절차

1. **계산 엔진 확인**: Investing 근사 또는 KIS 호환 중 어느 수치인지 먼저 적는다.
2. **MMRM 충족 확인**: 하락영역 MACD 상승흐름에서 Momentum>0, RSI>50, MFI>50을 처음 동시에 충족한 주봉인지 확인한다.
3. **기준봉 이전 52주만 보고 패턴 분류**: 이후 수익이나 결과 흐름으로 패턴을 고치지 않는다.
4. **장기선 구조 기록**: 150>200 정배열 여부, 간격, 가격과의 이격, 지지 후보를 적는다.
5. **MACD 위치 기록**: 0선 바로 아래인지, 깊은 하락영역인지, 상승영역 진입 여지가 있는지 적는다.
6. **저점 형태 기록**: V형, U형, 횡보지지, 첫 반등, 재시험 중 어느 쪽인지 적는다.
7. **ATH 기록**: 기준봉의 ATH 안전마진, ATH 회복 목표의 현실성, 신고가 확장 가능성을 적는다.
8. **워크포워드 확률 제시**: 패턴명, 표본 수, 8주 상승, 52주 후 +20%, 52주 내 +20%, 중앙 MAE와 신뢰구간을 함께 말한다.
9. **유사사례 검색**: `events_investing_proxy.csv`에서 같은 패턴을 찾고 성공사례와 실패사례를 모두 비교한다.
10. **무효화 조건 작성**: 사용자가 지지선으로 본 가격대를 주봉 음봉 종가로 이탈하면 지지가 깨진 것으로 기록한다.

## 8. 개별 사건 답변 형식

```text
종목 / 기준봉 / 계산 엔진
MMRM 충족 여부와 최초 충족 주
패턴명과 그렇게 분류한 근거
공식 표본 수와 확률(8주, 52주 +20%, 52주 내 +20%)
중앙 수익률·중앙 MAE·+20% 도달기간
ATH 안전마진과 회복/돌파 해석
150·200주선 및 지지선 해석
비슷한 성공사례 2~3건 / 실패사례 2~3건
상승 가설 / 지연 가설 / 실패 가설
관찰할 무효화 조건
```

## 9. GOOGL에서 얻은 교훈

- KIS 호환 계산에서 2022년 흐름의 최초 사건은 2022-08-01이다. 52주 수익률은 약 +9.1%, 52주 MAE는 약 -29.1%, +20% 도달에는 62주가 걸려 사용자의 1년 +20% 목표를 충족하지 못했다.
- 2023-01-30 사건은 26주 약 +22.3%, 52주 약 +35.9%, +20% 도달 15주로 훨씬 나았다.
- 2022년 사례는 첫 반등만으로 저점 확정을 가정하면 안 된다는 교훈이다. 사용자가 그은 지지선을 주봉 음봉 종가로 이탈하면 가설을 낮추고 다음 MMRM 충족이나 바닥 재확인을 기다린다.
- Investing 근사 계산에서는 2022-08 사건이 애초에 MFI 임계값을 넘지 않아 거래대상이 아닐 수 있었다. 이는 해당 한 사례에서 유리했지만 계산엔진 전체의 우월성을 증명하지는 않는다.

## 10. TFMR 리버스와 보조 판단

TFMR 리버스 규칙은 아직 수치화된 공식 모델 입력이 아니다. MMRM 매수와 리버스 TFMR 매도가 겹치면 바로 같은 확률을 적용해 공격적으로 매수하지 않는다. 별도 경고로 표시하고 5·20주선 하락돌파 음봉, 재지지, 다음 회복 신호 등을 확인하는 보수적 가설로 기록한다. TFMR 공식 규칙이 코드화되기 전에는 패턴 성공확률과 섞어 새로운 확률을 만들지 않는다.

## 11. 프로그램 확장 규격

MMRM 결과표 우측에 다음 컬럼을 추가한다.

- `calculation_engine`
- `pattern_id`, `pattern_name`, `pattern_distance`
- `pattern_official_n`, `pattern_ticker_n`
- `prob_rise_8w`, `prob_close_20_at_52w`, `prob_touch_20_within_52w`
- 각 확률의 `ci_low`, `ci_high`
- `median_return_8w`, `median_return_52w`, `median_mae_52w`
- `median_weeks_to_20`
- `goal_52w_rank`, `official_overall_rank`, `sample_status`
- `ath_safety_margin_pct`, `ma_150_200_state`, `support_break_warning`
- `tfmr_reverse_warning`

실시간 확률은 새 기준봉 날짜에서 결과가 이미 성숙한 과거 사건만 조회해 계산한다. 표본 30건 미만이면 숫자 대신 `표본부족`을 보여준다. 공식 프로그램에는 `investing_proxy` 결과를 기본 표시하고 KIS 결과는 별도 토글 또는 민감도 컬럼으로 둔다.

## 12. 해석 한계

- 현재 시가총액 상위 100개 종목을 과거로 거슬러 분석해 생존편향이 있다.
- 같은 종목의 반복 사건은 완전히 독립적이지 않다.
- Investing 근사판과 KIS 호환판 모두 공식 공급자의 전체 원천 데이터를 완전 복제한 것은 아니다.
- 수수료, 세금, 환율, 슬리피지, 다음 주 실제 체결가는 반영하지 않았다.
- 워크포워드는 미래정보 누수를 줄이지만 미래 시장환경 변화까지 제거하지는 못한다.
- 확률은 집단의 과거 빈도이지 한 종목의 확정 예언이 아니다.
"""


def outcome_flow_metrics(events_by_engine: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for engine, frame in events_by_engine.items():
        evaluation = frame[
            (frame["buy_point_date"] >= EVALUATION_START)
            & frame["outcome_flow_name"].fillna("").ne("")
        ].copy()
        for (name, family), group in evaluation.groupby(["outcome_flow_name", "flow_family"], sort=True):
            row: dict[str, object] = {
                "engine": engine,
                "engine_label": ENGINES[engine]["label"],
                "outcome_flow_name": name,
                "flow_family": family,
                "event_n": int(len(group)),
                "ticker_n": int(group["ticker"].nunique()),
            }
            for horizon in (4, 8, 13, 26, 52):
                values = pd.to_numeric(group[f"return_{horizon}w_pct"], errors="coerce").dropna()
                row[f"return_{horizon}w_median_pct"] = values.median()
                row[f"rise_{horizon}w_probability"] = (values > 0).mean() if len(values) else np.nan
            values52 = pd.to_numeric(group["return_52w_pct"], errors="coerce").dropna()
            row["close_20_at_52w_valid_n"] = int(len(values52))
            row["close_20_at_52w_probability"] = (values52 >= 20).mean() if len(values52) else np.nan
            row["mae_8w_median_pct"] = pd.to_numeric(group["mae_8w_pct"], errors="coerce").median()
            row["mae_52w_median_pct"] = pd.to_numeric(group["mae_52w_pct"], errors="coerce").median()
            rows.append(row)
    return pd.DataFrame(rows)


def pattern_to_flow_metrics(events_by_engine: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for engine, frame in events_by_engine.items():
        evaluation = frame[
            (frame["buy_point_date"] >= EVALUATION_START)
            & frame["outcome_flow_name"].fillna("").ne("")
            & frame["pattern_id"].ne("unclassified")
        ].copy()
        totals = evaluation.groupby("pattern_id").size().to_dict()
        for keys, group in evaluation.groupby(
            ["pattern_id", "pattern_name", "flow_family", "outcome_flow_name"], sort=True
        ):
            pattern_id, pattern_name, family, flow_name = keys
            total = int(totals[pattern_id])
            rows.append(
                {
                    "engine": engine,
                    "engine_label": ENGINES[engine]["label"],
                    "pattern_id": pattern_id,
                    "pattern_name": pattern_name,
                    "flow_family": family,
                    "outcome_flow_name": flow_name,
                    "event_n": int(len(group)),
                    "pattern_flow_valid_n": total,
                    "probability_within_pattern": len(group) / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def event_table(group: pd.DataFrame) -> str:
    work = group.copy()
    work["_hit"] = bool_series(work["hit_20_within_52w"]).fillna(-1)
    work["_weeks"] = pd.to_numeric(work["weeks_to_gain_20"], errors="coerce").fillna(9999)
    work["_ret52"] = pd.to_numeric(work["return_52w_pct"], errors="coerce").fillna(-9999)
    work = work.sort_values(["_hit", "_weeks", "_ret52"], ascending=[False, True, False])
    lines = [
        "<div class='table-wrap'><table><thead><tr><th>#</th><th>종목</th><th>기준봉</th><th>구간</th><th>8주</th><th>52주</th><th>52주 내 +20</th><th>+20 소요</th><th>52주 MAE</th><th>ATH 안전마진</th></tr></thead><tbody>"
    ]
    for rank, (_, row) in enumerate(work.iterrows(), 1):
        hit = bool_series(pd.Series([row["hit_20_within_52w"]])).iloc[0]
        hit_text = "성공" if hit == 1 else ("실패" if hit == 0 else "미성숙")
        split = "워크포워드" if pd.Timestamp(row["buy_point_date"]) >= EVALUATION_START else "분류학습"
        lines.append(
            "<tr>"
            f"<td>{rank}</td><td>{html.escape(str(row['ticker']))}</td>"
            f"<td>{pd.Timestamp(row['buy_point_date']).strftime('%Y-%m-%d')}</td><td>{split}</td>"
            f"<td>{num(row['return_8w_pct'])}</td><td>{num(row['return_52w_pct'])}</td>"
            f"<td>{hit_text}</td><td>{num(row['weeks_to_gain_20'], '주', 0)}</td>"
            f"<td>{num(row['mae_52w_pct'])}</td><td>{num(row['ath_safety_margin_pct'])}</td>"
            "</tr>"
        )
    lines.append("</tbody></table></div>")
    return "".join(lines)


def build_html(
    metrics: pd.DataFrame,
    periods: pd.DataFrame,
    conditions: pd.DataFrame,
    flow_metrics: pd.DataFrame,
    pattern_flows: pd.DataFrame,
    events: pd.DataFrame,
) -> str:
    defs = definitions().sort_values("display_order")
    inv = metrics[metrics["engine"] == "investing_proxy"].set_index("pattern_id")
    kis = metrics[metrics["engine"] == "kis_compatible"].set_index("pattern_id")
    ranked = metrics[(metrics["engine"] == "investing_proxy") & metrics["official_overall_rank"].notna()].sort_values("official_overall_rank")
    cards = []
    catalogs = []
    for _, definition in defs.iterrows():
        pid = definition["pattern_id"]
        row = inv.loc[pid]
        krow = kis.loc[pid]
        advice = PATTERN_ADVICE[pid]
        badge = rank_text(row["official_overall_rank"])
        cards.append(f"""
        <article class="pattern-card" id="pattern-{pid}">
          <div class="pattern-copy">
            <span class="badge">{badge}</span><h3>{html.escape(definition['pattern_name'])}</h3>
            <p class="lead">{html.escape(definition['short_description'])}</p>
            <p>{html.escape(definition['meaning'])}</p>
            <ul>{''.join(f'<li>{html.escape(x.strip())}</li>' for x in str(definition['classification_checklist']).split('|'))}</ul>
            <p><strong>판정:</strong> {html.escape(advice['action'])}</p>
            <p class="risk"><strong>위험:</strong> {html.escape(advice['risk'])}</p>
          </div>
          <img src="{definition['representative_chart']}" alt="{html.escape(definition['pattern_name'])} 대표 차트">
          <div class="metric-grid">
            <div><b>{int(row['evaluation_event_n'])}</b><span>워크포워드 사건</span></div>
            <div><b>{pct(row['rise_8w_probability'])}</b><span>8주 상승</span></div>
            <div><b>{pct(row['close_20_at_52w_probability'])}</b><span>52주 후 +20%</span></div>
            <div><b>{pct(row['touch_20_within_52w_probability'])}</b><span>52주 내 +20%</span></div>
            <div><b>{num(row['weeks_to_gain_20_median_among_hits'], '주', 0)}</b><span>+20% 중앙기간</span></div>
            <div><b>{num(row['mae_52w_median_pct'])}</b><span>52주 MAE</span></div>
          </div>
          <p class="sensitivity">95% 신뢰구간: 8주 상승 {pct(row['rise_8w_ci_low'])}~{pct(row['rise_8w_ci_high'])}, 52주 후 +20% {pct(row['close_20_at_52w_ci_low'])}~{pct(row['close_20_at_52w_ci_high'])}, 52주 내 +20% {pct(row['touch_20_within_52w_ci_low'])}~{pct(row['touch_20_within_52w_ci_high'])}.</p>
          <p class="sensitivity">KIS 호환 민감도: 8주 상승 {pct(krow['rise_8w_probability'])}, 52주 후 +20% {pct(krow['close_20_at_52w_probability'])}, 52주 내 +20% {pct(krow['touch_20_within_52w_probability'])}.</p>
        </article>""")
        group = events[events["pattern_id"] == pid]
        catalogs.append(f"<section id='events-{pid}'><h3>{html.escape(definition['pattern_name'])} 전체 사건 {len(group):,}건</h3><p>성공사례부터 정렬했다. 분류학습 구간과 워크포워드 평가구간을 구분해 표시한다.</p>{event_table(group)}</section>")

    unclassified = events[events["pattern_id"] == "unclassified"].copy()
    if not unclassified.empty:
        catalogs.append(
            f"<section id='events-unclassified'><h3>이력 부족·분류불가 {len(unclassified):,}건</h3>"
            "<p>상장 초기 등으로 기준봉 이전 장기 이력이 부족한 사건이다. 다섯 패턴과 공식 확률에서 제외한다.</p>"
            f"{event_table(unclassified)}</section>"
        )

    rank_rows = []
    for _, row in ranked.iterrows():
        rank_rows.append(
            f"<tr><td>{int(row['official_overall_rank'])}</td><td><a href='#pattern-{row['pattern_id']}'>{html.escape(row['pattern_name'])}</a></td>"
            f"<td>{int(row['evaluation_event_n'])}</td><td>{pct(row['rise_8w_probability'])}</td>"
            f"<td>{pct(row['close_20_at_52w_probability'])}</td><td>{pct(row['touch_20_within_52w_probability'])}</td>"
            f"<td>{num(row['mae_52w_median_pct'])}</td><td>{rank_text(row['goal_52w_rank'])}</td></tr>"
        )
    best = ranked.iloc[0]
    period_rows = []
    period_source = periods[periods["engine"] == "investing_proxy"].copy()
    order_map = dict(zip(defs["pattern_id"], defs["display_order"]))
    period_source["_order"] = period_source["pattern_id"].map(order_map)
    period_source["_period_order"] = period_source["period"].map(
        {"2015-2019": 1, "2020-2022": 2, "2023+": 3}
    )
    period_source = period_source.sort_values(["_period_order", "_order"])
    for _, row in period_source.iterrows():
        period_rows.append(
            f"<tr><td>{row['period']}</td><td>{html.escape(row['pattern_name'])}</td>"
            f"<td>{int(row['evaluation_event_n'])}</td><td>{pct(row['rise_8w_probability'])}</td>"
            f"<td>{pct(row['close_20_at_52w_probability'])}</td><td>{pct(row['touch_20_within_52w_probability'])}</td>"
            f"<td>{num(row['mae_52w_median_pct'])}</td></tr>"
        )
    condition_rows = []
    condition_source = conditions[conditions["engine"] == "investing_proxy"]
    condition_names = {
        "ma_order": "150·200주선 배열",
        "ma_tightness": "150·200주선 간격",
        "macd_zero": "가격 대비 MACD",
        "ath_margin": "ATH 안전마진",
    }
    for _, row in condition_source.iterrows():
        condition_rows.append(
            f"<tr><td>{condition_names[row['condition']]}</td><td>{html.escape(row['group'])}</td>"
            f"<td>{int(row['event_n'])}</td><td>{pct(row['rise_8w_probability'])}</td>"
            f"<td>{pct(row['close_20_at_52w_probability'])}</td><td>{pct(row['touch_20_within_52w_probability'])}</td>"
            f"<td>{num(row['mae_52w_median_pct'])}</td></tr>"
        )
    flow_rows = []
    for _, row in flow_metrics[flow_metrics["engine"] == "investing_proxy"].sort_values(
        ["flow_family", "return_52w_median_pct"], ascending=[True, False]
    ).iterrows():
        flow_rows.append(
            f"<tr><td>{html.escape(row['flow_family'])}</td><td>{html.escape(row['outcome_flow_name'])}</td>"
            f"<td>{int(row['event_n'])}</td><td>{num(row['return_8w_median_pct'])}</td>"
            f"<td>{num(row['return_26w_median_pct'])}</td><td>{num(row['return_52w_median_pct'])}</td>"
            f"<td>{pct(row['close_20_at_52w_probability'])}</td><td>{num(row['mae_52w_median_pct'])}</td></tr>"
        )
    family = (
        pattern_flows[pattern_flows["engine"] == "investing_proxy"]
        .groupby(["pattern_name", "flow_family"], as_index=False)
        .agg(event_n=("event_n", "sum"), pattern_flow_valid_n=("pattern_flow_valid_n", "max"))
    )
    family["probability"] = family["event_n"] / family["pattern_flow_valid_n"]
    family_rows = []
    for _, row in family.sort_values(["pattern_name", "probability"], ascending=[True, False]).iterrows():
        family_rows.append(
            f"<tr><td>{html.escape(row['pattern_name'])}</td><td>{html.escape(row['flow_family'])}</td>"
            f"<td>{int(row['event_n'])}/{int(row['pattern_flow_valid_n'])}</td><td>{pct(row['probability'])}</td></tr>"
        )
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMRM 패턴 연구 최종 보고서</title>
<style>
:root{{--bg:#f4f7fb;--paper:#fff;--ink:#142033;--muted:#637083;--line:#d9e1ec;--brand:#1d5fd1;--good:#087a55;--warn:#a55300}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;line-height:1.65}}
header{{background:linear-gradient(135deg,#10294d,#1d5fd1);color:white;padding:54px 24px}} header>div,main{{max-width:1180px;margin:auto}} h1{{font-size:clamp(2rem,5vw,3.7rem);line-height:1.1;margin:0 0 16px}} h2{{margin-top:0;font-size:1.8rem}} h3{{font-size:1.3rem}} .subtitle{{max-width:820px;font-size:1.1rem;opacity:.9}}
nav{{position:sticky;top:0;background:#ffffffee;backdrop-filter:blur(10px);border-bottom:1px solid var(--line);z-index:5;padding:10px 16px;overflow:auto;white-space:nowrap}} nav a{{color:var(--brand);text-decoration:none;margin-right:18px;font-weight:650}}
main{{padding:28px 18px 80px}} section,.pattern-card{{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:26px;margin:22px 0;box-shadow:0 8px 28px #2338580b}}
.hero-result{{border-left:7px solid var(--good);font-size:1.08rem}} .hero-result strong{{color:var(--good)}}
.pattern-card{{display:grid;grid-template-columns:1.25fr .9fr;gap:22px}} .pattern-card img{{width:100%;border:1px solid var(--line);border-radius:12px;background:#fff}} .pattern-copy ul{{padding-left:20px}} .badge{{float:right;background:#e9f1ff;color:var(--brand);border-radius:999px;padding:5px 11px;font-weight:800}} .lead{{font-size:1.1rem;color:var(--brand);font-weight:700}} .risk{{color:#7d3e00}}
.metric-grid{{grid-column:1/-1;display:grid;grid-template-columns:repeat(6,1fr);gap:10px}} .metric-grid div{{background:#f5f8fd;border-radius:12px;padding:12px;text-align:center}} .metric-grid b{{display:block;font-size:1.25rem;color:var(--brand)}} .metric-grid span{{font-size:.82rem;color:var(--muted)}} .sensitivity{{grid-column:1/-1;color:var(--muted);margin:0}}
.table-wrap{{overflow:auto;max-height:680px;border:1px solid var(--line);border-radius:12px}} table{{border-collapse:collapse;width:100%;font-size:.9rem;background:#fff}} th,td{{padding:9px 11px;border-bottom:1px solid #e8edf4;text-align:right;white-space:nowrap}} th{{position:sticky;top:0;background:#edf3fb;color:#233b60;z-index:1}} th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}} a{{color:var(--brand)}} .method li{{margin:7px 0}} code{{background:#edf2f8;padding:2px 5px;border-radius:5px}} .small{{color:var(--muted);font-size:.92rem}}
@media(max-width:850px){{.pattern-card{{grid-template-columns:1fr}}.metric-grid{{grid-template-columns:repeat(2,1fr)}}section,.pattern-card{{padding:18px}}}}
</style></head><body>
<header><div><p>MMRM WEEKLY BUY-POINT RESEARCH</p><h1>패턴 연구 최종 보고서</h1><p class="subtitle">고정된 다섯 가지 차트 패턴과 미래정보 없는 확장형 워크포워드 확률을 사용해 신규 기준봉의 즉각성, 1년 +20% 목표, 하락위험을 함께 판단한다.</p><p class="small">생성 {generated} · 공식 엔진 investing_proxy · 평가 시작 2015-10-12</p></div></header>
<nav><a href="#conclusion">결론</a><a href="#decision">판정</a><a href="#ranking">순위</a><a href="#method">검증법</a><a href="#stability">시기별</a><a href="#conditions">단일조건</a><a href="#flows">이후 흐름</a><a href="#patterns">패턴</a><a href="#catalogs">전체 사건</a><a href="#limits">한계</a></nav>
<main>
<section id="conclusion" class="hero-result"><h2>지금 무엇을 우선 볼 것인가</h2><p>공식 종합 1위는 <strong>{html.escape(best['pattern_name'])}</strong>이다. 이 순위는 52주 후 +20%만 보지 않고 즉각성 및 최대하락폭까지 반영한다. 52주 +20% 단일목표 순위는 별도 열로 확인해야 한다.</p><p>실전에서는 패턴명만으로 매수하지 않는다. 장기선 지지 유지, MACD 위치, V/U 저점 재시험, ATH 이격, TFMR 리버스 경고를 함께 기록한다.</p></section>
<section id="decision"><h2>신규 기준봉 판정 원칙</h2><p>사용자 기본 목표는 기준봉 종가 매수 후 52주 뒤 +20%다. 세션마다 결론이 달라지지 않도록 공식 MMRM 신호, 표본 30건 이상, 52주 +20% 역사확률 40% 이상, 패턴 거리 95백분위 이내, 심각한 데이터 품질 경고 없음의 다섯 조건을 기본 운영선으로 사용한다.</p><p>이 40%는 연구가 증명한 자연법칙이 아니라 사용자용 운영정책이다. 답변은 기준봉 당시 정보만으로 <strong>매수</strong> 또는 <strong>매수하지 않음</strong>을 먼저 말하고, 투자 논리가 성립하는 기대와 조심할 점을 함께 쓴다. 상세 규칙은 <code>MMRM_KNOWLEDGE_BASE.md</code>를 따른다.</p></section>
<section id="ranking"><h2>공식 우선순위</h2><p>가중치: 52주 후 +20% 40% · 52주 내 +20% 접촉 10% · 8주 상승 20% · 52주 MAE 방어 30%.</p><div class="table-wrap"><table><thead><tr><th>종합</th><th>패턴</th><th>n</th><th>8주 상승</th><th>52주 후 +20%</th><th>52주 내 +20%</th><th>52주 MAE</th><th>1년 목표순위</th></tr></thead><tbody>{''.join(rank_rows)}</tbody></table></div><p class="small">표본 30건 미만인 대폭락형은 결과가 좋아 보여도 공식순위에서 제외한다.</p></section>
<section id="method" class="method"><h2>검증법</h2><ol><li>2015-10-05까지 기준봉 이전 차트만으로 다섯 패턴을 정했다.</li><li>2015-10-12 이후에는 결과를 보지 않고 고정 패턴을 적용했다.</li><li>각 기준봉 시점의 확률은 그때 이미 성숙한 과거 결과만 사용한다. 52주 지표는 52주가 지나야 편입된다.</li><li>프로그램과 맞춘 Investing 근사판을 공식 기준으로, KIS 호환판을 민감도 비교로 사용한다.</li></ol><p>이는 같은 전체 자료를 학습과 검증에 중복 사용하는 방식이 아니다. 초기 구간은 분류체계 학습, 이후 구간은 시간순 평가로 역할이 분리된다.</p></section>
<section id="stability"><h2>시장 시기별 안정성</h2><p>한 번의 평균만으로 판단하지 않기 위해 워크포워드 평가구간을 세 시기로 나눴다. 최근 구간의 52주 표본은 아직 성숙 중이므로 사건 수와 함께 해석한다.</p><div class="table-wrap"><table><thead><tr><th>시기</th><th>패턴</th><th>사건</th><th>8주 상승</th><th>52주 후 +20%</th><th>52주 내 +20%</th><th>52주 MAE</th></tr></thead><tbody>{''.join(period_rows)}</tbody></table></div></section>
<section id="conditions"><h2>장기선·MACD·ATH 단일조건</h2><p>한 조건만 떼어 본 기술통계다. 다른 조건을 통제한 인과효과가 아니므로 패턴 판단을 대체하지 않는다.</p><div class="table-wrap"><table><thead><tr><th>조건</th><th>구간</th><th>사건</th><th>8주 상승</th><th>52주 후 +20%</th><th>52주 내 +20%</th><th>52주 MAE</th></tr></thead><tbody>{''.join(condition_rows)}</tbody></table></div></section>
<section id="flows"><h2>기준봉 이후 실제 흐름 연구</h2><p>이 표는 기준봉 뒤 결과를 설명하는 사후 분류다. 신규 기준봉에 이미 발생한 사실처럼 붙이거나 매수판정에 사용하지 않는다.</p><h3>이후 흐름 유형</h3><div class="table-wrap"><table><thead><tr><th>큰 흐름</th><th>세부 흐름</th><th>사건</th><th>8주 중앙</th><th>26주 중앙</th><th>52주 중앙</th><th>52주 +20%</th><th>52주 MAE</th></tr></thead><tbody>{''.join(flow_rows)}</tbody></table></div><h3>사전 패턴별 이후 흐름 빈도</h3><div class="table-wrap"><table><thead><tr><th>사전 패턴</th><th>이후 큰 흐름</th><th>사건</th><th>패턴 내 빈도</th></tr></thead><tbody>{''.join(family_rows)}</tbody></table></div></section>
<div id="patterns"><h2>다섯 패턴 상세</h2>{''.join(cards)}</div>
<div id="catalogs"><h2>패턴별 전체 사건</h2><p>아래 목록은 원형 MMRM 근사판 2,813건 전체다. 성공사례만 보지 말고 같은 패턴의 실패·지연 사례도 함께 비교해야 한다.</p>{''.join(catalogs)}</div>
<section id="limits"><h2>한계와 사용 원칙</h2><ul><li>현재 Top100 생존 종목을 과거로 거슬러 분석한 생존편향이 있다.</li><li>같은 종목의 반복 사건은 서로 독립적이지 않다.</li><li>두 계산 엔진 모두 공급자 공식 전체 원천의 완전 복제본은 아니다.</li><li>확률은 과거 집단빈도이며 한 사건의 수익을 보장하지 않는다.</li><li>TFMR 리버스와 사용자가 그은 지지선 이탈은 아직 공식 확률 입력이 아니라 별도 경고다.</li></ul></section>
</main></body></html>"""


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    events_by_engine = {engine: load_events(engine) for engine in ENGINES}
    predictions = pd.concat(
        [walk_forward_predictions(frame) for frame in events_by_engine.values()], ignore_index=True
    )
    metrics = attach_calibration(official_metrics(events_by_engine), predictions)
    periods = period_metrics(events_by_engine)
    conditions = condition_metrics(events_by_engine)
    flows = outcome_flow_metrics(events_by_engine)
    pattern_flows = pattern_to_flow_metrics(events_by_engine)

    predictions.to_csv(DATA / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(DATA / "pattern_metrics.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(DATA / "walk_forward_period_metrics.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(DATA / "condition_metrics.csv", index=False, encoding="utf-8-sig")
    flows.to_csv(DATA / "outcome_flow_metrics.csv", index=False, encoding="utf-8-sig")
    pattern_flows.to_csv(DATA / "pattern_to_flow_metrics.csv", index=False, encoding="utf-8-sig")

    # README, the knowledge base, the session prompt, and the manifest are curated
    # authority files. Rebuilding derived statistics must never overwrite them.
    (ROOT / "MMRM_PATTERN_REPORT.html").write_text(
        build_html(
            metrics,
            periods,
            conditions,
            flows,
            pattern_flows,
            events_by_engine["investing_proxy"],
        ),
        encoding="utf-8",
    )
    print(f"rebuilt {ROOT}")
    print(metrics[["engine", "pattern_name", "evaluation_event_n", "official_overall_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
