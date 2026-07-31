from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.odds.expected_value import attach_buy_score_columns


def compute_trifecta_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    odds_col: str = "odds",
    expected_value_thresholds: Iterable[float] = (1.0, 1.05, 1.1),
) -> dict[str, float | dict[str, float]]:
    if trifecta_df.empty:
        return {}

    required = {"race_id", "trifecta", probability_col, "is_actual"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing trifecta metric columns: {sorted(missing)}")

    race_count = int(trifecta_df["race_id"].nunique())
    top_hits = {1: 0, 3: 0, 5: 0, 10: 0, 12: 0}
    covered_races = 0
    log_losses: list[float] = []
    brier_scores: list[float] = []
    actual_probabilities: list[float] = []
    top_probabilities: list[float] = []

    for _, race_df in trifecta_df.groupby("race_id", sort=False):
        ordered = race_df.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue

        covered_races += 1
        actual_idx = int(actual_positions[0])
        actual_probability = max(float(ordered.loc[actual_idx, probability_col]), 1e-15)
        probs = ordered[probability_col].to_numpy(dtype=float)
        labels = ordered["is_actual"].to_numpy(dtype=float)

        for top_k in top_hits:
            top_hits[top_k] += int(actual_idx < top_k)

        log_losses.append(-np.log(actual_probability))
        brier_scores.append(float(np.mean((probs - labels) ** 2)))
        actual_probabilities.append(actual_probability)
        top_probabilities.append(float(ordered.iloc[0][probability_col]))

    metrics: dict[str, float | dict[str, float]] = {
        "race_count": float(race_count),
        "covered_races": float(covered_races),
        "candidate_coverage_rate": covered_races / race_count if race_count else 0.0,
        "top1_hit_rate": top_hits[1] / race_count if race_count else 0.0,
        "top3_hit_rate": top_hits[3] / race_count if race_count else 0.0,
        "top5_hit_rate": top_hits[5] / race_count if race_count else 0.0,
        "top10_hit_rate": top_hits[10] / race_count if race_count else 0.0,
        "top12_hit_rate": top_hits[12] / race_count if race_count else 0.0,
        "log_loss": float(np.mean(log_losses)) if log_losses else 0.0,
        "brier_score": float(np.mean(brier_scores)) if brier_scores else 0.0,
        "mean_actual_probability": float(np.mean(actual_probabilities)) if actual_probabilities else 0.0,
        "mean_top_probability": float(np.mean(top_probabilities)) if top_probabilities else 0.0,
    }

    if odds_col in trifecta_df.columns and trifecta_df[odds_col].notna().any():
        metrics["expected_value_backtest"] = compute_expected_value_backtest_metrics(
            trifecta_df,
            probability_col=probability_col,
            odds_col=odds_col,
            thresholds=expected_value_thresholds,
        )

    buy_signal_metrics = compute_buy_signal_top12_metrics(trifecta_df, probability_col=probability_col)
    if buy_signal_metrics:
        metrics["buy_signal_top12_metrics"] = buy_signal_metrics

    payout_proxy_metrics = compute_payout_proxy_buy_score_top12_metrics(
        trifecta_df,
        probability_col=probability_col,
    )
    if payout_proxy_metrics:
        metrics["payout_proxy_buy_score_top12_metrics"] = payout_proxy_metrics

    return metrics


def compute_payout_proxy_buy_score_top12_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
) -> dict[str, dict[str, float]]:
    if trifecta_df.empty or "trifecta_payout" not in trifecta_df.columns:
        return {}

    required = {"race_id", probability_col, "is_actual", "trifecta_payout"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing payout proxy metric columns: {sorted(missing)}")

    records: list[dict[str, float | str]] = []
    for race_id, race_df in trifecta_df.groupby("race_id", sort=False):
        ordered = race_df.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue
        actual_idx = int(actual_positions[0])
        actual_row = ordered.iloc[actual_idx]
        payout = pd.to_numeric(pd.Series([actual_row.get("trifecta_payout")]), errors="coerce").dropna()
        if payout.empty or float(payout.iloc[0]) <= 0.0:
            continue

        proxy_odds = float(payout.iloc[0]) / 100.0
        probability = float(actual_row[probability_col])
        proxy_frame = pd.DataFrame(
            {
                "race_id": [str(race_id)],
                "probability": [probability],
                "odds": [proxy_odds],
                "expected_value": [probability * proxy_odds],
                "ticket_priority_score": [_safe_float(actual_row.get("ticket_priority_score"), 0.0)],
                "race_upset_score": [_safe_float(actual_row.get("race_upset_score"), 0.0)],
            }
        )
        scored = attach_buy_score_columns(proxy_frame)
        records.append(
            {
                "score_label": _buy_score_label_key(scored.loc[0, "buy_score_label"]),
                "top12_hit": float(int(actual_idx < 12)),
                "buy_score": float(scored.loc[0, "buy_score"]),
                "proxy_expected_value": float(scored.loc[0, "expected_value"]),
                "proxy_odds": proxy_odds,
                "payout": float(payout.iloc[0]),
            }
        )

    grouped: dict[str, list[dict[str, float | str]]] = {}
    for record in records:
        grouped.setdefault(str(record["score_label"]), []).append(record)

    return {
        key: _summarize_payout_proxy_records(value)
        for key, value in sorted(grouped.items())
    }


def _safe_float(value: object, default: float = 0.0) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default)
    return float(converted.iloc[0])


def _summarize_payout_proxy_records(records: list[dict[str, float | str]]) -> dict[str, float]:
    if not records:
        return {
            "race_count": 0.0,
            "top12_hit_rate": 0.0,
            "mean_buy_score": 0.0,
            "mean_proxy_expected_value": 0.0,
            "mean_proxy_odds": 0.0,
            "mean_payout": 0.0,
        }
    return {
        "race_count": float(len(records)),
        "top12_hit_rate": float(np.mean([float(record["top12_hit"]) for record in records])),
        "mean_buy_score": float(np.mean([float(record["buy_score"]) for record in records])),
        "mean_proxy_expected_value": float(np.mean([float(record["proxy_expected_value"]) for record in records])),
        "mean_proxy_odds": float(np.mean([float(record["proxy_odds"]) for record in records])),
        "mean_payout": float(np.mean([float(record["payout"]) for record in records])),
    }


def compute_buy_signal_top12_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
) -> dict[str, dict[str, dict[str, float]]]:
    if trifecta_df.empty or "buy_score" not in trifecta_df.columns:
        return {}

    required = {"race_id", probability_col, "is_actual"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing buy signal metric columns: {sorted(missing)}")

    by_decision: dict[str, list[dict[str, float]]] = {}
    by_score_label: dict[str, list[dict[str, float]]] = {}

    for _, race_df in trifecta_df.groupby("race_id", sort=False):
        ordered = race_df.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue

        score_sort_columns = [column for column in ("buy_score", "expected_value", probability_col) if column in race_df.columns]
        score_sorted = race_df.sort_values(
            score_sort_columns,
            ascending=[False] * len(score_sort_columns),
        ).reset_index(drop=True)
        top_signal = score_sorted.iloc[0]
        buy_score = float(pd.to_numeric(pd.Series([top_signal.get("buy_score")]), errors="coerce").fillna(0.0).iloc[0])
        buy_candidate_count = _buy_candidate_count(race_df)
        top12_hit = float(int(int(actual_positions[0]) < 12))
        record = {
            "top12_hit": top12_hit,
            "buy_score": buy_score,
            "buy_candidate_count": float(buy_candidate_count),
        }

        by_decision.setdefault(_race_buy_decision_label(race_df), []).append(record)
        by_score_label.setdefault(_buy_score_label_key(top_signal.get("buy_score_label")), []).append(record)

    return {
        "by_buy_decision": {key: _summarize_buy_signal_records(records) for key, records in sorted(by_decision.items())},
        "by_score_label": {key: _summarize_buy_signal_records(records) for key, records in sorted(by_score_label.items())},
    }


def _buy_candidate_count(race_df: pd.DataFrame) -> int:
    if "buy_decision" not in race_df.columns:
        return int((pd.to_numeric(race_df["buy_score"], errors="coerce").fillna(0.0) >= 50.0).sum())
    decisions = race_df["buy_decision"].astype(str)
    return int((decisions != "見送り").sum())


def _race_buy_decision_label(race_df: pd.DataFrame) -> str:
    if "buy_decision" not in race_df.columns:
        score = pd.to_numeric(race_df["buy_score"], errors="coerce").fillna(0.0)
        return "buy" if bool((score >= 50.0).any()) else "skip"
    decisions = race_df["buy_decision"].astype(str)
    return "buy" if bool((decisions != "見送り").any()) else "skip"


def _buy_score_label_key(value: object) -> str:
    text = str(value)
    if text == "強く買い候補":
        return "strong_buy"
    if text == "買い候補":
        return "buy"
    if text == "抑え候補":
        return "keep"
    return "skip"


def _summarize_buy_signal_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {
            "race_count": 0.0,
            "top12_hit_rate": 0.0,
            "mean_buy_score": 0.0,
            "average_buy_candidate_count": 0.0,
        }
    top12_hits = [record["top12_hit"] for record in records]
    buy_scores = [record["buy_score"] for record in records]
    buy_candidate_counts = [record["buy_candidate_count"] for record in records]
    return {
        "race_count": float(len(records)),
        "top12_hit_rate": float(np.mean(top12_hits)),
        "mean_buy_score": float(np.mean(buy_scores)),
        "average_buy_candidate_count": float(np.mean(buy_candidate_counts)),
    }


def compute_trifecta_rerank_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    baseline_col: str = "probability_v1",
) -> dict[str, float]:
    if trifecta_df.empty:
        return {}

    required = {"race_id", probability_col, baseline_col, "is_actual"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing trifecta rerank metric columns: {sorted(missing)}")

    covered = 0
    rerank_top1 = 0
    rerank_mrr = 0.0
    baseline_mrr = 0.0
    mean_rank_improvement = 0.0

    for _, race_df in trifecta_df.groupby("race_id", sort=False):
        reranked = race_df.sort_values(probability_col, ascending=False).reset_index(drop=True)
        baseline = race_df.sort_values(baseline_col, ascending=False).reset_index(drop=True)
        rerank_actual = np.flatnonzero(reranked["is_actual"].to_numpy(dtype=bool))
        baseline_actual = np.flatnonzero(baseline["is_actual"].to_numpy(dtype=bool))
        if len(rerank_actual) != 1 or len(baseline_actual) != 1:
            continue

        covered += 1
        rerank_pos = int(rerank_actual[0]) + 1
        baseline_pos = int(baseline_actual[0]) + 1
        rerank_top1 += int(rerank_pos == 1)
        rerank_mrr += 1.0 / rerank_pos
        baseline_mrr += 1.0 / baseline_pos
        mean_rank_improvement += baseline_pos - rerank_pos

    if covered == 0:
        return {
            "coverage_races": 0.0,
            "rerank_top1_hit_rate": 0.0,
            "rerank_mrr": 0.0,
            "baseline_mrr": 0.0,
            "mean_rank_improvement": 0.0,
        }

    return {
        "coverage_races": float(covered),
        "rerank_top1_hit_rate": rerank_top1 / covered,
        "rerank_mrr": rerank_mrr / covered,
        "baseline_mrr": baseline_mrr / covered,
        "mean_rank_improvement": mean_rank_improvement / covered,
    }


def compute_expected_value_backtest_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    odds_col: str = "odds",
    thresholds: Iterable[float] = (1.0, 1.05, 1.1),
) -> dict[str, dict[str, float]]:
    if trifecta_df.empty or odds_col not in trifecta_df.columns:
        return {}

    frame = trifecta_df.dropna(subset=[odds_col]).copy()
    if frame.empty:
        return {}

    if "expected_value" not in frame.columns:
        frame["expected_value"] = frame[probability_col].astype(float) * frame[odds_col].astype(float)

    race_count = max(int(frame["race_id"].nunique()), 1)
    total_candidates = max(len(frame), 1)
    metrics: dict[str, dict[str, float]] = {}

    for threshold in thresholds:
        bought = frame[frame["expected_value"] >= float(threshold)].copy()
        ticket_count = len(bought)
        race_with_purchase = int(bought["race_id"].nunique()) if ticket_count else 0
        hit_count = int(bought["is_actual"].sum()) if ticket_count else 0
        total_return = float(bought.loc[bought["is_actual"].astype(bool), odds_col].sum()) if ticket_count else 0.0

        metrics[f"ev_gte_{threshold:.2f}"] = {
            "purchase_rate": ticket_count / total_candidates,
            "race_purchase_rate": race_with_purchase / race_count,
            "hit_rate": hit_count / ticket_count if ticket_count else 0.0,
            "recovery_rate": total_return / ticket_count if ticket_count else 0.0,
            "average_ticket_count": ticket_count / race_count,
        }

    return metrics


def compute_binary_classification_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    truth = pd.Series(y_true).astype(int)
    prob = np.asarray(y_prob, dtype=float)
    if len(truth) == 0:
        return {}

    prob = np.clip(prob, 1e-15, 1.0 - 1e-15)
    return {
        "sample_count": float(len(truth)),
        "positive_rate": float(truth.mean()),
        "log_loss": float(log_loss(truth, prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(truth, prob)),
    }


def compute_multiclass_classification_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    classes: list[str],
) -> dict[str, float]:
    truth = pd.Series(y_true).astype(str)
    prob = np.asarray(y_prob, dtype=float)
    if len(truth) == 0 or prob.size == 0:
        return {}

    encoded = pd.Categorical(truth, categories=classes)
    valid_mask = encoded.codes >= 0
    if not valid_mask.any():
        return {}

    filtered_truth = encoded.codes[valid_mask]
    filtered_prob = np.clip(prob[valid_mask], 1e-15, 1.0)
    filtered_prob = filtered_prob / filtered_prob.sum(axis=1, keepdims=True)
    one_hot = np.eye(len(classes), dtype=float)[filtered_truth]
    accuracy = float((filtered_prob.argmax(axis=1) == filtered_truth).mean())
    brier = float(np.mean(np.sum((filtered_prob - one_hot) ** 2, axis=1)))

    return {
        "sample_count": float(valid_mask.sum()),
        "accuracy": accuracy,
        "log_loss": float(log_loss(filtered_truth, filtered_prob, labels=list(range(len(classes))))),
        "brier_score": brier,
    }
