from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.top12_confidence import attach_top12_confidence_columns, top12_confidence_label_key


PAYOUT_BANDS: tuple[tuple[str, float | None, float | None, str], ...] = (
    ("lt_10000", None, 10000.0, "under_10000"),
    ("gte_10000_lt_50000", 10000.0, 50000.0, "10000_to_49999"),
    ("gte_50000_lt_100000", 50000.0, 100000.0, "50000_to_99999"),
    ("gte_100000", 100000.0, None, "100000_or_more"),
)


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

    top12_confidence_metrics = compute_top12_confidence_metrics(trifecta_df, probability_col=probability_col)
    if top12_confidence_metrics:
        metrics["top12_confidence_metrics"] = top12_confidence_metrics

    payout_band_metrics = compute_payout_band_metrics(trifecta_df, probability_col=probability_col)
    if payout_band_metrics:
        metrics["payout_band_metrics"] = payout_band_metrics

    return metrics


def compute_payout_band_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    payout_col: str = "trifecta_payout",
) -> dict[str, dict[str, float | str]]:
    if trifecta_df.empty or payout_col not in trifecta_df.columns:
        return {}

    by_band: dict[str, list[dict[str, float]]] = {}
    for _, race_df in trifecta_df.groupby("race_id", sort=False):
        payout_values = pd.to_numeric(race_df[payout_col], errors="coerce").dropna()
        if payout_values.empty:
            continue
        payout = float(payout_values.iloc[0])
        if payout <= 0.0:
            continue
        ordered = race_df.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue
        actual_idx = int(actual_positions[0])
        actual_probability = max(float(ordered.loc[actual_idx, probability_col]), 1e-15)
        record = {
            "payout": payout,
            "top1_hit": float(int(actual_idx < 1)),
            "top3_hit": float(int(actual_idx < 3)),
            "top5_hit": float(int(actual_idx < 5)),
            "top10_hit": float(int(actual_idx < 10)),
            "top12_hit": float(int(actual_idx < 12)),
            "log_loss": float(-np.log(actual_probability)),
        }
        by_band.setdefault(_payout_band_key(payout), []).append(record)

    total_races = sum(len(records) for records in by_band.values())
    return {
        key: _summarize_payout_band_records(key, records, total_races=total_races)
        for key, records in sorted(by_band.items())
    }


def _payout_band_key(payout: float) -> str:
    for key, lower, upper, _ in PAYOUT_BANDS:
        if lower is not None and payout < lower:
            continue
        if upper is not None and payout >= upper:
            continue
        return key
    return "unknown"


def _payout_band_label(key: str) -> str:
    for band_key, _, _, label in PAYOUT_BANDS:
        if band_key == key:
            return label
    return "unknown"


def _summarize_payout_band_records(
    key: str,
    records: list[dict[str, float]],
    total_races: int,
) -> dict[str, float | str]:
    if not records:
        return {
            "label": _payout_band_label(key),
            "race_count": 0.0,
            "race_rate": 0.0,
            "top1_hit_rate": 0.0,
            "top3_hit_rate": 0.0,
            "top5_hit_rate": 0.0,
            "top10_hit_rate": 0.0,
            "top12_hit_rate": 0.0,
            "log_loss": 0.0,
            "mean_payout": 0.0,
        }
    return {
        "label": _payout_band_label(key),
        "race_count": float(len(records)),
        "race_rate": float(len(records) / total_races) if total_races else 0.0,
        "top1_hit_rate": float(np.mean([record["top1_hit"] for record in records])),
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in records])),
        "top5_hit_rate": float(np.mean([record["top5_hit"] for record in records])),
        "top10_hit_rate": float(np.mean([record["top10_hit"] for record in records])),
        "top12_hit_rate": float(np.mean([record["top12_hit"] for record in records])),
        "log_loss": float(np.mean([record["log_loss"] for record in records])),
        "mean_payout": float(np.mean([record["payout"] for record in records])),
    }


def compute_top12_confidence_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
) -> dict[str, dict[str, float]]:
    if trifecta_df.empty:
        return {}

    required = {"race_id", probability_col, "is_actual"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing top12 confidence metric columns: {sorted(missing)}")

    frame = attach_top12_confidence_columns(trifecta_df, probability_col=probability_col)
    by_label: dict[str, list[dict[str, float]]] = {}

    for _, scored_race in frame.groupby("race_id", sort=False):
        ordered = scored_race.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue
        top_row = ordered.iloc[0]
        record = {
            "top12_hit": float(int(int(actual_positions[0]) < 12)),
            "score": float(top_row.get("top12_confidence_score", 0.0) or 0.0),
            "top12_probability_mass": float(top_row.get("top12_probability_mass", 0.0) or 0.0),
            "top5_probability_mass": float(top_row.get("top5_probability_mass", 0.0) or 0.0),
        }
        by_label.setdefault(top12_confidence_label_key(top_row.get("top12_confidence_label")), []).append(record)

    total_races = sum(len(records) for records in by_label.values())
    return {
        key: _summarize_top12_confidence_records(records, total_races=total_races)
        for key, records in sorted(by_label.items())
    }


def _summarize_top12_confidence_records(
    records: list[dict[str, float]],
    total_races: int,
) -> dict[str, float]:
    if not records:
        return {
            "race_count": 0.0,
            "race_rate": 0.0,
            "top12_hit_rate": 0.0,
            "mean_score": 0.0,
            "mean_top12_probability_mass": 0.0,
            "mean_top5_probability_mass": 0.0,
        }
    top12_hits = [record["top12_hit"] for record in records]
    scores = [record["score"] for record in records]
    top12_masses = [record["top12_probability_mass"] for record in records]
    top5_masses = [record["top5_probability_mass"] for record in records]
    return {
        "race_count": float(len(records)),
        "race_rate": float(len(records) / total_races) if total_races else 0.0,
        "top12_hit_rate": float(np.mean(top12_hits)),
        "mean_score": float(np.mean(scores)),
        "mean_top12_probability_mass": float(np.mean(top12_masses)),
        "mean_top5_probability_mass": float(np.mean(top5_masses)),
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
