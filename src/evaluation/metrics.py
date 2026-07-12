from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


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
    top_hits = {1: 0, 3: 0, 5: 0, 10: 0}
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

    return metrics


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
