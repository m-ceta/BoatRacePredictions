from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.top3_hit_probability import summarize_top3_hit_probability_bands
from src.top12_confidence import (
    attach_boat_top1_confidence_columns,
    attach_top3_confidence_columns,
    attach_top12_confidence_columns,
    boat_top1_confidence_label_key,
    recommended_ticket_count_from_top3_confidence,
    top3_confidence_label_key,
    top12_confidence_label_key,
)


PAYOUT_BANDS: tuple[tuple[str, float | None, float | None, str], ...] = (
    ("lt_10000", None, 10000.0, "under_10000"),
    ("gte_10000_lt_50000", 10000.0, 50000.0, "10000_to_49999"),
    ("gte_50000_lt_100000", 50000.0, 100000.0, "50000_to_99999"),
    ("gte_100000", 100000.0, None, "100000_or_more"),
)
TOP3_CONFIDENCE_SCORE_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("score_90_100", 90.0, None),
    ("score_85_90", 85.0, 90.0),
    ("score_80_85", 80.0, 85.0),
    ("score_75_80", 75.0, 80.0),
    ("score_70_75", 70.0, 75.0),
    ("score_65_70", 65.0, 70.0),
    ("score_60_65", 60.0, 65.0),
    ("score_lt_60", float("-inf"), 60.0),
)


def _trifecta_winner(value: object) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    return text.split("-", 1)[0].strip()


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
    boat_top1_hits = 0
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
        if _trifecta_winner(ordered.iloc[0].get("trifecta")) == _trifecta_winner(ordered.loc[actual_idx].get("trifecta")):
            boat_top1_hits += 1

        log_losses.append(-np.log(actual_probability))
        brier_scores.append(float(np.mean((probs - labels) ** 2)))
        actual_probabilities.append(actual_probability)
        top_probabilities.append(float(ordered.iloc[0][probability_col]))

    metrics: dict[str, float | dict[str, float]] = {
        "race_count": float(race_count),
        "covered_races": float(covered_races),
        "candidate_coverage_rate": covered_races / race_count if race_count else 0.0,
        "boat_top1_hit_rate": boat_top1_hits / race_count if race_count else 0.0,
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

    top3_confidence_metrics = compute_top3_confidence_metrics(trifecta_df, probability_col=probability_col)
    if top3_confidence_metrics:
        metrics["top3_confidence_metrics"] = top3_confidence_metrics

    top3_score_band_metrics = compute_top3_confidence_score_band_metrics(
        trifecta_df,
        probability_col=probability_col,
    )
    if top3_score_band_metrics:
        metrics["top3_confidence_score_band_metrics"] = top3_score_band_metrics

    boat_top1_confidence_metrics = compute_boat_top1_confidence_metrics(trifecta_df, probability_col=probability_col)
    if boat_top1_confidence_metrics:
        metrics["boat_top1_confidence_metrics"] = boat_top1_confidence_metrics

    top3_x_boat_top1_confidence_metrics = compute_top3_x_boat_top1_confidence_metrics(
        trifecta_df,
        probability_col=probability_col,
    )
    if top3_x_boat_top1_confidence_metrics:
        metrics["top3_x_boat_top1_confidence_metrics"] = top3_x_boat_top1_confidence_metrics

    top3_x_boat_top1_score_band_metrics = compute_top3_x_boat_top1_score_band_metrics(
        trifecta_df,
        probability_col=probability_col,
    )
    if top3_x_boat_top1_score_band_metrics:
        metrics["top3_x_boat_top1_score_band_metrics"] = top3_x_boat_top1_score_band_metrics

    top3_hit_probability_metrics = compute_top3_hit_probability_metrics(trifecta_df, probability_col=probability_col)
    if top3_hit_probability_metrics:
        metrics["top3_hit_probability_metrics"] = top3_hit_probability_metrics

    payout_band_metrics = compute_payout_band_metrics(trifecta_df, probability_col=probability_col)
    if payout_band_metrics:
        metrics["payout_band_metrics"] = payout_band_metrics

    uniform_ticket_recovery_metrics = compute_uniform_ticket_recovery_metrics(
        trifecta_df,
        probability_col=probability_col,
    )
    if uniform_ticket_recovery_metrics:
        metrics["uniform_ticket_recovery_metrics"] = uniform_ticket_recovery_metrics

    confidence_recovery_metrics = compute_top12_confidence_recovery_metrics(
        trifecta_df,
        probability_col=probability_col,
    )
    if confidence_recovery_metrics:
        metrics["top12_confidence_recovery_metrics"] = confidence_recovery_metrics

    confidence_strategy_recovery_metrics = compute_top12_confidence_strategy_recovery_metrics(
        trifecta_df,
        probability_col=probability_col,
    )
    if confidence_strategy_recovery_metrics:
        metrics["top12_confidence_strategy_recovery_metrics"] = confidence_strategy_recovery_metrics

    variable_ticket_recovery_metrics = compute_variable_ticket_recovery_metrics(
        trifecta_df,
        probability_col=probability_col,
    )
    if variable_ticket_recovery_metrics:
        metrics["variable_ticket_recovery_metrics"] = variable_ticket_recovery_metrics

    return metrics


def compute_uniform_ticket_recovery_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    payout_col: str = "trifecta_payout",
    top_ns: Iterable[int] = (1, 3, 5, 8, 12),
    bottom_ns: Iterable[int] = (8, 6),
    stake_per_ticket: float = 100.0,
) -> dict[str, dict[str, float]]:
    if trifecta_df.empty or payout_col not in trifecta_df.columns:
        return {}

    strategy_records: dict[str, list[dict[str, float]]] = {
        **{f"top{int(top_n)}": [] for top_n in top_ns},
        **{f"bottom{int(bottom_n)}": [] for bottom_n in bottom_ns},
    }
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
        for top_n in top_ns:
            ticket_count = min(int(top_n), len(ordered))
            stake = float(ticket_count) * float(stake_per_ticket)
            hit = float(actual_idx < ticket_count)
            strategy_records[f"top{int(top_n)}"].append(
                {
                    "hit": hit,
                    "stake": stake,
                    "return": payout * hit,
                    "payout": payout,
                    "ticket_count": float(ticket_count),
                }
            )
        for bottom_n in bottom_ns:
            ticket_count = min(int(bottom_n), len(ordered))
            candidate_pool = min(12, len(ordered))
            bottom_start = max(candidate_pool - ticket_count, 0)
            stake = float(ticket_count) * float(stake_per_ticket)
            hit = float(bottom_start <= actual_idx < candidate_pool)
            strategy_records[f"bottom{int(bottom_n)}"].append(
                {
                    "hit": hit,
                    "stake": stake,
                    "return": payout * hit,
                    "payout": payout,
                    "ticket_count": float(ticket_count),
                }
            )

    return {
        strategy: _summarize_ticket_recovery_records(records)
        for strategy, records in strategy_records.items()
        if records
    }


def compute_top12_confidence_recovery_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    payout_col: str = "trifecta_payout",
    stake_per_ticket: float = 100.0,
) -> dict[str, dict[str, float]]:
    if trifecta_df.empty or payout_col not in trifecta_df.columns:
        return {}

    frame = attach_top12_confidence_columns(trifecta_df, probability_col=probability_col)
    records_by_label: dict[str, list[dict[str, float]]] = {}
    for _, race_df in frame.groupby("race_id", sort=False):
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
        top_row = ordered.iloc[0]
        label = top12_confidence_label_key(top_row.get("top12_confidence_label"))
        ticket_count = min(12, len(ordered))
        hit = float(actual_idx < ticket_count)
        records_by_label.setdefault(label, []).append(
            {
                "hit": hit,
                "stake": float(ticket_count) * float(stake_per_ticket),
                "return": payout * hit,
                "payout": payout,
                "score": float(top_row.get("top12_confidence_score", 0.0) or 0.0),
            }
        )

    total_races = sum(len(records) for records in records_by_label.values())
    return {
        label: _summarize_ticket_recovery_records(records, total_races=total_races)
        for label, records in sorted(records_by_label.items())
        if records
    }


def compute_top12_confidence_strategy_recovery_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    payout_col: str = "trifecta_payout",
    top_ns: Iterable[int] = (1, 3, 5, 8, 12),
    bottom_ns: Iterable[int] = (8, 6),
    stake_per_ticket: float = 100.0,
) -> dict[str, dict[str, dict[str, float]]]:
    if trifecta_df.empty or payout_col not in trifecta_df.columns:
        return {}

    frame = attach_top12_confidence_columns(trifecta_df, probability_col=probability_col)
    strategy_names = [f"top{int(top_n)}" for top_n in top_ns] + [
        f"bottom{int(bottom_n)}" for bottom_n in bottom_ns
    ]
    records_by_label: dict[str, dict[str, list[dict[str, float]]]] = {
        label: {strategy: [] for strategy in strategy_names}
        for label in ("high", "middle", "low")
    }

    total_races = 0
    for _, race_df in frame.groupby("race_id", sort=False):
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

        total_races += 1
        actual_idx = int(actual_positions[0])
        top_row = ordered.iloc[0]
        label = top12_confidence_label_key(top_row.get("top12_confidence_label"))
        score = float(top_row.get("top12_confidence_score", 0.0) or 0.0)
        for top_n in top_ns:
            ticket_count = min(int(top_n), len(ordered))
            hit = float(actual_idx < ticket_count)
            records_by_label[label][f"top{int(top_n)}"].append(
                _ticket_recovery_record(
                    hit=hit,
                    ticket_count=ticket_count,
                    payout=payout,
                    score=score,
                    stake_per_ticket=stake_per_ticket,
                )
            )
        for bottom_n in bottom_ns:
            ticket_count = min(int(bottom_n), len(ordered))
            candidate_pool = min(12, len(ordered))
            bottom_start = max(candidate_pool - ticket_count, 0)
            hit = float(bottom_start <= actual_idx < candidate_pool)
            records_by_label[label][f"bottom{int(bottom_n)}"].append(
                _ticket_recovery_record(
                    hit=hit,
                    ticket_count=ticket_count,
                    payout=payout,
                    score=score,
                    stake_per_ticket=stake_per_ticket,
                )
            )

    if total_races <= 0:
        return {}

    result: dict[str, dict[str, dict[str, float]]] = {}
    for label in ("high", "middle", "low"):
        label_records = {
            strategy: _summarize_ticket_recovery_records(records, total_races=total_races)
            for strategy, records in records_by_label[label].items()
            if records
        }
        if label_records:
            result[label] = label_records
    return result


def compute_variable_ticket_recovery_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    payout_col: str = "trifecta_payout",
    stake_per_ticket: float = 100.0,
) -> dict[str, object]:
    if trifecta_df.empty or payout_col not in trifecta_df.columns:
        return {}

    frame = attach_top3_confidence_columns(trifecta_df, probability_col=probability_col)
    records: list[dict[str, float | str]] = []
    records_by_decision: dict[str, list[dict[str, float | str]]] = {}
    for _, race_df in frame.groupby("race_id", sort=False):
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

        top_row = ordered.iloc[0]
        requested_tickets = recommended_ticket_count_from_top3_confidence(top_row.get("top3_confidence_label"))
        decision = _variable_ticket_decision(requested_tickets)
        ticket_count = min(requested_tickets, len(ordered))
        actual_idx = int(actual_positions[0])
        hit = float(ticket_count > 0 and actual_idx < ticket_count)
        record = {
            "decision": decision,
            "hit": hit,
            "stake": float(ticket_count) * float(stake_per_ticket),
            "return": payout * hit,
            "payout": payout,
            "score": float(top_row.get("top3_confidence_score", 0.0) or 0.0),
            "ticket_count": float(ticket_count),
        }
        records.append(record)
        records_by_decision.setdefault(decision, []).append(record)

    if not records:
        return {}

    return {
        "summary": _summarize_variable_ticket_records(records),
        "by_decision": {
            decision: _summarize_variable_ticket_records(decision_records, total_races=len(records))
            for decision, decision_records in sorted(records_by_decision.items())
        },
        "rule": {
            "high": "top3",
            "middle": "top3",
            "low": "skip",
        },
        "confidence_type": "top3",
    }


def _variable_ticket_decision(ticket_count: int) -> str:
    count = int(ticket_count)
    if count > 0:
        return f"top{count}"
    return "skip"


def _ticket_recovery_record(
    hit: float,
    ticket_count: int,
    payout: float,
    score: float | None = None,
    stake_per_ticket: float = 100.0,
) -> dict[str, float]:
    record = {
        "hit": float(hit),
        "stake": float(ticket_count) * float(stake_per_ticket),
        "return": float(payout) * float(hit),
        "payout": float(payout),
        "ticket_count": float(ticket_count),
    }
    if score is not None:
        record["score"] = float(score)
    return record


def _summarize_ticket_recovery_records(
    records: list[dict[str, float]],
    total_races: int | None = None,
) -> dict[str, float]:
    if not records:
        return {
            "race_count": 0.0,
            "race_rate": 0.0,
            "hit_rate": 0.0,
            "total_stake": 0.0,
            "total_return": 0.0,
            "recovery_rate": 0.0,
            "mean_payout_all": 0.0,
            "mean_payout_hit": 0.0,
            "mean_score": 0.0,
        }
    race_count = len(records)
    total_stake = float(sum(record["stake"] for record in records))
    total_return = float(sum(record["return"] for record in records))
    hits = [record["hit"] for record in records]
    payouts = [record["payout"] for record in records]
    hit_payouts = [record["payout"] for record in records if record["hit"] > 0.0]
    scores = [record["score"] for record in records if "score" in record]
    ticket_counts = [record["ticket_count"] for record in records if "ticket_count" in record]
    summary = {
        "race_count": float(race_count),
        "race_rate": float(race_count / total_races) if total_races else 1.0,
        "hit_rate": float(np.mean(hits)),
        "total_stake": total_stake,
        "total_return": total_return,
        "recovery_rate": total_return / total_stake if total_stake else 0.0,
        "mean_payout_all": float(np.mean(payouts)) if payouts else 0.0,
        "mean_payout_hit": float(np.mean(hit_payouts)) if hit_payouts else 0.0,
    }
    if scores:
        summary["mean_score"] = float(np.mean(scores))
    if ticket_counts:
        summary["ticket_count"] = float(np.mean(ticket_counts))
    return summary


def _summarize_variable_ticket_records(
    records: list[dict[str, float | str]],
    total_races: int | None = None,
) -> dict[str, float]:
    if not records:
        return {
            "race_count": 0.0,
            "race_rate": 0.0,
            "purchased_race_count": 0.0,
            "purchase_rate": 0.0,
            "average_ticket_count": 0.0,
            "average_ticket_count_purchased": 0.0,
            "hit_rate": 0.0,
            "overall_hit_rate": 0.0,
            "total_stake": 0.0,
            "total_return": 0.0,
            "recovery_rate": 0.0,
            "mean_payout_all": 0.0,
            "mean_payout_hit": 0.0,
            "mean_score": 0.0,
        }

    race_count = len(records)
    ticket_counts = np.asarray([float(record["ticket_count"]) for record in records], dtype=float)
    purchase_mask = ticket_counts > 0.0
    hits = np.asarray([float(record["hit"]) for record in records], dtype=float)
    stakes = np.asarray([float(record["stake"]) for record in records], dtype=float)
    returns = np.asarray([float(record["return"]) for record in records], dtype=float)
    payouts = np.asarray([float(record["payout"]) for record in records], dtype=float)
    scores = np.asarray([float(record["score"]) for record in records], dtype=float)
    purchased_race_count = int(purchase_mask.sum())
    hit_payouts = payouts[hits > 0.0]
    total_stake = float(stakes.sum())
    total_return = float(returns.sum())
    return {
        "race_count": float(race_count),
        "race_rate": float(race_count / total_races) if total_races else 1.0,
        "purchased_race_count": float(purchased_race_count),
        "purchase_rate": float(purchased_race_count / race_count) if race_count else 0.0,
        "average_ticket_count": float(np.mean(ticket_counts)) if race_count else 0.0,
        "average_ticket_count_purchased": float(np.mean(ticket_counts[purchase_mask])) if purchased_race_count else 0.0,
        "hit_rate": float(hits[purchase_mask].mean()) if purchased_race_count else 0.0,
        "overall_hit_rate": float(hits.mean()) if race_count else 0.0,
        "total_stake": total_stake,
        "total_return": total_return,
        "recovery_rate": total_return / total_stake if total_stake else 0.0,
        "mean_payout_all": float(np.mean(payouts)) if race_count else 0.0,
        "mean_payout_hit": float(np.mean(hit_payouts)) if len(hit_payouts) else 0.0,
        "mean_score": float(np.mean(scores)) if race_count else 0.0,
    }


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


def compute_top3_confidence_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
) -> dict[str, dict[str, float]]:
    if trifecta_df.empty:
        return {}

    required = {"race_id", "trifecta", probability_col, "is_actual"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing top3 confidence metric columns: {sorted(missing)}")

    frame = attach_top3_confidence_columns(trifecta_df, probability_col=probability_col)
    by_label: dict[str, list[dict[str, float]]] = {}

    for _, scored_race in frame.groupby("race_id", sort=False):
        ordered = scored_race.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue
        actual_idx = int(actual_positions[0])
        top_row = ordered.iloc[0]
        record = {
            "boat_top1_hit": float(
                _trifecta_winner(top_row.get("trifecta")) == _trifecta_winner(ordered.loc[actual_idx].get("trifecta"))
            ),
            "top3_hit": float(int(actual_idx < 3)),
            "score": float(top_row.get("top3_confidence_score", 0.0) or 0.0),
            "top3_probability_mass": float(top_row.get("top3_probability_mass", 0.0) or 0.0),
            "top1_probability": float(top_row.get("top1_probability", 0.0) or 0.0),
            "top3_top4_probability_margin": float(top_row.get("top3_top4_probability_margin", 0.0) or 0.0),
        }
        by_label.setdefault(top3_confidence_label_key(top_row.get("top3_confidence_label")), []).append(record)

    total_races = sum(len(records) for records in by_label.values())
    return {
        key: _summarize_top3_confidence_records(records, total_races=total_races)
        for key, records in sorted(by_label.items())
    }


def _summarize_top3_confidence_records(
    records: list[dict[str, float]],
    total_races: int,
) -> dict[str, float]:
    if not records:
        return {
            "race_count": 0.0,
            "race_rate": 0.0,
            "boat_top1_hit_rate": 0.0,
            "top3_hit_rate": 0.0,
            "mean_score": 0.0,
            "mean_top3_probability_mass": 0.0,
            "mean_top1_probability": 0.0,
            "mean_top3_probability_margin": 0.0,
        }
    return {
        "race_count": float(len(records)),
        "race_rate": float(len(records) / total_races) if total_races else 0.0,
        "boat_top1_hit_rate": float(np.mean([record["boat_top1_hit"] for record in records])),
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in records])),
        "mean_score": float(np.mean([record["score"] for record in records])),
        "mean_top3_probability_mass": float(np.mean([record["top3_probability_mass"] for record in records])),
        "mean_top1_probability": float(np.mean([record["top1_probability"] for record in records])),
        "mean_top3_probability_margin": float(np.mean([record["top3_top4_probability_margin"] for record in records])),
    }


def compute_top3_confidence_score_band_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    payout_col: str = "trifecta_payout",
    stake_per_ticket: float = 100.0,
) -> dict[str, dict[str, float]]:
    if trifecta_df.empty:
        return {}

    required = {"race_id", "trifecta", probability_col, "is_actual"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing top3 confidence score band metric columns: {sorted(missing)}")

    frame = attach_top3_confidence_columns(trifecta_df, probability_col=probability_col)
    has_payout = payout_col in frame.columns
    by_band: dict[str, list[dict[str, float]]] = {band: [] for band, _, _ in TOP3_CONFIDENCE_SCORE_BANDS}

    for _, scored_race in frame.groupby("race_id", sort=False):
        ordered = scored_race.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue
        actual_idx = int(actual_positions[0])
        top_row = ordered.iloc[0]
        payout = 0.0
        if has_payout:
            payout_values = pd.to_numeric(ordered[payout_col], errors="coerce").dropna()
            payout = float(payout_values.iloc[0]) if not payout_values.empty else 0.0
        record = _top3_score_filter_record(
            ordered=ordered,
            top_row=top_row,
            actual_idx=actual_idx,
            payout=payout,
            stake_per_ticket=stake_per_ticket,
        )
        by_band[_top3_score_band_key(record["top3_confidence_score"])].append(record)

    total_races = sum(len(records) for records in by_band.values())
    if total_races <= 0:
        return {}
    return {
        band: _summarize_top3_score_filter_records(records, total_races=total_races)
        for band, records in by_band.items()
        if records
    }


def compute_boat_top1_confidence_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    payout_col: str = "trifecta_payout",
    stake_per_ticket: float = 100.0,
) -> dict[str, dict[str, float]]:
    if trifecta_df.empty:
        return {}

    required = {"race_id", "trifecta", probability_col, "is_actual"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing boat top1 confidence metric columns: {sorted(missing)}")

    frame = attach_boat_top1_confidence_columns(trifecta_df, probability_col=probability_col)
    has_payout = payout_col in frame.columns
    by_label: dict[str, list[dict[str, float]]] = {}

    for _, scored_race in frame.groupby("race_id", sort=False):
        ordered = scored_race.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue

        actual_idx = int(actual_positions[0])
        top_row = ordered.iloc[0]
        predicted_first_boat = str(top_row.get("predicted_first_boat", "")).split(".", 1)[0]
        actual_first_boat = _trifecta_winner(ordered.loc[actual_idx].get("trifecta"))
        top3_hit = float(actual_idx < 3)
        payout = 0.0
        if has_payout:
            payout_values = pd.to_numeric(ordered[payout_col], errors="coerce").dropna()
            payout = float(payout_values.iloc[0]) if not payout_values.empty else 0.0
        record = {
            "boat_top1_hit": float(predicted_first_boat == actual_first_boat),
            "top3_hit": top3_hit,
            "top12_hit": float(actual_idx < 12),
            "score": float(top_row.get("boat_top1_confidence_score", 0.0) or 0.0),
            "predicted_first_boat_probability": float(top_row.get("predicted_first_boat_probability", 0.0) or 0.0),
            "predicted_first_boat_gap": float(top_row.get("predicted_first_boat_gap", 0.0) or 0.0),
            "top3_return": payout * top3_hit if payout > 0.0 else 0.0,
            "top3_stake": min(3, len(ordered)) * float(stake_per_ticket) if payout > 0.0 else 0.0,
            "payout": payout,
        }
        by_label.setdefault(boat_top1_confidence_label_key(top_row.get("boat_top1_confidence_label")), []).append(record)

    total_races = sum(len(records) for records in by_label.values())
    return {
        key: _summarize_boat_top1_confidence_records(records, total_races=total_races)
        for key, records in sorted(by_label.items())
    }


def _summarize_boat_top1_confidence_records(
    records: list[dict[str, float]],
    total_races: int,
) -> dict[str, float]:
    if not records:
        return {
            "race_count": 0.0,
            "race_rate": 0.0,
            "boat_top1_hit_rate": 0.0,
            "top3_hit_rate": 0.0,
            "top12_hit_rate": 0.0,
            "top3_total_stake": 0.0,
            "top3_total_return": 0.0,
            "top3_recovery_rate": 0.0,
            "mean_score": 0.0,
            "mean_predicted_first_boat_probability": 0.0,
            "mean_predicted_first_boat_gap": 0.0,
            "mean_payout_hit": 0.0,
        }
    top3_stakes = [record["top3_stake"] for record in records]
    top3_returns = [record["top3_return"] for record in records]
    total_stake = float(sum(top3_stakes))
    total_return = float(sum(top3_returns))
    hit_payouts = [record["payout"] for record in records if record["top3_hit"] > 0.0 and record["payout"] > 0.0]
    return {
        "race_count": float(len(records)),
        "race_rate": float(len(records) / total_races) if total_races else 0.0,
        "boat_top1_hit_rate": float(np.mean([record["boat_top1_hit"] for record in records])),
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in records])),
        "top12_hit_rate": float(np.mean([record["top12_hit"] for record in records])),
        "top3_total_stake": total_stake,
        "top3_total_return": total_return,
        "top3_recovery_rate": total_return / total_stake if total_stake else 0.0,
        "mean_score": float(np.mean([record["score"] for record in records])),
        "mean_predicted_first_boat_probability": float(
            np.mean([record["predicted_first_boat_probability"] for record in records])
        ),
        "mean_predicted_first_boat_gap": float(np.mean([record["predicted_first_boat_gap"] for record in records])),
        "mean_payout_hit": float(np.mean(hit_payouts)) if hit_payouts else 0.0,
    }


def compute_top3_x_boat_top1_confidence_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    payout_col: str = "trifecta_payout",
    stake_per_ticket: float = 100.0,
) -> dict[str, dict[str, dict[str, float]]]:
    if trifecta_df.empty:
        return {}

    required = {"race_id", "trifecta", probability_col, "is_actual"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing top3 x boat top1 confidence metric columns: {sorted(missing)}")

    frame = attach_top3_confidence_columns(trifecta_df, probability_col=probability_col)
    frame = attach_boat_top1_confidence_columns(frame, probability_col=probability_col)
    has_payout = payout_col in frame.columns
    by_labels: dict[str, dict[str, list[dict[str, float]]]] = {
        top3_label: {boat_label: [] for boat_label in ("high", "middle", "low")}
        for top3_label in ("high", "middle", "low")
    }

    for _, scored_race in frame.groupby("race_id", sort=False):
        ordered = scored_race.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue

        actual_idx = int(actual_positions[0])
        top_row = ordered.iloc[0]
        top3_label = top3_confidence_label_key(top_row.get("top3_confidence_label"))
        boat_label = boat_top1_confidence_label_key(top_row.get("boat_top1_confidence_label"))
        predicted_first_boat = str(top_row.get("predicted_first_boat", "")).split(".", 1)[0]
        actual_first_boat = _trifecta_winner(ordered.loc[actual_idx].get("trifecta"))
        top1_hit = float(actual_idx < 1)
        top3_hit = float(actual_idx < 3)
        payout = 0.0
        if has_payout:
            payout_values = pd.to_numeric(ordered[payout_col], errors="coerce").dropna()
            payout = float(payout_values.iloc[0]) if not payout_values.empty else 0.0
        by_labels[top3_label][boat_label].append(
            {
                "boat_top1_hit": float(predicted_first_boat == actual_first_boat),
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "top12_hit": float(actual_idx < 12),
                "top3_confidence_score": float(top_row.get("top3_confidence_score", 0.0) or 0.0),
                "boat_top1_confidence_score": float(top_row.get("boat_top1_confidence_score", 0.0) or 0.0),
                "predicted_first_boat_probability": float(
                    top_row.get("predicted_first_boat_probability", 0.0) or 0.0
                ),
                "predicted_first_boat_gap": float(top_row.get("predicted_first_boat_gap", 0.0) or 0.0),
                "top3_return": payout * top3_hit if payout > 0.0 else 0.0,
                "top3_stake": min(3, len(ordered)) * float(stake_per_ticket) if payout > 0.0 else 0.0,
                "payout": payout,
            }
        )

    total_races = sum(len(records) for by_boat in by_labels.values() for records in by_boat.values())
    if total_races <= 0:
        return {}

    result: dict[str, dict[str, dict[str, float]]] = {}
    for top3_label in ("high", "middle", "low"):
        row: dict[str, dict[str, float]] = {}
        for boat_label in ("high", "middle", "low"):
            records = by_labels[top3_label][boat_label]
            if records:
                row[boat_label] = _summarize_confidence_cross_records(records, total_races=total_races)
        if row:
            result[top3_label] = row
    return result


def compute_top3_x_boat_top1_score_band_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
    payout_col: str = "trifecta_payout",
    stake_per_ticket: float = 100.0,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    if trifecta_df.empty:
        return {}

    required = {"race_id", "trifecta", probability_col, "is_actual"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing top3 x boat top1 score band metric columns: {sorted(missing)}")

    frame = attach_top3_confidence_columns(trifecta_df, probability_col=probability_col)
    frame = attach_boat_top1_confidence_columns(frame, probability_col=probability_col)
    has_payout = payout_col in frame.columns
    by_labels: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {
        top3_label: {
            boat_label: {band: [] for band, _, _ in TOP3_CONFIDENCE_SCORE_BANDS}
            for boat_label in ("high", "middle", "low")
        }
        for top3_label in ("high", "middle", "low")
    }

    for _, scored_race in frame.groupby("race_id", sort=False):
        ordered = scored_race.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue
        actual_idx = int(actual_positions[0])
        top_row = ordered.iloc[0]
        payout = 0.0
        if has_payout:
            payout_values = pd.to_numeric(ordered[payout_col], errors="coerce").dropna()
            payout = float(payout_values.iloc[0]) if not payout_values.empty else 0.0
        record = _top3_score_filter_record(
            ordered=ordered,
            top_row=top_row,
            actual_idx=actual_idx,
            payout=payout,
            stake_per_ticket=stake_per_ticket,
        )
        top3_label = top3_confidence_label_key(top_row.get("top3_confidence_label"))
        boat_label = boat_top1_confidence_label_key(top_row.get("boat_top1_confidence_label"))
        score_band = _top3_score_band_key(record["top3_confidence_score"])
        by_labels[top3_label][boat_label][score_band].append(record)

    total_races = sum(
        len(records)
        for by_boat in by_labels.values()
        for by_band in by_boat.values()
        for records in by_band.values()
    )
    if total_races <= 0:
        return {}

    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for top3_label in ("high", "middle", "low"):
        top3_row: dict[str, dict[str, dict[str, float]]] = {}
        for boat_label in ("high", "middle", "low"):
            boat_row = {
                band: _summarize_top3_score_filter_records(records, total_races=total_races)
                for band, records in by_labels[top3_label][boat_label].items()
                if records
            }
            if boat_row:
                top3_row[boat_label] = boat_row
        if top3_row:
            result[top3_label] = top3_row
    return result


def compute_top3_hit_probability_metrics(
    trifecta_df: pd.DataFrame,
    probability_col: str = "probability",
) -> dict[str, object]:
    if trifecta_df.empty or "top3_hit_probability" not in trifecta_df.columns:
        return {}
    required = {"race_id", probability_col, "is_actual"}
    missing = required - set(trifecta_df.columns)
    if missing:
        raise ValueError(f"Missing top3 hit probability metric columns: {sorted(missing)}")

    records: list[dict[str, float]] = []
    for _, race_df in trifecta_df.groupby("race_id", sort=False):
        ordered = race_df.sort_values(probability_col, ascending=False).reset_index(drop=True)
        actual_positions = np.flatnonzero(ordered["is_actual"].to_numpy(dtype=bool))
        if len(actual_positions) != 1:
            continue
        top_row = ordered.iloc[0]
        records.append(
            {
                "top3_hit_probability": float(top_row.get("top3_hit_probability", 0.0) or 0.0),
                "top3_hit": float(int(actual_positions[0]) < 3),
                "top3_confidence_score": float(top_row.get("top3_confidence_score", 0.0) or 0.0),
                "top3_structure_tightness_score": float(
                    top_row.get("top3_structure_tightness_score", 0.0) or 0.0
                ),
            }
        )
    if not records:
        return {}
    return summarize_top3_hit_probability_bands(pd.DataFrame(records))


def _top3_score_filter_record(
    ordered: pd.DataFrame,
    top_row: pd.Series,
    actual_idx: int,
    payout: float,
    stake_per_ticket: float,
) -> dict[str, float]:
    predicted_first_boat = str(top_row.get("predicted_first_boat", "")).split(".", 1)[0]
    if not predicted_first_boat:
        predicted_first_boat = str(_trifecta_winner(top_row.get("trifecta")) or "")
    actual_first_boat = _trifecta_winner(ordered.loc[actual_idx].get("trifecta"))
    top3_hit = float(actual_idx < 3)
    top3_stake = min(3, len(ordered)) * float(stake_per_ticket) if payout > 0.0 else 0.0
    return {
        "boat_top1_hit": float(predicted_first_boat == actual_first_boat),
        "top1_hit": float(actual_idx < 1),
        "top3_hit": top3_hit,
        "top5_hit": float(actual_idx < 5),
        "top12_hit": float(actual_idx < 12),
        "top3_confidence_score": float(top_row.get("top3_confidence_score", 0.0) or 0.0),
        "boat_top1_confidence_score": float(top_row.get("boat_top1_confidence_score", 0.0) or 0.0),
        "top3_probability_mass": float(top_row.get("top3_probability_mass", 0.0) or 0.0),
        "top1_probability": float(top_row.get("top1_probability", 0.0) or 0.0),
        "top1_top2_probability_gap": float(top_row.get("top1_top2_probability_gap", 0.0) or 0.0),
        "top3_top4_probability_margin": float(top_row.get("top3_top4_probability_margin", 0.0) or 0.0),
        "top3_same_first_boat_rate": float(top_row.get("top3_same_first_boat_rate", 0.0) or 0.0),
        "top3_same_first_second_pair_rate": float(top_row.get("top3_same_first_second_pair_rate", 0.0) or 0.0),
        "top3_unique_first_boat_count": float(top_row.get("top3_unique_first_boat_count", 0.0) or 0.0),
        "top3_unique_second_boat_count": float(top_row.get("top3_unique_second_boat_count", 0.0) or 0.0),
        "top3_unique_first_second_pair_count": float(
            top_row.get("top3_unique_first_second_pair_count", 0.0) or 0.0
        ),
        "top3_unique_boat_count": float(top_row.get("top3_unique_boat_count", 0.0) or 0.0),
        "top3_structure_tightness_score": float(top_row.get("top3_structure_tightness_score", 0.0) or 0.0),
        "probability_entropy": float(top_row.get("probability_entropy", 0.0) or 0.0),
        "top3_return": payout * top3_hit if payout > 0.0 else 0.0,
        "top3_stake": top3_stake,
        "payout": payout,
    }


def _summarize_top3_score_filter_records(
    records: list[dict[str, float]],
    total_races: int,
) -> dict[str, float]:
    total_stake = float(sum(record["top3_stake"] for record in records))
    total_return = float(sum(record["top3_return"] for record in records))
    hit_payouts = [record["payout"] for record in records if record["top3_hit"] > 0.0 and record["payout"] > 0.0]
    return {
        "race_count": float(len(records)),
        "race_rate": float(len(records) / total_races) if total_races else 0.0,
        "boat_top1_hit_rate": float(np.mean([record["boat_top1_hit"] for record in records])),
        "top1_hit_rate": float(np.mean([record["top1_hit"] for record in records])),
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in records])),
        "top5_hit_rate": float(np.mean([record["top5_hit"] for record in records])),
        "top12_hit_rate": float(np.mean([record["top12_hit"] for record in records])),
        "top3_total_stake": total_stake,
        "top3_total_return": total_return,
        "top3_recovery_rate": total_return / total_stake if total_stake else 0.0,
        "mean_payout_hit": float(np.mean(hit_payouts)) if hit_payouts else 0.0,
        "mean_top3_confidence_score": float(np.mean([record["top3_confidence_score"] for record in records])),
        "mean_boat_top1_confidence_score": float(
            np.mean([record["boat_top1_confidence_score"] for record in records])
        ),
        "mean_top3_probability_mass": float(np.mean([record["top3_probability_mass"] for record in records])),
        "mean_top1_probability": float(np.mean([record["top1_probability"] for record in records])),
        "mean_top1_top2_probability_gap": float(np.mean([record["top1_top2_probability_gap"] for record in records])),
        "mean_top3_top4_probability_margin": float(
            np.mean([record["top3_top4_probability_margin"] for record in records])
        ),
        "mean_top3_same_first_boat_rate": float(
            np.mean([record["top3_same_first_boat_rate"] for record in records])
        ),
        "mean_top3_same_first_second_pair_rate": float(
            np.mean([record["top3_same_first_second_pair_rate"] for record in records])
        ),
        "mean_top3_unique_first_boat_count": float(
            np.mean([record["top3_unique_first_boat_count"] for record in records])
        ),
        "mean_top3_unique_second_boat_count": float(
            np.mean([record["top3_unique_second_boat_count"] for record in records])
        ),
        "mean_top3_unique_first_second_pair_count": float(
            np.mean([record["top3_unique_first_second_pair_count"] for record in records])
        ),
        "mean_top3_unique_boat_count": float(np.mean([record["top3_unique_boat_count"] for record in records])),
        "mean_top3_structure_tightness_score": float(
            np.mean([record["top3_structure_tightness_score"] for record in records])
        ),
        "mean_probability_entropy": float(np.mean([record["probability_entropy"] for record in records])),
    }


def _top3_score_band_key(score: float) -> str:
    value = float(score)
    for key, lower, upper in TOP3_CONFIDENCE_SCORE_BANDS:
        if value < lower:
            continue
        if upper is not None and value >= upper:
            continue
        return key
    return "score_unknown"


def _summarize_confidence_cross_records(
    records: list[dict[str, float]],
    total_races: int,
) -> dict[str, float]:
    top3_stakes = [record["top3_stake"] for record in records]
    top3_returns = [record["top3_return"] for record in records]
    total_stake = float(sum(top3_stakes))
    total_return = float(sum(top3_returns))
    hit_payouts = [record["payout"] for record in records if record["top3_hit"] > 0.0 and record["payout"] > 0.0]
    return {
        "race_count": float(len(records)),
        "race_rate": float(len(records) / total_races) if total_races else 0.0,
        "boat_top1_hit_rate": float(np.mean([record["boat_top1_hit"] for record in records])),
        "top1_hit_rate": float(np.mean([record["top1_hit"] for record in records])),
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in records])),
        "top12_hit_rate": float(np.mean([record["top12_hit"] for record in records])),
        "top3_total_stake": total_stake,
        "top3_total_return": total_return,
        "top3_recovery_rate": total_return / total_stake if total_stake else 0.0,
        "mean_top3_confidence_score": float(np.mean([record["top3_confidence_score"] for record in records])),
        "mean_boat_top1_confidence_score": float(
            np.mean([record["boat_top1_confidence_score"] for record in records])
        ),
        "mean_predicted_first_boat_probability": float(
            np.mean([record["predicted_first_boat_probability"] for record in records])
        ),
        "mean_predicted_first_boat_gap": float(np.mean([record["predicted_first_boat_gap"] for record in records])),
        "mean_payout_hit": float(np.mean(hit_payouts)) if hit_payouts else 0.0,
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
