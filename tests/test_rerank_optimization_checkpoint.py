from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models import ranker


def test_optimize_rerank_saves_and_reuses_checkpoint(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "rerank_optimization_checkpoint.json"
    valid_df = pd.DataFrame({"race_id": ["R1", "R2"]})
    config = {
        "phase3": {
            "rerank": {
                "top_n_grid": [10],
                "weight_grid": [0.95, 0.99],
                "rank_penalty_strength_grid": [0.0, 0.01],
            }
        }
    }
    calls: list[tuple[float, float]] = []

    def fake_fit_model_trifecta_calibrator(*args, **kwargs):
        return object()

    def fake_evaluate_trifecta(*args, **kwargs):
        if not kwargs["use_v2"]:
            return {"log_loss": 0.1}
        model = kwargs["trifecta_v2_model"]
        weight = float(model["conservative_v1_weight"])
        penalty = float(model["rank_penalty_strength"])
        calls.append((weight, penalty))
        return {
            "top1_hit_rate": 0.0,
            "top3_hit_rate": 0.0,
            "top5_hit_rate": weight - penalty,
            "top10_hit_rate": 0.0,
            "top12_hit_rate": weight - penalty,
            "log_loss": 0.1,
            "rerank_metrics": {"rerank_mrr": 0.0},
        }

    monkeypatch.setattr(ranker, "fit_model_trifecta_calibrator", fake_fit_model_trifecta_calibrator)
    monkeypatch.setattr(ranker, "evaluate_trifecta", fake_evaluate_trifecta)

    first = ranker.optimize_rerank_inference_settings(
        models={},
        weights={},
        valid_df=valid_df,
        feature_columns=[],
        categorical_columns=[],
        trifecta_v2_model={"model_type": "lgbm_ranker"},
        config=config,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert first["best_conservative_weight"] == 0.99
    assert first["best_rank_penalty_strength"] == 0.0
    assert len(calls) == 4
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "completed"
    assert checkpoint["completed_count"] == 4

    calls.clear()
    second = ranker.optimize_rerank_inference_settings(
        models={},
        weights={},
        valid_df=valid_df,
        feature_columns=[],
        categorical_columns=[],
        trifecta_v2_model={"model_type": "lgbm_ranker"},
        config=config,
        checkpoint_path=checkpoint_path,
    )

    assert second == first
    assert calls == []


def test_two_stage_rerank_uses_small_coarse_sample_and_shortlisted_fine_search(monkeypatch) -> None:
    valid_df = pd.DataFrame(
        {
            "race_id": [f"R{race_no}" for race_no in range(20) for _ in range(2)],
        }
    )
    calls: list[dict[str, object]] = []

    def fake_optimize(*args, **kwargs):
        search_df = args[2]
        top_ns = list(kwargs["top_n_candidates"])
        weights = list(kwargs["conservative_weights"])
        penalties = list(kwargs["rank_penalty_strengths"])
        calls.append(
            {
                "races": search_df["race_id"].nunique(),
                "top_ns": top_ns,
                "weights": weights,
                "penalties": penalties,
            }
        )
        candidates = [
            {
                "top_n": float(top_n),
                "conservative_weight": float(weight),
                "rank_penalty_strength": float(penalties[0]),
                "objective": float(weight) + (0.001 * float(top_n)),
            }
            for top_n in top_ns
            for weight in weights
        ]
        best = max(candidates, key=lambda item: item["objective"])
        return {
            "best_top_n": best["top_n"],
            "best_conservative_weight": best["conservative_weight"],
            "best_rank_penalty_strength": best["rank_penalty_strength"],
            "objective": best["objective"],
            "ranked_candidates": sorted(candidates, key=lambda item: item["objective"], reverse=True),
        }

    monkeypatch.setattr(ranker, "optimize_rerank_inference_settings", fake_optimize)
    config = {
        "phase3": {
            "rerank": {
                "top_n_grid": [10, 24],
                "weight_grid": [0.5, 0.9],
                "rank_penalty_strength_grid": [0.0, 0.01],
                "coarse_penalty_grid": [0.0],
                "coarse_eval_max_races": 5,
                "fine_top_k": 2,
            }
        }
    }

    result = ranker.optimize_rerank_inference_settings_two_stage(
        models={},
        weights={},
        valid_df=valid_df,
        feature_columns=[],
        categorical_columns=[],
        trifecta_v2_model={"model_type": "lgbm_ranker"},
        config=config,
    )

    assert calls[0]["races"] == 5
    assert calls[0]["penalties"] == [0.0]
    assert all(call["races"] == 20 for call in calls[1:])
    assert sum(len(call["weights"]) * len(call["penalties"]) for call in calls[1:]) == 4
    assert result["search_mode"] == "two_stage"
    assert result["coarse_races"] == 5
