from __future__ import annotations

import pandas as pd

from src.models import ranker


def test_train_checkpoint_signature_mismatch_resets_completed_state(tmp_path) -> None:
    checkpoint_path = tmp_path / "train_checkpoint.json"
    ranker.save_train_checkpoint(
        checkpoint_path,
        {
            "signature": "old",
            "completed": {"catboost": True},
            "metrics": {"catboost": {"race_count": 1.0}},
        },
    )

    checkpoint = ranker.load_train_checkpoint(checkpoint_path, signature="new")

    assert checkpoint["signature"] == "new"
    assert checkpoint["completed"] == {}
    assert checkpoint["metrics"] == {}


def test_train_stage_completed_requires_checkpoint_and_artifacts(tmp_path) -> None:
    artifact_path = tmp_path / "model.txt"
    checkpoint = {"completed": {"lightgbm": True}}

    assert not ranker.train_stage_completed(checkpoint, "lightgbm", [artifact_path])

    artifact_path.write_text("model", encoding="utf-8")

    assert ranker.train_stage_completed(checkpoint, "lightgbm", [artifact_path])


def test_train_checkpoint_signature_changes_with_race_count() -> None:
    config = {"split": {"train_end_date": "2026-01-01"}}
    train_a = pd.DataFrame({"race_id": ["R1"], "race_date": [pd.Timestamp("2026-01-01")]})
    train_b = pd.DataFrame(
        {
            "race_id": ["R1", "R2"],
            "race_date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
        }
    )
    empty = pd.DataFrame({"race_id": [], "race_date": []})

    sig_a = ranker.train_checkpoint_signature(config, train_a, empty, empty)
    sig_b = ranker.train_checkpoint_signature(config, train_b, empty, empty)

    assert sig_a != sig_b

