from pathlib import Path

import pandas as pd

from src.cli import write_feature_correlation_outputs


def test_write_feature_correlation_outputs(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "race_id": ["r1"] * 6 + ["r2"] * 6,
            "finish_position": [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1],
            "target_rank": [6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6],
            "is_win": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "f1": [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1],
            "f2": [2, 4, 6, 8, 10, 12, 12, 10, 8, 6, 4, 2],
            "constant_feature": [1] * 12,
            "text_feature": ["a"] * 12,
        }
    )

    summary = write_feature_correlation_outputs(
        df,
        ["f1", "f2", "constant_feature", "text_feature"],
        tmp_path,
        split="valid",
        feature_set="full",
        sample_races=0,
        pair_threshold=0.99,
        max_pairs=10,
    )

    targets = pd.read_csv(tmp_path / "feature_correlation_targets.csv")
    pairs = pd.read_csv(tmp_path / "feature_correlation_pairs.csv")

    assert summary["race_count"] == 2
    assert summary["numeric_feature_count"] == 2
    assert summary["dropped_constant_count"] == 1
    assert summary["dropped_non_numeric_count"] == 1
    assert {"feature", "target", "pearson", "spearman"}.issubset(targets.columns)
    assert set(targets["feature"]) == {"f1", "f2"}
    assert {
        tuple(row)
        for row in pairs[["feature_left", "feature_right"]].to_numpy()
    } == {("f1", "f2")}
