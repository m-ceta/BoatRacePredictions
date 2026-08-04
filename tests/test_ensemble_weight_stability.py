import pandas as pd

from src.models.ranker import _split_base_by_race_hash, simplex_weight_vectors


def test_simplex_weight_vectors_respect_weight_limits():
    vectors = simplex_weight_vectors(
        model_count=5,
        steps=10,
        max_model_weight=0.5,
        min_nonzero_weight=0.1,
    )

    assert vectors
    for vector in vectors:
        assert abs(sum(vector) - 1.0) < 1e-9
        assert max(vector) <= 0.5
        assert all(weight == 0.0 or weight >= 0.1 for weight in vector)


def test_race_hash_fold_split_is_stable_and_race_atomic():
    base = pd.DataFrame(
        {
            "race_id": ["r1"] * 6 + ["r2"] * 6 + ["r3"] * 6 + ["r4"] * 6,
            "lane": [1, 2, 3, 4, 5, 6] * 4,
            "finish_position": [1, 2, 3, 4, 5, 6] * 4,
        }
    )

    first = _split_base_by_race_hash(base, folds=3)
    second = _split_base_by_race_hash(base, folds=3)

    first_sets = [set(part["race_id"].unique()) for part in first]
    second_sets = [set(part["race_id"].unique()) for part in second]
    assert first_sets == second_sets
    assert set().union(*first_sets) == {"r1", "r2", "r3", "r4"}
    assert sum(len(part) for part in first) == len(base)
