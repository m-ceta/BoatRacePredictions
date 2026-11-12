import numpy as np
import pandas as pd
import pytest

from src.live import apply_st_adjustments


def frame():
    return pd.DataFrame({
        "race_id": ["R1"] * 6, "lane": [1, 2, 3, 4, 5, 6],
        "course": [1, 3, 2, 4, 5, 6], "start_timing": [.16] * 6,
        "racer_prev_avg_st": [.16] * 6,
    })


def test_st_adjustment_updates_input_and_relative_features_without_mutation():
    source = frame()
    updated = apply_st_adjustments(source, [0, 0, -.03, .02, 0, 0])
    assert updated.loc[2, "start_timing"] == pytest.approx(.13)
    assert updated.loc[3, "start_timing"] == pytest.approx(.18)
    assert updated.loc[2, "start_timing_race_rank"] == 6
    assert updated.loc[2, "start_timing_race_rank_low"] == 1
    assert updated["racer_prev_avg_st"].eq(.16).all()
    assert source["start_timing"].eq(.16).all()


def test_zero_adjustments_preserve_features_exactly():
    source = frame()
    pd.testing.assert_frame_equal(source, apply_st_adjustments(source, [0] * 6))


@pytest.mark.parametrize("values", [[0] * 5, [np.nan] * 6, [np.inf] * 6, [.11] * 6])
def test_invalid_st_adjustments_are_rejected(values):
    with pytest.raises(ValueError):
        apply_st_adjustments(frame(), values)


def test_negative_resulting_st_is_not_silently_clipped():
    source = frame()
    source.loc[0, "start_timing"] = .02
    with pytest.raises(ValueError):
        apply_st_adjustments(source, [-.03, 0, 0, 0, 0, 0])
