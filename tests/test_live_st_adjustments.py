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


def test_st_preview_uses_live_feature_builder_without_loading_models(monkeypatch):
    from datetime import date
    from types import SimpleNamespace
    from pathlib import Path
    from src import api, live

    entries = frame().assign(race_no=1, race_id="20261112_15_01", race_date=date(2026, 11, 12))
    monkeypatch.setattr(api, "load_config", lambda _: {})
    monkeypatch.setattr(api, "get_artifact_paths", lambda _: {"features_path": Path("unused")})
    monkeypatch.setattr(api, "load_feature_columns", lambda _: ["start_timing"])
    monkeypatch.setattr(live, "fetch_mbrace_program_text", lambda _: "program")
    monkeypatch.setattr(live, "parse_entry_text", lambda _: [SimpleNamespace(to_dict=lambda r=r: r) for r in entries.to_dict("records")])
    monkeypatch.setattr(live, "load_live_history_frame", lambda *args: pd.DataFrame())
    calls = []

    def build(current, history, columns):
        calls.append(current.copy())
        return current.assign(start_timing=.17)

    monkeypatch.setattr(live, "build_live_feature_frame", build)
    monkeypatch.setattr(api, "load_bundle", lambda *args: pytest.fail("Preview must not load prediction models"))
    preview = api.preview_today_st("15", 1, race_date=date(2026, 11, 12), course_overrides=(2, 1, 3, 4, 5, 6))
    assert preview.lane.tolist() == [1, 2, 3, 4, 5, 6]
    assert preview.start_timing.eq(.17).all()
    assert calls[0].course.tolist() == [2, 1, 3, 4, 5, 6]
