from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    short_id: str
    label: str
    numeric_id: int


SCENARIO_DEFINITIONS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition("S0", "S0_C1_ESCAPE_IN_WIN", 0),
    ScenarioDefinition("S1", "S1_C2_SASHI_IN_REMAIN", 1),
    ScenarioDefinition("S2", "S2_C2_WIN_IN_OUT", 2),
    ScenarioDefinition("S3", "S3_C3_WIN_CENTER", 3),
    ScenarioDefinition("S4", "S4_C4_WIN_KADO", 4),
    ScenarioDefinition("S5", "S5_OUTER_WIN_OR_CHAIN", 5),
    ScenarioDefinition("S6", "S6_MIXED_OTHER", 6),
)

SCENARIO_SHORT_TO_LABEL = {item.short_id: item.label for item in SCENARIO_DEFINITIONS}
SCENARIO_LABEL_TO_SHORT = {item.label: item.short_id for item in SCENARIO_DEFINITIONS}
SCENARIO_LABEL_TO_NUMERIC = {item.label: item.numeric_id for item in SCENARIO_DEFINITIONS}
SCENARIO_NAMES = {item.label: item.label for item in SCENARIO_DEFINITIONS}

SCENARIO_DISPLAY_NAMES = {
    "S0_IN_CONTROL": "イン主導・逃げ展開",
    "S1_COURSE2_SASHI": "2コース差し展開",
    "S2_COURSE2_MAKURI": "2コースまくり展開",
    "S3_COURSE3_ATTACK": "3コース攻め展開",
    "S4_COURSE4_ATTACK": "4コース攻め展開",
    "S5_OUTSIDE_ATTACK": "5・6コース外攻め展開",
    "S6_CHAOS": "混戦・判定難",
}

SCENARIO_DESCRIPTIONS = {
    "S0_IN_CONTROL": "1号艇またはインコース艇が主導し、内側が残りやすい展開です。",
    "S1_COURSE2_SASHI": "2コース艇の差しが焦点で、1号艇も2・3着に残りやすい展開です。",
    "S2_COURSE2_MAKURI": "2コース艇が外から攻め、1号艇や内側が崩れる可能性がある展開です。",
    "S3_COURSE3_ATTACK": "3コース艇が攻めの起点になり、内側の崩れや外側の連動が起きやすい展開です。",
    "S4_COURSE4_ATTACK": "4コース艇、特にカド位置の攻めが展開を動かしやすい展開です。",
    "S5_OUTSIDE_ATTACK": "5・6コース艇の浮上や外枠連動が起きやすく、高配当化しやすい展開です。",
    "S6_CHAOS": "明確な主導艇を決めにくく、複数の攻めや崩れが絡みやすい展開です。",
}


SCENARIO_DISPLAY_NAMES = {
    "S0_C1_ESCAPE_IN_WIN": "1コース逃げ・イン勝ち型",
    "S1_C2_SASHI_IN_REMAIN": "2コース差し・イン残り型",
    "S2_C2_WIN_IN_OUT": "2コース勝ち・イン崩れ型",
    "S3_C3_WIN_CENTER": "3コース勝ち・センター型",
    "S4_C4_WIN_KADO": "4コース勝ち・カド型",
    "S5_OUTER_WIN_OR_CHAIN": "外コース勝ち・外連動型",
    "S6_MIXED_OTHER": "混合・その他型",
}

SCENARIO_DESCRIPTIONS = {
    "S0_C1_ESCAPE_IN_WIN": "1コース艇が勝ち、決まり手は逃げ寄り、上位も内側中心になりやすい決着型です。",
    "S1_C2_SASHI_IN_REMAIN": "2コース艇の差し勝ち、または2コース差し筋で1号艇が2・3着に残る決着型です。",
    "S2_C2_WIN_IN_OUT": "2コース艇が勝ち、1号艇が4着以下へ崩れる決着型です。",
    "S3_C3_WIN_CENTER": "3コース艇が勝ち、3・4コース中心に上位を作る決着型です。",
    "S4_C4_WIN_KADO": "4コース艇が勝ち、カドまたはセンター寄りで上位を作る決着型です。",
    "S5_OUTER_WIN_OR_CHAIN": "5・6コース艇の勝ち、または4～6コースが複数上位に絡む外連動型です。",
    "S6_MIXED_OTHER": "勝ち筋や上位構成が一つの典型に寄り切らない混合・その他の決着型です。",
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _row_numeric(row: pd.Series, *names: str, default: float = 0.0) -> float:
    for name in names:
        if name not in row:
            continue
        value = pd.to_numeric(row.get(name), errors="coerce")
        if pd.notna(value):
            return float(value)
    return float(default)


def _numeric_first_available(frame: pd.DataFrame, *names: str, default: float = 0.0) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype="float64")
    for name in names:
        if name not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[name], errors="coerce")
        values = values.where(values.notna(), candidate)
    return values.fillna(float(default)).astype(float)


def _race_scale(values: pd.Series, lower_is_better: bool = False) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    min_value = float(numeric.min()) if numeric.notna().any() else 0.0
    max_value = float(numeric.max()) if numeric.notna().any() else 0.0
    if np.isclose(max_value, min_value):
        return pd.Series(0.0, index=values.index)
    scaled = (numeric - min_value) / (max_value - min_value)
    if lower_is_better:
        scaled = 1.0 - scaled
    return scaled.fillna(0.0).astype(float)


def score_pre_race_scenarios(lane_frame: pd.DataFrame) -> dict[str, float]:
    """Build scenario context from pre-race features only.

    This intentionally does not use final rank, winning style, race ST, or mark
    passing order. It is safe for prediction-time reranking.
    """
    if lane_frame.empty:
        return _empty_context()

    frame = lane_frame.copy()
    frame["rank_prob_ctx"] = _numeric_first_available(frame, "win_probability_like")
    frame["top2_prob_ctx"] = _numeric_first_available(frame, "top2_prob")
    frame["top3_prob_ctx"] = _numeric_first_available(frame, "top3_prob")
    frame["exact1_prob_ctx"] = _numeric_first_available(frame, "exact1_prob", "win_prob")
    frame["nige_prob_ctx"] = _numeric_first_available(frame, "flow_prob_nige")
    frame["sashi_prob_ctx"] = _numeric_first_available(frame, "flow_prob_sashi")
    frame["makuri_prob_ctx"] = _numeric_first_available(frame, "flow_prob_makuri")
    frame["makurizashi_prob_ctx"] = _numeric_first_available(frame, "flow_prob_makurizashi")
    frame["venue_course_win_ctx"] = _numeric_first_available(
        frame, "venue_course_prev_win_rate", "venue_lane_prev_win_rate"
    )
    frame["venue_course_top2_ctx"] = _numeric_first_available(
        frame, "venue_course_prev_top2_rate", "venue_lane_prev_top2_rate"
    )
    frame["venue_course_top3_ctx"] = _numeric_first_available(
        frame, "venue_course_prev_top3_rate", "venue_lane_prev_top3_rate"
    )
    frame["machine_ctx"] = _numeric_first_available(
        frame, "motor_top2_rate_hist", "motor_prev_top3_rate", "motor_place_rate"
    ) + _numeric_first_available(frame, "boat_top2_rate_hist", "boat_prev_top3_rate", "boat_place_rate")
    frame["st_ctx"] = _numeric_first_available(frame, "avg_st_last5", "racer_prev_avg_st_5", "racer_prev_avg_st")
    frame["exhibition_ctx"] = _numeric_first_available(frame, "exhibition_time")
    frame["st_adv_ctx"] = _race_scale(frame["st_ctx"], lower_is_better=True)
    frame["exhibition_adv_ctx"] = _race_scale(frame["exhibition_ctx"], lower_is_better=True)
    frame["machine_adv_ctx"] = _race_scale(frame["machine_ctx"])

    attack_scores = _course_attack_scores(frame)
    outer_attack_scores = {lane: score for lane, score in attack_scores.items() if lane != 1}
    attack_lane = max(outer_attack_scores, key=outer_attack_scores.get) if outer_attack_scores else 0
    attack_pressure = float(outer_attack_scores.get(attack_lane, 0.0))

    lane1 = frame.loc[1] if 1 in frame.index else None
    if lane1 is None:
        escape_strength = 0.0
        lane1_top2 = 0.0
        lane1_venue_top2 = 0.0
    else:
        lane1_top2 = float(lane1["top2_prob_ctx"])
        lane1_venue_win = float(lane1["venue_course_win_ctx"])
        lane1_venue_top2 = float(lane1["venue_course_top2_ctx"])
        escape_strength = _clip01(
            (
                0.29 * float(lane1["exact1_prob_ctx"])
                + 0.18 * float(lane1["rank_prob_ctx"])
                + 0.21 * float(lane1["nige_prob_ctx"])
                + 0.07 * float(lane1["st_adv_ctx"])
                + 0.05 * float(lane1["exhibition_adv_ctx"])
                + 0.06 * float(lane1["machine_adv_ctx"])
                + 0.09 * lane1_venue_win
                + 0.05 * lane1_venue_top2
            )
            * (1.0 - 0.30 * attack_pressure)
        )

    sashi_pressure_2 = _lane_pressure(frame, 2, "sashi", venue_metric="top2")
    s2_makuri_pressure = _lane_pressure(frame, 2, "makuri", venue_metric="top2")
    s3_attack_pressure = max(
        _lane_pressure(frame, 3, "makuri", venue_metric="top2"),
        _lane_pressure(frame, 3, "makurizashi", venue_metric="top3"),
    )
    s4_course_attack_pressure = max(
        _lane_pressure(frame, 4, "makuri", venue_metric="top3"),
        _lane_pressure(frame, 4, "makurizashi", venue_metric="top3"),
    )
    s5_outside_attack_pressure = max(
        _lane_pressure(frame, 5, "makurizashi", venue_metric="top3"),
        _lane_pressure(frame, 6, "makurizashi", venue_metric="top3"),
    )
    makuri_pressure_3_4 = max(s3_attack_pressure, s4_course_attack_pressure)
    makurizashi_pressure = max(
        _lane_pressure(frame, lane, "makurizashi", venue_metric="top3") for lane in (3, 4, 5, 6)
    )
    outer_sweep_risk = _clip01(
        0.36 * max((attack_scores.get(lane, 0.0) for lane in (4, 5, 6)), default=0.0)
        + 0.28 * makurizashi_pressure
        + 0.22 * max((float(frame.loc[lane, "top3_prob_ctx"]) for lane in (4, 5, 6) if lane in frame.index), default=0.0)
        + 0.14 * max(
            (float(frame.loc[lane, "venue_course_top3_ctx"]) for lane in (4, 5, 6) if lane in frame.index),
            default=0.0,
        )
    )
    inner_collapse_risk = _clip01(
        (1.0 - escape_strength) * (0.36 + 0.26 * attack_pressure)
        + 0.22 * max(s2_makuri_pressure, s3_attack_pressure, s4_course_attack_pressure)
        + 0.10 * (1.0 - lane1_top2)
        + 0.06 * (1.0 - lane1_venue_top2)
    )
    venue_escape_win_rate = float(frame.loc[1, "venue_course_win_ctx"]) if 1 in frame.index else 0.0
    venue_outer_top3_rate = max(
        (float(frame.loc[lane, "venue_course_top3_ctx"]) for lane in (4, 5, 6) if lane in frame.index),
        default=0.0,
    )
    attack_values = sorted([float(score) for lane, score in attack_scores.items() if lane != 1], reverse=True)
    second_attack = attack_values[1] if len(attack_values) >= 2 else 0.0
    attack_margin = max(attack_pressure - second_attack, 0.0)
    close_attack_pressure = _clip01((0.12 - attack_margin) / 0.12) if attack_pressure >= 0.30 and second_attack >= 0.25 else 0.0
    chaos_pressure = _clip01(
        0.34 * close_attack_pressure
        + 0.24 * inner_collapse_risk * attack_pressure
        + 0.18 * max(s3_attack_pressure, s4_course_attack_pressure, s5_outside_attack_pressure)
        + 0.14 * outer_sweep_risk
        + 0.10 * (1.0 - escape_strength) * attack_pressure
    )

    return {
        "escape_strength": escape_strength,
        "inner_collapse_risk": inner_collapse_risk,
        "sashi_pressure_2": sashi_pressure_2,
        "makuri_pressure_3_4": makuri_pressure_3_4,
        "makurizashi_pressure": makurizashi_pressure,
        "outer_sweep_risk": outer_sweep_risk,
        "attack_lane": float(attack_lane),
        "attack_pressure": attack_pressure,
        "venue_escape_win_rate": venue_escape_win_rate,
        "venue_escape_top2_rate": lane1_venue_top2,
        "venue_outer_top3_rate": venue_outer_top3_rate,
        "s2_makuri_pressure": s2_makuri_pressure,
        "s3_attack_pressure": s3_attack_pressure,
        "s4_course_attack_pressure": s4_course_attack_pressure,
        "s4_makurizashi_pressure": s4_course_attack_pressure,
        "s5_outside_attack_pressure": s5_outside_attack_pressure,
        "s5_kado_makuri_pressure": s4_course_attack_pressure,
        "s6_outer_attack_pressure": s5_outside_attack_pressure,
        "s7_chain_pressure": chaos_pressure,
        "chaos_pressure": chaos_pressure,
        "attack_score_margin": attack_margin,
        "second_attack_pressure": second_attack,
    }


def _empty_context() -> dict[str, float]:
    return {
        "escape_strength": 0.0,
        "inner_collapse_risk": 0.0,
        "sashi_pressure_2": 0.0,
        "makuri_pressure_3_4": 0.0,
        "makurizashi_pressure": 0.0,
        "outer_sweep_risk": 0.0,
        "attack_lane": 0.0,
        "attack_pressure": 0.0,
        "venue_escape_win_rate": 0.0,
        "venue_escape_top2_rate": 0.0,
        "venue_outer_top3_rate": 0.0,
        "s2_makuri_pressure": 0.0,
        "s3_attack_pressure": 0.0,
        "s4_course_attack_pressure": 0.0,
        "s4_makurizashi_pressure": 0.0,
        "s5_outside_attack_pressure": 0.0,
        "s5_kado_makuri_pressure": 0.0,
        "s6_outer_attack_pressure": 0.0,
        "s7_chain_pressure": 0.0,
        "chaos_pressure": 0.0,
        "attack_score_margin": 0.0,
        "second_attack_pressure": 0.0,
    }


def _course_attack_scores(frame: pd.DataFrame) -> dict[int, float]:
    scores: dict[int, float] = {}
    for lane_value, row in frame.iterrows():
        lane = int(lane_value)
        lane_bias = {2: 0.07, 3: 0.10, 4: 0.12, 5: 0.06, 6: 0.04}.get(lane, 0.0)
        scores[lane] = _clip01(
            0.20 * float(row["rank_prob_ctx"])
            + 0.16 * float(row["top2_prob_ctx"])
            + 0.14 * float(row["sashi_prob_ctx"])
            + 0.22 * max(float(row["makuri_prob_ctx"]), float(row["makurizashi_prob_ctx"]))
            + 0.08 * float(row["st_adv_ctx"])
            + 0.05 * float(row["exhibition_adv_ctx"])
            + 0.04 * float(row["machine_adv_ctx"])
            + 0.07 * float(row["venue_course_top2_ctx"])
            + 0.04 * float(row["venue_course_top3_ctx"])
            + lane_bias
        )
    return scores


def _lane_pressure(frame: pd.DataFrame, lane: int, attack_kind: str, venue_metric: str) -> float:
    if lane not in frame.index:
        return 0.0
    row = frame.loc[lane]
    attack_prob = float(row["sashi_prob_ctx"]) if attack_kind == "sashi" else float(row[f"{attack_kind}_prob_ctx"])
    venue = float(row["venue_course_top2_ctx"] if venue_metric == "top2" else row["venue_course_top3_ctx"])
    return _clip01(
        0.42 * attack_prob
        + 0.17 * float(row["rank_prob_ctx"])
        + 0.14 * float(row["top2_prob_ctx"])
        + 0.09 * float(row["st_adv_ctx"])
        + 0.06 * float(row["exhibition_adv_ctx"])
        + 0.05 * float(row["machine_adv_ctx"])
        + 0.07 * venue
    )


def scenario_scores(scenario: dict[str, float]) -> dict[str, float]:
    model_scores = {
        f"S{i}": float(scenario.get(f"model_s{i}_score", np.nan))
        for i in range(7)
    }
    if all(np.isfinite(value) for value in model_scores.values()):
        total = sum(max(value, 0.0) for value in model_scores.values())
        if total > 0:
            return {key: _clip01(max(value, 0.0) / total) for key, value in model_scores.items()}

    escape = float(scenario.get("escape_strength", 0.0))
    collapse = float(scenario.get("inner_collapse_risk", 0.0))
    attack = float(scenario.get("attack_pressure", 0.0))
    sashi2 = float(scenario.get("sashi_pressure_2", 0.0))
    makuri2 = float(scenario.get("s2_makuri_pressure", 0.0))
    sashi_edge = max(sashi2 - 0.65 * makuri2, 0.0)
    makuri_edge = max(makuri2 - 0.75 * sashi2, 0.0)
    return {
        "S0": _clip01(0.68 * escape + 0.18 * (1.0 - collapse) + 0.14 * (1.0 - attack)),
        "S1": _clip01(
            0.50 * sashi_edge
            + 0.20 * sashi2
            + 0.14 * escape
            + 0.14 * float(scenario.get("venue_escape_top2_rate", 0.0))
            + 0.02 * (1.0 - collapse)
        ),
        "S2": _clip01(
            0.50 * makuri_edge
            + 0.22 * makuri2
            + 0.16 * collapse
            + 0.08 * attack
            + 0.04 * (1.0 - escape)
        ),
        "S3": _clip01(
            0.62 * float(scenario.get("s3_attack_pressure", 0.0))
            + 0.12 * float(scenario.get("makuri_pressure_3_4", 0.0))
            + 0.14 * collapse
            + 0.12 * attack
        ),
        "S4": _clip01(
            0.64 * float(scenario.get("s4_course_attack_pressure", 0.0))
            + 0.12 * float(scenario.get("makurizashi_pressure", 0.0))
            + 0.12 * attack
            + 0.12 * collapse
        ),
        "S5": _clip01(
            0.64 * float(scenario.get("s5_outside_attack_pressure", 0.0))
            + 0.18 * float(scenario.get("outer_sweep_risk", 0.0))
            + 0.10 * collapse
            + 0.08 * float(scenario.get("venue_outer_top3_rate", 0.0))
        ),
        "S6": _clip01(
            0.58 * float(scenario.get("chaos_pressure", scenario.get("s7_chain_pressure", 0.0)))
            + 0.18 * collapse
            + 0.14 * float(scenario.get("second_attack_pressure", 0.0))
            + 0.10 * (1.0 - escape) * attack
        ),
    }


def scenario_label(scenario: dict[str, float]) -> str:
    candidates = scenario_scores(scenario)
    if float(scenario.get("pattern_model_available", 0.0) or 0.0) > 0:
        short_id, _ = max(candidates.items(), key=lambda item: item[1])
        return SCENARIO_SHORT_TO_LABEL[short_id]
    normal_candidates = {key: value for key, value in candidates.items() if key != "S6"}
    ordered_normal = sorted(normal_candidates.items(), key=lambda item: item[1], reverse=True)
    short_id, strength = ordered_normal[0]
    second_strength = ordered_normal[1][1] if len(ordered_normal) > 1 else 0.0
    margin = float(strength - second_strength)
    chaos = float(scenario.get("chaos_pressure", scenario.get("s7_chain_pressure", 0.0)))

    if chaos >= 0.65:
        return "S6_MIXED_OTHER"
    if chaos >= 0.55 and margin < 0.05:
        return "S6_MIXED_OTHER"
    if strength < 0.28:
        return "S6_MIXED_OTHER"
    return SCENARIO_SHORT_TO_LABEL[short_id]


def normalize_scenario_label(label: str | None) -> str:
    if not label:
        return "S6_MIXED_OTHER"
    value = str(label)
    legacy_aliases = {
        "S0_IN_CONTROL": "S0_C1_ESCAPE_IN_WIN",
        "S1_COURSE2_SASHI": "S1_C2_SASHI_IN_REMAIN",
        "S2_COURSE2_MAKURI": "S2_C2_WIN_IN_OUT",
        "S3_COURSE3_ATTACK": "S3_C3_WIN_CENTER",
        "S4_COURSE4_ATTACK": "S4_C4_WIN_KADO",
        "S5_OUTSIDE_ATTACK": "S5_OUTER_WIN_OR_CHAIN",
        "S6_CHAOS": "S6_MIXED_OTHER",
    }
    if value in legacy_aliases:
        return legacy_aliases[value]
    if value in SCENARIO_LABEL_TO_SHORT:
        return value
    return SCENARIO_SHORT_TO_LABEL.get(value, "S6_MIXED_OTHER")


def scenario_display_name(label: str | None) -> str:
    return SCENARIO_DISPLAY_NAMES[normalize_scenario_label(label)]


def scenario_description(label: str | None) -> str:
    return SCENARIO_DESCRIPTIONS[normalize_scenario_label(label)]


def scenario_numeric_id(label: str | None) -> int:
    return SCENARIO_LABEL_TO_NUMERIC[normalize_scenario_label(label)]


def classify_result_pattern(race_df: pd.DataFrame) -> dict[str, object]:
    """Classify an observed race result into an objective finish pattern.

    This uses only post-race result columns and is not safe for prediction-time
    features. It is intended for validation and future target-label creation.
    """
    if race_df.empty or "finish_position" not in race_df.columns:
        return _unknown_result_pattern()

    frame = race_df.copy()
    frame["finish_position_num"] = pd.to_numeric(frame["finish_position"], errors="coerce")
    frame["lane_num"] = pd.to_numeric(frame.get("lane"), errors="coerce")
    frame["course_num"] = pd.to_numeric(frame.get("course"), errors="coerce")
    ranked = frame.dropna(subset=["finish_position_num"]).sort_values(["finish_position_num", "lane_num"])
    if ranked.empty:
        return _unknown_result_pattern()

    winner = ranked.iloc[0]
    winner_lane = _nullable_int(winner.get("lane_num"))
    winner_course = _nullable_int(winner.get("course_num"))
    winner_style = _style_code(winner.get("winning_style"))
    lane1_status = _lane1_result_status(frame)
    top3 = ranked.head(3)
    top3_lanes = [_nullable_int(value) for value in top3["lane_num"].tolist()]
    top3_courses = [_nullable_int(value) for value in top3["course_num"].tolist()]
    top3_shape = _top3_shape(top3_courses)
    scenario = _result_pattern_scenario(winner_course, winner_style, lane1_status, top3_shape)
    result_scenario = "__".join(
        [
            f"C{winner_course}" if winner_course is not None else "C_UNKNOWN",
            winner_style,
            lane1_status,
            top3_shape,
        ]
    )
    return {
        "scenario": scenario,
        "scenario_name": scenario_display_name(scenario),
        "result_scenario": result_scenario,
        "winner_lane": winner_lane,
        "winner_course": winner_course,
        "winner_style": winner_style,
        "lane1_status": lane1_status,
        "top3_shape": top3_shape,
        "top3_lanes": top3_lanes,
        "top3_courses": top3_courses,
        "is_outer_chain": top3_shape == "OUTER_CHAIN",
        "is_inner_dominant": top3_shape == "INNER_TOP3",
    }


def _unknown_result_pattern() -> dict[str, object]:
    return {
        "scenario": "S6_MIXED_OTHER",
        "scenario_name": scenario_display_name("S6_MIXED_OTHER"),
        "result_scenario": "C_UNKNOWN__UNKNOWN__UNKNOWN__UNKNOWN",
        "winner_lane": None,
        "winner_course": None,
        "winner_style": "UNKNOWN",
        "lane1_status": "UNKNOWN",
        "top3_shape": "UNKNOWN",
        "top3_lanes": [],
        "top3_courses": [],
        "is_outer_chain": False,
        "is_inner_dominant": False,
    }


def _nullable_int(value: object) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return int(numeric)


def _style_code(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    if "逃" in text:
        return "ESCAPE"
    if "まくり差" in text:
        return "MAKURI_SASHI"
    if "まくり" in text:
        return "MAKURI"
    if "差" in text:
        return "SASHI"
    if "抜" in text:
        return "NUKI"
    if "恵" in text:
        return "MEGUMARE"
    return "UNKNOWN"


def _lane1_result_status(frame: pd.DataFrame) -> str:
    if "lane" not in frame.columns:
        return "UNKNOWN"
    lane1 = frame[pd.to_numeric(frame["lane"], errors="coerce") == 1]
    if lane1.empty:
        return "UNKNOWN"
    finish = pd.to_numeric(lane1.iloc[0].get("finish_position"), errors="coerce")
    status = str(lane1.iloc[0].get("finish_status") or "")
    if any(token in status for token in ("F", "L", "K", "S", "転", "失")):
        return "IN_ACCIDENT"
    if pd.isna(finish):
        return "UNKNOWN"
    if int(finish) == 1:
        return "IN_WIN"
    if int(finish) in (2, 3):
        return "IN_REMAIN"
    return "IN_OUT"


def _top3_shape(courses: list[int | None]) -> str:
    known = [course for course in courses if course is not None]
    if len(known) < 3:
        return "UNKNOWN"
    outer_count = sum(course >= 4 for course in known)
    center_outer_count = sum(course >= 3 for course in known)
    if all(course <= 3 for course in known):
        return "INNER_TOP3"
    if outer_count >= 2:
        return "OUTER_CHAIN"
    if center_outer_count >= 2:
        return "CENTER_OUTER"
    if max(known) <= 4:
        return "INNER_CENTER"
    return "MIXED"


def _result_pattern_scenario(
    winner_course: int | None,
    winner_style: str,
    lane1_status: str,
    top3_shape: str,
) -> str:
    if winner_course == 1:
        return "S0_C1_ESCAPE_IN_WIN"
    if winner_course == 2 and winner_style == "SASHI" and lane1_status in {"IN_WIN", "IN_REMAIN"}:
        return "S1_C2_SASHI_IN_REMAIN"
    if winner_course == 2:
        return "S2_C2_WIN_IN_OUT"
    if winner_course == 3:
        return "S3_C3_WIN_CENTER"
    if winner_course == 4:
        return "S4_C4_WIN_KADO"
    if winner_course in {5, 6} or top3_shape == "OUTER_CHAIN":
        return "S5_OUTER_WIN_OR_CHAIN"
    return "S6_MIXED_OTHER"


def scenario_feature_values(scenario: dict[str, float]) -> dict[str, float]:
    label = scenario_label(scenario)
    scores = scenario_scores(scenario)
    legacy_aliases = {
        "S0": "scenario_s0_in_control_score",
        "S1": "scenario_s1_course2_sashi_score",
        "S2": "scenario_s2_course2_makuri_score",
        "S3": "scenario_s3_course3_attack_score",
        "S4": "scenario_s4_course4_attack_score",
        "S5": "scenario_s5_outside_attack_score",
        "S6": "scenario_s6_chaos_score",
    }
    features: dict[str, float] = {}
    for short_id, score in scores.items():
        canonical = SCENARIO_SHORT_TO_LABEL[short_id].lower()
        features[f"scenario_{canonical}_score"] = score
        features[f"scenario_{short_id.lower()}_score"] = score
        features[legacy_aliases[short_id]] = score
    features["scenario_s7_score"] = 0.0
    features["scenario_id_numeric"] = float(SCENARIO_LABEL_TO_NUMERIC[label])
    return features


def scenario_line_features(
    scenario: dict[str, float],
    first_lane: int,
    second_lane: int | None = None,
    third_lane: int | None = None,
) -> dict[str, float]:
    lanes = [lane for lane in (first_lane, second_lane, third_lane) if lane is not None]
    attack_lane = int(scenario.get("attack_lane", 0))
    attack_in_line = attack_lane in lanes
    escape_line_fit = float(first_lane == 1) * float(scenario["escape_strength"])
    sashi_line_fit = float(first_lane == 2 and (second_lane in (1, 3, 4))) * float(scenario["sashi_pressure_2"])
    makuri_line_fit = float(first_lane in (2, 3, 4)) * max(
        float(scenario.get("s2_makuri_pressure", 0.0)),
        float(scenario.get("s3_attack_pressure", 0.0)),
        float(scenario.get("s4_course_attack_pressure", 0.0)),
    )
    makurizashi_line_fit = (
        float(first_lane in (3, 4, 5, 6) and (second_lane is None or second_lane <= 4))
        * float(scenario["makurizashi_pressure"])
    )
    outer_follow_fit = float(any(lane >= 4 for lane in lanes[1:])) * max(
        float(scenario["outer_sweep_risk"]),
        float(scenario.get("s5_outside_attack_pressure", 0.0)),
    )
    attack_line_fit = (
        float(first_lane == attack_lane or (second_lane == attack_lane and first_lane in (1, 2)))
        * float(scenario["attack_pressure"])
    )
    inner_line = first_lane == 1 and second_lane in (2, 3, None)
    scenario_mismatch_penalty = _clip01(
        float(scenario["escape_strength"] > 0.45 and first_lane != 1) * float(scenario["escape_strength"])
        + float(scenario["inner_collapse_risk"] > 0.45 and inner_line) * float(scenario["inner_collapse_risk"])
        + float(scenario["attack_pressure"] > 0.45 and not attack_in_line) * float(scenario["attack_pressure"])
    )
    return {
        "escape_line_fit": escape_line_fit,
        "sashi_line_fit": sashi_line_fit,
        "makuri_line_fit": makuri_line_fit,
        "makurizashi_line_fit": makurizashi_line_fit,
        "outer_follow_fit": outer_follow_fit,
        "attack_line_fit": attack_line_fit,
        "scenario_mismatch_penalty": scenario_mismatch_penalty,
    }
