from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api import (  # noqa: E402
    backfill_rowdata_files,
    build_dataset_from_rowdata_streaming,
    load_bundle,
    predict_today,
)
from src.drive_restore import (  # noqa: E402
    DEFAULT_ARTIFACTS_DRIVE_FILE_URL,
    DEFAULT_DATA_DRIVE_FILE_URL,
    DEFAULT_ROWDATA_DRIVE_FILE_URL,
    download_and_restore_packages,
)
from src.models.ranker import get_artifact_paths, load_config  # noqa: E402
from src.today_schedule import (  # noqa: E402
    choose_default_today_race_no,
    choose_default_today_venue,
    current_jst_date,
    fetch_daily_race_schedule,
    filter_future_schedule,
)

VENUES = {
    "01": "桐生",
    "02": "戸田",
    "03": "江戸川",
    "04": "平和島",
    "05": "多摩川",
    "06": "浜名湖",
    "07": "蒲郡",
    "08": "常滑",
    "09": "津",
    "10": "三国",
    "11": "びわこ",
    "12": "住之江",
    "13": "尼崎",
    "14": "鳴門",
    "15": "丸亀",
    "16": "児島",
    "17": "宮島",
    "18": "徳山",
    "19": "下関",
    "20": "若松",
    "21": "芦屋",
    "22": "福岡",
    "23": "唐津",
    "24": "大村",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else repo_root() / target


@st.cache_resource(show_spinner=False)
def load_cached_bundle(config_path: str):
    return load_bundle(config_path)


@st.cache_data(show_spinner=False, ttl=300)
def load_model_accuracy_summary(config_path: str) -> dict[str, object] | None:
    try:
        config = load_config(_resolve_repo_path(config_path))
        artifacts = get_artifact_paths(config)
        metrics_path = artifacts["metrics_path"]
        if not metrics_path.is_absolute():
            metrics_path = repo_root() / metrics_path
        if not metrics_path.exists():
            return None
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    section = None
    for root_key in ("trifecta", "trifecta_v1_metrics"):
        root = metrics.get(root_key, {})
        if not isinstance(root, dict):
            continue
        for split_key in ("valid_calibrated", "valid_raw"):
            candidate = root.get(split_key)
            if isinstance(candidate, dict) and candidate:
                section = candidate
                break
        if section is not None:
            break
    if section is None:
        return None

    recovery_metrics = section.get("uniform_ticket_recovery_metrics", {})
    if not isinstance(recovery_metrics, dict):
        recovery_metrics = {}

    rows = []
    for label, key, recovery_key in (
        ("Top1", "top1_hit_rate", "top1"),
        ("Top3", "top3_hit_rate", "top3"),
        ("Top5", "top5_hit_rate", "top5"),
        ("Top10", "top10_hit_rate", "top10"),
        ("Top12", "top12_hit_rate", "top12"),
        ("Top20", "top20_hit_rate", "top20"),
        ("Top25", "top25_hit_rate", "top25"),
    ):
        recovery_entry = recovery_metrics.get(recovery_key, {})
        if not isinstance(recovery_entry, dict):
            recovery_entry = {}
        hit_rate = section.get(key, recovery_entry.get("hit_rate"))
        recovery_rate = recovery_entry.get("recovery_rate")
        if hit_rate is not None or recovery_rate is not None:
            rows.append(
                {
                    "label": label,
                    "hit_rate": float(hit_rate) if hit_rate is not None else None,
                    "recovery_rate": float(recovery_rate) if recovery_rate is not None else None,
                }
            )
    if not rows:
        return None
    return {
        "race_count": int(float(section.get("race_count", 0.0) or 0.0)),
        "rows": rows,
    }


@st.cache_data(show_spinner=False, ttl=300)
def load_today_schedule():
    return fetch_daily_race_schedule()


@st.cache_data(show_spinner=False, ttl=60)
def load_exhibition_courses(venue: str, race_no: int, race_date: date) -> tuple[int, int, int, int, int, int] | None:
    from src.live import fetch_boatrace_exhibition_courses

    return fetch_boatrace_exhibition_courses(race_date, venue, race_no)


def _prediction_venue_key() -> str:
    return "prediction_selected_venue"


def _prediction_race_key() -> str:
    return "prediction_selected_race_no"


def _parse_course_assignment_text(value: str) -> tuple[int, int, int, int, int, int] | None:
    normalized = value.strip().replace("，", ",").replace("、", ",").replace(" ", "")
    if not normalized:
        return None
    parts = normalized.split(",") if "," in normalized else list(normalized)
    if len(parts) != 6 or any(not part.isdigit() for part in parts):
        return None
    courses = tuple(int(part) for part in parts)
    if sorted(courses) != [1, 2, 3, 4, 5, 6]:
        return None
    return courses  # type: ignore[return-value]


def _format_courses(courses: tuple[int, int, int, int, int, int]) -> str:
    return "".join(str(course) for course in courses)


def _render_course_inputs(
    scope: str,
    exhibition_courses: tuple[int, int, int, int, int, int] | None,
) -> tuple[tuple[int, int, int, int, int, int], bool]:
    lane_courses = (1, 2, 3, 4, 5, 6)
    default_courses = exhibition_courses or lane_courses
    default_source = "展示進入" if exhibition_courses is not None else "枠なり"
    mode_options = ["展示進入を使う・編集", "枠なり"] if exhibition_courses is not None else ["枠なり", "進入を手入力"]
    with st.expander("進入コース設定", expanded=False):
        if exhibition_courses is not None:
            st.success(f"展示進入を取得しました: {_format_courses(exhibition_courses)}")
        else:
            st.info("展示進入を取得できない場合は、枠なりを初期値にします。")
        mode = st.radio(
            "予測に使う進入",
            options=mode_options,
            horizontal=True,
            key=f"prediction_course_mode_{scope}_{default_source}_{_format_courses(default_courses)}",
        )
        if mode == "枠なり":
            courses = lane_courses
            is_valid = True
            st.info("枠番と同じ進入コースで予測します。")
        else:
            raw_value = st.text_input(
                "展示進入（1号艇から順に入力）",
                value=_format_courses(default_courses),
                help="例: 213456 は 1号艇が2コース、2号艇が1コースです。カンマ区切りの 2,1,3,4,5,6 も使えます。",
                key=f"prediction_course_text_{scope}_{default_source}_{_format_courses(default_courses)}",
            )
            parsed = _parse_course_assignment_text(raw_value)
            is_valid = parsed is not None
            courses = parsed if parsed is not None else default_courses
            if not is_valid:
                st.error("1〜6を重複なく6つ指定してください。例: 213456")

        preview = [
            {"艇番": f"{lane}号艇", "予測に使う進入": f"{course}コース"}
            for lane, course in enumerate(courses, start=1)
        ]
        st.dataframe(preview, use_container_width=True, hide_index=True)
        st.caption("この進入コースを使って、場別コース成績・攻め圧・三連単確率などを再計算します。")
    return courses, is_valid


def _set_default_prediction_race(schedule: dict[str, dict[int, object]]) -> None:
    venue = st.session_state.get(_prediction_venue_key(), "15")
    st.session_state[_prediction_race_key()] = choose_default_today_race_no(schedule, venue)


def _format_percent(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _percent_value(value: object) -> float | None:
    try:
        return float(value) * 100.0
    except (TypeError, ValueError):
        return None


def _dataframe_height(row_count: int, row_height: int = 34, header_height: int = 38) -> int:
    return header_height + max(int(row_count), 1) * row_height + 8


def _inject_table_style() -> None:
    st.markdown(
        """
<style>
div[data-testid="stDataFrame"] [role="gridcell"],
div[data-testid="stDataFrame"] [role="columnheader"] {
  font-size: 1rem;
  justify-content: center;
  text-align: center;
}
div[data-testid="stDataFrame"] [role="gridcell"] > div,
div[data-testid="stDataFrame"] [role="columnheader"] > div {
  justify-content: center;
  text-align: center;
}
.centered-dataframe table {
  width: 100%;
  border-collapse: collapse;
}
.centered-dataframe th,
.centered-dataframe td {
  padding: 0.45rem 0.6rem;
  text-align: center !important;
  vertical-align: middle !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _select_and_rename_columns(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> pd.DataFrame:
    available = [(source, label) for source, label in columns if source in frame.columns]
    extra_labels = {
        "is_darkhorse_candidate": "穴候補",
    }
    selected_sources = {source for source, _ in available}
    available.extend(
        (source, label)
        for source, label in extra_labels.items()
        if source in frame.columns and source not in selected_sources
    )
    if not available:
        return frame.copy()
    selected = frame[[source for source, _ in available]].copy()
    return selected.rename(columns={source: label for source, label in available})


def _format_ranking_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _select_and_rename_columns(
        frame,
        [
            ("lane", "艇番"),
            ("course", "進入"),
            ("predicted_rank", "予想順位"),
            ("class_name", "級別"),
            ("racer_name", "氏名"),
            ("branch", "支部"),
            ("age", "年齢"),
            ("motor_no", "モーター"),
            ("boat_no", "ボート"),
            ("exhibition_time", "展示タイム"),
            ("start_timing", "ST"),
            ("score", "総合スコア"),
        ],
    )


def _format_trifecta_frame(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy().reset_index(drop=True)
    probability_col = "adjusted_probability" if "adjusted_probability" in formatted.columns else "probability"
    display_probability_col = "display_prediction_probability"
    formatted["display_rank_no"] = range(1, len(formatted) + 1)
    if probability_col in formatted.columns:
        formatted[display_probability_col] = pd.to_numeric(formatted[probability_col], errors="coerce") * 100.0
    columns = [
        ("display_rank_no", "No"),
        ("trifecta", "買い目"),
        (display_probability_col, "予想確率(%)"),
        ("odds", "現在オッズ"),
    ]
    for source, _label in columns:
        if source not in formatted.columns:
            formatted[source] = None
    formatted = formatted[[source for source, _label in columns]]
    return formatted.rename(columns={source: label for source, label in columns})


def _ranking_column_config(frame: pd.DataFrame) -> dict[str, Any]:
    scores = pd.to_numeric(frame.get("総合スコア", pd.Series(dtype=float)), errors="coerce").dropna()
    if scores.empty:
        min_value, max_value = 0.0, 1.0
    else:
        min_value = min(0.0, float(scores.min()))
        max_value = max(1.0, float(scores.max()))
        if min_value == max_value:
            max_value = min_value + 1.0
    return {
        "総合スコア": st.column_config.ProgressColumn(
            "総合スコア",
            format="%.4f",
            min_value=min_value,
            max_value=max_value,
        )
    }


def _trifecta_column_config(frame: pd.DataFrame) -> dict[str, Any]:
    probabilities = pd.to_numeric(frame.get("予想確率(%)", pd.Series(dtype=float)), errors="coerce").dropna()
    probability_max = 30.0 if probabilities.empty or float(probabilities.max()) <= 30.0 else 100.0
    return {
        "予想確率(%)": st.column_config.ProgressColumn(
            "予想確率(%)",
            format="%.2f%%",
            min_value=0.0,
            max_value=probability_max,
        ),
        "現在オッズ": st.column_config.NumberColumn("現在オッズ", format="%.1f"),
    }


def _render_centered_prediction_table(
    frame: pd.DataFrame,
    *,
    formats: dict[str, str] | None = None,
    bars: dict[str, tuple[float, float, str]] | None = None,
) -> None:
    styler = frame.style.hide(axis="index").format(formats or {}, na_rep="-")
    for column, (min_value, max_value, color) in (bars or {}).items():
        if column in frame.columns:
            styler = styler.bar(subset=[column], vmin=min_value, vmax=max_value, color=color)
    styler = styler.set_properties(
        **{
            "font-size": "1rem",
            "text-align": "center",
            "vertical-align": "middle",
        }
    )
    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("font-size", "1rem"),
                    ("font-weight", "700"),
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                ],
            },
        ]
    )
    st.markdown(f'<div class="centered-dataframe">{styler.to_html()}</div>', unsafe_allow_html=True)


def _render_ranking_table(frame: pd.DataFrame) -> None:
    score_col = frame.columns[-1] if len(frame.columns) else None
    bars: dict[str, tuple[float, float, str]] = {}
    formats: dict[str, str] = {}
    if score_col is not None:
        scores = pd.to_numeric(frame[score_col], errors="coerce").dropna()
        if scores.empty:
            min_value, max_value = 0.0, 1.0
        else:
            min_value = min(0.0, float(scores.min()))
            max_value = max(1.0, float(scores.max()))
            if min_value == max_value:
                max_value = min_value + 1.0
        bars[str(score_col)] = (min_value, max_value, "#d7e9ff")
        formats[str(score_col)] = "{:.4f}"
    _render_centered_prediction_table(frame, formats=formats, bars=bars)


def _render_trifecta_table(frame: pd.DataFrame) -> None:
    probability_col = frame.columns[2] if len(frame.columns) > 2 else None
    odds_col = frame.columns[3] if len(frame.columns) > 3 else None
    formats: dict[str, str] = {}
    bars: dict[str, tuple[float, float, str]] = {}
    if probability_col is not None:
        probabilities = pd.to_numeric(frame[probability_col], errors="coerce").dropna()
        probability_max = 30.0 if probabilities.empty or float(probabilities.max()) <= 30.0 else 100.0
        bars[str(probability_col)] = (0.0, probability_max, "#dff3df")
        formats[str(probability_col)] = "{:.2f}%"
    if odds_col is not None:
        formats[str(odds_col)] = "{:.1f}"
    _render_centered_prediction_table(frame, formats=formats, bars=bars)


def _trifecta_display_frame(prediction: Any) -> pd.DataFrame:
    frame = prediction.odds if prediction.odds is not None and not prediction.odds.empty else prediction.trifecta
    probability_col = "adjusted_probability" if "adjusted_probability" in frame.columns else "probability"
    return frame.sort_values(probability_col, ascending=False).head(20).reset_index(drop=True)


def _prediction_race_value(prediction: Any, column: str, default: Any = 0.0) -> Any:
    frame = prediction.odds if prediction.odds is not None and not prediction.odds.empty else prediction.trifecta
    if frame is None or frame.empty or column not in frame.columns:
        return default
    values = frame[column].dropna()
    if values.empty:
        return default
    return values.iloc[0]


def _render_prediction_guide() -> None:
    with st.expander("買い目判断と予測の見方", expanded=False):
        st.markdown(
            """
**3連単予想一覧**

- `No` は予想確率が高い順の順位です。
- `予想確率(%)` は補正後の3連単確率です。1レース120通りを合計すると100%になるように正規化しています。
- `オッズ` は取得できた場合だけ表示します。

**上部の指標**

- `Top3的中確率` は、予想Top3内に正解3連単が入る見込みを過去データから学習した確率です。
- `イン勝ち確率` は、1コース艇が1着になる見込みです。
- `イン沈み確率` は、1コース艇が3着以下になる見込みです。

**期待値と買い判断**

- 内部の期待値は、一覧の `予想確率(%)` を確率へ戻し、現在オッズを掛けて計算しています。
- 買い判断は期待値計算には残していますが、一覧では予想順位・確率・オッズを優先して表示します。
- Top3的中確率が低いレースや、イン勝ち確率が低く展開が割れやすいレースは、予想順位が下がりやすいため慎重に見てください。
"""
        )


def _render_model_accuracy_summary(config_path: str) -> None:
    summary = load_model_accuracy_summary(config_path)
    if not summary:
        return
    rows = list(summary.get("rows", []))
    if not rows:
        return

    with st.expander("現在のモデル精度（検証データ）", expanded=False):
        _render_model_accuracy_summary_content(summary, rows)


def _render_model_accuracy_summary_content(summary: dict[str, Any], rows: list[Any]) -> None:
    race_count = int(summary.get("race_count", 0) or 0)
    st.caption(f"評価レース数: {race_count:,}")
    chart_rows = []
    for row in rows:
        label = str(row["label"])
        hit_rate = _percent_value(row.get("hit_rate"))
        recovery_rate = _percent_value(row.get("recovery_rate"))
        if hit_rate is not None:
            chart_rows.append({"TopN": label, "指標": "的中率", "値": hit_rate})
        if recovery_rate is not None:
            chart_rows.append({"TopN": label, "指標": "回収率", "値": recovery_rate})
    if chart_rows:
        chart_frame = pd.DataFrame(chart_rows)
        top_order = [str(row["label"]) for row in rows]
        y_axis = alt.Axis(labelOverlap=False, labelLimit=80)
        chart_height = _dataframe_height(len(rows), row_height=30, header_height=42)
        chart_columns = st.columns(2)
        for column, (metric_name, color) in zip(chart_columns, (("的中率", "#16a34a"), ("回収率", "#f97316"))):
            metric_frame = chart_frame[chart_frame["指標"] == metric_name]
            if metric_frame.empty:
                continue
            with column:
                st.caption(metric_name)
                bars = (
                    alt.Chart(metric_frame)
                    .mark_bar(color=color, cornerRadiusEnd=3)
                    .encode(
                        x=alt.X("値:Q", title=f"{metric_name}(%)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("TopN:N", title="", sort=top_order, axis=y_axis),
                        tooltip=[
                            alt.Tooltip("TopN:N", title="TopN"),
                            alt.Tooltip("指標:N", title="指標"),
                            alt.Tooltip("値:Q", title="値(%)", format=".1f"),
                        ],
                    )
                )
                labels = (
                    alt.Chart(metric_frame)
                    .mark_text(align="left", baseline="middle", dx=4, color="#111827")
                    .encode(
                        x=alt.X("値:Q", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("TopN:N", sort=top_order, axis=y_axis),
                        text=alt.Text("値:Q", format=".1f"),
                    )
                )
                st.altair_chart((bars + labels).properties(height=chart_height), use_container_width=True)
    st.caption("的中率は正解3連単が予測TopN以内に入った割合、回収率は各TopNを均等買いした場合の検証値です。実際の当日レースごとの結果を保証するものではありません。")


def _run_python_cli(command_label: str, args: Sequence[str]) -> tuple[int, str]:
    placeholder = st.empty()
    lines: list[str] = []
    full_command = [sys.executable, *args]

    with st.spinner(f"{command_label} を実行しています..."):
        process = subprocess.Popen(
            full_command,
            cwd=repo_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip("\n"))
            placeholder.code("\n".join(lines[-200:]) or "(no output yet)")
        return_code = process.wait()

    output_text = "\n".join(lines)
    if not lines:
        placeholder.code("(no output)")
    return return_code, output_text


def _render_command_result(command_label: str, return_code: int, output_text: str) -> None:
    if return_code == 0:
        st.success(f"{command_label} が完了しました。")
    else:
        st.error(f"{command_label} が失敗しました。終了コード: {return_code}")

    with st.expander("実行ログ", expanded=return_code != 0):
        st.code(output_text or "(no output)")


def _format_recent_backtest_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _select_and_rename_columns(
        frame,
        [
            ("race_date", "日付"),
            ("race_count", "対象レース数"),
            ("hit_races", "的中レース数"),
            ("hit_rate", "正解率"),
            ("total_stake", "購入額"),
            ("total_return", "払戻額"),
            ("recovery_rate", "回収率"),
        ],
    )


def _format_recent_backtest_race_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _select_and_rename_columns(
        frame,
        [
            ("race_date", "日付"),
            ("venue", "レース場"),
            ("race_no", "レースNo"),
            ("predicted_tickets", "購入買い目"),
            ("actual_trifecta", "結果3連単"),
            ("actual_payout", "結果払戻"),
            ("race_hit", "的中"),
            ("total_stake", "購入額"),
            ("total_return", "払戻額"),
            ("recovery_rate", "回収率"),
        ],
    )


def _format_recent_backtest_ticket_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _select_and_rename_columns(
        frame,
        [
            ("race_date", "日付"),
            ("venue", "レース場"),
            ("race_no", "レースNo"),
            ("prediction_rank", "予想順位"),
            ("trifecta", "購入買い目"),
            ("probability", "予想確率"),
            ("actual_trifecta", "結果3連単"),
            ("trifecta_payout", "結果払戻"),
            ("hit", "的中"),
            ("stake_amount", "購入額"),
            ("return_amount", "払戻額"),
        ],
    )


def _render_recent_backtest_report(report: dict) -> None:
    summary = report.get("summary", {})
    race_count = int(summary.get("race_count", 0))
    hit_races = int(summary.get("hit_races", 0))
    ticket_count = int(summary.get("ticket_count", 0))
    total_stake = float(summary.get("total_stake", 0.0))
    total_return = float(summary.get("total_return", 0.0))

    st.success("過去1週間分の予測と結果比較が完了しました。")
    st.caption(
        f"対象期間: {summary.get('start_date', '-')} ～ {summary.get('end_date', '-')} / "
        f"利用データ最終日: {summary.get('latest_available_date', '-')}"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("対象レース数", f"{race_count}")
    col2.metric("的中レース数", f"{hit_races}")
    col3.metric("正解率", f"{float(summary.get('race_hit_rate', 0.0)) * 100:.1f}%")
    col4.metric("回収率", f"{float(summary.get('recovery_rate', 0.0)) * 100:.1f}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("購入点数", f"{ticket_count}")
    col6.metric("購入額", f"{total_stake:,.0f}円")
    col7.metric("払戻額", f"{total_return:,.0f}円")
    col8.metric("Top12内率", f"{float(summary.get('top12_hit_rate', 0.0)) * 100:.1f}%")

    missing_payout_files = summary.get("missing_payout_files", [])
    if missing_payout_files:
        st.warning(f"払戻ファイルが見つからなかった日があります: {len(missing_payout_files)}件")

    daily_df = pd.DataFrame(report.get("daily_summary", []))
    if not daily_df.empty:
        st.markdown("**日別集計**")
        st.dataframe(_format_recent_backtest_daily_frame(daily_df), use_container_width=True, hide_index=True)

    race_df = pd.DataFrame(report.get("race_summary", []))
    if not race_df.empty:
        st.markdown("**レース別結果**")
        st.dataframe(_format_recent_backtest_race_frame(race_df), use_container_width=True, hide_index=True)

    ticket_df = pd.DataFrame(report.get("ticket_details", []))
    if not ticket_df.empty:
        st.markdown("**購入明細**")
        st.dataframe(_format_recent_backtest_ticket_frame(ticket_df), use_container_width=True, hide_index=True)


def render_prediction_tab() -> None:
    st.subheader("当日レース予測")
    st.caption("本日の開催レースから予測対象を選び、順位予測と三連単予想を表示します。")

    schedule_fetch_error: Exception | None = None
    try:
        schedule = filter_future_schedule(load_today_schedule())
    except Exception as exc:  # pragma: no cover
        schedule_fetch_error = exc
        schedule = {}

    if schedule_fetch_error is not None:
        st.warning(
            f"本日の開催情報の取得に失敗しました。mbrace への接続または応答に問題があります: {schedule_fetch_error}"
        )
        if st.button("mbrace取得を再試行", key="retry_today_schedule_local"):
            load_today_schedule.clear()
            st.rerun()
        return

    if not schedule:
        st.info("現在時刻以降に本日開催予定のレースはありません。")
        return

    venue_options = sorted(schedule.keys())
    venue_key = _prediction_venue_key()
    race_key = _prediction_race_key()
    if st.session_state.get(venue_key) not in venue_options:
        st.session_state[venue_key] = choose_default_today_venue(schedule)

    config_path = st.text_input("設定ファイル", value="configs/train.yaml")
    selected = st.selectbox(
        "レース場",
        options=venue_options,
        format_func=lambda code: f"{code} {VENUES.get(code, code)}",
        key=venue_key,
        on_change=_set_default_prediction_race,
        args=(schedule,),
    )

    race_options = sorted(schedule.get(selected, {}).keys())
    if st.session_state.get(race_key) not in race_options:
        st.session_state[race_key] = choose_default_today_race_no(schedule, selected)
        if st.session_state[race_key] not in race_options:
            st.session_state[race_key] = race_options[-1]

    race_no = st.selectbox(
        "レースNo",
        options=race_options,
        format_func=lambda value: f"{int(value)}R",
        key=race_key,
    )
    target_date = current_jst_date()
    exhibition_courses = load_exhibition_courses(selected, int(race_no), target_date)
    course_overrides, course_overrides_valid = _render_course_inputs(
        f"{selected}_{int(race_no):02d}_{target_date.isoformat()}",
        exhibition_courses,
    )
    submitted = st.button("予測を実行", disabled=not course_overrides_valid)

    if not submitted:
        return

    try:
        load_cached_bundle.clear()
        with st.spinner("予測を実行しています..."):
            prediction = predict_today(
                venue=selected,
                race_no=int(race_no),
                config_path=config_path,
                course_overrides=course_overrides,
            )
    except Exception as exc:  # pragma: no cover
        st.error(f"予測に失敗しました: {exc}")
        return

    st.success("予測が完了しました。")
    _render_model_accuracy_summary(config_path)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(
            "Top3的中確率",
            f"{float(_prediction_race_value(prediction, 'top3_hit_probability', 0.0)) * 100:.1f}%",
            str(_prediction_race_value(prediction, "top3_hit_probability_label", "-")),
        )
    with metric_col2:
        st.metric(
            "イン勝ち確率",
            f"{float(_prediction_race_value(prediction, 'in_win_probability', 0.0)) * 100:.1f}%",
            str(_prediction_race_value(prediction, "in_win_probability_label", "-")),
        )
    with metric_col3:
        st.metric(
            "イン沈み確率",
            f"{float(_prediction_race_value(prediction, 'in_collapse_probability', 0.0)) * 100:.1f}%",
            str(_prediction_race_value(prediction, "in_collapse_probability_label", "-")),
        )
    _render_prediction_guide()

    st.markdown("**順位予測**")
    ranking_frame = _format_ranking_frame(prediction.ranking)
    _render_ranking_table(ranking_frame)

    st.markdown("**三連単候補**")
    trifecta_display = _trifecta_display_frame(prediction)
    trifecta_frame = _format_trifecta_frame(trifecta_display)
    _render_trifecta_table(trifecta_frame)


def render_download_tab() -> None:
    st.subheader("共有データ取得")
    st.caption("Google Drive 共有リンクから rowdata / data / artifacts を復元します。")

    with st.form("download_form"):
        project_root_value = st.text_input("project-root", value=".")
        rowdata_url = st.text_input("rowdata.zip URL", value=DEFAULT_ROWDATA_DRIVE_FILE_URL)
        data_url = st.text_input("data.zip URL", value=DEFAULT_DATA_DRIVE_FILE_URL)
        artifacts_url = st.text_input("artifacts.zip URL", value=DEFAULT_ARTIFACTS_DRIVE_FILE_URL)
        restore_rowdata = st.checkbox("rowdata を復元", value=True)
        restore_data = st.checkbox("data を復元", value=True)
        restore_artifacts = st.checkbox("artifacts を復元", value=True)
        submitted = st.form_submit_button("ダウンロードして復元")

    if not submitted:
        return

    try:
        with st.spinner("ダウンロードして復元しています..."):
            report = download_and_restore_packages(
                project_root=Path(project_root_value),
                rowdata_drive_file_url=rowdata_url,
                data_drive_file_url=data_url,
                artifacts_drive_file_url=artifacts_url,
                restore_rowdata=restore_rowdata,
                restore_data=restore_data,
                restore_artifacts=restore_artifacts,
            )
    except Exception as exc:  # pragma: no cover
        st.error(f"復元に失敗しました: {exc}")
        return

    st.success("復元が完了しました。")
    st.json(report.to_dict())


def render_backfill_tab() -> None:
    st.subheader("rowdata 更新")
    st.caption("mbrace から不足している B/K テキストを取得します。")

    with st.form("backfill_form"):
        rowdata_dir = st.text_input("rowdata フォルダ", value="rowdata")
        start_date_text = st.text_input(
            "開始日",
            value="",
            help="未入力なら既存ファイルの最新日の翌日から更新します。例: 2026-05-14",
        )
        end_date = st.date_input("終了日", value=date.today())
        kinds = st.multiselect(
            "対象種別",
            options=["B", "K"],
            default=["B", "K"],
            help="B は番組表、K はレース結果です。",
        )
        overwrite = st.checkbox("既存ファイルを上書きする", value=False)
        submitted = st.form_submit_button("更新を実行")

    if not submitted:
        return
    if not kinds:
        st.warning("B または K を選択してください。")
        return

    try:
        normalized_start = start_date_text.strip() or None
        with st.spinner("rowdata を更新しています..."):
            report = backfill_rowdata_files(
                rowdata_dir=rowdata_dir,
                start_date=normalized_start,
                end_date=end_date,
                kinds=kinds,
                overwrite=overwrite,
            )
    except Exception as exc:  # pragma: no cover
        st.error(f"更新に失敗しました: {exc}")
        return

    st.success("更新が完了しました。")
    st.json(report.to_dict())


def render_dataset_tab() -> None:
    st.subheader("学習データ更新")
    st.caption("rowdata から race_entries / race_results / training_table を生成します。")

    with st.form("dataset_form"):
        rowdata_dir = st.text_input("rowdata フォルダ", value="rowdata", key="dataset_rowdata")
        output_dir = st.text_input("出力フォルダ", value="data/processed")
        max_date = st.text_input(
            "最大日付",
            value="",
            help="未入力なら rowdata に存在する最新日まで処理します。例: 2026-05-24",
        )
        submitted = st.form_submit_button("学習データを生成")

    if not submitted:
        return

    try:
        with st.spinner("学習データを生成しています..."):
            summary = build_dataset_from_rowdata_streaming(
                rowdata_dir=rowdata_dir,
                output_dir=output_dir,
                max_date=max_date.strip() or None,
            )
    except Exception as exc:  # pragma: no cover
        st.error(f"学習データ生成に失敗しました: {exc}")
        return

    st.success("学習データ生成が完了しました。")
    st.json(summary.to_dict())


def render_train_tab() -> None:
    st.subheader("モデル再学習")
    st.caption("ranker / classifier / 三連単v1 calibrator を再学習します。")

    with st.form("train_form"):
        config_path = st.text_input("設定ファイル", value="configs/train.yaml", key="train_config")
        training_device = st.selectbox(
            "学習デバイス",
            options=["cpu", "gpu"],
            index=0,
            help="WebUI から実行する学習ジョブのデバイスを選択します。既定値は CPU です。",
        )
        submitted = st.form_submit_button("モデル再学習を実行")

    if not submitted:
        return

    return_code, output_text = _run_python_cli(
        "モデル再学習",
        [
            "-c",
            "from src.cli import train_main; train_main()",
            "--config",
            config_path,
            "--training-device",
            training_device,
        ],
    )
    _render_command_result("モデル再学習", return_code, output_text)


def render_recent_backtest_tab() -> None:
    st.subheader("過去1週間分予測")
    st.caption("学習済みモデルで直近7日間のレースを再予測し、結果と比較して正解率と回収率を集計します。")

    with st.form("recent_backtest_form"):
        config_path = st.text_input("設定ファイル", value="configs/train.yaml", key="recent_backtest_config")
        rowdata_dir = st.text_input("rowdata フォルダ", value="rowdata", key="recent_backtest_rowdata")
        days = st.number_input("対象日数", min_value=1, max_value=31, value=7, step=1)
        top_k = st.number_input("1レースあたりの購入点数", min_value=1, max_value=10, value=1, step=1)
        stake = st.number_input("1点あたり購入額", min_value=100, max_value=10000, value=100, step=100)
        submitted = st.form_submit_button("過去1週間分予測を実行")

    if not submitted:
        return

    return_code, output_text = _run_python_cli(
        "過去1週間分予測",
        [
            "-c",
            "from src.cli import backtest_recent_week_main; backtest_recent_week_main()",
            "--config",
            config_path,
            "--rowdata",
            rowdata_dir,
            "--days",
            str(int(days)),
            "--stake",
            str(int(stake)),
            "--top-k",
            str(int(top_k)),
        ],
    )
    if return_code != 0:
        _render_command_result("過去1週間分予測", return_code, output_text)
        return

    try:
        report = json.loads(output_text)
    except json.JSONDecodeError:
        _render_command_result("過去1週間分予測", return_code, output_text)
        st.error("コマンド出力を JSON として解釈できませんでした。")
        return

    _render_recent_backtest_report(report)
    with st.expander("実行ログ"):
        st.code(output_text or "(no output)")


def main() -> None:
    st.set_page_config(page_title="BoatRace Predictions", page_icon="🚤", layout="wide")
    _inject_table_style()
    st.title("BoatRace Predictions")
    st.caption("当日予測、rowdata 更新、学習データ生成、モデル再学習をブラウザから実行できます。")

    tabs = st.tabs(
        [
            "当日予測",
            "共有データ取得",
            "rowdata更新",
            "学習データ更新",
            "モデル再学習",
            "過去1週間分予測",
        ]
    )
    with tabs[0]:
        render_prediction_tab()
    with tabs[1]:
        render_download_tab()
    with tabs[2]:
        render_backfill_tab()
    with tabs[3]:
        render_dataset_tab()
    with tabs[4]:
        render_train_tab()
    with tabs[5]:
        render_recent_backtest_tab()


if __name__ == "__main__":
    main()
