import logging
import json
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.drive_restore import (  # noqa: E402
    download_and_restore_packages,
)
from src.models.ranker import get_artifact_paths, load_config  # noqa: E402

LOGGER = logging.getLogger(__name__)

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


def default_config_path() -> str:
    return "configs/train.yaml"


def log_exception_to_stderr(context: str) -> None:
    print(f"[community_today_app] {context}", file=sys.stderr, flush=True)
    print(traceback.format_exc(), file=sys.stderr, flush=True)


def render_exception_details(exc: Exception) -> None:
    with st.expander("詳細エラー", expanded=False):
        st.exception(exc)


def default_secret(name: str, fallback: str) -> str:
    try:
        return str(st.secrets.get(name, fallback))
    except Exception:
        return fallback


def get_shared_data_urls() -> tuple[str, str]:
    data_url = st.session_state.get(
        "community_data_drive_url",
        default_secret("data_drive_file_url", ""),
    )
    artifacts_url = st.session_state.get(
        "community_artifacts_drive_url",
        default_secret("artifacts_drive_file_url", ""),
    )
    return str(data_url).strip(), str(artifacts_url).strip()


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


@st.cache_resource(show_spinner=False)
def ensure_shared_data(data_url: str, artifacts_url: str) -> dict[str, object]:
    data_url = data_url.strip()
    artifacts_url = artifacts_url.strip()
    restore_data = bool(data_url)
    restore_artifacts = bool(artifacts_url)
    if not restore_data and not restore_artifacts:
        return {
            "rowdata_zip": None,
            "data_zip": None,
            "artifacts_zip": None,
            "restored_targets": [],
        }

    report = download_and_restore_packages(
        project_root=repo_root(),
        data_drive_file_url=data_url,
        artifacts_drive_file_url=artifacts_url,
        restore_rowdata=False,
        restore_data=restore_data,
        restore_artifacts=restore_artifacts,
    )
    return report.to_dict()


@st.cache_data(show_spinner=False, ttl=300)
def load_today_schedule():
    from src.today_schedule import fetch_daily_race_schedule, filter_future_schedule

    return filter_future_schedule(fetch_daily_race_schedule())


@st.cache_data(show_spinner=False, ttl=60)
def load_exhibition_courses(venue: str, race_no: int, race_date: date) -> tuple[int, int, int, int, int, int] | None:
    from src.live import fetch_boatrace_exhibition_courses

    return fetch_boatrace_exhibition_courses(race_date, venue, race_no)


@st.cache_data(show_spinner=False, ttl=300)
def predict_today_cached(
    config_path: str,
    venue: str,
    race_no: int,
    race_date: date,
    course_overrides: tuple[int, int, int, int, int, int],
) -> Any:
    from src.api import predict_today

    return predict_today(
        venue=venue,
        race_no=race_no,
        config_path=config_path,
        race_date=race_date,
        course_overrides=course_overrides,
    )


def _prediction_venue_key() -> str:
    return "community_prediction_selected_venue"


def _prediction_race_key() -> str:
    return "community_prediction_selected_race_no"


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
            key=f"community_prediction_course_mode_{scope}_{default_source}_{_format_courses(default_courses)}",
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
                key=f"community_prediction_course_text_{scope}_{default_source}_{_format_courses(default_courses)}",
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
    from src.today_schedule import choose_default_today_race_no

    venue = st.session_state.get(_prediction_venue_key(), "15")
    st.session_state[_prediction_race_key()] = choose_default_today_race_no(schedule, venue)


def _select_and_rename_columns(frame: Any, columns: list[tuple[str, str]]):
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


def _format_ranking_frame(frame):
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


def _format_trifecta_frame(frame):
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


def _ranking_column_config(frame) -> dict[str, Any]:
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


def _trifecta_column_config(frame) -> dict[str, Any]:
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


def _trifecta_display_frame(prediction: Any):
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
}
</style>
""",
        unsafe_allow_html=True,
    )


def _confidence_label(score: float) -> str:
    if score >= 0.22:
        return "高"
    if score >= 0.16:
        return "中"
    return "低"


def _build_prediction_summary(prediction: Any) -> str:
    return "\n".join(
        [
            f"予想信頼度: {_confidence_label(float(prediction.confidence_score))} ({_format_percent(prediction.confidence_score)})",
        ]
    )


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

    st.markdown("**現在のモデル精度（検証データ）**")
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


def bootstrap_shared_data_from_secrets() -> None:
    data_url = default_secret("data_drive_file_url", "").strip()
    artifacts_url = default_secret("artifacts_drive_file_url", "").strip()
    if not data_url and not artifacts_url:
        return

    state_key = f"community_bootstrap_done::{data_url}::{artifacts_url}"
    if st.session_state.get(state_key):
        return

    st.session_state["community_data_drive_url"] = data_url
    st.session_state["community_artifacts_drive_url"] = artifacts_url

    with st.spinner("共有データを起動時に読み込んでいます..."):
        ensure_shared_data(data_url, artifacts_url)

    st.session_state[state_key] = True


def render_prediction_tab() -> None:
    st.subheader("当日レース予測")
    st.caption("本日のレースの着順予想と3連単予想を表示します。")

    schedule_fetch_error: Exception | None = None
    try:
        schedule = load_today_schedule()
    except Exception as exc:  # pragma: no cover
        schedule_fetch_error = exc
        schedule = {}

    if schedule_fetch_error is not None:
        st.warning(
            f"本日の開催情報の取得に失敗しました。mbrace への接続または応答に問題があります: {schedule_fetch_error}"
        )
        if st.button("mbrace取得を再試行", key="retry_today_schedule_cloud"):
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
        from src.today_schedule import choose_default_today_venue

        st.session_state[venue_key] = choose_default_today_venue(schedule)

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
        from src.today_schedule import choose_default_today_race_no

        st.session_state[race_key] = choose_default_today_race_no(schedule, selected)
        if st.session_state[race_key] not in race_options:
            st.session_state[race_key] = race_options[-1]

    race_no = st.selectbox(
        "レースNo",
        options=race_options,
        format_func=lambda value: f"{int(value)}R",
        key=race_key,
    )
    from src.today_schedule import current_jst_date

    target_date = current_jst_date()
    exhibition_courses = load_exhibition_courses(selected, int(race_no), target_date)
    course_overrides, course_overrides_valid = _render_course_inputs(
        f"{selected}_{int(race_no):02d}_{target_date.isoformat()}",
        exhibition_courses,
    )
    submitted = st.button("予測する", disabled=not course_overrides_valid)

    if not submitted:
        return

    data_url, artifacts_url = get_shared_data_urls()

    try:
        if data_url or artifacts_url:
            with st.spinner("共有データを確認しています..."):
                ensure_shared_data(data_url, artifacts_url)
        with st.spinner("予測を実行しています..."):
            prediction = predict_today_cached(
                config_path=default_config_path(),
                venue=selected,
                race_no=int(race_no),
                race_date=current_jst_date(),
                course_overrides=course_overrides,
            )
    except Exception as exc:  # pragma: no cover
        log_exception_to_stderr(
            f"Prediction failed in Community Cloud app (venue={selected}, race_no={int(race_no)})"
        )
        LOGGER.exception(
            "Prediction failed in Community Cloud app (venue=%s, race_no=%s)",
            selected,
            int(race_no),
        )
        st.error(f"予測に失敗しました: {exc}")
        render_exception_details(exc)
        return

    st.success("予測が完了しました。")
    _render_model_accuracy_summary(default_config_path())

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
    st.dataframe(
        ranking_frame,
        use_container_width=True,
        hide_index=True,
        height=_dataframe_height(len(ranking_frame)),
        column_config=_ranking_column_config(ranking_frame),
    )

    st.markdown("**3連単予想**")
    trifecta_display = _trifecta_display_frame(prediction)
    trifecta_frame = _format_trifecta_frame(trifecta_display)
    st.dataframe(
        trifecta_frame,
        use_container_width=True,
        hide_index=True,
        height="auto",
        row_height=38,
        column_config=_trifecta_column_config(trifecta_frame),
    )


def main() -> None:
    st.set_page_config(page_title="BoatRace Today", page_icon="🚤", layout="wide")
    _inject_table_style()
    st.title("BoatRace Today")
    st.caption("Streamlit Community Cloud 向けの当日予測アプリです。")

    try:
        bootstrap_shared_data_from_secrets()
    except Exception as exc:  # pragma: no cover
        log_exception_to_stderr("Startup shared data bootstrap failed in Community Cloud app")
        LOGGER.exception("Startup shared data bootstrap failed in Community Cloud app")
        st.warning(f"起動時の共有データ初期化に失敗しました: {exc}")
        render_exception_details(exc)

    render_prediction_tab()


if __name__ == "__main__":
    main()
