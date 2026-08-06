import logging
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.drive_restore import (  # noqa: E402
    download_and_restore_packages,
)

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
        "race_scenario_name": "決着パターン",
        "race_scenario_id": "決着ID",
        "race_scenario_description": "決着イメージ",
        "race_upset_score": "荒れ度",
        "race_upset_label": "荒れ判定",
        "trifecta_darkhorse_score": "穴度",
        "is_darkhorse_candidate": "穴候補",
        "ticket_priority_score": "買い目優先度",
        "ticket_hint": "買い目目安",
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
            ("predicted_rank", "予想着順"),
            ("lane", "艇番"),
            ("racer_name", "選手名"),
            ("class_name", "級別"),
            ("branch", "支部"),
            ("age", "年齢"),
            ("motor_no", "モーター"),
            ("boat_no", "ボート"),
            ("exhibition_time", "展示タイム"),
            ("course", "進入"),
            ("start_timing", "ST"),
            ("win_probability_like", "1着期待度"),
            ("score", "総合スコア"),
        ],
    )


def _format_trifecta_frame(frame):
    columns = [
        ("trifecta", "買い目"),
        ("probability", "予想確率"),
        ("odds", "現在オッズ"),
        ("expected_value", "期待値"),
        ("recommended_bet_amount", "推奨購入金額"),
        ("top12_confidence_score", "Top12信頼スコア"),
        ("top12_confidence_label", "Top12信頼"),
        ("recommended_ticket_label", "推奨点数"),
        ("trifecta_darkhorse_score", "穴度"),
    ]
    formatted = frame.copy()
    for source, _label in columns:
        if source not in formatted.columns:
            formatted[source] = None
    formatted = formatted[[source for source, _label in columns]]
    return formatted.rename(columns={source: label for source, label in columns})


def _trifecta_display_frame(prediction: Any):
    if prediction.odds is not None and not prediction.odds.empty:
        return prediction.odds.sort_values("probability", ascending=False).head(20)
    return prediction.trifecta.sort_values("probability", ascending=False).head(20)


def _format_percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


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
**買い目の判断方法**

- `期待値` は `現在オッズ × 予想確率` です。
- `期待値` が 1.0 以上、かつ `現在オッズ` が 12.0 倍以上なら `買い`、それ以外は `見送り` です。
- `Top12信頼スコア` は 0〜100 の総合指標です。Top12内の確率合計、上位確率の集中度、1位と2位の差、Top12境界の余裕、荒れ度を合成しています。
- `Top12信頼` は 75以上が高、60以上が中、60未満が低です。

**予想信頼度**

- 本命艇の `1着期待度` が高いほど上がります。
- 1位と2位の差、3連単1位候補と2位候補の差が大きいほど上がります。
- `高 / 中 / 低` は、予想がどれだけ絞れているかの目安です。

**1着期待度**

- その艇が 1 着になる見込みを、順位モデルのスコアから相対的に表した値です。
- 同じレース内で、どの艇が頭候補として強いかを見るための指標です。

**基本モデル確率と最適化後確率**

- `基本モデル確率` は、順位予想からそのまま作った 3連単確率です。
- `最適化後確率` は、追加モデルで 3連単の並び順を調整した後の確率です。
- `予想確率` は、通常この最適化後の値を使っています。
"""
        )


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
    st.text(_build_prediction_summary(prediction))
    st.metric("決着パターン", prediction.race_scenario_name, prediction.race_scenario_id)
    st.caption(f"決着イメージ: {prediction.race_scenario_description}")
    st.metric("レース荒れ度", f"{float(prediction.race_upset_score) * 100:.1f}%", prediction.race_upset_label)
    _render_prediction_guide()

    st.markdown("**順位予測**")
    st.dataframe(_format_ranking_frame(prediction.ranking), use_container_width=True, hide_index=True)

    st.markdown("**3連単予想**")
    st.dataframe(
        _format_trifecta_frame(_trifecta_display_frame(prediction)),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="BoatRace Today", page_icon="🚤", layout="wide")
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
