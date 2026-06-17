from __future__ import annotations

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

LOGGER = logging.getLogger(__name__)


def log_exception_to_stderr(context: str) -> None:
    print(f"[community_today_app] {context}", file=sys.stderr, flush=True)
    print(traceback.format_exc(), file=sys.stderr, flush=True)


def render_exception_details(exc: Exception) -> None:
    with st.expander("詳細エラー", expanded=False):
        st.exception(exc)

from src.drive_restore import (
    DEFAULT_ARTIFACTS_DRIVE_FILE_URL,
    DEFAULT_DATA_DRIVE_FILE_URL,
    download_and_restore_packages,
)
from src.today_schedule import choose_default_today_race_no, choose_default_today_venue, fetch_daily_race_schedule


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


def default_secret(name: str, fallback: str) -> str:
    try:
        return str(st.secrets.get(name, fallback))
    except Exception:
        return fallback


def get_shared_data_urls() -> tuple[str, str]:
    data_url = st.session_state.get(
        "community_data_drive_url",
        default_secret("data_drive_file_url", DEFAULT_DATA_DRIVE_FILE_URL),
    )
    artifacts_url = st.session_state.get(
        "community_artifacts_drive_url",
        default_secret("artifacts_drive_file_url", DEFAULT_ARTIFACTS_DRIVE_FILE_URL),
    )
    return str(data_url).strip(), str(artifacts_url).strip()


@st.cache_resource(show_spinner=False)
def ensure_shared_data(data_url: str, artifacts_url: str) -> dict[str, object]:
    report = download_and_restore_packages(
        project_root=repo_root(),
        data_drive_file_url=data_url,
        artifacts_drive_file_url=artifacts_url,
        restore_rowdata=False,
        restore_data=True,
        restore_artifacts=True,
    )
    return report.to_dict()


@st.cache_data(show_spinner=False, ttl=300)
def load_today_schedule():
    return fetch_daily_race_schedule()


@st.cache_data(show_spinner=False, ttl=300)
def predict_today_cached(
    config_path: str,
    venue: str,
    race_no: int,
    race_date: date,
) -> Any:
    from src.api import predict_today

    return predict_today(
        venue=venue,
        race_no=race_no,
        config_path=config_path,
        race_date=race_date,
    )


def clear_prediction_caches() -> None:
    predict_today_cached.clear()


def _prediction_venue_key() -> str:
    return "community_prediction_selected_venue"


def _prediction_race_key() -> str:
    return "community_prediction_selected_race_no"


def _set_default_prediction_race(schedule: dict[str, dict[int, object]]) -> None:
    venue = st.session_state.get(_prediction_venue_key(), "15")
    st.session_state[_prediction_race_key()] = choose_default_today_race_no(schedule, venue)


def bootstrap_shared_data_from_secrets() -> None:
    data_url = default_secret("data_drive_file_url", DEFAULT_DATA_DRIVE_FILE_URL).strip()
    artifacts_url = default_secret("artifacts_drive_file_url", DEFAULT_ARTIFACTS_DRIVE_FILE_URL).strip()
    if not data_url or not artifacts_url:
        return

    state_key = f"community_bootstrap_done::{data_url}::{artifacts_url}"
    if st.session_state.get(state_key):
        return

    st.session_state["community_data_drive_url"] = data_url
    st.session_state["community_artifacts_drive_url"] = artifacts_url

    with st.spinner("共有データを初期化しています..."):
        report = ensure_shared_data(data_url, artifacts_url)

    st.session_state["community_bootstrap_report"] = report
    st.session_state[state_key] = True


def render_data_setup_tab() -> None:
    st.subheader("共有データ取得")
    st.caption(
        "Google Drive の共有リンクから `data.zip` と `artifacts.zip` を取得し、"
        "アプリ内で再利用します。ダウンロード済みデータは `st.cache_resource` で保持します。"
    )

    default_data_url, default_artifacts_url = get_shared_data_urls()

    bootstrap_report = st.session_state.get("community_bootstrap_report")
    if bootstrap_report:
        st.info("Secrets に設定された共有リンクから初期データを読み込み済みです。")

    with st.form("community_data_download_form"):
        data_url = st.text_input("data.zip URL", value=default_data_url)
        artifacts_url = st.text_input("artifacts.zip URL", value=default_artifacts_url)
        submitted = st.form_submit_button("共有データを取得")

    col1, col2 = st.columns(2)
    with col1:
        force_refresh = st.button("キャッシュをクリアして再取得", use_container_width=True)
    with col2:
        clear_only = st.button("予測キャッシュのみクリア", use_container_width=True)

    if clear_only:
        predict_today_cached.clear()
        st.success("予測キャッシュをクリアしました。")

    if force_refresh:
        ensure_shared_data.clear()
        clear_prediction_caches()
        st.info("共有データキャッシュをクリアしました。続けて再取得してください。")

    if not submitted:
        return

    st.session_state["community_data_drive_url"] = data_url
    st.session_state["community_artifacts_drive_url"] = artifacts_url

    try:
        with st.spinner("共有データをダウンロードして展開しています..."):
            report = ensure_shared_data(data_url.strip(), artifacts_url.strip())
        clear_prediction_caches()
        st.session_state["community_bootstrap_report"] = report
    except Exception as exc:  # pragma: no cover
        log_exception_to_stderr("Failed to restore shared data in Community Cloud app")
        LOGGER.exception("Failed to restore shared data in Community Cloud app")
        st.error(f"共有データ取得に失敗しました: {exc}")
        render_exception_details(exc)
        return

    st.success("共有データの取得が完了しました。")
    st.json(report)


def render_prediction_tab() -> None:
    st.subheader("当日レース予測")
    st.caption(
        "番組表やオッズを含む当日予測結果は `st.cache_data` で 5 分間キャッシュし、"
        "同じ条件での再実行時の通信量を抑えます。"
    )

    try:
        schedule = load_today_schedule()
    except Exception as exc:  # pragma: no cover
        schedule = {}
        st.warning(f"本日の開催情報の取得に失敗したため、手動選択に切り替えます: {exc}")

    venue_options = sorted(schedule.keys()) if schedule else list(VENUES.keys())
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

    race_options = sorted(schedule.get(selected, {}).keys()) if schedule else list(range(1, 13))
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
    submitted = st.button("予測する")

    if not submitted:
        return

    data_url, artifacts_url = get_shared_data_urls()
    if not data_url or not artifacts_url:
        st.warning("先に『共有データ取得』タブで data.zip / artifacts.zip の URL を設定してください。")
        return

    try:
        with st.spinner("共有データを確認しています..."):
            ensure_shared_data(data_url, artifacts_url)
        with st.spinner("予測を実行しています..."):
            prediction = predict_today_cached(
                config_path=config_path,
                venue=selected,
                race_no=int(race_no),
                race_date=date.today(),
            )
    except Exception as exc:  # pragma: no cover
        log_exception_to_stderr(
            f"Prediction failed in Community Cloud app (venue={selected}, race_no={int(race_no)}, race_date={date.today()})"
        )
        LOGGER.exception(
            "Prediction failed in Community Cloud app (venue=%s, race_no=%s, race_date=%s)",
            selected,
            int(race_no),
            date.today(),
        )
        st.error(f"予測に失敗しました: {exc}")
        render_exception_details(exc)
        return

    st.success("予測が完了しました。")
    st.text(prediction.text)
    st.markdown("**順位予測**")
    st.dataframe(prediction.ranking, use_container_width=True, hide_index=True)
    st.markdown("**三連単候補**")
    st.dataframe(prediction.trifecta.head(20), use_container_width=True, hide_index=True)

    if prediction.odds is not None and not prediction.odds.empty:
        st.markdown("**オッズ評価**")
        st.dataframe(prediction.odds.head(20), use_container_width=True, hide_index=True)

    if prediction.buy_candidates is not None and not prediction.buy_candidates.empty:
        st.markdown("**買い候補**")
        st.dataframe(prediction.buy_candidates, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="BoatRace Today", page_icon="🚤", layout="wide")
    st.title("BoatRace Today")
    st.caption("Streamlit Community Cloud 向けの当日予測専用アプリです。")

    try:
        bootstrap_shared_data_from_secrets()
    except Exception as exc:  # pragma: no cover
        log_exception_to_stderr("Startup shared data bootstrap failed in Community Cloud app")
        LOGGER.exception("Startup shared data bootstrap failed in Community Cloud app")
        st.warning(f"起動時の共有データ初期化に失敗しました: {exc}")
        render_exception_details(exc)

    tabs = st.tabs(["当日レース予測", "共有データ取得"])
    with tabs[0]:
        render_prediction_tab()
    with tabs[1]:
        render_data_setup_tab()


if __name__ == "__main__":
    main()
