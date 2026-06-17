from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api import load_bundle, predict_today
from src.drive_restore import (
    DEFAULT_ARTIFACTS_DRIVE_FILE_URL,
    DEFAULT_DATA_DRIVE_FILE_URL,
    download_and_restore_packages,
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


@st.cache_resource(show_spinner=False)
def load_cached_bundle(config_path: str):
    return load_bundle(config_path)


@st.cache_data(show_spinner=False, ttl=300)
def predict_today_cached(
    config_path: str,
    venue: str,
    race_no: int,
    race_date: date,
) -> Any:
    return predict_today(
        venue=venue,
        race_no=race_no,
        config_path=config_path,
        race_date=race_date,
    )


def clear_prediction_caches() -> None:
    load_cached_bundle.clear()
    predict_today_cached.clear()


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
        st.error(f"共有データ取得に失敗しました: {exc}")
        return

    st.success("共有データの取得が完了しました。")
    st.json(report)


def render_prediction_tab() -> None:
    st.subheader("当日レース予測")
    st.caption(
        "番組表やオッズを含む当日予測結果は `st.cache_data` で 5 分間キャッシュし、"
        "同じ条件での再実行時の通信量を抑えます。"
    )

    with st.form("community_prediction_form"):
        config_path = st.text_input("設定ファイル", value="configs/train.yaml")
        selected = st.selectbox(
            "レース場",
            options=list(VENUES.keys()),
            format_func=lambda code: f"{code} {VENUES[code]}",
        )
        race_no = st.number_input("レースNo", min_value=1, max_value=12, value=12, step=1)
        race_date = st.date_input("日付", value=date.today())
        submitted = st.form_submit_button("予測する")

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
                race_date=race_date,
            )
    except Exception as exc:  # pragma: no cover
        st.error(f"予測に失敗しました: {exc}")
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
        st.warning(f"起動時の共有データ初期化に失敗しました: {exc}")

    tabs = st.tabs(["当日レース予測", "共有データ取得"])
    with tabs[0]:
        render_prediction_tab()
    with tabs[1]:
        render_data_setup_tab()


if __name__ == "__main__":
    main()
