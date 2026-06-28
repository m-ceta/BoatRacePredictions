from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

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
from src.today_schedule import (  # noqa: E402
    choose_default_today_race_no,
    choose_default_today_venue,
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


@st.cache_resource(show_spinner=False)
def load_cached_bundle(config_path: str):
    return load_bundle(config_path)


@st.cache_data(show_spinner=False, ttl=300)
def load_today_schedule():
    return fetch_daily_race_schedule()


def _prediction_venue_key() -> str:
    return "prediction_selected_venue"


def _prediction_race_key() -> str:
    return "prediction_selected_race_no"


def _set_default_prediction_race(schedule: dict[str, dict[int, object]]) -> None:
    venue = st.session_state.get(_prediction_venue_key(), "15")
    st.session_state[_prediction_race_key()] = choose_default_today_race_no(schedule, venue)


def _select_and_rename_columns(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> pd.DataFrame:
    available = [(source, label) for source, label in columns if source in frame.columns]
    if not available:
        return frame.copy()
    selected = frame[[source for source, _ in available]].copy()
    return selected.rename(columns={source: label for source, label in available})


def _format_ranking_frame(frame: pd.DataFrame) -> pd.DataFrame:
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


def _format_trifecta_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _select_and_rename_columns(
        frame,
        [
            ("trifecta", "買い目"),
            ("probability", "予想確率"),
            ("probability_v1", "基本モデル確率"),
            ("probability_v2", "最適化後確率"),
            ("raw_probability_v1", "基本モデル生スコア"),
            ("raw_probability_v2", "最適化後生スコア"),
        ],
    )


def _format_odds_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _select_and_rename_columns(
        frame,
        [
            ("trifecta", "買い目"),
            ("probability", "予想確率"),
            ("odds", "現在オッズ"),
            ("expected_value", "期待値"),
            ("buy_decision", "判定"),
        ],
    )


def _format_buy_candidates_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _select_and_rename_columns(
        frame,
        [
            ("trifecta", "買い目"),
            ("probability", "予想確率"),
            ("odds", "現在オッズ"),
            ("expected_value", "期待値"),
            ("buy_decision", "判定"),
        ],
    )


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


def render_prediction_tab() -> None:
    st.subheader("当日レース予測")
    st.caption("本日の開催レースから予測対象を選び、順位予測、三連単候補、オッズ評価を表示します。")

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
    submitted = st.button("予測を実行")

    if not submitted:
        return

    try:
        load_cached_bundle.clear()
        with st.spinner("予測を実行しています..."):
            prediction = predict_today(
                venue=selected,
                race_no=int(race_no),
                config_path=config_path,
            )
    except Exception as exc:  # pragma: no cover
        st.error(f"予測に失敗しました: {exc}")
        return

    st.success("予測が完了しました。")
    st.text(prediction.text)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**順位予測**")
        st.dataframe(_format_ranking_frame(prediction.ranking), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**三連単候補**")
        st.dataframe(_format_trifecta_frame(prediction.trifecta.head(20)), use_container_width=True, hide_index=True)

    if prediction.odds is not None and not prediction.odds.empty:
        st.markdown("**オッズ評価**")
        st.dataframe(_format_odds_frame(prediction.odds.head(20)), use_container_width=True, hide_index=True)

    if prediction.buy_candidates is not None and not prediction.buy_candidates.empty:
        st.markdown("**買い候補**")
        st.dataframe(_format_buy_candidates_frame(prediction.buy_candidates), use_container_width=True, hide_index=True)


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
    st.caption("ranker / classifier / flow / staged / Phase3 基本モデルを再学習します。")

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


def render_trifecta_train_tab() -> None:
    st.subheader("三連単最適化学習")
    st.caption("三連単 Phase3 rerank の追加学習と評価を実行します。")

    with st.form("trifecta_train_form"):
        config_path = st.text_input("設定ファイル", value="configs/train.yaml", key="trifecta_train_config")
        training_device = st.selectbox(
            "学習デバイス",
            options=["cpu", "gpu"],
            index=0,
            key="trifecta_training_device",
            help="WebUI から実行する学習ジョブのデバイスを選択します。既定値は CPU です。",
        )
        max_races = st.number_input("max-races", min_value=100, max_value=10000, value=1000, step=100)
        eval_max_races = st.number_input("eval-max-races", min_value=100, max_value=10000, value=1000, step=100)
        eval_rerank_top_n = st.number_input("eval-rerank-top-n", min_value=3, max_value=120, value=10, step=1)
        optimize_rerank = st.checkbox("optimize-rerank を有効化", value=True)
        submitted = st.form_submit_button("三連単最適化学習を実行")

    if not submitted:
        return

    args = [
        "-c",
        "from src.cli import train_trifecta_v2_main; train_trifecta_v2_main()",
        "--config",
        config_path,
        "--training-device",
        training_device,
        "--max-races",
        str(int(max_races)),
        "--eval-max-races",
        str(int(eval_max_races)),
        "--eval-rerank-top-n",
        str(int(eval_rerank_top_n)),
    ]
    if optimize_rerank:
        args.append("--optimize-rerank")

    return_code, output_text = _run_python_cli("三連単最適化学習", args)
    _render_command_result("三連単最適化学習", return_code, output_text)


def main() -> None:
    st.set_page_config(page_title="BoatRace Predictions", page_icon="🚤", layout="wide")
    st.title("BoatRace Predictions")
    st.caption("当日予測、rowdata 更新、学習データ生成、モデル再学習をブラウザから実行できます。")

    tabs = st.tabs(
        [
            "当日予測",
            "共有データ取得",
            "rowdata更新",
            "学習データ更新",
            "モデル再学習",
            "三連単最適化学習",
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
        render_trifecta_train_tab()


if __name__ == "__main__":
    main()
