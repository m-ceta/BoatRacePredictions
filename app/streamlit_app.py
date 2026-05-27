from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd
import streamlit as st

from src.api import backfill_rowdata_files, build_dataset_from_rowdata, load_bundle, predict_today
from src.drive_backup import DEFAULT_DRIVE_FOLDER_URL


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


@st.cache_resource(show_spinner=False)
def load_cached_bundle(config_path: str):
    return load_bundle(config_path)


def _run_python_cli(command_label: str, args: Sequence[str]) -> tuple[int, str]:
    placeholder = st.empty()
    lines: list[str] = []
    full_command = [sys.executable, *args]

    with st.spinner(f"{command_label} を実行しています..."):
        process = subprocess.Popen(
            full_command,
            cwd=Path(__file__).resolve().parents[1],
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
    st.caption("対象レースを指定して、順位予測・三連単候補・オッズ評価を表示します。")

    with st.form("prediction_form"):
        config_path = st.text_input("設定ファイル", value="configs/train.yaml")
        selected = st.selectbox(
            "レース場",
            options=list(VENUES.keys()),
            format_func=lambda code: f"{code} {VENUES[code]}",
        )
        race_no = st.number_input("レースNo", min_value=1, max_value=12, value=12, step=1)
        race_date = st.date_input("日付", value=date.today())
        submitted = st.form_submit_button("予測を実行")

    if not submitted:
        return

    try:
        load_cached_bundle.clear()
        with st.spinner("当日データを取得して予測しています..."):
            prediction = predict_today(
                venue=selected,
                race_no=int(race_no),
                config_path=config_path,
                race_date=race_date,
            )
    except Exception as exc:  # pragma: no cover - streamlit runtime path
        st.error(f"予測に失敗しました: {exc}")
        return

    st.success("予測が完了しました。")
    st.text(prediction.text)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**順位予測**")
        st.dataframe(prediction.ranking, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**三連単候補**")
        st.dataframe(prediction.trifecta.head(20), use_container_width=True, hide_index=True)

    if prediction.odds is not None and not prediction.odds.empty:
        st.markdown("**オッズ評価**")
        st.dataframe(prediction.odds.head(20), use_container_width=True, hide_index=True)

    if prediction.buy_candidates is not None and not prediction.buy_candidates.empty:
        st.markdown("**買い候補**")
        st.dataframe(prediction.buy_candidates, use_container_width=True, hide_index=True)


def render_backfill_tab() -> None:
    st.subheader("rowdata 補完")
    st.caption("mbrace から不足する B/K テキストを取得します。")

    with st.form("backfill_form"):
        rowdata_dir = st.text_input("rowdata フォルダ", value="rowdata")
        start_date_text = st.text_input(
            "開始日",
            value="",
            help="未入力なら既存ファイルの最新日の翌日から補完します。例: 2026-05-14",
        )
        end_date = st.date_input("終了日", value=date.today())
        kinds = st.multiselect(
            "対象種別",
            options=["B", "K"],
            default=["B", "K"],
            help="B は番組表、K はレース結果です。",
        )
        overwrite = st.checkbox("既存ファイルを上書きする", value=False)
        submitted = st.form_submit_button("補完を実行")

    if not submitted:
        return

    if not kinds:
        st.warning("少なくとも B または K を選択してください。")
        return

    try:
        normalized_start = start_date_text.strip() or None
        with st.spinner("rowdata を補完しています..."):
            report = backfill_rowdata_files(
                rowdata_dir=rowdata_dir,
                start_date=normalized_start,
                end_date=end_date,
                kinds=kinds,
                overwrite=overwrite,
            )
    except Exception as exc:  # pragma: no cover - streamlit runtime path
        st.error(f"補完に失敗しました: {exc}")
        return

    st.success("補完が完了しました。")
    st.json(report.to_dict())

    if report.downloaded_files:
        st.markdown("**取得ファイル**")
        st.dataframe(
            pd.DataFrame({"path": [str(path) for path in report.downloaded_files]}),
            use_container_width=True,
            hide_index=True,
        )

    if report.unavailable_files:
        st.markdown("**未取得ファイル**")
        st.dataframe(
            pd.DataFrame({"file": report.unavailable_files}),
            use_container_width=True,
            hide_index=True,
        )


def render_dataset_tab() -> None:
    st.subheader("学習データ更新")
    st.caption("rowdata から race_entries / race_results / training_table を再生成します。")

    with st.form("dataset_form"):
        rowdata_dir = st.text_input("rowdata フォルダ", value="rowdata", key="dataset_rowdata")
        output_dir = st.text_input("出力フォルダ", value="data/processed")
        submitted = st.form_submit_button("学習データを再生成")

    if not submitted:
        return

    try:
        with st.spinner("学習データを再生成しています..."):
            tables = build_dataset_from_rowdata(rowdata_dir=rowdata_dir, output_dir=output_dir)
    except Exception as exc:  # pragma: no cover - streamlit runtime path
        st.error(f"学習データ生成に失敗しました: {exc}")
        return

    summary = {
        "entries_rows": int(len(tables["entries"])),
        "results_rows": int(len(tables["results"])),
        "training_rows": int(len(tables["training_table"])),
        "output_dir": str(Path(output_dir)),
    }
    st.success("学習データ生成が完了しました。")
    st.json(summary)


def render_train_tab() -> None:
    st.subheader("モデル再学習")
    st.caption("ranker / classifier / flow / staged / Phase3 基本モデルを再学習します。")

    with st.form("train_form"):
        config_path = st.text_input("設定ファイル", value="configs/train.yaml", key="train_config")
        training_device = st.selectbox(
            "学習デバイス",
            options=["cpu", "gpu"],
            index=0,
            help="WebUI から実行する学習ジョブのデバイスを選択します。既定は CPU です。",
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
    st.caption("三連単 Phase3 rerank の追加最適化と専用評価を実行します。")

    with st.form("trifecta_train_form"):
        config_path = st.text_input("設定ファイル", value="configs/train.yaml", key="trifecta_train_config")
        training_device = st.selectbox(
            "学習デバイス",
            options=["cpu", "gpu"],
            index=0,
            key="trifecta_training_device",
            help="WebUI から実行する学習ジョブのデバイスを選択します。既定は CPU です。",
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


def render_upload_tab() -> None:
    st.subheader("学習成果物アップロード")
    st.caption("rowdata.zip と drp.zip を作成し、Google Drive に同名上書きアップロードします。")

    with st.form("package_upload_form"):
        project_root = st.text_input("project-root", value=".")
        drive_folder = st.text_input("drive-folder", value=DEFAULT_DRIVE_FOLDER_URL)
        credentials = st.text_input("credentials", value="google_drive_credentials.json")
        token = st.text_input("token", value="artifacts/google-drive-token.json")
        rowdata_zip_name = st.text_input("rowdata-zip-name", value="rowdata.zip")
        drp_zip_name = st.text_input("drp-zip-name", value="drp.zip")
        submitted = st.form_submit_button("学習成果物アップロードを実行")

    if not submitted:
        return

    args = [
        "-c",
        "from src.cli import package_and_upload_main; package_and_upload_main()",
        "--project-root",
        project_root,
        "--drive-folder",
        drive_folder,
        "--credentials",
        credentials,
        "--token",
        token,
        "--rowdata-zip-name",
        rowdata_zip_name,
        "--drp-zip-name",
        drp_zip_name,
    ]
    return_code, output_text = _run_python_cli("学習成果物アップロード", args)
    _render_command_result("学習成果物アップロード", return_code, output_text)


def main() -> None:
    st.set_page_config(page_title="BoatRace Predictions", page_icon="🚤", layout="wide")
    st.title("BoatRace Predictions")
    st.caption(
        "当日予測、rowdata 補完、学習データ更新、モデル再学習、Google Drive への成果物アップロードをブラウザから実行できます。"
    )

    tabs = st.tabs(
        [
            "当日予測",
            "rowdata補完",
            "学習データ更新",
            "モデル再学習",
            "三連単最適化学習",
            "学習成果物アップロード",
        ]
    )
    with tabs[0]:
        render_prediction_tab()
    with tabs[1]:
        render_backfill_tab()
    with tabs[2]:
        render_dataset_tab()
    with tabs[3]:
        render_train_tab()
    with tabs[4]:
        render_trifecta_train_tab()
    with tabs[5]:
        render_upload_tab()


if __name__ == "__main__":
    main()
