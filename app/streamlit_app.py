from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.api import backfill_rowdata_files, build_dataset_from_rowdata, load_bundle, predict_today


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


def render_prediction_tab() -> None:
    st.subheader("当日レース予測")
    st.caption("当日番組表とオッズを取得し、順位予測と三連単候補を表示します。")

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
        bundle = load_cached_bundle(config_path)
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
    st.caption("mbrace の日次アーカイブから不足している B/K テキストを補完します。")

    with st.form("backfill_form"):
        rowdata_dir = st.text_input("rowdata フォルダ", value="rowdata")
        start_date_text = st.text_input("開始日", value="", help="未入力なら既存最新日の翌日から補完します。例: 2026-05-14")
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

    st.success("補完処理が完了しました。")
    st.json(report.to_dict())

    if report.downloaded_files:
        st.markdown("**新規取得ファイル**")
        st.dataframe(
            pd.DataFrame({"path": [str(path) for path in report.downloaded_files]}),
            use_container_width=True,
            hide_index=True,
        )

    if report.unavailable_files:
        st.markdown("**取得不可ファイル**")
        st.dataframe(
            pd.DataFrame({"file": report.unavailable_files}),
            use_container_width=True,
            hide_index=True,
        )


def render_dataset_tab() -> None:
    st.subheader("学習データ更新")
    st.caption("rowdata から `race_entries` / `race_results` / `training_table` を再生成します。")

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
        st.error(f"学習データ再生成に失敗しました: {exc}")
        return

    summary = {
        "entries_rows": int(len(tables["entries"])),
        "results_rows": int(len(tables["results"])),
        "training_rows": int(len(tables["training_table"])),
        "output_dir": str(Path(output_dir)),
    }
    st.success("学習データ再生成が完了しました。")
    st.json(summary)


def main() -> None:
    st.set_page_config(page_title="BoatRace Predictions", page_icon="🚤", layout="wide")
    st.title("BoatRace Predictions")
    st.caption("当日レース予測、rowdata 補完、学習データ更新をブラウザから実行します。")

    prediction_tab, backfill_tab, dataset_tab = st.tabs(
        ["当日予測", "rowdata補完", "学習データ更新"]
    )
    with prediction_tab:
        render_prediction_tab()
    with backfill_tab:
        render_backfill_tab()
    with dataset_tab:
        render_dataset_tab()


if __name__ == "__main__":
    main()
