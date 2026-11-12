"""Shared hypothetical start-timing controls for the prediction UIs."""

import streamlit as st
import pandas as pd


@st.cache_data(show_spinner=False, ttl=300)
def load_st_reference(config_path, venue, race_no, race_date, courses):
    from src.api import preview_today_st

    return preview_today_st(venue, race_no, config_path, race_date, courses)


def render_st_adjustments(key: str, *, config_path=None, venue=None, race_no=None, race_date=None, courses=None) -> tuple[float | None, ...] | None:
    with st.expander("スタートの早い・遅いを仮定して予測する", expanded=False):
        st.caption("各艇の予測に使うSTを直接入力してください。小さい値ほど早いスタートです。初期値は通常STです。")
        reference = {}
        if config_path is not None:
            try:
                with st.spinner("通常STの参考値を確認しています..."):
                    preview = load_st_reference(config_path, venue, race_no, race_date, courses)
                reference = preview.set_index("lane")["start_timing"].to_dict()
            except Exception:
                st.caption("通常STを取得できませんでした。学習データ・出走表の取得状態をご確認ください。通常予測は引き続き利用できます。")
        revision = repr((courses, tuple(reference.get(lane) for lane in range(1, 7))))
        keys = [f"{key}_{revision}_st_absolute_{lane}" for lane in range(1, 7)]
        if st.button("すべて通常に戻す", key=f"{key}_reset_st"):
            for lane, widget_key in enumerate(keys, 1):
                normal = reference.get(lane)
                st.session_state[widget_key] = float(normal) if pd.notna(normal) else 0.0
        columns = st.columns(3)
        values = []
        for lane, widget_key in enumerate(keys, 1):
            with columns[(lane - 1) % 3]:
                normal = reference.get(lane)
                st.caption(f"{lane}号艇の通常ST：{normal:.2f}秒" if pd.notna(normal) else f"{lane}号艇の通常ST：取得できません")
                available = pd.notna(normal) and 0 <= float(normal) <= 1
                entered = st.number_input(
                    f"{lane}号艇の予測に使うST（秒）", min_value=0.0, max_value=1.0,
                    value=float(normal) if available else 0.0, step=0.01,
                    format="%.2f", key=widget_key, disabled=not available,
                )
                changed = available and entered != float(normal)
                values.append(float(entered) if changed else None)
                if changed:
                    st.caption(f"通常STとの差：{entered - float(normal):+.2f}秒")
        st.caption("通常STは過去成績から作る予測用の参考値で、当日の実測STではありません（取得結果は最大5分キャッシュ）。")
        if any(value is not None for value in values):
            st.warning("仮定を置いた参考予測です。指定したSTが実現する確率や、厳密な条件付き的中確率ではありません。")
    return tuple(values) if any(value is not None for value in values) else None


def render_st_comparison(prediction) -> None:
    comparison = getattr(prediction, "st_comparison", None)
    if comparison is None:
        return
    st.info("以下はSTを変更した参考予測です。通常予測との違いを確認してください。")
    st.subheader("スタート条件を変えた結果")
    frame = comparison.sort_values("艇番")
    st.dataframe(frame.style.format({
        "通常ST（秒）": "{:.2f}", "変更後ST（秒）": "{:.2f}",
        "通常の予想順位": "{:.0f}", "変更後の予想順位": "{:.0f}",
    }).set_properties(**{"text-align": "center"}), hide_index=True, use_container_width=True)
