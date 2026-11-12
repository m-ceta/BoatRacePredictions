"""Shared hypothetical start-timing controls for the prediction UIs."""

import streamlit as st


def render_st_adjustments(key: str) -> tuple[float, ...] | None:
    with st.expander("スタートの早い・遅いを仮定して予測する", expanded=False):
        st.caption("0.00なら通常予測です。マイナスは早いスタート、プラスは遅いスタートを仮定します。")
        keys = [f"{key}_st_delta_{lane}" for lane in range(1, 7)]
        if st.button("すべて通常に戻す", key=f"{key}_reset_st"):
            for widget_key in keys:
                st.session_state[widget_key] = 0.0
        columns = st.columns(3)
        values = []
        for lane, widget_key in enumerate(keys, 1):
            with columns[(lane - 1) % 3]:
                values.append(st.number_input(
                    f"{lane}号艇の補正（秒）", min_value=-0.10, max_value=0.10,
                    value=0.0, step=0.01, format="%+.2f", key=widget_key,
                ))
        if any(values):
            st.warning("仮定を置いた参考予測です。指定したSTが実現する確率や、厳密な条件付き的中確率ではありません。")
    return tuple(values) if any(values) else None


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
