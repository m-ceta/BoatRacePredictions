import tempfile

import pytest


def test_st_controls_default_to_reference_accept_absolute_input_and_reset(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import pandas as pd
    import app.st_scenario as ui

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(ui, "load_st_reference", lambda *args: pd.DataFrame({
        "lane": range(1, 7), "start_timing": [.16] * 6,
    }))
    monkeypatch.setattr(ui, "load_exhibition_st", lambda *args: {1: "0.15", 2: "F0.02"})
    app = AppTest.from_string('''
import pandas as pd
import streamlit as st
import app.st_scenario as ui
values = ui.render_st_adjustments("test", config_path="unused", venue="15", race_no=1, race_date="2026-11-12")
st.json(list(values) if values is not None else [])
''').run()
    assert not app.exception
    assert [control.value for control in app.number_input] == [.16] * 6
    captions = [caption.value for caption in app.caption]
    assert "展示ST：0.15秒" in captions
    assert "展示ST：F0.02秒" in captions
    assert "展示ST：未発表・取得できません" in captions
    app.number_input[0].set_value(.13).run()
    assert not app.exception
    assert app.number_input[0].value == .13
    assert app.json[0].value == '[0.13, null, null, null, null, null]'
    app.button[0].click().run()
    assert not app.exception
    assert [control.value for control in app.number_input] == [.16] * 6
