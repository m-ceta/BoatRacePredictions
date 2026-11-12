from datetime import date

from src import live


def block(lane, value):
    return f'<div class="table1_boatImage1"><span class="table1_boatImage1Number is-type{lane}">{lane}</span><span class="table1_boatImage1Time">{value}</span></div>'


def test_parse_st_maps_to_boat_not_exhibition_order_and_preserves_flags():
    html = block(4, ".09") + block(1, "Ｆ.０２") + block(6, "L.50") + block(2, "0.15")
    assert live.parse_start_exhibition_st_html(html) == {4: "0.09", 1: "F0.02", 6: "L0.50", 2: "0.15"}


def test_missing_invalid_and_duplicate_st_are_not_guessed():
    html = block(1, "-") + block(2, "ST未発表") + block(3, "2.00") + block(4, ".12") + block(4, ".13") + block(5, ".00")
    assert live.parse_start_exhibition_st_html(html) == {5: "0.00"}
    assert live.parse_start_exhibition_st_html("<html>システムエラー</html>") == {}


def test_fetch_exhibition_st_failure_is_nonfatal(monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError()
    monkeypatch.setattr(live, "fetch_text", fail)
    assert live.fetch_boatrace_exhibition_st(date(2026, 11, 12), "15", 1) == {}
