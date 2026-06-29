from __future__ import annotations

from src.recent_backtest import parse_trifecta_payouts_from_lines


def test_parse_trifecta_payouts_from_lines_reads_race_detail_payouts() -> None:
    lines = """
STARTK
24KBGN
大　村［成績］      5/24

   第 1日          2026/ 5/24                             ボートレース大　村

   1R       予選　　　　                 H1800m  晴　  風  北西　 2m  波　  1cm
-------------------------------------------------------------------------------
        ３連単   1-4-2     7650  人気    36

   2R       予選　　　　                 H1800m  晴　  風  北西　 1m  波　  1cm
-------------------------------------------------------------------------------
        ３連単   3-6-2    23210  人気    81
""".splitlines()

    payouts = parse_trifecta_payouts_from_lines(lines)

    assert payouts["race_id"].tolist() == [
        "2026-05-24_24_01",
        "2026-05-24_24_02",
    ]
    assert payouts["actual_trifecta"].tolist() == ["1-4-2", "3-6-2"]
    assert payouts["trifecta_payout"].tolist() == [7650.0, 23210.0]
