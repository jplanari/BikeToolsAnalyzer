import numpy as np
import pytest

from src.analysis import workout as wo


def B(reps=1, work_min=0, work_w=0, rest_min=0, rest_w=0, name=''):
    return dict(name=name, reps=reps, work_min=work_min, work_w=work_w, rest_min=rest_min, rest_w=rest_w)


def test_steady_hour_at_ftp_is_100_tss():
    s = wo.session_stats([B(work_min=60, work_w=250)], ftp=250)
    assert (s['duration_s'], round(s['np']), round(s['if'], 3), round(s['tss'])) == (3600, 250, 1.0, 100)
    assert s['kj'] == pytest.approx(900)


def test_tss_scales_with_intensity_squared():
    s = wo.session_stats([B(work_min=60, work_w=175)], ftp=250)               # IF 0.7
    assert s['tss'] == pytest.approx(49, abs=0.1)
    half = wo.session_stats([B(work_min=30, work_w=250)], ftp=250)
    assert half['tss'] == pytest.approx(50, abs=0.1)


def test_rest_goes_between_repeats_not_after_the_last():
    p = wo.expand([B(reps=3, work_min=1, work_w=300, rest_min=2, rest_w=100)])
    assert len(p) == 3 * 60 + 2 * 120
    assert p[:60].tolist() == [300] * 60 and p[60:180].tolist() == [100] * 120 and p[-1] == 300


def test_variable_session_has_np_above_its_average_power():
    blocks = [B(reps=4, work_min=5, work_w=300, rest_min=5, rest_w=100)]       # 35 min
    s = wo.session_stats(blocks, 250)
    assert s['duration_s'] == 4 * 300 + 3 * 300
    assert s['avg_power'] == pytest.approx((4 * 300 * 300 + 3 * 300 * 100) / 2100)      # 214.3 W
    assert s['np'] > s['avg_power'] + 20
    steady = wo.session_stats([B(work_min=35, work_w=214.3)], 250)
    assert steady['np'] == pytest.approx(steady['avg_power'])
    assert s['tss'] > steady['tss']


def test_bad_rows_are_ignored_and_empty_session_is_zero():
    rows = [B(work_min=0, work_w=300), {'name': None, 'reps': None, 'work_min': 'x', 'work_w': None},
            B(reps=0, work_min=10, work_w=-50)]
    clean = wo.clean_blocks(rows)
    assert len(clean) == 1 and clean[0]['reps'] == 1 and clean[0]['work_w'] == 0
    assert wo.session_stats([], 250)['tss'] == 0
    assert wo.session_stats([B(work_min=10, work_w=200)], 0)['tss'] == 0           # no FTP: no score


def test_description_and_defaults():
    text = wo.describe([B(work_min=15, work_w=150), B(reps=4, work_min=8, work_w=285, rest_min=4, rest_w=140)])
    assert text == "15' @150 W + 4×8' @285 W (4' @140 W)"
    d = wo.default_blocks(250)
    assert [b['work_w'] for b in d] == [140, 240, 125]                              # 55 %, 95 %, 50 % of 250
    assert 50 < wo.session_stats(d, 250)['tss'] < 90


def test_session_is_capped_at_12_hours():
    assert len(wo.expand([B(work_min=20 * 60, work_w=100)])) == 12 * 3600
