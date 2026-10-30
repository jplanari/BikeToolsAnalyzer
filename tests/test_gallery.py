"""Ride gallery: titles, search, newest-first paging and the last-7-days strip."""
import pandas as pd
import pytest

pytest.importorskip("pandas")
from src.viz import gallery as g


def rides():
    return [
        {'id': 1, 'filename': 'strava_111.json', 'date_time': '2026-10-01T07:00:00', 'moving_s': 3600, 'tss': 50},
        {'id': 2, 'filename': 'Sunday-loop.fit', 'date_time': '2026-10-25T09:30:00', 'moving_s': 7200, 'tss': 120},
        {'id': 3, 'filename': 'wahoo_77.fit', 'date_time': '2026-10-28T18:00:00', 'moving_s': 1800, 'tss': 30},
        {'id': 4, 'filename': 'mystery.gpx', 'date_time': None, 'moving_s': None, 'tss': None},
    ]


def test_titles_are_readable():
    assert g.ride_title('strava_111.json') == 'Strava ride 111'
    assert g.ride_title('wahoo_77.fit') == 'Wahoo ride 77'
    assert g.ride_title('Sunday-loop.fit') == 'Sunday-loop' and g.ride_title(None) == 'ride'


def test_newest_first_and_paging():
    shown, total = g.filter_rides(rides(), limit=2)
    assert [r['id'] for r in shown] == [3, 2] and total == 4
    shown, _ = g.filter_rides(rides(), limit=10)
    assert [r['id'] for r in shown][-1] == 4                      # a ride without a date goes last


def test_search_matches_title_month_and_weekday_words():
    ids = lambda q: [r['id'] for r in g.filter_rides(rides(), q)[0]]
    assert ids('sunday') == [2]                                   # title
    assert ids('october') == [3, 2, 1]                            # month name
    assert ids('wahoo oct') == [3]                                # every word must match
    assert ids('zzz') == []


def test_last_seven_days_strip():
    s = g.week_summary(rides(), today='2026-10-29')
    assert s['rides'] == 2 and s['hours'] == pytest.approx(2.5) and s['tss'] == pytest.approx(150)
    assert g.week_summary([], today='2026-10-29') == {'rides': 0, 'hours': 0.0, 'tss': 0.0}
