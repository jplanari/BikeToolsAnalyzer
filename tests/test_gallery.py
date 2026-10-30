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


def test_the_same_ride_from_two_sources_appears_once():
    dup = [
        {'id': 1, 'filename': 'strava_5.json', 'date_time': '2026-10-28T12:38:01', 'moving_s': 7000, 'tss': 99},
        {'id': 2, 'filename': 'wahoo_9.fit', 'date_time': '2026-10-28T12:38:20', 'moving_s': 7000, 'tss': 99},
        {'id': 3, 'filename': 'other.fit', 'date_time': '2026-10-27T08:00:00', 'moving_s': 3000, 'tss': 40},
    ]
    kept = g.collapse_duplicates(dup)
    assert sorted(r['id'] for r in kept) == [2, 3]                  # the recorder's FIT file beats Strava's copy
    shown, total = g.filter_rides(dup)
    assert total == 2 and [r['id'] for r in shown] == [2, 3]
    assert g.week_summary(dup, today='2026-10-29')['rides'] == 2    # and it is not counted twice either
    assert g.collapse_duplicates([dup[0], {**dup[0], 'id': 8, 'filename': 'x.gpx'}])[0]['filename'] == 'x.gpx'
    far = {**dup[1], 'id': 4, 'date_time': '2026-10-28T15:00:00'}
    assert len(g.collapse_duplicates([dup[0], far])) == 2            # different start times are different rides
