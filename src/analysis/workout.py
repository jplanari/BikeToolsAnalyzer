"""
Structured workouts: blocks of target power x time -> duration, NP, IF and TSS.

A block is a dict {name, reps, work_min, work_w, rest_min, rest_w}: `reps` repeats of
`work_min` at `work_w`, with `rest_min` at `rest_w` between the repeats (not after the
last one). A warm-up is one rep with no rest; "5 x 5 min at 300 W, 3 min easy" is
reps=5, work_min=5, work_w=300, rest_min=3, rest_w=120.

The session is turned into a 1 Hz power trace and scored exactly like a recorded ride
(Normalized Power, IF = NP / FTP, TSS = hours x IF^2 x 100), so planned and ridden load
are comparable on the same chart.
"""
import numpy as np
import pandas as pd

from src.analysis.power import NP

BLOCK_COLUMNS = ['name', 'reps', 'work_min', 'work_w', 'rest_min', 'rest_w']
MAX_SESSION_S = 12 * 3600


def _num(x, default=0.0):
    v = pd.to_numeric(x, errors='coerce')
    return default if pd.isna(v) else float(v)


def clean_blocks(blocks):
    """Valid blocks only, with numbers coerced; rows that last no time are dropped."""
    out = []
    for b in (blocks.to_dict('records') if hasattr(blocks, 'to_dict') else (blocks or [])):
        reps = max(1, int(round(_num(b.get('reps'), 1))))
        work_min, rest_min = max(0.0, _num(b.get('work_min'))), max(0.0, _num(b.get('rest_min')))
        work_w, rest_w = max(0.0, _num(b.get('work_w'))), max(0.0, _num(b.get('rest_w')))
        if work_min <= 0 and (rest_min <= 0 or reps < 2):
            continue
        name = b.get('name')
        out.append({'name': '' if name is None or (isinstance(name, float) and pd.isna(name)) else str(name),
                    'reps': reps, 'work_min': work_min, 'work_w': work_w,
                    'rest_min': rest_min, 'rest_w': rest_w})
    return out


def expand(blocks):
    """1 Hz target power (W) for the whole session."""
    parts = []
    for b in clean_blocks(blocks):
        work = np.full(int(round(b['work_min'] * 60)), b['work_w'])
        rest = np.full(int(round(b['rest_min'] * 60)), b['rest_w'])
        for i in range(b['reps']):
            parts.append(work)
            if i < b['reps'] - 1:
                parts.append(rest)
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)[:MAX_SESSION_S]


def session_stats(blocks, ftp):
    """{'duration_s', 'avg_power', 'np', 'if', 'tss', 'kj'}; zeros for an empty session."""
    p = expand(blocks)
    if len(p) == 0:
        return {'duration_s': 0, 'avg_power': 0.0, 'np': 0.0, 'if': 0.0, 'tss': 0.0, 'kj': 0.0}
    np_w = float(NP(pd.Series(p)))
    if_val = np_w / ftp if ftp else 0.0
    return {'duration_s': int(len(p)), 'avg_power': float(p.mean()), 'np': np_w, 'if': if_val,
            'tss': len(p) / 3600.0 * if_val ** 2 * 100, 'kj': float(p.sum() / 1000.0)}


def _min(x):
    return f"{x:g}'"


def describe(blocks):
    """Short text such as "15' @150 W + 4×8' @285 W (4' @140 W)"."""
    parts = []
    for b in clean_blocks(blocks):
        s = f"{b['reps']}×" if b['reps'] > 1 else ""
        s += f"{_min(b['work_min'])} @{b['work_w']:g} W"
        if b['reps'] > 1 and b['rest_min'] > 0:
            s += f" ({_min(b['rest_min'])} @{b['rest_w']:g} W)"
        parts.append(s)
    return " + ".join(parts)


def default_blocks(ftp):
    """A starting point for the editor: warm-up, 4 x 8 min threshold, cool-down."""
    ftp = ftp or 200
    w = lambda pct: int(round(ftp * pct / 100 / 5) * 5)
    return [
        {'name': 'Warm-up', 'reps': 1, 'work_min': 15, 'work_w': w(55), 'rest_min': 0, 'rest_w': 0},
        {'name': 'Intervals', 'reps': 4, 'work_min': 8, 'work_w': w(95), 'rest_min': 4, 'rest_w': w(50)},
        {'name': 'Cool-down', 'reps': 1, 'work_min': 10, 'work_w': w(50), 'rest_min': 0, 'rest_w': 0},
    ]
