"""
Solo or group: the label that separates drafting from the other explanations of a head-wind / tail-wind CdA difference.

A group ride shelters the rider from the head wind, so the CdA fitted on its head-wind stretches comes out low even if
the wind model is right. The reports therefore split by ride type when they know it. Two sources, the second wins:

- Strava rides imported after `athlete_count` was added to the stored fields carry it: 1 is solo, more is group.
- A CSV with the columns `ride_id` (or `name`, the stored file name) and `type` (solo or group). Make a template with
  `--tags-template FILE`: it lists every ride, with the type filled in where Strava gave it, ready to complete.

Rides with no label are reported as `unknown` and left out of the solo/group comparison.
"""
import csv
import json

SOLO, GROUP, UNKNOWN = "solo", "group", "unknown"
KNOWN = (SOLO, GROUP)


def type_from_content(content):
    """'solo' / 'group' from a stored Strava payload's athlete_count; None when the file has no such information."""
    try:
        act = json.loads(content).get("activity", {})
        n = act.get("athlete_count")
    except (ValueError, TypeError, AttributeError):
        return None
    if isinstance(n, bool) or not isinstance(n, (int, float)) or n < 1:
        return None
    return SOLO if n == 1 else GROUP


def read_tags(path):
    """{ride_id (int) or file name (str): 'solo' | 'group'} from a CSV; raises ValueError naming the bad line."""
    tags = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = [c.strip().lower() for c in (reader.fieldnames or [])]
        if "type" not in fields or not ({"ride_id", "name"} & set(fields)):
            raise ValueError("the tags file needs a header with `type` and `ride_id` (or `name`)")
        for n, row in enumerate(reader, start=2):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            kind = row["type"].lower()
            if not kind:
                continue                              # an unfilled row of the template: leave the ride unlabelled
            if kind not in KNOWN:
                raise ValueError(f"line {n}: type must be solo or group, not {row['type']!r}")
            if row.get("ride_id"):
                try:
                    tags[int(row["ride_id"])] = kind
                except ValueError:
                    raise ValueError(f"line {n}: ride_id must be a number, not {row['ride_id']!r}")
            elif row.get("name"):
                tags[row["name"]] = kind
    return tags


def apply_tags(prepared, tags=None):
    """Set p['ride_type'] on each prepared ride: the file's label, else what the import recorded, else unknown."""
    tags = tags or {}
    for p in prepared:
        p['ride_type'] = tags.get(p['ride_id']) or tags.get(p.get('name')) or p.get('ride_type') or UNKNOWN
    return prepared


def write_template(prepared, path):
    """A CSV to fill in: every ride with the type known so far (blank when unknown)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ride_id", "date", "name", "type"])
        for p in prepared:
            kind = p.get('ride_type')
            w.writerow([p['ride_id'], (p.get('date') or "")[:10], p.get('name') or "", kind if kind in KNOWN else ""])
    return len(prepared)


def summary(prepared):
    counts = {SOLO: 0, GROUP: 0, UNKNOWN: 0}
    for p in prepared:
        counts[p.get('ride_type') if p.get('ride_type') in KNOWN else UNKNOWN] += 1
    return counts
