"""Garmin TCX (Training Center XML) reader: the format of older Garmin Connect, Polar, TrainingPeaks and many exports."""
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _first(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _num(el, *path):
    for name in path:
        el = _first(el, name) if el is not None else None
    if el is None or el.text is None:
        return np.nan
    try:
        return float(el.text.strip())
    except ValueError:
        return np.nan


def _root(file_content):
    if isinstance(file_content, str):
        file_content = file_content.encode("utf-8")
    elif not isinstance(file_content, (bytes, bytearray)):
        file_content = file_content.read()
    try:
        return ET.fromstring(bytes(file_content))
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse TCX file: {e}")


def tcx_sport(file_content):
    """The activity's Sport attribute ('Biking', 'Running', 'Other') of the first activity, or None."""
    try:
        for act in _root(file_content).iter():
            if _local(act.tag) == "Activity":
                return act.attrib.get("Sport")
    except ValueError:
        pass
    return None


def parse_tcx(file_content):
    """DataFrame like parse_gpx / parse_fit: time, lat, lon, ele, hr, power, cadence, plus dist_m and speed when recorded."""
    rows = []
    for tp in _root(file_content).iter():
        if _local(tp.tag) != "Trackpoint":
            continue
        t = _first(tp, "Time")
        if t is None or not t.text:
            continue
        time = pd.to_datetime(t.text.strip(), utc=True, errors="coerce")
        if pd.isna(time):
            continue
        row = {"time": time.tz_localize(None),
               "lat": _num(tp, "Position", "LatitudeDegrees"),
               "lon": _num(tp, "Position", "LongitudeDegrees"),
               "ele": _num(tp, "AltitudeMeters"),
               "dist_m": _num(tp, "DistanceMeters"),
               "hr": _num(tp, "HeartRateBpm", "Value"),
               "cadence": _num(tp, "Cadence"),
               "power": np.nan, "speed": np.nan}
        ext = _first(tp, "Extensions")
        if ext is not None:
            for node in ext.iter():           # <TPX><Watts>, <Speed>, <RunCadence> under any namespace
                name = _local(node.tag)
                try:
                    val = float(node.text.strip()) if node.text and node.text.strip() else np.nan
                except ValueError:
                    continue
                if name == "Watts":
                    row["power"] = val
                elif name == "Speed":
                    row["speed"] = val
        rows.append(row)

    df = pd.DataFrame(rows, columns=["time", "lat", "lon", "ele", "dist_m", "hr", "cadence", "power", "speed"])
    if df.empty:
        return df
    df = df.sort_values("time").reset_index(drop=True)
    for col in ("speed", "dist_m"):
        if df[col].isna().all():
            df = df.drop(columns=col)
    return df
