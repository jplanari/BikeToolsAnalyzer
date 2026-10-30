"""
Turns whatever a rider drops into the uploader into plain activity files.

Bike computers and platforms hand out rides in different wrappers: a `.fit` straight from a Garmin, Hammerhead,
Magene, Wahoo or Zwift; a `.fit.gz` from Strava or Garmin Connect; `.gpx` / `.tcx` exports; and `.zip` bundles such
as Garmin's bulk export (zips inside zips) or Zwift's activity folder. `expand_upload` opens them all and returns
the activities inside with the file type recognised from the content, not from the name.
"""
import gzip
import io
import os
import re
import zipfile
from dataclasses import dataclass

MAX_ACTIVITY_BYTES = 200 * 1024 * 1024        # one unpacked file
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024      # everything unpacked from one upload (zip-bomb guard)
MAX_DEPTH = 3                                 # zip in zip in gz ...
ACTIVITY_KINDS = ("fit", "gpx", "tcx")
UPLOAD_TYPES = ["fit", "gpx", "tcx", "gz", "zip"]


@dataclass
class Activity:
    name: str      # file name with the extension that matches the content
    data: bytes
    kind: str      # 'fit' | 'gpx' | 'tcx'


def sniff_kind(data):
    """'fit', 'gpx', 'tcx', 'gzip' or 'zip' from the first bytes of the file, else None."""
    head = bytes(data[:4096])
    if len(head) >= 12 and head[8:12] == b".FIT":
        return "fit"
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:4] == b"PK\x03\x04" or head[:4] == b"PK\x05\x06":
        return "zip"
    text = head.decode("utf-8", errors="ignore").lstrip("﻿ \t\r\n").lower()
    if text.startswith("<") or "<?xml" in text[:100]:
        if "<trainingcenterdatabase" in text or "trainingcenterdatabase" in text[:2000]:
            return "tcx"
        if "<gpx" in text:
            return "gpx"
    return None


def _stem(name):
    base = os.path.basename((name or "").replace("\\", "/")) or "ride"
    base = re.sub(r"\.(gz|gzip|zip)$", "", base, flags=re.I)
    return re.sub(r"\.(fit|gpx|tcx|xml)$", "", base, flags=re.I) or "ride"


def _hidden(member):
    parts = member.replace("\\", "/").split("/")
    return any(p.startswith(".") or p == "__MACOSX" for p in parts if p)


def expand_upload(name, data, _depth=0, _budget=None):
    """
    Returns (activities, notes). `notes` explains what was skipped (unknown file types, unreadable archives);
    an upload that holds nothing usable gives an empty list and a note, never an exception.
    """
    budget = _budget if _budget is not None else [MAX_TOTAL_BYTES]
    kind = sniff_kind(data)
    notes = []

    if kind in ACTIVITY_KINDS:
        if len(data) > MAX_ACTIVITY_BYTES:
            return [], [f"{name}: file is too large."]
        budget[0] -= len(data)
        return [Activity(f"{_stem(name)}.{kind}", bytes(data), kind)], notes

    if _depth >= MAX_DEPTH:
        return [], [f"{name}: archives are nested too deeply."]

    if kind == "gzip":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                inner = gz.read(MAX_ACTIVITY_BYTES + 1)
        except (OSError, EOFError) as e:
            return [], [f"{name}: could not unzip it ({e})."]
        if len(inner) > MAX_ACTIVITY_BYTES:
            return [], [f"{name}: file is too large."]
        return expand_upload(_stem(name), inner, _depth + 1, budget)

    if kind == "zip":
        out = []
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                members = [i for i in zf.infolist() if not i.is_dir() and not _hidden(i.filename)]
                for info in sorted(members, key=lambda i: i.filename):
                    if info.file_size > MAX_ACTIVITY_BYTES or info.file_size > budget[0]:
                        notes.append(f"{info.filename}: too large, skipped.")
                        continue
                    try:
                        inner = zf.read(info)
                    except Exception as e:
                        notes.append(f"{info.filename}: could not be read ({e}).")
                        continue
                    found, sub = expand_upload(info.filename, inner, _depth + 1, budget)
                    out.extend(found)
                    # unknown files inside archives (json, csv, pdf, ...) are normal in bulk exports: stay quiet
                    notes.extend(n for n in sub if "not a ride file" not in n)
        except zipfile.BadZipFile:
            return [], [f"{name}: the zip file is damaged."]
        if not out and not notes:
            notes.append(f"{name}: no rides found inside.")
        return out, notes

    return [], [f"{name}: not a ride file (expected FIT, GPX or TCX, or a zip / gz of them)."]
