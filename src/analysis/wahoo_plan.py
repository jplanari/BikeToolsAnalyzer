"""
Interval-built sessions -> Wahoo plan.json (schema 1.0.0, https://cloud-api.wahooligan.com/docs/plan-json-format.pdf).

Power targets are absolute watts (a +/-3 % band so the head unit shows a range), so a session that
was already sent does not change if your FTP does. Rest between repeats only, like the planner:
"4 x 8' (4' rest)" is a repeat of [work, rest] run 3 times, then one last work step.
"""
import json

from src.analysis import workout as wo

SCHEMA_VERSION = "1.0.0"
BAND = 0.03                      # +/-3 % around the target power
MAX_DESCRIPTION = 5000


def intensity_type(watts, ftp, rest=False):
    """Wahoo's step label, from power relative to FTP."""
    if rest:
        return "recover"
    if not ftp:
        return "active"
    f = watts / ftp
    if f >= 1.06:
        return "map"
    if f >= 0.91:
        return "ftp"
    if f >= 0.76:
        return "tempo"
    return "active"


def _target(watts):
    watts = max(1, int(round(watts)))
    return [{"type": "watts", "low": int(round(watts * (1 - BAND))), "high": int(round(watts * (1 + BAND)))}]


def _step(name, minutes, watts, ftp, rest=False, label=None):
    return {"name": name, "exit_trigger_type": "time", "exit_trigger_value": int(round(minutes * 60)),
            "intensity_type": label or intensity_type(watts, ftp, rest), "targets": _target(watts)}


def intervals(blocks, ftp):
    """The plan's interval list for cleaned blocks."""
    out = []
    for n, b in enumerate(wo.clean_blocks(blocks), 1):
        name = b['name'] or f"Block {n}"
        work = _step(name, b['work_min'], b['work_w'], ftp) if b['work_min'] > 0 else None
        rest = _step("Recover", b['rest_min'], b['rest_w'], ftp, rest=True) if b['rest_min'] > 0 else None
        reps = b['reps']
        if reps == 1:
            if work:
                out.append(work)
        elif rest:
            if work:
                if reps > 2:                          # reps-1 passes of [work, rest], then the last work
                    out.append({"name": f"{reps} x {name}", "exit_trigger_type": "repeat",
                                "exit_trigger_value": reps - 2, "intervals": [work, rest]})
                else:
                    out.extend([work, rest])
                out.append(work)
            else:       # rest-only block: nothing to repeat around
                out.extend([rest] * (reps - 1))
        elif work:
            out.append({"name": f"{reps} x {name}", "exit_trigger_type": "repeat",
                        "exit_trigger_value": reps - 1, "intervals": [work]})
    return out


def plan_dict(blocks, ftp, name, description=None, outdoor=True):
    """The plan.json content."""
    desc = (description or wo.describe(blocks) or name)[:MAX_DESCRIPTION]
    header = {"name": name, "version": SCHEMA_VERSION, "description": desc,
              "workout_type_family": 0, "workout_type_location": 1 if outdoor else 0}
    if ftp:
        header["ftp"] = int(ftp)
    return {"header": header, "intervals": intervals(blocks, ftp)}


def plan_json(blocks, ftp, name, description=None, outdoor=True):
    return json.dumps(plan_dict(blocks, ftp, name, description, outdoor), separators=(",", ":"))


def total_seconds(items):
    """Duration of an interval list, counting every repeat pass (to check against the planner)."""
    t = 0
    for i in items:
        if i["exit_trigger_type"] == "repeat":
            t += (i["exit_trigger_value"] + 1) * total_seconds(i["intervals"])
        else:
            t += i["exit_trigger_value"]
    return t
