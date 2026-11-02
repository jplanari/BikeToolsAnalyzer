"""
The weekly e-mail: one PDF with the previous Monday-to-Sunday week (training analytics, all the deeper
analytics, then one section per ride) and a short summary in the body.

Nothing here needs Streamlit's UI, so the send can run in a background thread when the app is opened
(maybe_send_weekly). Which week goes out is decided by the database: claim_weekly_email lets exactly one
caller send each week.
"""
import threading
from datetime import datetime

import pandas as pd

from src.data import db
from src.data.reports import mailer
from src.analysis import nutrition as nut
from src.i18n import current_language, t, tn, use_language

MAX_RIDES = 14                     # ride sections per report: keeps the PDF and the build time bounded


def previous_week(today=None):
    """(monday, sunday) of the last complete Monday-Sunday week before `today`."""
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    this_monday = today - pd.Timedelta(days=today.weekday())
    return this_monday - pd.Timedelta(days=7), this_monday - pd.Timedelta(days=1)


def week_rides(user_name, monday, sunday):
    """Stored rides that started in the week, oldest first."""
    rides = db.get_user_rides(user_name)
    out = []
    for r in rides:
        when = pd.to_datetime(r.get('start_local') or r.get('date_time'), errors='coerce')   # the rider's own clock
        if pd.isna(when):
            continue
        day = (when.tz_convert(None) if when.tzinfo is not None else when).normalize()
        if monday <= day <= sunday:
            out.append(r)
    return sorted(out, key=lambda r: r['date_time'])


def _with_weather(df, user_name, ride_id):
    """The ride with its weather, as on the ride page: the stored one, else fetched now and kept for next time.
    A ride whose weather cannot be had (offline, no GPS) is reported without wind, never failed."""
    from src.data.providers import weather
    from src.data.parsers.gpx import calculate_bearing
    try:
        df['bearing'] = calculate_bearing(df)
        stored = db.get_ride_weather(user_name, ride_id)
        out, payload, _ = weather.load_or_fetch(df, stored)
        if payload is not None and stored is None:
            db.save_ride_weather(user_name, ride_id, payload)
        return out
    except Exception:
        return df


def _ride_figures(user, ride):
    """Figures of one ride's report, rebuilt from the stored file with the ride's own FTP / LTHR."""
    from src.analysis import power_profile as pp
    from src.analysis.climbs import detect_climbs
    from src.analysis.power import calculate_w_prime_balance
    from src.data.processing.history import ftp_used_for_ride
    from src.data.processing.pipeline import prepare_ride, summarize_ride
    from src.data.reports.ride_report_data import collect_ride_report
    from src.viz.ride_report_pdf import ride_figures

    filename, content = db.get_ride_file(user['name'], ride['id'])
    if content is None:
        raise ValueError(t("original file not stored"))
    df, dist_m, ascent_m = prepare_ride(filename or "", content)
    df = _with_weather(df, user['name'], ride['id'])
    ftp = ftp_used_for_ride(ride, user['ftp'])
    lthr = ride.get('lthr_at_ride') or user['lthr']
    summary = summarize_ride(df, ftp, lthr)
    if 'power' in df.columns and df['power'].notna().any() and ftp:
        df['w_prime_balance'] = calculate_w_prime_balance(df['power'], ftp, w_prime_cap_j=user.get('w_prime') or 20000)
    settings = {'ftp': ftp, 'lthr': lthr, 'weight': user['weight'], 'height': user.get('height')}
    best = recent = None
    if 'power' in df.columns:
        best = db.get_user_best_power(user['name'])
        today = pd.Timestamp.today().normalize()
        recent = pp.as_dict(pp.best_curve(db.get_power_records(user['name'], today - pd.Timedelta(days=89), today)))
    data = collect_ride_report(df, summary, settings, (filename or f"ride {ride['id']}").rsplit('.', 1)[0], dist_m,
                               ascent_m, climbs=detect_climbs(df, min_length=1000, join_dist=1000),
                               best_curve=best, recent_curve=recent, user_name=user['name'])
    return ride_figures(data)


def _cover(user_name, monday, sunday, rides, failures):
    from src.viz.report_pdf import INK, LEFT, SOFT, _new_page, _note
    fig = _new_page(t("Weekly report: {monday:%d %b} – {sunday:%d %b %Y}", monday=monday, sunday=sunday), user_name)
    lines = [t("Contents"), t("1. Training analytics of the week: load, power profile, durability, intensity, rides."),
             t("2. Deeper analytics: power and heart rate, power and cadence, recovery and efficiency "
               "(this week against the week before)."),
             t("3. One section per ride ({n} this week): summary, power and zones, route and climbs, "
               "aerodynamics, heat and laps.", n=len(rides))]
    y = 0.88
    fig.text(LEFT, y, lines[0], fontsize=11, fontweight='bold', color=INK, va='top')
    y -= 0.03
    for line in lines[1:]:
        y = _note(fig, y, line, size=9, color=INK)
    if failures:
        y = _note(fig, y - 0.01, t("Not included: {items}", items="; ".join(failures)), size=8, color=SOFT)
    return fig


def build_weekly_pdf(user_name, monday=None, sunday=None, today=None):
    """
    Returns (pdf_bytes, summary dict). Raises ValueError for an unknown user. A ride that cannot be read is
    listed on the first page instead of stopping the report.
    """
    from src.data.reports.deeper_data import collect_deeper_data
    from src.data.reports.report_data import collect_report_data
    from src.viz.deeper_pdf import deeper_figures
    from src.viz.report_pdf import report_figures, write_pdf

    if monday is None:
        monday, sunday = previous_week(today)
    user = db.get_user_data(user_name)
    if not user:
        raise ValueError(t("unknown user {user_name}", user_name=repr(user_name)))
    report = collect_report_data(user_name, period_days=7, today=sunday)
    rides = week_rides(user_name, monday, sunday)
    shown, failures, ride_figs = rides[:MAX_RIDES], [], []
    if len(rides) > MAX_RIDES:
        failures.append(t("{n} more rides (only the first {max} get a section)", n=len(rides) - MAX_RIDES, max=MAX_RIDES))
    for r in shown:
        try:
            ride_figs.extend(_ride_figures(user, r))
        except Exception as e:                       # unreadable file etc.: report it, keep going
            failures.append(f"{r.get('filename') or r['id']}: {e}")
    figures = ([_cover(user_name, monday, sunday, rides, failures)] + report_figures(report)
               + deeper_figures(collect_deeper_data(user_name, monday, sunday)) + ride_figs)
    pdf = write_pdf(figures, t("BikeTools Analyzer  ·  {user_name}  ·  week of {monday:%d %b %Y}", user_name=user_name, monday=monday),
                    t("Weekly report - {user_name} - {monday:%Y-%m-%d}", user_name=user_name, monday=monday))
    summary = {'rides': len(rides), 'distance_km': sum(r.get('distance_km') or 0 for r in rides),
               'ascent_m': sum(r.get('elevation_m') or 0 for r in rides),
               'hours': sum(r.get('moving_s') or 0 for r in rides) / 3600.0,
               'tss': sum(r.get('tss') or 0 for r in rides),
               'kcal': sum(nut.kcal_from_kj(r.get('kj')) or 0 for r in rides),
               'carb_g': sum(r.get('carb_g') or 0 for r in rides), 'pages': len(figures), 'failures': failures}
    return pdf, summary


def body_text(user_name, monday, sunday, summary):
    s = summary
    if s['rides']:
        stats = tn("{n} ride, {km:.0f} km, {m:.0f} m of climbing, {h:.1f} h moving time, {tss:.0f} TSS.",
                   "{n} rides, {km:.0f} km, {m:.0f} m of climbing, {h:.1f} h moving time, {tss:.0f} TSS.",
                   s['rides'], km=s['distance_km'], m=s['ascent_m'], h=s['hours'], tss=s['tss'])
        if s.get('kcal'):
            stats += " " + (t("About {kcal:,.0f} kcal burned on the bike, of which roughly {carb:,.0f} g carbohydrate.",
                              kcal=s['kcal'], carb=s['carb_g']) if s.get('carb_g') else
                            t("About {kcal:,.0f} kcal burned on the bike.", kcal=s['kcal']))
    else:
        stats = t("No rides this week.")
    return t("Hi {user_name},\n\nYour week {monday:%d %b} – {sunday:%d %b %Y}: {stats}\n\n"
             "The attached PDF ({pages} pages) has the training analytics, all the deeper analytics and a "
             "section for each ride.\n\nBikeTools Analyzer", user_name=user_name, monday=monday, sunday=sunday,
             stats=stats, pages=s['pages'])


def _user_language(user_name):
    """The rider's saved language (the e-mail goes out from a background thread, which has no browser language)."""
    try:
        return (db.get_user_data(user_name) or {}).get('language') or current_language()
    except Exception:
        return current_language()


def send_weekly(user_name, monday=None, sunday=None):
    """Builds and sends the report of that week (default: the previous one) to the user's address, in the rider's
    language."""
    with use_language(_user_language(user_name)):
        return _send_weekly(user_name, monday, sunday)


def _send_weekly(user_name, monday, sunday):
    cfg = db.get_weekly_email(user_name)
    if not cfg or not cfg.get('recipient'):
        raise mailer.MailError(t("No e-mail address is set for the weekly report."))
    smtp = db.get_smtp_settings()
    if not smtp:
        raise mailer.MailError(t("No mail server is set up yet."))
    if monday is None:
        monday, sunday = previous_week()
    pdf, summary = build_weekly_pdf(user_name, monday, sunday)
    msg = mailer.build_message(
        smtp['sender'], cfg['recipient'], t("Your training week {monday:%d %b} – {sunday:%d %b %Y}", monday=monday, sunday=sunday),
        body_text(user_name, monday, sunday, summary),
        attachments=[(f"weekly-report-{user_name}-{monday:%Y-%m-%d}.pdf", pdf, 'application/pdf')])
    mailer.send_message(msg, settings=smtp)
    return summary


def send_if_due(user_name, today=None):
    """
    Sends last week's report if it is switched on and not sent yet. Returns True when a message went out.
    The week is claimed in the database first; a failure gives the claim back so the next visit retries.
    """
    cfg = db.get_weekly_email(user_name)
    if not cfg or not cfg['enabled'] or not db.get_smtp_settings():
        return False
    monday, sunday = previous_week(today)
    if not db.claim_weekly_email(user_name, f"{monday:%Y-%m-%d}"):
        return False
    try:
        send_weekly(user_name, monday, sunday)
    except Exception as e:
        with use_language(_user_language(user_name)):
            db.release_weekly_email(user_name, t("failed {now:%d %b %H:%M}: {e}", now=datetime.now(), e=e), cfg['last_week'])
        return False
    with use_language(_user_language(user_name)):
        db.finish_weekly_email(user_name, t("sent {now:%d %b %Y %H:%M} (week of {monday:%d %b})", now=datetime.now(), monday=monday))
    return True


_started = False
_lock = threading.Lock()


def maybe_send_weekly():
    """
    Called when the app is opened: one background thread per server process checks every subscribed user.
    It does nothing visible; the outcome is shown in the User Corner.
    """
    global _started
    with _lock:
        if _started:
            return False
        _started = True

    def run():
        global _started
        try:
            for name in db.users_with_weekly_email():
                send_if_due(name)
        finally:
            _started = False              # next app visit may check again (cheap when nothing is due)

    threading.Thread(target=run, name="weekly-email", daemon=True).start()
    return True
