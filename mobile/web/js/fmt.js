import { language, t } from './i18n.js';

const nf = (d = 0) => new Intl.NumberFormat(language(), { maximumFractionDigits: d, minimumFractionDigits: d });
export const num = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? '–' : nf(d).format(v));
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function dur(s) {                       // 3725 -> 1:02 h ; 1500 -> 25 min
  if (s === null || s === undefined) return '–';
  const m = Math.round(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)}:${String(m % 60).padStart(2, '0')} h` : `${m} min`;
}
export function clock(s) {                     // axis label: 1:30:00 / 12:30
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${String(m).padStart(2, '0')}` : `${m} min`;
}
export function span(sec) {                    // 5 -> 5 s, 60 -> 1 min, 3600 -> 1 h
  if (sec < 60) return `${sec} s`;
  if (sec < 3600) return `${sec / 60} min`;
  return `${sec / 3600} h`;
}
const parse = (iso) => (iso && iso.length <= 10 ? new Date(iso + 'T12:00:00') : new Date(iso));
export function dayLong(iso) {
  if (!iso) return '';
  return new Intl.DateTimeFormat(language(), { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' }).format(parse(iso));
}
export function dayShort(iso) {
  return iso ? new Intl.DateTimeFormat(language(), { day: 'numeric', month: 'short' }).format(parse(iso)) : '';
}
export function hour(iso) {
  return iso ? new Intl.DateTimeFormat(language(), { hour: '2-digit', minute: '2-digit' }).format(parse(iso)) : '';
}
export function title(ride) {                  // 'strava_123.json' -> "Strava ride 123"
  const m = /^(strava|wahoo)_(.+)$/i.exec(ride.name || '');
  if (m) return t(m[1].toLowerCase() === 'strava' ? 'Strava ride {id}' : 'Wahoo ride {id}', { id: m[2] });
  return ride.name || t('Ride');
}
