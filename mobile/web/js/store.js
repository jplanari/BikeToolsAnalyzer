// localStorage that never throws (private windows and blocked storage just forget).
export function load(key, fallback = null) {
  try { const v = localStorage.getItem('bt.' + key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
}
export function save(key, value) {
  try { localStorage.setItem('bt.' + key, JSON.stringify(value)); return true; } catch { return false; }
}
export function drop(key) { try { localStorage.removeItem('bt.' + key); } catch { /* nothing to do */ } }
