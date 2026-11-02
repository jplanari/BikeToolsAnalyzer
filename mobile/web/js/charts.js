// Small hand-drawn SVG charts: no library, works offline. Colours come from CSS variables, so dark mode just works.
import { esc } from './fmt.js';

const NS = 'http://www.w3.org/2000/svg';

function niceTicks(lo, hi, n = 4) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const start = Math.floor(lo / step) * step, out = [];
  for (let v = start; out.length < 50; v += step) { out.push(+v.toFixed(10)); if (v >= hi - step * 1e-9) break; }   // the top tick covers the data
  return out;
}

function frame(el, height) {
  const w = Math.max(240, el.clientWidth || 320);
  return { w, h: height, m: { l: 38, r: 8, t: 8, b: 22 } };
}

/**
 * spec: { xs:[numbers], series:[{label,color,values:[number|null],width?}], height, xFmt(x), yFmt(v), xTicks:[{x,label}],
 *         logX?, zero? (include 0 on y), band?:{from,to} (shaded y range), hline?:{value,label} }
 * Dragging a finger (or the mouse) over the chart shows a cursor and the values at that point.
 */
export function lineChart(el, spec) {
  const { xs, series } = spec;
  const f = frame(el, spec.height || 200);
  const iw = f.w - f.m.l - f.m.r, ih = f.h - f.m.t - f.m.b;
  const all = series.flatMap((s) => s.values).filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
  if (!all.length || xs.length < 2) { el.innerHTML = ''; return; }
  let lo = Math.min(...all, spec.zero ? 0 : Infinity), hi = Math.max(...all, spec.zero ? 0 : -Infinity);
  const pad = (hi - lo) * 0.06 || 1;
  if (!spec.zero) lo -= pad;
  hi += pad;
  const ticks = niceTicks(lo, hi);
  const y0 = ticks[0], y1 = ticks[ticks.length - 1];
  const tx = spec.logX ? (x) => Math.log(x) : (x) => x;
  const xmin = tx(xs[0]), xmax = tx(xs[xs.length - 1]);
  const X = (x) => f.m.l + ((tx(x) - xmin) / (xmax - xmin || 1)) * iw;
  const Y = (v) => f.m.t + (1 - (v - y0) / (y1 - y0 || 1)) * ih;

  let g = '';
  for (const tk of ticks) g += `<line class="grid-line" x1="${f.m.l}" x2="${f.w - f.m.r}" y1="${Y(tk)}" y2="${Y(tk)}"/><text x="${f.m.l - 6}" y="${Y(tk) + 4}" text-anchor="end">${esc(spec.yFmt ? spec.yFmt(tk) : tk)}</text>`;
  for (const tk of spec.xTicks || []) g += `<text x="${X(tk.x)}" y="${f.h - 4}" text-anchor="middle">${esc(tk.label)}</text>`;
  if (spec.band) g += `<rect x="${f.m.l}" width="${iw}" y="${Y(spec.band.to)}" height="${Math.max(0, Y(spec.band.from) - Y(spec.band.to))}" fill="var(--s3)" opacity=".12"/>`;
  if (spec.hline) g += `<line x1="${f.m.l}" x2="${f.w - f.m.r}" y1="${Y(spec.hline)}" y2="${Y(spec.hline)}" stroke="var(--text-3)" stroke-dasharray="3 3"/>`;
  for (const s of series) {
    let d = '', pen = false;
    s.values.forEach((v, i) => {
      if (v === null || v === undefined || Number.isNaN(v)) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`;
      pen = true;
    });
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" stroke-linejoin="round" stroke-linecap="round"/>`;
  }
  g += '<line class="cursor" y1="' + f.m.t + '" y2="' + (f.h - f.m.b) + '" stroke="var(--text-3)" stroke-width="1" visibility="hidden"/>';
  for (const s of series) g += `<circle class="dot" r="4" fill="${s.color}" stroke="var(--surface)" stroke-width="2" visibility="hidden" data-label="${esc(s.label)}"/>`;
  const legend = series.length > 1
    ? `<div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join('')}</div>` : '';
  el.classList.add('chart');
  el.innerHTML = `<svg viewBox="0 0 ${f.w} ${f.h}" role="img" aria-label="${esc(spec.label || '')}">${g}</svg><div class="tip" hidden></div>${legend}`;

  const svg = el.querySelector('svg'), tip = el.querySelector('.tip'), cur = el.querySelector('.cursor');
  const dots = [...el.querySelectorAll('.dot')];
  const at = (clientX) => {
    const r = svg.getBoundingClientRect();
    const px = ((clientX - r.left) / r.width) * f.w;
    let best = 0, bd = Infinity;
    xs.forEach((x, i) => { const d = Math.abs(X(x) - px); if (d < bd) { bd = d; best = i; } });
    return best;
  };
  const show = (i) => {
    const x = X(xs[i]);
    cur.setAttribute('x1', x); cur.setAttribute('x2', x); cur.removeAttribute('visibility');
    let rows = '';
    series.forEach((s, k) => {
      const v = s.values[i];
      if (v === null || v === undefined || Number.isNaN(v)) { dots[k].setAttribute('visibility', 'hidden'); return; }
      dots[k].setAttribute('cx', x); dots[k].setAttribute('cy', Y(v)); dots[k].removeAttribute('visibility');
      rows += `<div><i style="background:${s.color}"></i>${esc(s.label)}: <b>${esc(spec.vFmt ? spec.vFmt(v, s) : (spec.yFmt ? spec.yFmt(v) : v))}</b></div>`;
    });
    tip.innerHTML = `<b>${esc(spec.xFmt ? spec.xFmt(xs[i]) : xs[i])}</b>${rows}`;
    tip.hidden = false;
    const scale = svg.getBoundingClientRect().width / f.w;
    tip.style.left = Math.min(Math.max(x * scale, 60), el.clientWidth - 60) + 'px';
    tip.style.top = '0px';
  };
  const hide = () => { tip.hidden = true; cur.setAttribute('visibility', 'hidden'); dots.forEach((d) => d.setAttribute('visibility', 'hidden')); };
  svg.addEventListener('pointerdown', (e) => show(at(e.clientX)));
  svg.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse' || e.buttons) show(at(e.clientX)); });
  svg.addEventListener('pointerleave', hide);
  svg.addEventListener('pointerup', () => setTimeout(hide, 1800));
}

/** Vertical bars from the baseline, rounded at the top; tap a bar to read it. spec: {labels, values, color, yFmt, vFmt, height} */
export function barChart(el, spec) {
  const f = frame(el, spec.height || 150);
  const iw = f.w - f.m.l - f.m.r, ih = f.h - f.m.t - f.m.b;
  const top = Math.max(...spec.values, 1);
  const ticks = niceTicks(0, top, 3), y1 = ticks[ticks.length - 1];
  const n = spec.values.length, slot = iw / n, bw = Math.max(3, Math.min(26, slot - 2));
  let g = '';
  for (const tk of ticks) {
    const y = f.m.t + (1 - tk / y1) * ih;
    g += `<line class="grid-line" x1="${f.m.l}" x2="${f.w - f.m.r}" y1="${y}" y2="${y}"/><text x="${f.m.l - 6}" y="${y + 4}" text-anchor="end">${esc(spec.yFmt ? spec.yFmt(tk) : tk)}</text>`;
  }
  spec.values.forEach((v, i) => {
    const h = (v / y1) * ih, x = f.m.l + i * slot + (slot - bw) / 2, y = f.m.t + ih - h, r = Math.min(4, bw / 2, h);
    if (h > 0) g += `<path data-i="${i}" fill="${spec.color || 'var(--s1)'}" d="M${x},${f.m.t + ih}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${f.m.t + ih}Z"/>`;
  });
  const every = Math.ceil(n / 6);
  spec.labels.forEach((l, i) => { if (i % every === 0 || i === n - 1) g += `<text x="${f.m.l + i * slot + slot / 2}" y="${f.h - 4}" text-anchor="middle">${esc(l)}</text>`; });
  el.classList.add('chart');
  el.innerHTML = `<svg viewBox="0 0 ${f.w} ${f.h}" role="img" aria-label="${esc(spec.label || '')}">${g}</svg><div class="tip" hidden></div>`;
  const tip = el.querySelector('.tip');
  el.querySelector('svg').addEventListener('pointerdown', (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const i = Math.max(0, Math.min(n - 1, Math.floor((((e.clientX - r.left) / r.width) * f.w - f.m.l) / slot)));
    tip.innerHTML = `<b>${esc(spec.labels[i])}</b><div>${esc(spec.vFmt ? spec.vFmt(spec.values[i], i) : spec.values[i])}</div>`;
    tip.hidden = false;
    tip.style.left = Math.min(Math.max((f.m.l + i * slot + slot / 2) * (r.width / f.w), 50), el.clientWidth - 50) + 'px';
    tip.style.top = '0px';
    clearTimeout(tip._t); tip._t = setTimeout(() => { tip.hidden = true; }, 2200);
  });
}

/** The route as a line: no map tiles (so it works offline and sends nothing anywhere). points: [[lat, lon], ...] */
export function trackMap(el, points) {
  if (!points || points.length < 2) { el.innerHTML = ''; return; }
  const lat0 = points.reduce((a, p) => a + p[0], 0) / points.length, k = Math.cos((lat0 * Math.PI) / 180);
  const xy = points.map((p) => [p[1] * k, -p[0]]);
  const xs = xy.map((p) => p[0]), ys = xy.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const w = 320, pad = 14, span = Math.max(x1 - x0, y1 - y0, 1e-9);
  const sc = (w - 2 * pad) / span, h = Math.max(120, Math.min(260, (y1 - y0) * sc + 2 * pad));
  const P = ([x, y]) => [pad + (x - x0) * sc + ((w - 2 * pad) - (x1 - x0) * sc) / 2, pad + (y - y0) * sc + ((h - 2 * pad) - (y1 - y0) * sc) / 2];
  const d = xy.map((p, i) => (i ? 'L' : 'M') + P(p).map((v) => v.toFixed(1)).join(',')).join('');
  const [sx, sy] = P(xy[0]), [ex, ey] = P(xy[xy.length - 1]);
  el.innerHTML = `<svg class="map" viewBox="0 0 ${w} ${h}" role="img" aria-label=""><path d="${d}"/><circle cx="${sx}" cy="${sy}" r="5" fill="var(--good)" stroke="var(--surface)" stroke-width="2"/><circle cx="${ex}" cy="${ey}" r="5" fill="var(--bad)" stroke="var(--surface)" stroke-width="2"/></svg>`;
}
