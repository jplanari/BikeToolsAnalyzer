import { api, ApiError } from './api.js';
import { lineChart, barChart, trackMap } from './charts.js';
import { clock, dayLong, dayShort, dur, esc, hour, num, span, title } from './fmt.js';
import { t } from './i18n.js';

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#2a78d6';

const kpi = (label, value, unit = '', sub = '') =>
  `<div class="kpi"><div class="l">${esc(label)}</div><div class="v">${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ''}</div>${sub ? `<div class="s">${esc(sub)}</div>` : ''}</div>`;
const seg = (id, options, current) =>
  `<div class="seg" id="${id}" role="group">${options.map(([k, l]) => `<button data-k="${esc(k)}" aria-pressed="${k === current}">${esc(l)}</button>`).join('')}</div>`;
function wireSeg(root, id, on) {
  const box = root.querySelector('#' + id);
  box.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    box.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b));
    on(b.dataset.k);
  });
}
const skeleton = (n = 4) => '<div class="skeleton"></div>'.repeat(n);
export const errorMessage = (e) => ({
  auth: t('The server did not accept this phone. Pair it again.'),
  throttled: t('Too many attempts. Wait a minute and try again.'),
  network: t('Cannot reach the server. Check the address and your connection.'),
}[e && e.kind] || t('Something went wrong on the server.'));

/** Shows a skeleton, runs `load`, then `paint`. Errors become a message with a retry. */
async function show(root, ctx, load, paint) {
  root.innerHTML = skeleton();
  try {
    const res = await load();
    if (ctx.stale()) return;
    ctx.offline(res.stale || (Array.isArray(res) && res.some((r) => r && r.stale)));
    paint(res);
  } catch (e) {
    if (ctx.stale()) return;
    if (e instanceof ApiError && e.kind === 'auth') { ctx.signedOut(); return; }
    root.innerHTML = `<div class="empty"><p>${esc(errorMessage(e))}</p><button class="btn secondary" id="retry">${esc(t('Try again'))}</button></div>`;
    root.querySelector('#retry').addEventListener('click', () => show(root, ctx, load, paint));
  }
}

// --- Rides -----------------------------------------------------------------------------------------------------

export async function ridesView(root, ctx) {
  ctx.title(t('Rides'));
  const PAGE = 30;
  let shown = [], total = 0;
  const card = (r) => `<a class="card ride" href="#/ride/${r.id}">
    <div class="row"><span class="when">${esc(dayLong(r.when))}</span><span class="faint">${esc(hour(r.when))}</span></div>
    <div class="name">${esc(title(r))}</div>
    <div class="stats"><span><b>${num(r.distance_km, 1)}</b> km</span><span><b>${esc(dur(r.moving_s))}</b></span>
    ${r.elevation_m ? `<span><b>${num(r.elevation_m)}</b> m</span>` : ''}${r.np ? `<span><b>${num(r.np)}</b> W</span>` : ''}${r.tss ? `<span><b>${num(r.tss)}</b> TSS</span>` : ''}</div></a>`;
  const paint = () => {
    root.innerHTML = (shown.length ? shown.map(card).join('') : `<div class="empty">${esc(t('No rides yet. Upload or import rides in the desktop app.'))}</div>`)
      + (shown.length < total ? `<button class="btn secondary" id="more">${esc(t('Load more'))}</button>` : '');
    const more = root.querySelector('#more');
    if (more) more.addEventListener('click', async () => {
      more.disabled = true;
      try { const { data } = await api(`/rides?limit=${PAGE}&offset=${shown.length}`); shown = shown.concat(data.rides); total = data.total; paint(); }
      catch (e) { more.disabled = false; more.textContent = errorMessage(e); }
    });
  };
  await show(root, ctx, () => api(`/rides?limit=${PAGE}`), ({ data }) => { shown = data.rides; total = data.total; paint(); });
}

// --- One ride --------------------------------------------------------------------------------------------------

function zoneBars(zones) {
  const total = zones.reduce((a, z) => a + (z.seconds || 0), 0);
  if (!total) return '';
  const w = Math.min(120, Math.max(28, ...zones.map((z) => String(z.zone).length * 7)));
  return zones.map((z) => `<div class="bar"><span style="width:${w}px;flex:none">${esc(z.zone)}</span><span class="track"><span class="fill" style="display:block;width:${(100 * z.seconds / total).toFixed(1)}%"></span></span><span>${esc(dur(z.seconds))}</span></div>`).join('');
}

function energyCard(e) {
  if (!e) return '';
  let h = `<h2>${esc(t('Energy and fuel'))}</h2><div class="grid">${kpi(t('Energy burned'), num(e.kcal), 'kcal', `${num(e.kj)} kJ`)}`;
  if (e.carb_g !== null) h += kpi(t('Carbohydrate'), num(e.carb_g), 'g', e.carb_pct !== null ? t('{pct} % of the energy', { pct: num(e.carb_pct) }) : '');
  if (e.fat_g !== null) h += kpi(t('Fat'), num(e.fat_g), 'g');
  if (e.store_used_pct !== null) h += kpi(t('Glycogen store used'), num(e.store_used_pct), '%', e.load ? t(e.load) : '');
  h += '</div>';
  if (e.guide && e.guide.rate_high) {
    h += `<div class="card"><div class="muted">${esc(t('Guide for a ride like this'))}</div><div><b>${num(e.guide.rate_low)}–${num(e.guide.rate_high)}</b> ${esc(t('g/h'))} · ${num(e.guide.total_low_g)}–${num(e.guide.total_high_g)} g</div></div>`;
  }
  if (e.intake) {
    const i = e.intake;
    h += `<div class="card"><div class="row"><span class="muted">${esc(t('What you ate'))}</span>${i.status ? `<span class="pill">${esc(t(i.status))}</span>` : ''}</div>
      <div><b>${num(i.eaten_g)}</b> g${i.carb_rate !== null ? ` · ${num(i.carb_rate)} ${esc(t('g/h'))}` : ''}${i.fluid_ml !== null ? ` · ${num(i.fluid_ml)} ml` : ''}${i.coverage_pct !== null ? ` · ${esc(t('{pct} % of the carbohydrate burned', { pct: num(i.coverage_pct) }))}` : ''}</div></div>`;
  }
  h += `<p class="note">${esc(t('Estimated from the power on the pedals, not measured. Log what you ate in the desktop app.'))}</p>`;
  return h;
}

export async function rideView(root, ctx, id) {
  ctx.title(t('Ride'));
  ctx.back('#/rides');
  await show(root, ctx, async () => {
    const [ride, series] = await Promise.all([api(`/rides/${id}`), api(`/rides/${id}/series`).catch(() => null)]);
    return Object.assign([ride, series], { stale: ride.stale });
  }, ([ride, series]) => {
    const r = ride.data, s = series && series.data && !series.data.error ? series.data : null;
    ctx.title(dayShort((r.when || '').slice(0, 10)) || t('Ride'));
    const channels = [];
    if (s && s.power) channels.push(['power', t('Power'), 'W', 0]);
    if (s && s.hr) channels.push(['hr', t('Heart rate'), 'bpm', 0]);
    if (s && s.speed_kph) channels.push(['speed_kph', t('Speed'), 'km/h', 1]);
    if (s && s.elevation) channels.push(['elevation', t('Elevation'), 'm', 0]);
    root.innerHTML = `
      <div class="row"><div><div class="when" style="font-weight:650">${esc(dayLong(r.when))}</div><div class="faint">${esc(hour(r.when))} · ${esc(title(r))}</div></div></div>
      <h2>${esc(t('Summary'))}</h2>
      <div class="grid">${kpi(t('Distance'), num(r.distance_km, 1), 'km')}${kpi(t('Moving time'), dur(r.moving_s))}${kpi(t('Elevation'), num(r.elevation_m), 'm')}${kpi(t('Avg speed'), num(r.avg_speed_kph, 1), 'km/h')}
      ${r.np ? kpi(t('Normalized power'), num(r.np), 'W', r.avg_power ? t('avg {w} W', { w: num(r.avg_power) }) : '') : ''}
      ${r.tss ? kpi('TSS', num(r.tss), '', r.if ? `IF ${num(r.if, 2)}` : '') : ''}
      ${r.kj ? kpi(t('Work'), num(r.kj), 'kJ') : ''}${r.avg_hr ? kpi(t('Avg heart rate'), num(r.avg_hr), 'bpm') : ''}
      ${r.vi ? kpi(t('Variability index'), num(r.vi, 2)) : ''}${r.ef ? kpi(t('Efficiency factor'), num(r.ef, 2)) : ''}${r.decoupling_pct !== null ? kpi(t('Aerobic decoupling'), num(r.decoupling_pct, 1), '%') : ''}</div>
      ${channels.length ? `<h2>${esc(t('Ride profile'))}</h2>${seg('chan', channels.map((c) => [c[0], c[1]]), channels[0][0])}<div id="chart"></div>` : ''}
      ${s && s.track ? `<h2>${esc(t('Route'))}</h2><div id="map"></div>` : ''}
      ${r.zones.power.length && r.zones.power.some((z) => z.seconds) ? `<h2>${esc(t('Time in power zones'))}</h2><div class="card">${zoneBars(r.zones.power)}</div>` : ''}
      ${r.zones.hr.length && r.zones.hr.some((z) => z.seconds) ? `<h2>${esc(t('Time in heart-rate zones'))}</h2><div class="card">${zoneBars(r.zones.hr)}</div>` : ''}
      ${r.power_curve.length ? `<h2>${esc(t('Best efforts in this ride'))}</h2><div class="card"><table><thead><tr><th>${esc(t('Duration'))}</th><th>${esc(t('Watts'))}</th><th>W/kg</th></tr></thead><tbody>${r.power_curve.map((p) => `<tr><td>${esc(span(p.seconds))}</td><td>${num(p.watts)}</td><td>${r.weight_kg ? num(p.watts / r.weight_kg, 2) : '–'}</td></tr>`).join('')}</tbody></table></div>` : ''}
      ${energyCard(r.energy)}
      ${s === null && series && series.data && series.data.error ? `<p class="note">${esc(t('The ride file could not be read, so there is no chart.'))}</p>` : ''}`;
    if (channels.length) {
      const draw = (key) => {
        const [, label, unit, digits] = channels.find((c) => c[0] === key);
        const n = s.points, step = s.seconds / n;
        const colour = css(key === 'power' ? '--s1' : key === 'hr' ? '--s2' : '--s3');
        lineChart(root.querySelector('#chart'), {
          xs: Array.from({ length: n }, (_, i) => Math.round(i * step)), label,
          series: [{ label, color: colour, values: s[key] }],
          xTicks: [0, 0.25, 0.5, 0.75, 1].map((f) => ({ x: Math.round(f * (n - 1) * step), label: clock(Math.round(f * (n - 1) * step)) })),
          xFmt: (x) => clock(x), yFmt: (v) => num(v, 0), vFmt: (v) => `${num(v, digits)} ${unit}`, zero: key === 'power',
        });
      };
      draw(channels[0][0]);
      wireSeg(root, 'chan', draw);
      ctx.onResize(() => { const b = root.querySelector('#chan [aria-pressed="true"]'); if (b) draw(b.dataset.k); });
    }
    if (s && s.track) trackMap(root.querySelector('#map'), s.track);
  });
}

// --- Training load ---------------------------------------------------------------------------------------------

export async function trainingView(root, ctx) {
  ctx.title(t('Training'));
  let range = 90;
  await show(root, ctx, () => api('/training/load?days=365'), ({ data: d }) => {
    if (d.empty) { root.innerHTML = `<div class="empty">${esc(t('No rides with TSS yet. Upload a ride or import your Strava history.'))}</div>`; return; }
    const n = d.now, st = d.status;
    root.innerHTML = `
      <div class="grid">
        ${kpi(t('Fitness (CTL)'), num(n.ctl))}
        ${kpi(t('Fatigue (ATL)'), num(n.atl))}
        ${kpi(t('Form (TSB)'), (n.tsb > 0 ? '+' : '') + num(n.tsb), '', st.form ? t(st.form) : '')}
        ${kpi(t('Ramp rate'), n.ramp === null ? t('N/A') : (n.ramp > 0 ? '+' : '') + num(n.ramp, 1), t('/wk'), st.ramp ? t(st.ramp) : '')}
        ${kpi(t('Acute:chronic'), n.acwr === null ? t('N/A') : num(n.acwr, 2), '', st.acwr ? t(st.acwr) : '')}
        ${kpi(t('Monotony (7 d)'), n.monotony === null ? t('N/A') : num(n.monotony, 2), '', st.monotony ? t(st.monotony) : '')}
      </div>
      <h2>${esc(t('Fitness, fatigue and form'))}</h2>
      ${seg('range', [['42', t('6 weeks')], ['90', t('3 months')], ['182', t('6 months')], ['365', t('1 year')]], '90')}
      <div id="pmc"></div>
      <h2>${esc(t('Volume'))}</h2><div id="vol"></div>
      <p class="note">${esc(t('Form is yesterday’s fitness minus fatigue.'))}</p>`;
    const draw = () => {
      const days = d.days.slice(-range);
      if (days.length < 2) { root.querySelector('#pmc').innerHTML = `<p class="note">${esc(t('Not enough days yet.'))}</p>`; return; }
      const xs = days.map((_, i) => i);
      const stepTick = Math.max(1, Math.floor(days.length / 4));
      lineChart(root.querySelector('#pmc'), {
        xs, label: t('Fitness, fatigue and form'), height: 220,
        series: [{ label: t('Fitness (CTL)'), color: css('--s1'), values: days.map((x) => x.ctl) },
                 { label: t('Fatigue (ATL)'), color: css('--s2'), values: days.map((x) => x.atl) },
                 { label: t('Form (TSB)'), color: css('--s3'), values: days.map((x) => x.tsb), width: 1.5 }],
        xTicks: xs.filter((i) => i % stepTick === 0 && i < days.length - stepTick / 2).map((i) => ({ x: i, label: dayShort(days[i].d) })),
        xFmt: (i) => dayLong(days[i].d), yFmt: (v) => num(v, 0), vFmt: (v) => num(v, 1), hline: 0,
      });
    };
    draw();
    wireSeg(root, 'range', (k) => { range = +k; draw(); });
    ctx.onResize(draw);
    const w = d.weeks;
    const drawVolume = () => barChart(root.querySelector('#vol'), {
      labels: w.map((x) => dayShort(x.d)), values: w.map((x) => x.hours || 0), color: css('--s1'), label: t('Volume'),
      yFmt: (v) => num(v, 0) + ' h',
      vFmt: (v, i) => `${num(v, 1)} h · ${num(w[i].km)} km · ${num(w[i].tss)} TSS · ${w[i].rides} ${t('rides')}`,
    });
    if (w.length) { drawVolume(); ctx.onResize(drawVolume); }
    else root.querySelector('#vol').innerHTML = `<p class="note">${esc(t('No rides in this period.'))}</p>`;
  });
}

// --- Power -----------------------------------------------------------------------------------------------------

const windowName = (label) => (label.startsWith('Season ') ? t('Season {year}', { year: label.slice(7) }) : t(label));

export async function powerView(root, ctx) {
  ctx.title(t('Power'));
  let win = 1, unit = 'w';
  await show(root, ctx, () => api('/training/power'), ({ data: d }) => {
    if (d.empty) { root.innerHTML = `<div class="empty">${esc(t('No power data yet.'))}</div>`; return; }
    const cp = d.cp, two = cp.two, three = cp.three;
    root.innerHTML = `
      <h2>${esc(t('Power curve'))}</h2>
      ${seg('win', d.windows.map((w, i) => [String(i), windowName(w.label)]), String(win))}
      ${d.weight_kg ? seg('unit', [['w', 'W'], ['kg', 'W/kg']], unit) : ''}
      <div id="curve"></div>
      <h2>${esc(t('Critical power (from your last 90 days)'))}</h2>
      <div class="grid">${kpi(t('CP (2-parameter)'), two ? num(two.cp) : t('N/A'), two ? 'W' : '')}${kpi('W′', two ? num(two.w_prime_kj, 1) : t('N/A'), two ? 'kJ' : '')}
      ${kpi(t('CP (3-parameter)'), three ? num(three.cp) : t('N/A'), three ? 'W' : '')}${kpi(t('Pmax (model)'), three ? num(three.pmax) : t('N/A'), three ? 'W' : '')}</div>
      <h2>${esc(t('Power profile'))}</h2>
      <div class="card"><table><thead><tr><th>${esc(t('Duration'))}</th><th>${esc(t('Watts (90 d)'))}</th><th>${esc(t('W/kg (90 d)'))}</th><th>${esc(t('× FTP (90 d)'))}</th></tr></thead><tbody>
      ${d.profile.map((p) => `<tr><td>${esc(t(p.label))}</td><td>${num(p.watts)}</td><td>${num(p.w_per_kg, 2)}</td><td>${num(p.x_ftp, 2)}</td></tr>`).join('')}</tbody></table></div>
      <h2>${esc(t('🏆 Records with dates'))}</h2>
      <div class="card"><table><thead><tr><th>${esc(t('Duration'))}</th><th>${esc(t('Watts'))}</th><th>W/kg</th><th>${esc(t('Set on'))}</th></tr></thead><tbody>
      ${d.records.map((p) => `<tr><td>${esc(span(p.seconds))}</td><td>${num(p.watts)}</td><td>${num(p.w_per_kg, 2)}</td><td>${esc(dayShort(p.date))}</td></tr>`).join('')}</tbody></table></div>`;
    const draw = () => {
      const sel = d.windows[win], all = d.windows[d.windows.length - 1];
      const secs = all.points.map((p) => p.seconds);
      const val = (pts) => secs.map((s) => { const p = pts.find((q) => q.seconds === s); return p ? (unit === 'w' ? p.watts : p.w_per_kg) : null; });
      const series = [{ label: windowName(sel.label), color: css('--s1'), values: val(sel.points) }];
      if (sel !== all) series.push({ label: windowName(all.label), color: css('--s2'), values: val(all.points), width: 1.5 });
      const ticks = [5, 60, 300, 1200, 3600].filter((s) => secs.includes(s));
      lineChart(root.querySelector('#curve'), {
        xs: secs, logX: true, label: t('Power curve'), height: 220, series,
        xTicks: ticks.map((s) => ({ x: s, label: span(s) })), xFmt: (s) => span(s),
        yFmt: (v) => num(v, unit === 'w' ? 0 : 1), vFmt: (v) => `${num(v, unit === 'w' ? 0 : 2)} ${unit === 'w' ? 'W' : 'W/kg'}`,
      });
    };
    draw();
    wireSeg(root, 'win', (k) => { win = +k; draw(); });
    if (d.weight_kg) wireSeg(root, 'unit', (k) => { unit = k; draw(); });
    ctx.onResize(draw);
  });
}
