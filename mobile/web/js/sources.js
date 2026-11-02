import { act, live } from './api.js';
import { esc } from './fmt.js';
import { t } from './i18n.js';
import { errorMessage } from './views.js';

const PROVIDERS = [['strava', 'Strava'], ['wahoo', 'Wahoo']];
const ranges = () => [[30, t('30 days')], [90, t('90 days')], [365, t('A year')], [0, t('Only new rides')]];

function jobText(job) {
  if (!job) return '';
  if (job.state === 'queued') return t('Waiting to start…');
  if (job.state === 'running') {
    return job.total ? t('Importing {done} of {total}…', { done: job.done || 0, total: job.total }) : t('Importing…');
  }
  if (job.state === 'done') return t('Done: {n} new rides imported.', { n: job.imported });
  return job.error === 'reconnect' ? t('The connection was refused. Connect again.') : t('The import failed. Try again later.');
}

export async function sourcesView(root, ctx) {
  ctx.title(t('Data sources')); ctx.back('#/settings');
  let timer = null;
  const stop = () => { clearTimeout(timer); timer = null; };
  async function draw(first) {
    if (ctx.stale()) return stop();
    let info;
    try { info = await live('/connections'); }
    catch (e) {
      if (e.kind === 'auth') return ctx.signedOut();
      if (first) root.innerHTML = `<div class="empty"><p>${esc(errorMessage(e))}</p><button class="btn secondary" id="retry">${esc(t('Try again'))}</button></div>`;
      if (first) document.getElementById('retry').addEventListener('click', () => draw(true));
      return;
    }
    if (ctx.stale()) return;
    const busy = info.job && (info.job.state === 'queued' || info.job.state === 'running');
    root.innerHTML = `<p class="muted">${esc(t('Connect your accounts and the server fetches your rides. Your passwords for these services are never seen by BikeTools.'))}</p>`
      + PROVIDERS.map(([k, label]) => {
        const p = info[k];
        const body = !p.configured
          ? `<p class="note">${esc(t('This server is not set up for {provider}.', { provider: label }))}</p>`
          : p.connected
            ? `<div>${esc(t('Connected as {name}.', { name: p.name || label }))}</div>
               <label for="r-${k}">${esc(t('Import'))}</label>
               <select id="r-${k}">${ranges().map(([d, l]) => `<option value="${d}">${esc(l)}</option>`).join('')}</select>
               <button class="btn" data-act="import" data-p="${k}" ${busy ? 'disabled' : ''}>${esc(t('Import rides'))}</button>
               <button class="btn secondary" data-act="disconnect" data-p="${k}">${esc(t('Disconnect'))}</button>`
            : `<button class="btn" data-act="start" data-p="${k}">${esc(t('Connect {provider}', { provider: label }))}</button>`;
        return `<div class="card"><h2>${label}</h2>${body}</div>`;
      }).join('')
      + (info.job ? `<p class="note" role="status" id="job">${esc(jobText(info.job))}</p>` : '')
      + `<p class="error" id="err" role="alert" hidden></p>`;
    root.querySelectorAll('[data-act]').forEach((b) => b.addEventListener('click', () => run(b.dataset.act, b.dataset.p)));
    stop();
    if (busy) timer = setTimeout(() => draw(false), 2000);
  }
  async function run(action, provider) {
    const err = document.getElementById('err'); err.hidden = true;
    try {
      if (action === 'start') {
        const { url } = await act(`/connections/${provider}/start`, {});
        window.open(url, '_blank', 'noopener');
        root.insertAdjacentHTML('afterbegin', `<p class="note" role="status">${esc(t('Finish in the browser, then come back here.'))}</p>`);
        const check = async () => {                    // the page that finishes the sign-in runs elsewhere: look until it shows up
          if (ctx.stale()) return;
          try { const i = await live('/connections'); if (i[provider].connected) return draw(true); } catch { /* keep looking */ }
          setTimeout(check, 3000);
        };
        setTimeout(check, 3000);
        return;
      }
      if (action === 'import') {
        const days = Number(document.getElementById('r-' + provider).value);
        await act(`/connections/${provider}/import`, { days: days || null });
      } else await act(`/connections/${provider}/disconnect`, {});
      draw(true);
    } catch (e) {
      err.textContent = e.kind === 'conflict' ? t('That is not possible right now: an import may already be running.') : errorMessage(e);
      err.hidden = false;
    }
  }
  root.innerHTML = '<p class="muted">…</p>';
  draw(true);
}
