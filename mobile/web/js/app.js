import { baseUrl, config, connect, login, saveConfig, signOut } from './api.js';
import { esc } from './fmt.js';
import { LANGUAGES, detect, language, setLanguage, t } from './i18n.js';
import { errorMessage, powerView, rideView, ridesView, trainingView } from './views.js';

const $ = (id) => document.getElementById(id);
const view = $('view'), titleEl = $('title'), backEl = $('back'), bannerEl = $('banner'), tabsEl = $('tabs');
let routeId = 0, resizers = [], backTarget = null;

const TABS = [['rides', '🚴', 'Rides'], ['training', '📈', 'Training'], ['power', '⚡', 'Power']];

function ctxFor(id) {
  return {
    stale: () => id !== routeId,
    title: (s) => { titleEl.textContent = s; document.title = s + ' · BikeTools'; },
    back: (to) => { backTarget = to; backEl.hidden = false; },
    offline: (on) => { bannerEl.hidden = !on; bannerEl.textContent = on ? t('Offline: showing the last data saved on this phone.') : ''; },
    onResize: (fn) => resizers.push(fn),
    signedOut: () => { signOut(); location.hash = '#/connect'; },
  };
}

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [path, query] = raw.split('?');
  return { parts: path.split('/').filter(Boolean), params: new URLSearchParams(query || '') };
}

function drawTabs(current) {
  tabsEl.hidden = !current;
  if (!current) return;
  tabsEl.innerHTML = TABS.map(([k, icon, label]) =>
    `<a href="#/${k}" ${k === current ? 'aria-current="page"' : ''}><b aria-hidden="true">${icon}</b>${esc(t(label))}</a>`).join('');
}

// --- Connect / settings ----------------------------------------------------------------------------------------

const PRIVATE_HOST = /^(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.|169\.254\.)/;
function unencryptedAndPublic(address) {
  try {
    const u = new URL(baseUrl(address));
    return u.protocol === 'http:' && !(PRIVATE_HOST.test(u.hostname) || u.hostname.endsWith('.local') || u.hostname.endsWith('.ts.net') || !u.hostname.includes('.'));
  } catch { return false; }
}

function connectView(prefill = {}) {
  const c = config();
  const sameOrigin = location.protocol.startsWith('http') && !['localhost', '127.0.0.1'].includes(location.hostname) && !window.Capacitor;
  const server = prefill.server || c.server || (sameOrigin ? location.origin : '');
  let mode = prefill.token ? 'token' : 'password';
  titleEl.textContent = 'BikeTools';
  document.title = 'BikeTools';
  backEl.hidden = true; drawTabs(null); bannerEl.hidden = true;
  view.innerHTML = `
    <h2>${esc(t('Connect to your BikeTools'))}</h2>
    <p class="muted">${esc(t('This app shows the data of your BikeTools Analyzer. Start the phone server on your computer, then sign in with your rider name and password, or create an access token in the desktop app (User Corner → Phone app).'))}</p>
    <form id="f" autocomplete="off">
      <label for="server">${esc(t('Server address'))}</label>
      <input id="server" inputmode="url" autocapitalize="off" autocorrect="off" spellcheck="false" placeholder="http://192.168.1.20:8765" value="${esc(server)}">
      <div id="pw-fields">
        <label for="name">${esc(t('Rider name'))}</label>
        <input id="name" autocapitalize="off" autocorrect="off" spellcheck="false" autocomplete="username">
        <label for="password">${esc(t('Password'))}</label>
        <input id="password" type="password" autocomplete="current-password">
      </div>
      <div id="tk-fields">
        <label for="token">${esc(t('Access token'))}</label>
        <input id="token" type="password" autocapitalize="off" autocorrect="off" spellcheck="false" placeholder="bt_…" value="${esc(prefill.token || '')}">
      </div>
      <p class="error" id="err" role="alert" hidden></p>
      <p class="error" id="block" hidden>${esc(t('Passwords are not sent over an unencrypted connection outside a private network. Use a VPN such as Tailscale or an https address.'))}</p>
      <button class="btn" id="go">${esc(t('Connect'))}</button>
      <button class="btn secondary" id="swap" type="button"></button>
    </form>
    <p class="note" id="warn" hidden>${esc(t('This address is not encrypted (http) and is not on a private network. Anyone on the way could read your data and token. Prefer a VPN such as Tailscale or an https address.'))}</p>`;
  const refresh = () => {
    $('pw-fields').hidden = mode !== 'password';
    $('tk-fields').hidden = mode !== 'token';
    $('swap').textContent = mode === 'password' ? t('Use an access token instead') : t('Sign in with a password instead');
    const risky = unencryptedAndPublic($('server').value);
    $('warn').hidden = !risky || mode === 'password';
    $('block').hidden = !(risky && mode === 'password');
    $('go').disabled = risky && mode === 'password';
  };
  $('server').addEventListener('input', refresh);
  $('swap').addEventListener('click', () => { mode = mode === 'password' ? 'token' : 'password'; refresh(); });
  refresh();
  $('f').addEventListener('submit', async (e) => {
    e.preventDefault();
    const err = $('err'), btn = $('go');
    err.hidden = true; btn.disabled = true; btn.textContent = t('Connecting…');
    try {
      let token, me;
      if (mode === 'password') {
        ({ token, me } = await login($('server').value, $('name').value.trim(), $('password').value));
      } else {
        token = $('token').value.trim();
        me = await connect($('server').value, token);
      }
      saveConfig({ server: baseUrl($('server').value), token });
      setLanguage(detect(me.language));
      location.hash = '#/rides';
      route();
    } catch (ex) {
      err.textContent = ex.kind === 'auth' ? t('That token was not accepted.') : ex.kind === 'login' ? t('Wrong name or password.') : errorMessage(ex);
      err.hidden = false; btn.textContent = t('Connect'); refresh();
    }
  });
}

function settingsView(ctx) {
  ctx.title(t('Settings')); ctx.back('#/rides'); drawTabs(null);
  const c = config();
  view.innerHTML = `
    <div class="card"><div class="muted">${esc(t('Server address'))}</div><div style="word-break:break-all">${esc(c.server || location.origin)}</div></div>
    <label for="lang">${esc(t('Language'))}</label>
    <select id="lang">${Object.entries(LANGUAGES).map(([k, n]) => `<option value="${k}" ${k === language() ? 'selected' : ''}>${esc(n)}</option>`).join('')}</select>
    <button class="btn secondary" id="refresh">${esc(t('Check connection'))}</button>
    <p class="note" id="status" role="status"></p>
    <button class="btn secondary" id="out">${esc(t('Disconnect this phone'))}</button>
    <p class="note">${esc(t('The phone app only reads your data. Uploading, editing and logging what you ate are done in the desktop app.'))}</p>`;
  $('lang').addEventListener('change', (e) => { setLanguage(e.target.value, true); route(); });
  $('refresh').addEventListener('click', async () => {
    const s = $('status'); s.textContent = '…';
    try { const me = await connect(config().server, config().token); s.textContent = t('Connected as {name}.', { name: me.name }); }
    catch (ex) { s.textContent = errorMessage(ex); }
  });
  $('out').addEventListener('click', () => { signOut(); location.hash = '#/connect'; });
}

// --- Router ----------------------------------------------------------------------------------------------------

function route() {
  const id = ++routeId;
  resizers = []; backEl.hidden = true; backTarget = null; bannerEl.hidden = true;
  const { parts, params } = parseHash();
  const ctx = ctxFor(id);
  const [page, arg] = parts;
  if (page === 'pair') {                                      // biketools://pair?server=…&token=…, or a link from the desktop app
    return connectView({ server: params.get('server') || '', token: params.get('token') || '' });
  }
  if (page === 'connect') return connectView();
  if (!config().token) { location.hash = '#/connect'; return; }
  window.scrollTo(0, 0);
  if (page === 'settings') return settingsView(ctx);
  const tab = ['rides', 'training', 'power'].includes(page) ? page : (page === 'ride' ? 'rides' : 'rides');
  drawTabs(page === 'ride' ? null : tab);
  if (page === 'ride' && /^\d+$/.test(arg || '')) return rideView(view, ctx, arg);
  if (page === 'training') return trainingView(view, ctx);
  if (page === 'power') return powerView(view, ctx);
  return ridesView(view, ctx);
}

backEl.addEventListener('click', () => { if (history.length > 1) history.back(); else location.hash = backTarget || '#/rides'; });
$('gear').addEventListener('click', () => { location.hash = '#/settings'; });
window.addEventListener('hashchange', route);
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => resizers.forEach((fn) => fn()), 150); });

// The Android wrapper hands over biketools://pair links.
const cap = window.Capacitor;
if (cap && cap.Plugins && cap.Plugins.App) {
  cap.Plugins.App.addListener('appUrlOpen', ({ url }) => {
    try { const u = new URL(url); if (u.host === 'pair' || u.pathname.includes('pair')) location.hash = '#/pair?' + u.searchParams.toString(); } catch { /* ignore */ }
  });
}

setLanguage(detect(null));
$('back').setAttribute('aria-label', t('Back'));
$('gear').setAttribute('aria-label', t('Settings'));
if ('serviceWorker' in navigator && location.protocol.startsWith('http') && !window.Capacitor) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
route();
