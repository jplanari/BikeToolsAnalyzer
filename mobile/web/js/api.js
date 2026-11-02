import { load, save, drop } from './store.js';

export class ApiError extends Error {
  constructor(kind, status) { super(kind); this.kind = kind; this.status = status; }
}
const MAX_CACHED = 24;
export const config = () => load('cfg', { server: '', token: '' });
export const saveConfig = (c) => save('cfg', c);
export const signOut = () => { const c = config(); saveConfig({ server: c.server, token: '' }); clearCache(); };

export const baseUrl = (server) => {
  const s = (server || '').trim().replace(/\/+$/, '');
  if (!s) return location.protocol.startsWith('http') && location.hostname !== 'localhost' ? location.origin : '';
  return /^https?:\/\//i.test(s) ? s : 'http://' + s;
};

// The last answer of each address is kept, so the app still shows something on a train or in a tunnel.
function cacheGet(url) { return load('c:' + url); }
function cacheSet(url, data) {
  const index = load('cidx', []).filter((u) => u !== url);
  index.push(url);
  while (index.length > MAX_CACHED) drop('c:' + index.shift());
  if (!save('c:' + url, data)) { while (index.length > 1) { drop('c:' + index.shift()); if (save('c:' + url, data)) break; } }
  save('cidx', index);
}
export function clearCache() { for (const u of load('cidx', [])) drop('c:' + u); drop('cidx'); }

async function get(url, token, timeoutMs = 20000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { headers: token ? { Authorization: 'Bearer ' + token } : {}, signal: ctrl.signal });
    if (r.status === 401) throw new ApiError('auth', 401);
    if (r.status === 429) throw new ApiError('throttled', 429);
    if (!r.ok) throw new ApiError('http', r.status);
    return await r.json();
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError('network');
  } finally { clearTimeout(timer); }
}

/** {data, stale}: fresh from the server, or the last saved answer (stale) when the server cannot be reached. */
export async function api(path) {
  const c = config();
  const url = baseUrl(c.server) + '/api/v1' + path;
  try {
    const data = await get(url, c.token);
    cacheSet(url, data);
    return { data, stale: false };
  } catch (e) {
    if (e.kind === 'network' || e.kind === 'http') {
      const saved = cacheGet(url);
      if (saved) return { data: saved, stale: true };
    }
    throw e;
  }
}

/** Checks an address and a token before they are saved. Returns the rider's profile. */
export async function connect(server, token) {
  const base = baseUrl(server);
  if (!base) throw new ApiError('network');
  const ping = await get(base + '/api/v1/ping', null, 8000);
  if (!ping || ping.app !== 'BikeTools') throw new ApiError('network');
  return get(base + '/api/v1/me', token, 8000);
}
