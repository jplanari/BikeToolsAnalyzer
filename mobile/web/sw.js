// Keeps the app's own files so it opens without a connection. Data is never cached here (api.js keeps the last answers).
const NAME = 'biketools-v2';
const FILES = ['./', 'index.html', 'app.css', 'icon.svg', 'manifest.webmanifest', 'js/app.js', 'js/api.js', 'js/charts.js', 'js/fmt.js', 'js/i18n.js',
  'js/locales.js', 'js/sources.js', 'js/store.js', 'js/views.js'];
self.addEventListener('install', (e) => { e.waitUntil(caches.open(NAME).then((c) => c.addAll(FILES)).then(() => self.skipWaiting())); });
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== NAME).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin || u.pathname.startsWith('/api/')) return;
  e.respondWith(fetch(e.request).then((r) => { const copy = r.clone(); caches.open(NAME).then((c) => c.put(e.request, copy)); return r; })
    .catch(() => caches.match(e.request)));
});
