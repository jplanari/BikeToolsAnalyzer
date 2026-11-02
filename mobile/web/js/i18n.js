// The English text is the key, as on the desktop. locales.js is generated from the desktop catalogs
// (python -m mobile.build_locales); a string with no entry shows in English.
import { LOCALES } from './locales.js';
import { load, save } from './store.js';

export const LANGUAGES = { en: 'English', es: 'Español', ca: 'Català' };
let lang = 'en';
let cat = {};

export function detect(profileLanguage) {
  const saved = load('lang');
  if (saved && LANGUAGES[saved]) return saved;
  for (const c of [profileLanguage, ...(navigator.languages || [navigator.language])]) {
    const base = String(c || '').toLowerCase().split('-')[0];
    if (LANGUAGES[base]) return base;
  }
  return 'en';
}
export function setLanguage(code, remember = false) {
  lang = LANGUAGES[code] ? code : 'en';
  cat = LOCALES[lang] || {};
  document.documentElement.lang = lang;
  if (remember) save('lang', lang);
  return lang;
}
export const language = () => lang;
export function t(text, values) {
  let out = Object.prototype.hasOwnProperty.call(cat, text) ? cat[text] : text;
  if (values) out = out.replace(/\{(\w+)\}/g, (m, k) => (k in values ? values[k] : m));
  return out;
}
