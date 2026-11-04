# Switches for trying BikeTools Analyzer in different modes from your terminal (Linux and macOS; bash or zsh).
#
#   source tools/devenv.sh          (once per terminal; it only sets variables in THAT terminal)
#   bt_help                         list the commands
#
# Nothing here is permanent: close the terminal, or run  bt_clear , and every setting is gone. Your secrets (mail password,
# database address) go in tools/devenv.local, which git ignores (copy tools/devenv.local.example).

if [ -n "${BASH_VERSION:-}" ]; then _bt_here="${BASH_SOURCE[0]}"; elif [ -n "${ZSH_VERSION:-}" ]; then _bt_here="${(%):-%x}"; else _bt_here="$0"; fi
BT_ROOT="$(cd "$(dirname "$_bt_here")/.." && pwd)"
unset _bt_here

# Where the server test data lives. Never the same place as your real desktop data.
BT_SERVER_DATA="${BT_SERVER_DATA:-$HOME/.local/share/BikeToolsAnalyzer-server-test}"
BT_PORT="${BT_PORT:-8501}"
[ -f "$BT_ROOT/tools/devenv.local" ] && . "$BT_ROOT/tools/devenv.local"

_bt_flag() { eval "[ -n \"\${$1:-}\" ]" && echo on || echo off; }
_bt_say() { printf '%s\n' "$*"; }

bt_help() {
cat <<'H'
bt_server on|off      server mode (sign-in screens). 'on' uses a separate test data folder; 'off' is the normal desktop app
bt_mail on|off        e-mail sending from the BT_SMTP_* values in tools/devenv.local ('off' = no mail leaves the machine)
bt_db sqlite|postgres sqlite (default) or PostgreSQL from BT_PG_URL in tools/devenv.local
bt_proxy on|off       trust X-Forwarded-For (only when a reverse proxy is in front)
bt_limits strict|default   tiny upload/storage limits to test them (1 MB file, 5 MB unpacked, 2 MB per rider) / the defaults
bt_confirm [email]    mark server-test accounts as e-mail confirmed (for when mail is off); no argument = all accounts
bt_reset              delete the server test data (asks first)
bt_run [desktop|web]  start the app: desktop window (default) or 'streamlit run app.py' in your browser
bt_status             show every current setting
bt_clear              unset everything this file set
H
}

bt_server() {
  case "${1:-}" in
    on)  export BIKETOOLS_SERVER_MODE=1
         export BIKETOOLS_DATA="$BT_SERVER_DATA"; mkdir -p "$BIKETOOLS_DATA"
         [ -n "${BIKETOOLS_DATABASE_URL:-}" ] || export BIKETOOLS_DB="$BIKETOOLS_DATA/biketools.db"
         export BIKETOOLS_PUBLIC_URL="http://localhost:$BT_PORT"; export BIKETOOLS_PORT="$BT_PORT"
         _bt_say "server mode ON, test data in $BIKETOOLS_DATA" ;;
    off) unset BIKETOOLS_SERVER_MODE BIKETOOLS_DATA BIKETOOLS_DB BIKETOOLS_PUBLIC_URL BIKETOOLS_PORT
         _bt_say "server mode OFF: the normal desktop app with your real data" ;;
    *)   _bt_say "usage: bt_server on|off"; return 1 ;;
  esac
}

bt_mail() {
  case "${1:-}" in
    on)  if [ -z "${BT_SMTP_HOST:-}" ] || [ -z "${BT_SMTP_SENDER:-}" ]; then
           _bt_say "set BT_SMTP_HOST and BT_SMTP_SENDER (and the login) in tools/devenv.local first"; return 1; fi
         export BIKETOOLS_SMTP_HOST="$BT_SMTP_HOST" BIKETOOLS_SMTP_SENDER="$BT_SMTP_SENDER"
         export BIKETOOLS_SMTP_PORT="${BT_SMTP_PORT:-587}" BIKETOOLS_SMTP_SECURITY="${BT_SMTP_SECURITY:-starttls}"
         export BIKETOOLS_SMTP_USER="${BT_SMTP_USER:-$BT_SMTP_SENDER}" BIKETOOLS_SMTP_PASSWORD="${BT_SMTP_PASSWORD:-}"
         _bt_say "mail ON ($BT_SMTP_HOST as $BT_SMTP_SENDER)" ;;
    off) unset BIKETOOLS_SMTP_HOST BIKETOOLS_SMTP_PORT BIKETOOLS_SMTP_SECURITY BIKETOOLS_SMTP_USER BIKETOOLS_SMTP_PASSWORD BIKETOOLS_SMTP_SENDER
         _bt_say "mail OFF: confirmation links are not sent; use bt_confirm to confirm accounts" ;;
    *)   _bt_say "usage: bt_mail on|off"; return 1 ;;
  esac
}

bt_db() {
  case "${1:-}" in
    postgres) if [ -z "${BT_PG_URL:-}" ]; then _bt_say "set BT_PG_URL=postgresql://user:password@host:5432/dbname in tools/devenv.local"; return 1; fi
              export BIKETOOLS_DATABASE_URL="$BT_PG_URL"; unset BIKETOOLS_DB
              _bt_say "database: PostgreSQL (tests of the app will write to it; use a throwaway database)" ;;
    sqlite)   unset BIKETOOLS_DATABASE_URL
              [ -n "${BIKETOOLS_DATA:-}" ] && export BIKETOOLS_DB="$BIKETOOLS_DATA/biketools.db"
              _bt_say "database: SQLite" ;;
    *)        _bt_say "usage: bt_db sqlite|postgres"; return 1 ;;
  esac
}

bt_proxy() {
  case "${1:-}" in
    on)  export BIKETOOLS_TRUST_PROXY=1; _bt_say "trusting X-Forwarded-For (only safe behind your own proxy)" ;;
    off) unset BIKETOOLS_TRUST_PROXY; _bt_say "not trusting X-Forwarded-For" ;;
    *)   _bt_say "usage: bt_proxy on|off"; return 1 ;;
  esac
}

bt_limits() {
  case "${1:-}" in
    strict)  export BIKETOOLS_MAX_ACTIVITY_MB=1 BIKETOOLS_MAX_UNPACKED_MB=5 BIKETOOLS_RIDER_QUOTA_MB=2
             _bt_say "strict limits: 1 MB per file, 5 MB per upload, 2 MB stored per rider (server mode only for the quota)" ;;
    default) unset BIKETOOLS_MAX_ACTIVITY_MB BIKETOOLS_MAX_UNPACKED_MB BIKETOOLS_RIDER_QUOTA_MB; _bt_say "default limits" ;;
    *)       _bt_say "usage: bt_limits strict|default"; return 1 ;;
  esac
}

bt_confirm() {
  if [ -n "${BIKETOOLS_DATABASE_URL:-}" ]; then
    _bt_say "PostgreSQL: run  psql \"\$BIKETOOLS_DATABASE_URL\" -c \"UPDATE users SET email_verified_at = extract(epoch from now())::bigint WHERE email_verified_at IS NULL AND password_hash IS NOT NULL\""; return 0
  fi
  python3 - "${BIKETOOLS_DB:-}" "${1:-}" <<'P'
import sqlite3, sys, time
path, email = sys.argv[1], sys.argv[2].strip().lower()
if not path:
    sys.exit("no test database: run  bt_server on  first")
c = sqlite3.connect(path)
try:
    sql, args = "UPDATE users SET email_verified_at = ? WHERE email_verified_at IS NULL AND password_hash IS NOT NULL", [int(time.time())]
    if email:
        sql, args = sql + " AND lower(email) = ?", args + [email]
    n = c.execute(sql, args).rowcount
    c.commit()
except sqlite3.OperationalError as e:
    sys.exit(f"{e}: start the app once and sign up first")
print(f"{n} account(s) confirmed")
P
}

bt_reset() {
  [ -n "${BT_SERVER_DATA:-}" ] && [ "$BT_SERVER_DATA" != "/" ] && [ "$BT_SERVER_DATA" != "$HOME" ] || { _bt_say "refusing: BT_SERVER_DATA looks unsafe"; return 1; }
  [ -d "$BT_SERVER_DATA" ] || { _bt_say "nothing to delete ($BT_SERVER_DATA does not exist)"; return 0; }
  printf 'Delete ALL server test data in %s ? [y/N] ' "$BT_SERVER_DATA"; read -r _bt_a
  case "$_bt_a" in y|Y|yes) rm -rf -- "$BT_SERVER_DATA"; _bt_say "deleted"; [ -n "${BIKETOOLS_DATA:-}" ] && mkdir -p "$BIKETOOLS_DATA" ;; *) _bt_say "kept" ;; esac
  unset _bt_a
}

bt_run() {
  cd "$BT_ROOT" || return 1
  case "${1:-desktop}" in
    desktop) python3 desktop/launcher.py ;;
    web)     streamlit run app.py --server.port "$BT_PORT" ;;
    *)       _bt_say "usage: bt_run [desktop|web]"; return 1 ;;
  esac
}

bt_status() {
  _bt_say "server mode : $(_bt_flag BIKETOOLS_SERVER_MODE)"
  _bt_say "mail        : $([ -n "${BIKETOOLS_SMTP_HOST:-}" ] && echo "on ($BIKETOOLS_SMTP_HOST)" || echo off)"
  _bt_say "database    : $([ -n "${BIKETOOLS_DATABASE_URL:-}" ] && echo PostgreSQL || echo "SQLite ${BIKETOOLS_DB:-(default file)}")"
  _bt_say "data folder : ${BIKETOOLS_DATA:-(your normal desktop data)}"
  _bt_say "proxy trust : $(_bt_flag BIKETOOLS_TRUST_PROXY)"
  _bt_say "limits      : file ${BIKETOOLS_MAX_ACTIVITY_MB:-default} MB, upload ${BIKETOOLS_MAX_UNPACKED_MB:-default} MB, per rider ${BIKETOOLS_RIDER_QUOTA_MB:-default} MB"
  _bt_say "public url  : ${BIKETOOLS_PUBLIC_URL:-(not set)}   port: ${BIKETOOLS_PORT:-8501}"
}

bt_clear() {
  unset BIKETOOLS_SERVER_MODE BIKETOOLS_DATA BIKETOOLS_DB BIKETOOLS_DATABASE_URL BIKETOOLS_PUBLIC_URL BIKETOOLS_PORT BIKETOOLS_TRUST_PROXY \
        BIKETOOLS_MAX_ACTIVITY_MB BIKETOOLS_MAX_UNPACKED_MB BIKETOOLS_RIDER_QUOTA_MB \
        BIKETOOLS_SMTP_HOST BIKETOOLS_SMTP_PORT BIKETOOLS_SMTP_SECURITY BIKETOOLS_SMTP_USER BIKETOOLS_SMTP_PASSWORD BIKETOOLS_SMTP_SENDER
  _bt_say "all BikeTools settings cleared in this terminal"
}

_bt_say "BikeTools dev switches loaded. Type bt_help. Current state:"
bt_status
