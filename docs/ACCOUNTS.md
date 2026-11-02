# Accounts (shared server)

The desktop app has no accounts: a rider is just a name. On a shared server a rider is an **account** with an e-mail
and a password. This is the backend (`src/data/db/accounts.py`, `src/data/accounts_mail.py`); the screens that use it
come with the server front end.

| | |
|---|---|
| Sign-up | `register(name, email, password)` creates the rider and e-mails a confirmation link (3 days, once). |
| Sign-in | `login(email, password)` → a session token; refused until the address is confirmed. |
| Session | `session_user(token)` → the rider name or `None`. 30 days, renewed while in use, 90 at most. |
| Forgot password | `request_password_reset(email)` e-mails a link (1 hour, once); `reset_password(token, new)` signs the rider out everywhere. |
| Change / delete | `change_password`, `logout`, `logout_everywhere`, `delete_account(name, password)` (removes all their data). |

**What is stored.** Passwords as salted scrypt hashes (`scrypt$n$r$p$salt$hash`, cost raised by changing `COST`). Confirmation,
reset and session tokens only as SHA-256, so a database copy cannot be used to sign in. `get_user_data` says whether a
rider has a password (`has_password`) but never returns the hash.

**What it does not reveal.** Sign-up, reset and resend answer the same whether or not the e-mail has an account (the
owner of an existing address is told by e-mail); an unknown e-mail, a rider without a password and a wrong password all
answer `wrong` and take the same time. Guessing is throttled per e-mail and per client address (pass the real client
address as `client=`).

**Server settings.** `BIKETOOLS_PUBLIC_URL` (the address in the e-mail links, e.g. `https://bike.example.net`) and an
SMTP account (Weekly e-mail → mail server settings). Without a mail server nobody can confirm an address; a failed send
is not reported to the person asking.

**Not here yet.** The sign-in / sign-up screens and a session cookie, a CAPTCHA or e-mail-domain policy against
abuse, two-factor sign-in, and an administrator view.
