"""Runs the phone server inside another program (the desktop app's User Corner starts and stops it) or from the
command line. One server per process; it ends with the process."""
import logging
import socketserver
import threading
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from .app import application

DEFAULT_PORT = 8765
_lock = threading.Lock()
_server = None


class ThreadingServer(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


class QuietHandler(WSGIRequestHandler):
    def log_message(self, fmt, *args):          # never log the query string or headers: they can carry tokens
        line = str(args[0]).split("?")[0] if args else ""
        logging.getLogger("biketools.api").info("%s %s %s", self.address_string(), line, " ".join(str(a) for a in args[1:]))


def make(host, port):
    return make_server(host, port, application, server_class=ThreadingServer, handler_class=QuietHandler)


def start(host="0.0.0.0", port=DEFAULT_PORT):
    """Starts the server in a background thread. Returns (ok, message); starting twice is not an error."""
    global _server
    with _lock:
        if _server is not None:
            return True, "already running"
        try:
            _server = make(host, port)
        except OSError as e:
            return False, str(e)
        threading.Thread(target=_server.serve_forever, name="biketools-phone-server", daemon=True).start()
        return True, ""


def stop():
    global _server
    with _lock:
        srv, _server = _server, None
    if srv is not None:
        srv.shutdown()
        srv.server_close()


def running():
    """(host, port) while the server runs, else None."""
    with _lock:
        return None if _server is None else (_server.server_name, _server.server_port)
