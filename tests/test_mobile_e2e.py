"""The phone web app in a real browser at phone size. Skipped where Playwright/Chromium is missing (e.g. CI)."""
import threading
from wsgiref.simple_server import WSGIRequestHandler, make_server

import pytest

from src.api.app import application
from src.data import db
from tests.test_nutrition import seeded  # noqa: F401
from tests.test_strava import real_distances  # noqa: F401

sync_api = pytest.importorskip("playwright.sync_api")


class Quiet(WSGIRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture
def server(seeded):
    token, _ = db.create_api_token("Rider", "e2e")
    httpd = make_server("127.0.0.1", 0, application, handler_class=Quiet)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", token
    httpd.shutdown()


@pytest.fixture
def page(server):
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:                      # no browser installed
            pytest.skip(f"Chromium not available: {e}")
        ctx = browser.new_context(viewport={"width": 390, "height": 800}, device_scale_factor=2, has_touch=True, is_mobile=True)
        pg = ctx.new_page()
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
        yield pg
        browser.close()


def pair(page, server):
    url, token = server
    page.goto(f"{url}/#/pair?server={url}&token={token}")
    page.click("#go")
    page.wait_for_selector(".ride")


def test_pairing_lists_rides_and_opens_one(page, server):
    pair(page, server)
    assert page.locator(".ride").count() == 4
    page.locator(".ride").first.click()
    page.wait_for_selector("#chart svg path", state="attached")
    assert page.locator(".kpi").count() >= 8
    assert page.locator("#map svg path").count() == 1
    assert "Energy and fuel" in page.inner_text("#view")
    assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")      # no sideways scroll
    assert page.errors == []


def test_training_and_power_pages_draw(page, server):
    pair(page, server)
    page.click("#tabs a[href='#/training']")
    page.wait_for_selector("#pmc svg path", state="attached")
    assert "Fitness (CTL)" in page.inner_text("#view")
    page.click("#tabs a[href='#/power']")
    page.wait_for_selector("#curve svg path", state="attached")
    assert page.locator("table").count() == 2
    page.click("#unit button[data-k='kg']")
    assert page.locator("#curve svg path").count() >= 1
    assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
    assert page.errors == []


def test_spanish_and_a_wrong_token(page, server):
    url, token = server
    page.goto(f"{url}/#/pair?server={url}&token=bt_nope")
    page.click("#go")
    page.wait_for_selector("#err:not([hidden])")
    assert "token" in page.inner_text("#err").lower()
    page.fill("#token", token)
    page.click("#go")
    page.wait_for_selector(".ride")
    page.click("#gear")
    page.select_option("#lang", "es")
    page.click("#back")
    page.wait_for_selector(".ride")
    assert "Salidas" in page.inner_text("#title") or "Salidas" in page.inner_text("#tabs")


def test_offline_shows_the_last_data(page, server):
    pair(page, server)
    page.context.set_offline(True)
    page.reload()
    page.wait_for_selector(".ride", timeout=30000)
    assert page.is_visible("#banner")
