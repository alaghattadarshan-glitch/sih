"""End-to-End Integration Tests for Interactive Demo HTTP Server and Sequence."""

import time
import json
import urllib.request
import pytest
from src.demo.server import DemoServer


@pytest.fixture(scope="module")
def live_server():
    """Start embedded test server on high ephemeral port."""
    server = DemoServer(host="127.0.0.1", port=18080)
    server.start(auto_replay_loop=False)
    yield server
    server.shutdown()


def test_http_static_assets_serving(live_server):
    """Verify HTTP server returns HTML, CSS, and JS static assets."""
    url = live_server.get_url()

    # Index HTML
    with urllib.request.urlopen(f"{url}/index.html", timeout=3.0) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "SIH-26168" in content
        assert "AI-ML INTELLIGENT DEAD RECKONING" in content

    # CSS
    with urllib.request.urlopen(f"{url}/styles.css", timeout=3.0) as resp:
        assert resp.status == 200

    # JS
    with urllib.request.urlopen(f"{url}/app.js", timeout=3.0) as resp:
        assert resp.status == 200


def test_http_live_navigation_cycle(live_server):
    """Test full interactive sequence over live HTTP network connection:
    GNSS_AIDED -> Outage Trigger -> GNSS_OUTAGE -> Restore -> RECOVERING -> GNSS_AIDED.
    """
    url = live_server.get_url()

    # 1. Reset & Start
    req = urllib.request.Request(f"{url}/api/reset", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200

    # 2. Advance 20 samples in nominal GNSS_AIDED mode
    step_req = urllib.request.Request(
        f"{url}/api/step",
        data=json.dumps({"samples": 20}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(step_req, timeout=3.0) as resp:
        assert resp.status == 200

    # Verify initial GNSS_AIDED mode
    with urllib.request.urlopen(f"{url}/api/telemetry", timeout=3.0) as resp:
        telem = json.loads(resp.read().decode("utf-8"))
        assert telem["navigation_mode"] == "GNSS_AIDED"
        assert telem["position"]["uncertainty_m"] > 0.0

    # 3. Simulate GNSS Outage
    outage_req = urllib.request.Request(f"{url}/api/outage/start", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(outage_req, timeout=3.0) as resp:
        assert resp.status == 200

    # Advance 40 samples during outage
    with urllib.request.urlopen(step_req, timeout=3.0) as resp:
        assert resp.status == 200

    # Verify transition to GNSS_OUTAGE
    with urllib.request.urlopen(f"{url}/api/telemetry", timeout=3.0) as resp:
        telem_out = json.loads(resp.read().decode("utf-8"))
        assert telem_out["navigation_mode"] == "GNSS_OUTAGE"
        assert telem_out["error_metrics"]["outage_timer_s"] > 0.0

    # 4. Restore GNSS
    restore_req = urllib.request.Request(f"{url}/api/outage/restore", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(restore_req, timeout=3.0) as resp:
        assert resp.status == 200

    # Advance 250 samples to allow multiple GNSS fix epochs to be ingested and filter to re-acquire
    step_large_req = urllib.request.Request(
        f"{url}/api/step",
        data=json.dumps({"samples": 250}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(step_large_req, timeout=3.0) as resp:
        assert resp.status == 200

    # Verify mode is no longer GNSS_OUTAGE
    with urllib.request.urlopen(f"{url}/api/telemetry", timeout=3.0) as resp:
        telem_rec = json.loads(resp.read().decode("utf-8"))
        assert telem_rec["navigation_mode"] in ["RECOVERING", "GNSS_AIDED"]
