"""Unit Tests for Demo REST API and Session Handlers."""

import json
import pytest
from src.demo.session_manager import NavigationSessionManager
from src.demo.replay_controller import ReplayController
from src.demo.api import DemoAPIHandler


@pytest.fixture
def demo_api():
    """Create API handler fixture."""
    session = NavigationSessionManager()
    controller = ReplayController(session)
    handler = DemoAPIHandler(session, controller)
    return handler, session, controller


def test_api_status_endpoint(demo_api):
    """Verify /api/status returns expected system state."""
    handler, session, _ = demo_api
    status_code, headers, body = handler.handle_request("GET", "/api/status")

    assert status_code == 200
    assert headers["Content-Type"] == "application/json"
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "OK"
    assert data["scenario"] == "mixed_urban"
    assert "provenance" in data


def test_api_telemetry_endpoint(demo_api):
    """Verify /api/telemetry returns complete structured navigation payload."""
    handler, session, _ = demo_api
    session.step_forward(num_samples=10)

    status_code, headers, body = handler.handle_request("GET", "/api/telemetry")
    assert status_code == 200
    data = json.loads(body.decode("utf-8"))

    # Required payload sections
    assert "provenance" in data
    assert "session_state" in data
    assert "navigation_mode" in data
    assert "subsystems" in data
    assert "position" in data
    assert "motion" in data
    assert "sensor_stream" in data
    assert "error_metrics" in data
    assert "diagnostics" in data
    assert "trajectories" in data
    assert "events" in data

    # Check numerical bounds
    assert data["position"]["uncertainty_m"] >= 0.0
    assert data["motion"]["speed_kmh"] >= 0.0


def test_api_control_flow(demo_api):
    """Verify start, pause, step, and reset control actions."""
    handler, session, _ = demo_api

    # Start
    code, _, _ = handler.handle_request("POST", "/api/start")
    assert code == 200
    assert session.is_running is True

    # Pause
    code, _, _ = handler.handle_request("POST", "/api/pause")
    assert code == 200
    assert session.is_paused is True

    # Step
    code, _, body = handler.handle_request("POST", "/api/step", json.dumps({"samples": 5}).encode("utf-8"))
    assert code == 200
    assert session.current_step > 0

    # Reset
    code, _, _ = handler.handle_request("POST", "/api/reset")
    assert code == 200
    assert session.current_step == 0


def test_api_outage_and_restore(demo_api):
    """Verify manual outage injection and restoration commands."""
    handler, session, _ = demo_api

    # Trigger outage
    code, _, _ = handler.handle_request("POST", "/api/outage/start")
    assert code == 200
    assert session.manual_outage_override is True

    # Step through
    session.step_forward(num_samples=20)
    assert session.pipeline.current_mode.value == "GNSS_OUTAGE"

    # Restore GNSS
    code, _, _ = handler.handle_request("POST", "/api/outage/restore")
    assert code == 200
    assert session.manual_outage_override is False


def test_api_config_update(demo_api):
    """Verify runtime configuration updates for scenario and feature toggles."""
    handler, session, _ = demo_api

    config_payload = {
        "scenario": "straight",
        "outage_duration": 10.0,
        "enable_ai": False,
        "enable_nhc": True,
        "enable_zupt": False,
        "playback_speed": 5.0,
    }
    code, _, _ = handler.handle_request("POST", "/api/config", json.dumps(config_payload).encode("utf-8"))
    assert code == 200
    assert session.scenario_name == "straight"
    assert session.outage_duration_s == 10.0
    assert session.enable_ai is False
    assert session.enable_nhc is True
    assert session.enable_zupt is False
    assert session.playback_speed == 5.0


def test_api_benchmark_endpoint(demo_api):
    """Verify /api/benchmark returns Step 17 benchmark dataset."""
    handler, _, _ = demo_api
    code, _, body = handler.handle_request("GET", "/api/benchmark")
    assert code == 200
    data = json.loads(body.decode("utf-8"))
    assert "benchmark" in data
    assert "sih_check" in data
