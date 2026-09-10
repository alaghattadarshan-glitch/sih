"""Demo REST API Routing & Request Dispatcher."""

import os
import json
from typing import Dict, Any, Tuple
from src.demo.session_manager import NavigationSessionManager
from src.demo.replay_controller import ReplayController


class DemoAPIHandler:
    """Dispatches REST API endpoints for the navigation dashboard."""

    def __init__(self, session: NavigationSessionManager, controller: ReplayController):
        """Initialize API dispatcher.

        Args:
            session: Navigation session manager instance.
            controller: Background replay controller instance.
        """
        self.session = session
        self.controller = controller

    def handle_request(self, method: str, path: str, body_bytes: bytes = b"") -> Tuple[int, Dict[str, str], bytes]:
        """Dispatch HTTP request to appropriate handler.

        Returns:
            Tuple[int, Dict[str, str], bytes]: (status_code, headers, response_body)
        """
        headers = {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

        if method == "OPTIONS":
            return 200, headers, b"{}"

        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            body = {}

            # Strip query parameters for routing
        clean_path = path.split("?")[0].rstrip("/")

        # Route table
        if method == "GET" and clean_path == "/api/status":
            resp = {
                "status": "OK",
                "is_running": self.session.is_running,
                "is_paused": self.session.is_paused,
                "scenario": self.session.scenario_name,
                "provenance": self.session.provenance,
                "current_time": self.session.time_history[-1] if self.session.time_history else 0.0,
            }
            return 200, headers, json.dumps(resp).encode("utf-8")

        elif method == "GET" and clean_path == "/api/telemetry":
            telemetry = self.session.get_telemetry()
            return 200, headers, json.dumps(telemetry).encode("utf-8")

        elif method == "GET" and clean_path == "/api/events":
            return 200, headers, json.dumps({"events": self.session.events}).encode("utf-8")

        elif method == "GET" and clean_path == "/api/benchmark":
            bench_file = os.path.join("results", "final_benchmark", "benchmark_summary.json")
            sih_file = os.path.join("results", "final_benchmark", "sih_target_check.json")

            bench_data = {}
            if os.path.exists(bench_file):
                with open(bench_file, "r") as f:
                    bench_data = json.load(f)

            sih_data = {}
            if os.path.exists(sih_file):
                with open(sih_file, "r") as f:
                    sih_data = json.load(f)

            return 200, headers, json.dumps({
                "benchmark": bench_data,
                "sih_check": sih_data,
            }).encode("utf-8")

        elif method == "POST" and clean_path == "/api/start":
            self.session.start()
            return 200, headers, json.dumps({"status": "STARTED"}).encode("utf-8")

        elif method == "POST" and clean_path == "/api/pause":
            self.session.pause()
            return 200, headers, json.dumps({"status": "PAUSED"}).encode("utf-8")

        elif method == "POST" and clean_path == "/api/reset":
            self.session.reset()
            return 200, headers, json.dumps({"status": "RESET"}).encode("utf-8")

        elif method == "POST" and clean_path == "/api/step":
            samples = int(body.get("samples", 10))
            has_more = self.session.step_forward(num_samples=samples)
            return 200, headers, json.dumps({"status": "STEPPED", "has_more": has_more}).encode("utf-8")

        elif method == "POST" and clean_path == "/api/outage/start":
            self.session.trigger_outage()
            return 200, headers, json.dumps({"status": "OUTAGE_TRIGGERED"}).encode("utf-8")

        elif method == "POST" and clean_path == "/api/outage/restore":
            self.session.restore_gnss()
            return 200, headers, json.dumps({"status": "GNSS_RESTORED"}).encode("utf-8")

        elif method == "POST" and clean_path == "/api/config":
            # Scenario change requires reloading
            if "scenario" in body or "outage_duration" in body:
                scen = body.get("scenario", self.session.scenario_name)
                dur = float(body.get("outage_duration", self.session.outage_duration_s))
                self.session.load_scenario(scen, dur)

            self.session.set_config(
                enable_ai=body.get("enable_ai"),
                enable_nhc=body.get("enable_nhc"),
                enable_zupt=body.get("enable_zupt"),
                enable_map=body.get("enable_map"),
                playback_speed=body.get("playback_speed"),
            )
            return 200, headers, json.dumps({"status": "CONFIG_UPDATED"}).encode("utf-8")

        elif method == "POST" and clean_path == "/api/demo/run":
            self.controller.run_guided_demo()
            return 200, headers, json.dumps({"status": "GUIDED_DEMO_STARTED"}).encode("utf-8")

        return 404, headers, json.dumps({"error": f"Endpoint '{clean_path}' not found"}).encode("utf-8")
