"""Background Replay Controller and Guided Demo Engine."""

import time
import threading
from typing import Optional
from src.demo.session_manager import NavigationSessionManager


class ReplayController:
    """Controls the asynchronous playback loop and automated SIH demonstration sequence."""

    def __init__(self, session: NavigationSessionManager):
        """Initialize controller with session manager."""
        self.session = session
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_guided_demo_active = False

    def start_loop(self):
        """Start background execution loop."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop_loop(self):
        """Stop background execution loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run_loop(self):
        """Main background ticker executing sample batches."""
        while not self._stop_event.is_set():
            if self.session.is_running and not self.session.is_paused:
                # 100 Hz simulation rate; batch size scaled by playback speed
                # Target UI update rate ~20 Hz -> 5 IMU samples per 50ms tick at 1x
                speed = max(0.1, self.session.playback_speed)
                batch_size = max(1, int(5 * speed))

                has_more = self.session.step_forward(num_samples=batch_size)
                if not has_more:
                    self.session.is_running = False
                    self.session.log_event("Reached end of trajectory.", "SYSTEM")

                # Sleep to maintain ~20 Hz tick
                time.sleep(0.05)
            else:
                time.sleep(0.1)

    def run_guided_demo(self):
        """Execute automated, judge-ready SIH demo sequence in a background thread."""
        def _guided_worker():
            self.is_guided_demo_active = True
            self.session.log_event("--- STARTING GUIDED SIH DEMO SEQUENCE ---", "DEMO")

            # 1. Load Mixed Urban Scenario
            self.session.load_scenario("mixed_urban", outage_duration_s=30.0)
            self.session.playback_speed = 3.0  # Run at 3x for snappy presentation
            self.session.start()

            # 2. Let vehicle cruise in GNSS_AIDED mode for ~10 seconds of sim time
            while self.session.is_running and (self.session.time_history[-1] if self.session.time_history else 0) < 30.0:
                time.sleep(0.1)

            # 3. Simulate GNSS Outage (30s to 60s)
            self.session.log_event("[DEMO] Outage window entered. AI + NHC + ZUPT + Map activated.", "DEMO")
            while self.session.is_running and (self.session.time_history[-1] if self.session.time_history else 0) < 60.0:
                time.sleep(0.1)

            # 4. Observe GNSS Recovery
            self.session.log_event("[DEMO] GNSS signal restored. Monitoring filter re-acquisition transient.", "DEMO")
            while self.session.is_running and (self.session.time_history[-1] if self.session.time_history else 0) < 80.0:
                time.sleep(0.1)

            self.session.log_event("--- GUIDED SIH DEMO SEQUENCE COMPLETED SUCCESSFULLY ---", "DEMO")
            self.is_guided_demo_active = False

        threading.Thread(target=_guided_worker, daemon=True).start()
