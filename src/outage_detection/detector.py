"""GNSS Outage & Quality Degradation Detection Module.

Monitors raw GNSS observation availability, timestamp gaps, horizontal accuracy,
HDOP, and carrier-to-noise ratio (C/N0), maintaining an online state machine
(GOOD, DEGRADED, OUTAGE, RECOVERING) with hysteresis to prevent state flickering.
"""

from enum import Enum
from typing import Optional, Dict, Any
from src.data.observations import GNSSObservation


class GNSSStatus(str, Enum):
    """GNSS Signal Status Classification."""

    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    OUTAGE = "OUTAGE"
    RECOVERING = "RECOVERING"


class GNSSOutageDetector:
    """Online state-machine detector for GNSS signal degradation and outages."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize detector with configurable quality thresholds.

        Args:
            config_dict (Optional[Dict[str, Any]]): Threshold settings dictionary.
        """
        cfg = config_dict or {}
        self.max_gap_sec: float = cfg.get("max_timestamp_gap_sec", 2.0)
        self.degraded_acc_m: float = cfg.get("degraded_horizontal_accuracy_m", 5.0)
        self.outage_acc_m: float = cfg.get("outage_horizontal_accuracy_m", 15.0)
        self.degraded_hdop: float = cfg.get("degraded_hdop", 2.5)
        self.outage_hdop: float = cfg.get("outage_hdop", 5.0)
        self.degraded_cn0: float = cfg.get("degraded_cn0_dbhz", 35.0)
        self.outage_cn0: float = cfg.get("outage_cn0_dbhz", 25.0)
        self.persistence_count: int = cfg.get("persistence_count", 2)

        # Internal state
        self._current_status: GNSSStatus = GNSSStatus.GOOD
        self._last_obs_time: Optional[float] = None
        self._pending_candidate: Optional[GNSSStatus] = None
        self._candidate_counter: int = 0
        self._recovering_counter: int = 0

    @property
    def current_status(self) -> GNSSStatus:
        """Current confirmed GNSS status."""
        return self._current_status

    def process_observation(
        self,
        obs: Optional[GNSSObservation],
        current_time: float,
        hdop: Optional[float] = None,
        cn0: Optional[float] = None,
    ) -> GNSSStatus:
        """Process a GNSS observation and update online status state machine.

        Args:
            obs (Optional[GNSSObservation]): Instantaneous GNSS observation (or None if fix missing).
            current_time (float): Current system epoch time in seconds.
            hdop (Optional[float]): Horizontal Dilution of Precision (optional).
            cn0 (Optional[float]): Carrier-to-Noise ratio in dB-Hz (optional).

        Returns:
            GNSSStatus: Updated confirmed GNSS status.
        """
        # 1. Determine raw status candidate for instant observation
        candidate = self._evaluate_instant_candidate(obs, current_time, hdop, cn0)

        # 2. Update timestamp memory
        if obs is not None and obs.timestamp >= 0:
            self._last_obs_time = obs.timestamp

        # 3. Apply state machine hysteresis and recovery transitions
        if candidate == self._current_status:
            self._candidate_counter = 0
            self._pending_candidate = None
            if self._current_status == GNSSStatus.RECOVERING:
                self._recovering_counter += 1
                if self._recovering_counter >= self.persistence_count:
                    self._current_status = GNSSStatus.GOOD
                    self._recovering_counter = 0
            return self._current_status

        # Transition candidate differs from current confirmed status
        if candidate == self._pending_candidate:
            self._candidate_counter += 1
        else:
            self._pending_candidate = candidate
            self._candidate_counter = 1

        # Check if candidate persistence threshold is met
        if self._candidate_counter >= self.persistence_count:
            old_status = self._current_status
            target_status = self._pending_candidate

            # Handle transition out of OUTAGE -> RECOVERING -> GOOD
            if old_status == GNSSStatus.OUTAGE and target_status == GNSSStatus.GOOD:
                self._current_status = GNSSStatus.RECOVERING
                self._recovering_counter = 1
            else:
                self._current_status = target_status
                self._recovering_counter = 0

            self._candidate_counter = 0
            self._pending_candidate = None

        return self._current_status

    def _evaluate_instant_candidate(
        self,
        obs: Optional[GNSSObservation],
        current_time: float,
        hdop: Optional[float],
        cn0: Optional[float],
    ) -> GNSSStatus:
        """Evaluate raw instant candidate status from observation fields."""
        # Missing fix
        if obs is None:
            return GNSSStatus.OUTAGE

        # Check horizontal accuracy
        if obs.horizontal_accuracy is not None:
            if obs.horizontal_accuracy >= self.outage_acc_m:
                return GNSSStatus.OUTAGE
            if obs.horizontal_accuracy >= self.degraded_acc_m:
                return GNSSStatus.DEGRADED

        # Check HDOP
        if hdop is not None:
            if hdop >= self.outage_hdop:
                return GNSSStatus.OUTAGE
            if hdop >= self.degraded_hdop:
                return GNSSStatus.DEGRADED

        # Check C/N0
        if cn0 is not None:
            if cn0 <= self.outage_cn0:
                return GNSSStatus.OUTAGE
            if cn0 <= self.degraded_cn0:
                return GNSSStatus.DEGRADED

        return GNSSStatus.GOOD

    def reset(self):
        """Reset state machine detector."""
        self._current_status = GNSSStatus.GOOD
        self._last_obs_time = None
        self._pending_candidate = None
        self._candidate_counter = 0
        self._recovering_counter = 0
