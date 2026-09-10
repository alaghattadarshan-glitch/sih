"""Unit tests for GNSS Outage and Quality Degradation Detector."""

import pytest
from src.data.observations import GNSSObservation
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus


@pytest.fixture
def detector_cfg():
    return {
        "max_timestamp_gap_sec": 2.0,
        "degraded_horizontal_accuracy_m": 5.0,
        "outage_horizontal_accuracy_m": 15.0,
        "degraded_hdop": 2.5,
        "outage_hdop": 5.0,
        "degraded_cn0_dbhz": 35.0,
        "outage_cn0_dbhz": 25.0,
        "persistence_count": 2,
    }


def make_gnss(ts=1.0, lat=12.97, lon=77.59, alt=920.0, acc=2.0):
    return GNSSObservation(
        timestamp=ts,
        latitude=lat,
        longitude=lon,
        altitude=alt,
        horizontal_accuracy=acc,
        vertical_accuracy=3.0,
    )


def test_normal_gnss_good_status(detector_cfg):
    det = GNSSOutageDetector(detector_cfg)
    obs = make_gnss(ts=1.0, acc=1.5)
    
    # Process twice to satisfy persistence count if needed
    s1 = det.process_observation(obs, current_time=1.0, hdop=1.0, cn0=45.0)
    s2 = det.process_observation(make_gnss(ts=2.0, acc=1.5), current_time=2.0, hdop=1.0, cn0=45.0)
    assert s1 == GNSSStatus.GOOD
    assert s2 == GNSSStatus.GOOD


def test_missing_gnss_outage(detector_cfg):
    det = GNSSOutageDetector(detector_cfg)
    # Give initial good fix
    det.process_observation(make_gnss(ts=1.0), current_time=1.0)
    det.process_observation(make_gnss(ts=2.0), current_time=2.0)
    assert det.current_status == GNSSStatus.GOOD

    # Missing observations (obs=None)
    s1 = det.process_observation(None, current_time=3.0)
    assert s1 == GNSSStatus.GOOD  # Hysteresis persistence=1
    s2 = det.process_observation(None, current_time=4.0)
    assert s2 == GNSSStatus.OUTAGE  # Hysteresis persistence=2 -> confirmed OUTAGE


def test_timestamp_gap(detector_cfg):
    det = GNSSOutageDetector(detector_cfg)
    det.process_observation(make_gnss(ts=1.0), current_time=1.0)
    det.process_observation(make_gnss(ts=2.0), current_time=2.0)

    # Big gap in time without receiving new fix (gap = current_time - last_obs_time)
    det.process_observation(None, current_time=5.0)  # gap = 3.0s > max_gap=2.0s
    s_gap = det.process_observation(None, current_time=5.1)  # gap = 3.1s persistence count=2
    assert s_gap == GNSSStatus.OUTAGE


def test_poor_horizontal_accuracy(detector_cfg):
    det = GNSSOutageDetector(detector_cfg)
    det.process_observation(make_gnss(ts=1.0, acc=2.0), current_time=1.0)
    det.process_observation(make_gnss(ts=2.0, acc=2.0), current_time=2.0)

    # Degraded accuracy (8.0m > 5.0m)
    obs_deg = make_gnss(ts=3.0, acc=8.0)
    det.process_observation(obs_deg, current_time=3.0)
    s_deg = det.process_observation(make_gnss(ts=4.0, acc=8.0), current_time=4.0)
    assert s_deg == GNSSStatus.DEGRADED

    # Outage accuracy (20.0m > 15.0m)
    obs_out = make_gnss(ts=5.0, acc=20.0)
    det.process_observation(obs_out, current_time=5.0)
    s_out = det.process_observation(make_gnss(ts=6.0, acc=20.0), current_time=6.0)
    assert s_out == GNSSStatus.OUTAGE


def test_poor_hdop(detector_cfg):
    det = GNSSOutageDetector(detector_cfg)
    det.process_observation(make_gnss(ts=1.0), current_time=1.0, hdop=1.0)
    det.process_observation(make_gnss(ts=2.0), current_time=2.0, hdop=1.0)

    # Poor HDOP degraded (3.0 > 2.5)
    det.process_observation(make_gnss(ts=3.0), current_time=3.0, hdop=3.0)
    s_deg = det.process_observation(make_gnss(ts=4.0), current_time=4.0, hdop=3.0)
    assert s_deg == GNSSStatus.DEGRADED

    # Poor HDOP outage (6.0 > 5.0)
    det.process_observation(make_gnss(ts=5.0), current_time=5.0, hdop=6.0)
    s_out = det.process_observation(make_gnss(ts=6.0), current_time=6.0, hdop=6.0)
    assert s_out == GNSSStatus.OUTAGE


def test_poor_cn0(detector_cfg):
    det = GNSSOutageDetector(detector_cfg)
    det.process_observation(make_gnss(ts=1.0), current_time=1.0, cn0=45.0)
    det.process_observation(make_gnss(ts=2.0), current_time=2.0, cn0=45.0)

    # Low C/N0 degraded (30 dB-Hz < 35 dB-Hz)
    det.process_observation(make_gnss(ts=3.0), current_time=3.0, cn0=30.0)
    s_deg = det.process_observation(make_gnss(ts=4.0), current_time=4.0, cn0=30.0)
    assert s_deg == GNSSStatus.DEGRADED

    # Low C/N0 outage (20 dB-Hz < 25 dB-Hz)
    det.process_observation(make_gnss(ts=5.0), current_time=5.0, cn0=20.0)
    s_out = det.process_observation(make_gnss(ts=6.0), current_time=6.0, cn0=20.0)
    assert s_out == GNSSStatus.OUTAGE


def test_full_state_transitions(detector_cfg):
    """Test full cycle: GOOD -> DEGRADED -> OUTAGE -> RECOVERING -> GOOD."""
    det = GNSSOutageDetector(detector_cfg)

    # 1. Start GOOD
    det.process_observation(make_gnss(ts=1.0, acc=1.5), current_time=1.0)
    det.process_observation(make_gnss(ts=2.0, acc=1.5), current_time=2.0)
    assert det.current_status == GNSSStatus.GOOD

    # 2. Transition GOOD -> DEGRADED
    det.process_observation(make_gnss(ts=3.0, acc=7.0), current_time=3.0)
    s_deg = det.process_observation(make_gnss(ts=4.0, acc=7.0), current_time=4.0)
    assert s_deg == GNSSStatus.DEGRADED

    # 3. Transition DEGRADED -> OUTAGE
    det.process_observation(make_gnss(ts=5.0, acc=25.0), current_time=5.0)
    s_out = det.process_observation(make_gnss(ts=6.0, acc=25.0), current_time=6.0)
    assert s_out == GNSSStatus.OUTAGE

    # 4. Transition OUTAGE -> RECOVERING
    det.process_observation(make_gnss(ts=7.0, acc=1.5), current_time=7.0)
    s_rec = det.process_observation(make_gnss(ts=8.0, acc=1.5), current_time=8.0)
    assert s_rec == GNSSStatus.RECOVERING

    # 5. Transition RECOVERING -> GOOD
    det.process_observation(make_gnss(ts=9.0, acc=1.5), current_time=9.0)
    s_good = det.process_observation(make_gnss(ts=10.0, acc=1.5), current_time=10.0)
    assert s_good == GNSSStatus.GOOD


def test_noisy_threshold_persistence(detector_cfg):
    """Single noisy sample should not falsely trigger status change when persistence_count=2."""
    det = GNSSOutageDetector(detector_cfg)
    det.process_observation(make_gnss(ts=1.0, acc=1.5), current_time=1.0)
    det.process_observation(make_gnss(ts=2.0, acc=1.5), current_time=2.0)
    assert det.current_status == GNSSStatus.GOOD

    # Single bad sample
    s_single = det.process_observation(None, current_time=3.0)
    assert s_single == GNSSStatus.GOOD  # Prevent false alarm

    # Immediately followed by good sample
    s_recovered = det.process_observation(make_gnss(ts=4.0, acc=1.5), current_time=4.0)
    assert s_recovered == GNSSStatus.GOOD
