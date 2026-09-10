"""Comprehensive unit test suite for Geodetic, ECEF, and ENU coordinate transformations."""

import math
import pytest
from src.coordinate_transforms import (
    WGS84,
    llh_to_ecef,
    ecef_to_llh,
    ecef_to_enu,
    enu_to_ecef,
    llh_to_enu,
    enu_to_llh,
    LocalFrame,
)

# Reference geographic locations for validation
TEST_LOCATIONS = [
    # (name, lat_deg, lon_deg, h_m)
    ("Equator_PrimeMeridian", 0.0, 0.0, 0.0),
    ("Bengaluru_India", 12.9716, 77.5946, 920.0),
    ("London_UK", 51.5074, -0.1278, 15.0),
    ("Sydney_Australia", -33.8688, 151.2093, 30.0),
    ("HighLatitude_Svalbard", 78.2232, 15.6267, 20.0),
    ("NegativeLongitude_NewYork", 40.7128, -74.0060, 10.0),
    ("HighAltitude_Himalayas", 27.9881, 86.9250, 5364.0),
    ("NegativeAltitude_DeadSea", 31.5000, 35.5000, -430.0),
]


def test_llh_to_ecef():
    """Test LLH to ECEF conversion for known analytical reference points."""
    # Point on equator at prime meridian (0, 0, 0)
    x, y, z = llh_to_ecef(0.0, 0.0, 0.0)
    assert pytest.approx(x, abs=1e-3) == WGS84.a
    assert pytest.approx(y, abs=1e-3) == 0.0
    assert pytest.approx(z, abs=1e-3) == 0.0

    # Point at North Pole (90, 0, 0)
    x_north, y_north, z_north = llh_to_ecef(90.0, 0.0, 0.0)
    assert pytest.approx(x_north, abs=1e-3) == 0.0
    assert pytest.approx(y_north, abs=1e-3) == 0.0
    assert pytest.approx(z_north, abs=1e-3) == WGS84.b


def test_ecef_to_llh():
    """Test ECEF to LLH conversion for known analytical reference points."""
    # ECEF point corresponding to (0, 0, 0)
    lat, lon, h = ecef_to_llh(WGS84.a, 0.0, 0.0)
    assert pytest.approx(lat, abs=1e-7) == 0.0
    assert pytest.approx(lon, abs=1e-7) == 0.0
    assert pytest.approx(h, abs=1e-4) == 0.0

    # ECEF point corresponding to North Pole
    lat_n, lon_n, h_n = ecef_to_llh(0.0, 0.0, WGS84.b)
    assert pytest.approx(lat_n, abs=1e-7) == 90.0
    assert pytest.approx(h_n, abs=1e-4) == 0.0


@pytest.mark.parametrize("name, lat, lon, h", TEST_LOCATIONS)
def test_llh_ecef_roundtrip(name, lat, lon, h):
    """Test LLH -> ECEF -> LLH roundtrip precision across global test locations."""
    x, y, z = llh_to_ecef(lat, lon, h)
    lat_rec, lon_rec, h_rec = ecef_to_llh(x, y, z)

    assert pytest.approx(lat_rec, abs=1e-8) == lat, f"Latitude mismatch for {name}"
    assert pytest.approx(lon_rec, abs=1e-8) == lon, f"Longitude mismatch for {name}"
    assert pytest.approx(h_rec, abs=1e-4) == h, f"Height mismatch for {name}"


def test_ecef_to_enu():
    """Test ECEF to ENU transformation at a local reference origin."""
    ref_lat, ref_lon, ref_h = 12.9716, 77.5946, 920.0
    x0, y0, z0 = llh_to_ecef(ref_lat, ref_lon, ref_h)

    # Origin point should map to (0, 0, 0) ENU
    e, n, u = ecef_to_enu(x0, y0, z0, ref_lat, ref_lon, ref_h)
    assert pytest.approx(e, abs=1e-6) == 0.0
    assert pytest.approx(n, abs=1e-6) == 0.0
    assert pytest.approx(u, abs=1e-6) == 0.0


def test_enu_to_ecef():
    """Test ENU to ECEF transformation for zero displacement."""
    ref_lat, ref_lon, ref_h = 12.9716, 77.5946, 920.0
    x0, y0, z0 = llh_to_ecef(ref_lat, ref_lon, ref_h)

    x, y, z = enu_to_ecef(0.0, 0.0, 0.0, ref_lat, ref_lon, ref_h)
    assert pytest.approx(x, abs=1e-6) == x0
    assert pytest.approx(y, abs=1e-6) == y0
    assert pytest.approx(z, abs=1e-6) == z0


@pytest.mark.parametrize("name, lat, lon, h", TEST_LOCATIONS)
def test_enu_roundtrip(name, lat, lon, h):
    """Test LLH -> ECEF -> ENU -> ECEF -> LLH roundtrip consistency."""
    # Offset target point by a small displacement
    target_lat = lat + 0.001
    target_lon = lon + 0.001
    target_h = h + 10.0

    east, north, up = llh_to_enu(target_lat, target_lon, target_h, lat, lon, h)
    rec_lat, rec_lon, rec_h = enu_to_llh(east, north, up, lat, lon, h)

    assert pytest.approx(rec_lat, abs=1e-7) == target_lat
    assert pytest.approx(rec_lon, abs=1e-7) == target_lon
    assert pytest.approx(rec_h, abs=1e-4) == target_h


def test_equator():
    """Test transformation behavior specifically at the equator."""
    e, n, u = llh_to_enu(0.0, 0.001, 0.0, 0.0, 0.0, 0.0)
    # Moving east along equator should result in positive East displacement and near zero North/Up
    assert e > 100.0  # Approx 111 meters
    assert pytest.approx(n, abs=0.1) == 0.0


def test_high_latitude():
    """Test transformations near extreme polar latitudes (89.9° N)."""
    ref_lat, ref_lon, ref_h = 89.9, 10.0, 50.0
    target_lat, target_lon, target_h = 89.901, 10.001, 55.0

    e, n, u = llh_to_enu(target_lat, target_lon, target_h, ref_lat, ref_lon, ref_h)
    rec_lat, rec_lon, rec_h = enu_to_llh(e, n, u, ref_lat, ref_lon, ref_h)

    assert pytest.approx(rec_lat, abs=1e-7) == target_lat
    assert pytest.approx(rec_lon, abs=1e-7) == target_lon
    assert pytest.approx(rec_h, abs=1e-4) == target_h


def test_negative_longitude():
    """Test transformation handling across negative longitudes (e.g. New York / Americas)."""
    ref_lat, ref_lon, ref_h = 40.7128, -74.0060, 10.0
    x, y, z = llh_to_ecef(ref_lat, ref_lon, ref_h)
    lat_rec, lon_rec, h_rec = ecef_to_llh(x, y, z)

    assert pytest.approx(lat_rec, abs=1e-8) == ref_lat
    assert pytest.approx(lon_rec, abs=1e-8) == ref_lon
    assert pytest.approx(h_rec, abs=1e-4) == ref_h


def test_altitude():
    """Test positive and negative altitude transformations."""
    # High altitude test
    x_high, y_high, z_high = llh_to_ecef(12.9716, 77.5946, 10000.0)
    _, _, h_high = ecef_to_llh(x_high, y_high, z_high)
    assert pytest.approx(h_high, abs=1e-4) == 10000.0

    # Negative altitude test (sub-surface / subterranean / Dead Sea)
    x_low, y_low, z_low = llh_to_ecef(31.5, 35.5, -430.0)
    _, _, h_low = ecef_to_llh(x_low, y_low, z_low)
    assert pytest.approx(h_low, abs=1e-4) == -430.0


def test_small_local_displacement():
    """Test sub-meter small local displacement (10cm movement)."""
    ref_lat, ref_lon, ref_h = 12.9716, 77.5946, 920.0
    frame = LocalFrame(ref_lat, ref_lon, ref_h)

    # 10 cm East, 20 cm North, 5 cm Up
    e_in, n_in, u_in = 0.10, 0.20, 0.05
    lat, lon, h = frame.from_enu(e_in, n_in, u_in)
    e_out, n_out, u_out = frame.to_enu(lat, lon, h)

    assert pytest.approx(e_out, abs=1e-5) == e_in
    assert pytest.approx(n_out, abs=1e-5) == n_in
    assert pytest.approx(u_out, abs=1e-5) == u_in


def test_large_local_displacement():
    """Test 10km local displacement within reasonable tangent plane limits."""
    frame = LocalFrame(12.9716, 77.5946, 920.0)
    e_in, n_in, u_in = 10000.0, 10000.0, 50.0

    lat, lon, h = frame.from_enu(e_in, n_in, u_in)
    e_out, n_out, u_out = frame.to_enu(lat, lon, h)

    assert pytest.approx(e_out, abs=1e-3) == e_in
    assert pytest.approx(n_out, abs=1e-3) == n_in
    assert pytest.approx(u_out, abs=1e-3) == u_in


def test_longitude_wraparound():
    """Test longitude normalization around +/-180 degrees boundaries."""
    x1, y1, z1 = llh_to_ecef(0.0, 180.0, 0.0)
    x2, y2, z2 = llh_to_ecef(0.0, -180.0, 0.0)

    assert pytest.approx(x1, abs=1e-4) == x2
    assert pytest.approx(y1, abs=1e-4) == y2
    assert pytest.approx(z1, abs=1e-4) == z2


def test_local_frame_class():
    """Test object-oriented LocalFrame class methods."""
    frame = LocalFrame(12.9716, 77.5946, 920.0)

    assert frame.ref_lat == 12.9716
    assert frame.ref_lon == 77.5946
    assert frame.ref_height == 920.0

    # Test roundtrip via ECEF
    e, n, u = 50.0, 100.0, 10.0
    x, y, z = frame.to_ecef(e, n, u)
    e_rec, n_rec, u_rec = frame.from_ecef(x, y, z)

    assert pytest.approx(e_rec, abs=1e-6) == e
    assert pytest.approx(n_rec, abs=1e-6) == n
    assert pytest.approx(u_rec, abs=1e-6) == u
