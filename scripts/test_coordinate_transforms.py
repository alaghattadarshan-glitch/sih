#!/usr/bin/env python3
"""Demonstration script for WGS84 Geodetic, ECEF, and Local ENU Transformations.

This script demonstrates coordinate transformations by:
1. Defining a reference geographic position (e.g. Bengaluru, India).
2. Defining a nearby target geographic position.
3. Converting both points from LLH to ECEF.
4. Converting the target point into local ENU displacement relative to the reference origin.
"""

import sys
from pathlib import Path

# Ensure project root is in Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coordinate_transforms import (
    llh_to_ecef,
    LocalFrame,
)


def main():
    # 1. Define reference geographic position (Origin: Bengaluru, India)
    ref_lat = 12.9716  # degrees North
    ref_lon = 77.5946  # degrees East
    ref_h = 920.0  # meters above WGS84 ellipsoid

    # 2. Define nearby target geographic position
    target_lat = 12.9750  # degrees North
    target_lon = 77.5980  # degrees East
    target_h = 925.0  # meters

    # 3. Convert both points to ECEF
    ref_x, ref_y, ref_z = llh_to_ecef(ref_lat, ref_lon, ref_h)
    target_x, target_y, target_z = llh_to_ecef(target_lat, target_lon, target_h)

    # 4. Convert target point into local ENU relative to reference
    frame = LocalFrame(ref_lat, ref_lon, ref_h)
    east, north, up = frame.to_enu(target_lat, target_lon, target_h)

    # 5. Print results in requested clean format
    print("==========================================================")
    print("   SIH26168 COORDINATE TRANSFORMATION DEMONSTRATION")
    print("==========================================================")
    print("Reference:")
    print(f"Lat:    {ref_lat:.6f}°")
    print(f"Lon:    {ref_lon:.6f}°")
    print(f"Height: {ref_h:.2f} m")
    print()
    print("Target:")
    print(f"Lat:    {target_lat:.6f}°")
    print(f"Lon:    {target_lon:.6f}°")
    print(f"Height: {target_h:.2f} m")
    print()
    print("Reference ECEF:")
    print(f"X: {ref_x:14.3f} m")
    print(f"Y: {ref_y:14.3f} m")
    print(f"Z: {ref_z:14.3f} m")
    print()
    print("Target ECEF:")
    print(f"X: {target_x:14.3f} m")
    print(f"Y: {target_y:14.3f} m")
    print(f"Z: {target_z:14.3f} m")
    print()
    print("Local displacement:")
    print(f"East:  {east:10.3f} m")
    print(f"North: {north:10.3f} m")
    print(f"Up:    {up:10.3f} m")
    print("==========================================================")


if __name__ == "__main__":
    main()
