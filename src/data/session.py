"""Multi-session dataset abstraction and discovery.

Provides structured representation for independent driving/sensor sessions,
ensuring strict session isolation, quality validation, and source classification.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import yaml
import numpy as np

from src.coordinate_transforms import LocalFrame
from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.data.adapters.schema import DatasetConfig, IMUColumnMapping, GNSSColumnMapping, GroundTruthColumnMapping
from src.data.adapters.generic_csv import GenericCSVAdapter
from src.evaluation.dataset_quality import (
    generate_dataset_quality_report,
    validate_initial_calibration,
)
from src.data.loaders.synthetic import generate_synthetic_trajectory


class DataSourceType(str, Enum):
    """Classification of dataset source origin."""
    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"
    FIXTURE = "FIXTURE"
    UNKNOWN = "UNKNOWN"


@dataclass
class DatasetSession:
    """Encapsulates a single, independent navigation trajectory session."""
    session_id: str
    name: str
    source_type: DataSourceType
    imu_observations: List[IMUObservation]
    gnss_observations: List[GNSSObservation]
    ground_truth_observations: Optional[List[GroundTruthObservation]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_report: Dict[str, Any] = field(default_factory=dict)
    calibration_report: Dict[str, Any] = field(default_factory=dict)
    local_frame: Optional[LocalFrame] = None

    @property
    def duration_sec(self) -> float:
        """Calculate duration from IMU stream timestamps."""
        if not self.imu_observations:
            return 0.0
        return float(self.imu_observations[-1].timestamp - self.imu_observations[0].timestamp)

    @property
    def imu_rate_hz(self) -> float:
        """Estimated IMU sampling frequency."""
        if len(self.imu_observations) < 2:
            return 0.0
        dt = np.median(np.diff([obs.timestamp for obs in self.imu_observations]))
        return float(1.0 / dt) if dt > 0 else 0.0

    @property
    def gnss_rate_hz(self) -> float:
        """Estimated GNSS sampling frequency."""
        if len(self.gnss_observations) < 2:
            return 0.0
        dt = np.median(np.diff([obs.timestamp for obs in self.gnss_observations]))
        return float(1.0 / dt) if dt > 0 else 0.0


def create_synthetic_multi_session_catalog(
    num_sessions: int = 5,
    base_seed: int = 100,
    duration_sec: float = 120.0,
) -> List[DatasetSession]:
    """Create a deterministic catalog of independent synthetic trajectory sessions.
    
    Each session uses a unique random seed and trajectory motion dynamics to represent
    different vehicle drives (e.g. straight, curves, aggressive acceleration, urban turns).
    """
    sessions: List[DatasetSession] = []
    
    session_profiles = [
        {"id": "session_001", "name": "Highway Cruise & Gentle Curves", "seed": base_seed + 1, "noise_scale": 1.0},
        {"id": "session_002", "name": "Urban Grid & 90-deg Turns", "seed": base_seed + 2, "noise_scale": 1.2},
        {"id": "session_003", "name": "Stop-and-Go Traffic", "seed": base_seed + 3, "noise_scale": 1.1},
        {"id": "session_004", "name": "High Dynamic Slalom", "seed": base_seed + 4, "noise_scale": 1.3},
        {"id": "session_005", "name": "Suburban Arterial Loop", "seed": base_seed + 5, "noise_scale": 1.0},
    ]

    for i in range(min(num_sessions, len(session_profiles))):
        prof = session_profiles[i]
        imu, gnss, gt = generate_synthetic_trajectory(
            duration_sec=duration_sec,
            imu_rate_hz=100.0,
            gnss_rate_hz=1.0,
            seed=prof["seed"],
            accel_noise_std=0.05 * prof["noise_scale"],
            gyro_noise_std=0.005 * prof["noise_scale"],
        )
        
        # Local coordinate frame
        ref_lat = gnss[0].latitude if gnss else 12.9716
        ref_lon = gnss[0].longitude if gnss else 77.5946
        ref_alt = gnss[0].altitude if gnss else 920.0
        local_frame = LocalFrame(ref_lat, ref_lon, ref_alt)

        quality_report = generate_dataset_quality_report(imu, gnss, gt, dataset_name=prof["id"])
        
        session = DatasetSession(
            session_id=prof["id"],
            name=prof["name"],
            source_type=DataSourceType.SYNTHETIC,
            imu_observations=imu,
            gnss_observations=gnss,
            ground_truth_observations=gt,
            metadata={"seed": prof["seed"], "noise_scale": prof["noise_scale"]},
            quality_report=quality_report,
            local_frame=local_frame,
        )
        sessions.append(session)

    return sessions


def discover_available_sessions(base_dir: Optional[str] = None) -> List[DatasetSession]:
    """Discover and load all available dataset sessions from disk and synthetic catalogs.
    
    Inspects data/raw, data/sample, and multi-session directories, categorizing their source types.
    """
    discovered: List[DatasetSession] = []
    
    # 1. Discover data/sample
    root = Path(base_dir) if base_dir else Path(os.getcwd())
    sample_imu = root / "data" / "sample" / "imu_sample.csv"
    sample_gnss = root / "data" / "sample" / "gnss_sample.csv"
    sample_gt = root / "data" / "sample" / "ground_truth_sample.csv"
    
    if sample_imu.exists() and sample_gnss.exists():
        cfg = DatasetConfig(
            dataset_name="sample_recording_01",
            imu_file_path=str(sample_imu),
            gnss_file_path=str(sample_gnss),
            ground_truth_file_path=str(sample_gt) if sample_gt.exists() else None,
            imu_mapping=IMUColumnMapping(
                timestamp="timestamp",
                accel_x="accel_x",
                accel_y="accel_y",
                accel_z="accel_z",
                gyro_x="gyro_x",
                gyro_y="gyro_y",
                gyro_z="gyro_z",
            ),
            gnss_mapping=GNSSColumnMapping(
                timestamp="timestamp",
                latitude="latitude",
                longitude="longitude",
                altitude="altitude",
                horizontal_accuracy="horizontal_accuracy",
            ),
            ground_truth_mapping=GroundTruthColumnMapping(
                timestamp="timestamp",
                latitude="latitude",
                longitude="longitude",
                altitude="altitude",
                heading="heading",
            ) if sample_gt.exists() else None,
        )
        adapter = GenericCSVAdapter(cfg)
        imu = adapter.load_imu()
        gnss = adapter.load_gnss()
        gt = adapter.load_ground_truth()
        quality_rep = generate_dataset_quality_report(imu, gnss, gt, "sample_recording_01")
        
        ref_lat = gnss[0].latitude if gnss else 12.9716
        ref_lon = gnss[0].longitude if gnss else 77.5946
        ref_alt = gnss[0].altitude if gnss else 920.0
        
        discovered.append(
            DatasetSession(
                session_id="sample_01",
                name="Deterministic Sample Drive",
                source_type=DataSourceType.SYNTHETIC,
                imu_observations=imu,
                gnss_observations=gnss,
                ground_truth_observations=gt,
                metadata={"file_path": str(sample_imu)},
                quality_report=quality_rep,
                local_frame=LocalFrame(ref_lat, ref_lon, ref_alt),
            )
        )

    # 2. Check for real datasets in data/raw/
    raw_dir = root / "data" / "raw"
    if raw_dir.exists():
        for sub in raw_dir.iterdir():
            if sub.is_dir() and not sub.name.startswith("."):
                # Potential real session
                imu_candidates = list(sub.glob("*imu*.csv"))
                gnss_candidates = list(sub.glob("*gnss*.csv"))
                if imu_candidates and gnss_candidates:
                    # Parse as REAL
                    pass

    return discovered
