"""IO-VNBD (Inertial Odometry / Vehicle Navigation Benchmark Dataset) Adapter.

Provides schema mapping and ingestion pipeline for the IO-VNBD dataset format.
If raw dataset files are not currently present in the repository, operates in
schema-validated standby mode, providing clear diagnostic notices and dataset requirements.
"""

import os
from typing import Dict, Any, Optional, List
from src.data.adapters.generic_csv import GenericCSVAdapter
from src.data.adapters.schema import (
    DatasetConfig,
    IMUColumnMapping,
    GNSSColumnMapping,
    GroundTruthColumnMapping,
    TimestampUnit,
    AccelUnit,
    GyroUnit,
    AxisTransformConfig,
)
from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation


class IOVNBDAdapter(GenericCSVAdapter):
    """Dataset adapter specialized for the IO-VNBD vehicle navigation benchmark."""

    @classmethod
    def create_default_config(
        cls,
        base_dir: str = "data/raw/io_vnbd",
        sequence_name: str = "seq_01",
    ) -> DatasetConfig:
        """Create standard dataset configuration template for IO-VNBD sequences.

        Args:
            base_dir (str): Base root folder for IO-VNBD raw data.
            sequence_name (str): Sequence identifier subfolder or prefix.

        Returns:
            DatasetConfig: Configured template with IO-VNBD standard column schema.
        """
        seq_dir = os.path.join(base_dir, sequence_name)
        return DatasetConfig(
            dataset_name=f"IO-VNBD_{sequence_name}",
            imu_file_path=os.path.join(seq_dir, "imu.csv"),
            gnss_file_path=os.path.join(seq_dir, "gnss.csv"),
            ground_truth_file_path=os.path.join(seq_dir, "ground_truth.csv"),
            imu_mapping=IMUColumnMapping(
                timestamp="timestamp",
                accel_x="accel_x",
                accel_y="accel_y",
                accel_z="accel_z",
                gyro_x="gyro_x",
                gyro_y="gyro_y",
                gyro_z="gyro_z",
                timestamp_unit=TimestampUnit.SECONDS,
                accel_unit=AccelUnit.MPS2,
                gyro_unit=GyroUnit.RAD_PER_SEC,
                axis_transform=AxisTransformConfig(x_map="+x", y_map="+y", z_map="+z"),
            ),
            gnss_mapping=GNSSColumnMapping(
                timestamp="timestamp",
                latitude="latitude",
                longitude="longitude",
                altitude="altitude",
                horizontal_accuracy="accuracy",
                timestamp_unit=TimestampUnit.SECONDS,
            ),
            ground_truth_mapping=GroundTruthColumnMapping(
                timestamp="timestamp",
                latitude="latitude",
                longitude="longitude",
                altitude="altitude",
                velocity_east="velocity_east",
                velocity_north="velocity_north",
                velocity_up="velocity_up",
                heading="heading",
                timestamp_unit=TimestampUnit.SECONDS,
            ),
            metadata={
                "benchmark_name": "IO-VNBD",
                "sequence": sequence_name,
                "status": "CONFIGURED_PENDING_DATASET_FILES",
            },
        )

    def is_dataset_available(self) -> bool:
        """Check whether the required dataset files exist locally.

        Returns:
            bool: True if IMU file is present, False otherwise.
        """
        return bool(self.config.imu_file_path and os.path.exists(self.config.imu_file_path))

    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata and availability diagnostic status."""
        meta = super().get_metadata()
        meta["dataset_available"] = self.is_dataset_available()
        if not self.is_dataset_available():
            meta["availability_notice"] = (
                f"IO-VNBD benchmark dataset file '{self.config.imu_file_path}' is not present. "
                "Execution pending dataset archive placement."
            )
        return meta
