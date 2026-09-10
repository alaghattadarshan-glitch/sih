"""Basic environment and project scaffolding verification tests."""

import importlib
import os
import sys
import yaml
import pytest


def test_python_environment():
    """Verify Python runtime version is compatible."""
    assert sys.version_info >= (3, 9), "Python version must be >= 3.9"


@pytest.mark.parametrize(
    "module_name",
    [
        "src",
        "src.data",
        "src.data.loaders",
        "src.data.synchronization",
        "src.data.preprocessing",
        "src.calibration",
        "src.coordinate_transforms",
        "src.navigation",
        "src.navigation.imu",
        "src.navigation.ins",
        "src.navigation.dead_reckoning",
        "src.navigation.ekf",
        "src.navigation.state",
        "src.ml",
        "src.ml.datasets",
        "src.ml.models",
        "src.ml.training",
        "src.ml.inference",
        "src.ml.evaluation",
        "src.map_matching",
        "src.outage_detection",
        "src.evaluation",
    ],
)
def test_module_imports(module_name):
    """Verify that all project modules can be imported without errors."""
    module = importlib.import_module(module_name)
    assert module is not None


def test_configuration_loading():
    """Verify that the central configuration file exists and can be loaded."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml"
    )
    assert os.path.exists(config_path), f"Config file not found at {config_path}"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), "Config file content is not a valid YAML dictionary"
    
    required_sections = [
        "project",
        "data",
        "navigation",
        "imu",
        "gnss",
        "machine_learning",
        "evaluation",
    ]
    for section in required_sections:
        assert section in config, f"Missing required configuration section: '{section}'"


def test_baseline_dependencies_importable():
    """Verify core third-party dependencies are available."""
    dependencies = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "sklearn",
        "yaml",
        "torch",
        "pytest",
    ]
    for dep in dependencies:
        mod = importlib.import_module(dep)
        assert mod is not None
