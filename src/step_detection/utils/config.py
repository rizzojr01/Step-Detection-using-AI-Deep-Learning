"""
Configuration management for Step Detection project.
Loads and provides access to configuration parameters from config.yaml.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for step detection project."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml file. If None, searches in standard locations.
        """
        if config_path is None:
            config_path = self._find_config_file()

        self.config_path = config_path
        self._config = self._load_config()

    def _find_config_file(self) -> str:
        """Find config.yaml in standard locations."""
        possible_paths = [
            "config/config.yaml",
            "config.yaml",
            "../config/config.yaml",
            "../../config/config.yaml",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Could not find config.yaml in any of these locations: {possible_paths}"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'detection.confidence_threshold')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name (e.g., 'detection', 'training')

        Returns:
            Dictionary with section configuration
        """
        return self._config.get(section, {})

    # Convenience properties for commonly used configurations

    @property
    def detection(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self.get_section("detection")

    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get_section("training")

    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get_section("model")

    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get_section("api")

    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get_section("data")

    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get_section("logging")

    # Detection-specific convenience methods

    def get_confidence_threshold(self) -> float:
        """Get confidence threshold for step detection."""
        return self.get("detection.confidence_threshold", 0.7)

    def get_magnitude_threshold(self) -> float:
        """Get magnitude threshold for filtering small movements."""
        return self.get("detection.magnitude_threshold", 15.0)

    def get_start_threshold(self) -> float:
        """Get legacy start threshold."""
        return self.get("detection.start_threshold", 0.07)

    def get_end_threshold(self) -> float:
        """Get legacy end threshold."""
        return self.get("detection.end_threshold", 0.07)

    # Detection filter settings

    def use_confidence_filter(self) -> bool:
        """Check if confidence filtering should be used."""
        return self.get("detection.use_confidence_filter", True)

    def use_magnitude_filter(self) -> bool:
        """Check if magnitude filtering should be used."""
        return self.get("detection.use_magnitude_filter", True)

    def is_magnitude_filter_enabled(self) -> bool:
        """Check if magnitude filter is enabled (alias for use_magnitude_filter)."""
        return self.use_magnitude_filter()

    def is_confidence_filter_enabled(self) -> bool:
        """Check if confidence filter is enabled (alias for use_confidence_filter)."""
        return self.use_confidence_filter()

    def is_time_filter_enabled(self) -> bool:
        """Check if time filter is enabled."""
        return self.get("detection.enable_time_filter", True)

    def get_min_step_interval(self) -> float:
        """Get minimum time interval between step detections in seconds."""
        return self.get("detection.min_step_interval", 0.3)

    def get_step_timeout(self) -> float:
        """Get maximum time a step can remain in progress in seconds."""
        return self.get("detection.step_timeout", 2.0)

    def get_max_step_rate(self) -> float:
        """Get maximum steps per second allowed."""
        return self.get("detection.max_step_rate", 4.0)

    def get_step_rate_window(self) -> float:
        """Get time window for rate calculation in seconds."""
        return self.get("detection.step_rate_window", 1.0)

    def get_min_step_duration(self) -> float:
        """Get minimum step duration in seconds."""
        return self.get("detection.min_step_duration", 0.1)

    def get_max_step_duration(self) -> float:
        """Get maximum step duration in seconds."""
        return self.get("detection.max_step_duration", 2.0)

    # Debug configuration methods

    def is_step_detection_logs_enabled(self) -> bool:
        """Check if detailed step detection logging is enabled."""
        return self.get("debug.enable_step_detection_logs", True)

    def is_raw_model_tracking_enabled(self) -> bool:
        """Check if raw model tracking is enabled."""
        return self.get("debug.enable_raw_model_tracking", True)

    def should_log_only_on_activity(self) -> bool:
        """Check if logging should only happen on step activity."""
        return self.get("debug.log_only_on_activity", True)

    def should_show_confidence_threshold(self) -> bool:
        """Check if confidence threshold should be shown in logs."""
        return self.get("debug.show_confidence_threshold", True)

    # Training-specific convenience methods

    def get_dropout_rate(self) -> float:
        """Get dropout rate for model training."""
        return self.get("training.dropout_rate", 0.3)

    def get_regularization(self) -> float:
        """Get L2 regularization factor."""
        return self.get("training.regularization", 0.001)

    def use_class_weights(self) -> bool:
        """Check if class weights should be used in training."""
        return self.get("training.use_class_weights", True)

    def get_epochs(self) -> int:
        """Get number of training epochs."""
        return self.get("training.epochs", 50)

    def get_batch_size(self) -> int:
        """Get training batch size."""
        return self.get("training.batch_size", 32)

    def get_learning_rate(self) -> float:
        """Get learning rate."""
        return self.get("training.learning_rate", 0.001)

    # Model-specific convenience methods

    def get_input_shape(self) -> list:
        """Get model input shape."""
        return self.get("model.input_shape", [6])

    def get_output_classes(self) -> int:
        """Get number of output classes."""
        return self.get("model.output_classes", 3)

    # Data paths

    def get_raw_data_path(self) -> str:
        """Get raw data path."""
        return self.get("data.raw_data_path", "data/raw")

    def get_processed_data_path(self) -> str:
        """Get processed data path."""
        return self.get("data.processed_data_path", "data/processed")

    def get_model_save_path(self) -> str:
        """Get model save path."""
        return self.get("data.model_save_path", "models")

    # API configuration

    def get_api_host(self) -> str:
        """Get API host."""
        return self.get("api.host", "0.0.0.0")

    def get_api_port(self) -> int:
        """Get API port."""
        return self.get("api.port", 8000)

    def get_api_reload(self) -> bool:
        """Get API reload setting."""
        return self.get("api.reload", True)

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(path={self.config_path})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Config(path='{self.config_path}', sections={list(self._config.keys())})"
        )


# Global config instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Path to config file. Only used on first call.

    Returns:
        Config instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_path)

    return _config_instance


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Reload configuration.

    Args:
        config_path: Path to config file

    Returns:
        New Config instance
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance


# Convenience functions for direct access
def get_detection_config() -> Dict[str, Any]:
    """Get detection configuration."""
    return get_config().detection


def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    return get_config().training


def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return get_config().model
