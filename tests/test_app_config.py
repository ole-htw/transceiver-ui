from __future__ import annotations

from transceiver.app_config import MissionRuntimeConfig


def test_mission_runtime_config_reads_remote_ros_setup_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCEIVER_REMOTE_ROS_SETUP", "/opt/ros/humble/setup.bash")

    config = MissionRuntimeConfig.from_env()

    assert config.remote_ros_setup == "/opt/ros/humble/setup.bash"
