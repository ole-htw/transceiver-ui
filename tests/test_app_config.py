from __future__ import annotations

from transceiver.app_config import MissionRuntimeConfig


def test_mission_runtime_config_reads_remote_ros_setup_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCEIVER_REMOTE_ROS_SETUP", "/opt/ros/humble/setup.bash")

    config = MissionRuntimeConfig.from_env()

    assert config.remote_ros_setup == "/opt/ros/humble/setup.bash"


def test_mission_runtime_config_reads_values_from_dotenv(tmp_path, monkeypatch) -> None:
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "TRANSCEIVER_ROBOT_HOST=robot@10.0.0.2\n"
        "TRANSCEIVER_NAV_RETRY_ATTEMPTS=2\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRANSCEIVER_ROBOT_HOST", raising=False)
    monkeypatch.delenv("TRANSCEIVER_NAV_RETRY_ATTEMPTS", raising=False)

    config = MissionRuntimeConfig.from_env()

    assert config.robot_host == "robot@10.0.0.2"
    assert config.navigation_retry_attempts == 2
