from __future__ import annotations

from transceiver.app_config import MissionRuntimeConfig


def test_mission_runtime_config_reads_remote_ros_setup_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCEIVER_REMOTE_ROS_SETUP", "/opt/ros/jazzy/setup.bash")

    config = MissionRuntimeConfig.from_env()

    assert config.remote_ros_setup == "/opt/ros/jazzy/setup.bash"


def test_mission_runtime_config_reads_remote_ros_env_cmd_from_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "TRANSCEIVER_REMOTE_ROS_ENV_CMD",
        "source /opt/ros/jazzy/setup.bash && source ~/ws/install/setup.bash",
    )

    config = MissionRuntimeConfig.from_env()

    assert (
        config.remote_ros_env_cmd
        == "source /opt/ros/jazzy/setup.bash && source ~/ws/install/setup.bash"
    )


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


def test_mission_runtime_config_reads_fastdds_profile_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCEIVER_FASTDDS_PROFILES_FILE", "/etc/nav2/fastdds/nav2.xml")

    config = MissionRuntimeConfig.from_env()

    assert config.fastdds_profiles_file == "/etc/nav2/fastdds/nav2.xml"


def test_mission_runtime_config_reads_lidar_runtime_values_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCEIVER_LIDAR_TOPIC", "/robot1/scan")
    monkeypatch.setenv("TRANSCEIVER_LIDAR_REFERENCE_TIMEOUT_S", "21")

    config = MissionRuntimeConfig.from_env()

    assert config.lidar_topic == "/robot1/scan"
    assert config.lidar_reference_timeout_s == 21.0
