from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_defaults(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries from a .env file without overriding existing env vars."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, parsed)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed


@dataclass(frozen=True)
class MissionRuntimeConfig:
    robot_host: str = "ole@192.168.10.10"
    ros2_namespace: str = ""
    ros2_action_name: str = "/navigate_to_pose"
    remote_ros_env_cmd: str = ""
    remote_ros_setup: str = ""
    goal_acceptance_timeout_s: float = 8.0
    goal_reached_timeout_s: float = 120.0
    navigation_retry_attempts: int = 0

    @classmethod
    def from_env(cls) -> "MissionRuntimeConfig":
        _load_dotenv_defaults()
        return cls(
            robot_host=os.getenv("TRANSCEIVER_ROBOT_HOST", cls.robot_host),
            ros2_namespace=os.getenv("TRANSCEIVER_ROS2_NAMESPACE", cls.ros2_namespace),
            ros2_action_name=os.getenv("TRANSCEIVER_ROS2_ACTION_NAME", cls.ros2_action_name),
            remote_ros_env_cmd=os.getenv("TRANSCEIVER_REMOTE_ROS_ENV_CMD", cls.remote_ros_env_cmd),
            remote_ros_setup=os.getenv("TRANSCEIVER_REMOTE_ROS_SETUP", cls.remote_ros_setup),
            goal_acceptance_timeout_s=_env_float(
                "TRANSCEIVER_NAV_GOAL_ACCEPT_TIMEOUT_S",
                cls.goal_acceptance_timeout_s,
            ),
            goal_reached_timeout_s=_env_float(
                "TRANSCEIVER_NAV_GOAL_REACHED_TIMEOUT_S",
                cls.goal_reached_timeout_s,
            ),
            navigation_retry_attempts=max(
                0,
                _env_int(
                    "TRANSCEIVER_NAV_RETRY_ATTEMPTS",
                    cls.navigation_retry_attempts,
                ),
            ),
        )
