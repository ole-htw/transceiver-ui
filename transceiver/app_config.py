from __future__ import annotations

import os
from dataclasses import dataclass


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
    goal_acceptance_timeout_s: float = 8.0
    goal_reached_timeout_s: float = 120.0
    navigation_retry_attempts: int = 0

    @classmethod
    def from_env(cls) -> "MissionRuntimeConfig":
        return cls(
            robot_host=os.getenv("TRANSCEIVER_ROBOT_HOST", cls.robot_host),
            ros2_namespace=os.getenv("TRANSCEIVER_ROS2_NAMESPACE", cls.ros2_namespace),
            ros2_action_name=os.getenv("TRANSCEIVER_ROS2_ACTION_NAME", cls.ros2_action_name),
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
