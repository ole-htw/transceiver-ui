from __future__ import annotations

from dataclasses import dataclass

from transceiver.navigation_adapter import (
    NavigationAdapter,
    NavigationAdapterConfig,
    NavigationOutcome,
    NavigationPoint,
    Ros2CliNavigationTransport,
)


@dataclass
class _FakeTransport:
    outcomes: list[NavigationOutcome]

    def __post_init__(self) -> None:
        self.send_calls = 0
        self.cancel_calls = 0

    def send_goal(self, *, point, config, on_feedback):
        self.send_calls += 1
        on_feedback({"remaining_distance": 1.23})
        idx = min(self.send_calls - 1, len(self.outcomes) - 1)
        return self.outcomes[idx]

    def cancel_current_goal(self) -> None:
        self.cancel_calls += 1


def test_build_goal_payload_uses_navigate_to_pose_in_map_frame() -> None:
    payload = Ros2CliNavigationTransport.build_goal_payload(
        NavigationPoint(x=1.0, y=2.0, z=0.5, qx=0.1, qy=0.2, qz=0.3, qw=0.9)
    )

    assert payload["pose"]["header"]["frame_id"] == "map"
    assert payload["pose"]["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 0.5}
    assert payload["pose"]["pose"]["orientation"] == {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "w": 0.9,
    }


def test_adapter_retries_on_connection_error_and_emits_events() -> None:
    fake = _FakeTransport(
        outcomes=[
            NavigationOutcome("connection_error", accepted=False, message="ssh failed"),
            NavigationOutcome("succeeded", accepted=True, message="done"),
        ]
    )
    adapter = NavigationAdapter(
        transport=fake,
        config=NavigationAdapterConfig(retry_attempts=1),
    )

    events: list[str] = []
    state = adapter.navigate_to_point(
        NavigationPoint(0.0, 0.0),
        on_event=lambda event: events.append(event.type),
    )

    assert state == "succeeded"
    assert fake.send_calls == 2
    assert events.count("goal_sent") == 2
    assert "connection_error" in events
    assert "feedback" in events
    assert "accepted" in events
    assert "succeeded" in events


def test_timeout_can_trigger_cancel_event() -> None:
    fake = _FakeTransport(outcomes=[NavigationOutcome("timeout", accepted=True)])
    adapter = NavigationAdapter(
        transport=fake,
        config=NavigationAdapterConfig(retry_attempts=0, cancel_on_timeout=True),
    )

    events: list[str] = []
    state = adapter.navigate_to_point(
        NavigationPoint(3.0, 4.0),
        on_event=lambda event: events.append(event.type),
    )

    assert state == "timeout"
    assert fake.cancel_calls == 1
    assert "timeout" in events
    assert "canceled" in events
