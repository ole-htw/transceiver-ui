import unittest

from transceiver.measurement_mission import measurement_mission_from_dict


class TestMeasurementMission(unittest.TestCase):
    def test_valid_mission_with_mixed_orientations(self) -> None:
        mission = measurement_mission_from_dict(
            {
                "name": "scan-1",
                "repeat": 3,
                "wait_after_arrival_s": 1.25,
                "points": [
                    {
                        "id": "p1",
                        "x": 0.0,
                        "y": 0.0,
                        "yaw": 0.5,
                    },
                    {
                        "id": "p2",
                        "name": "second",
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0,
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                        "qw": 1.0,
                        "notes": "ok",
                        "measurement_profile": "dense",
                    },
                ],
            }
        )

        self.assertEqual(mission.name, "scan-1")
        self.assertEqual(len(mission.points), 2)
        self.assertEqual(mission.repeat, 3)
        self.assertAlmostEqual(mission.wait_after_arrival_s, 1.25)
        self.assertEqual(mission.points[0].z, 0.0)

    def test_rejects_duplicate_point_ids(self) -> None:
        with self.assertRaises(ValueError):
            measurement_mission_from_dict(
                {
                    "name": "scan-1",
                    "points": [
                        {"id": "p1", "x": 0.0, "y": 0.0, "yaw": 0.0},
                        {"id": "p1", "x": 1.0, "y": 1.0, "yaw": 0.0},
                    ],
                }
            )

    def test_rejects_yaw_and_quaternion_together(self) -> None:
        with self.assertRaises(ValueError):
            measurement_mission_from_dict(
                {
                    "name": "scan-1",
                    "points": [
                        {
                            "id": "p1",
                            "x": 0.0,
                            "y": 0.0,
                            "yaw": 0.0,
                            "qx": 0.0,
                            "qy": 0.0,
                            "qz": 0.0,
                            "qw": 1.0,
                        }
                    ],
                }
            )

    def test_rejects_missing_orientation(self) -> None:
        with self.assertRaises(ValueError):
            measurement_mission_from_dict(
                {
                    "name": "scan-1",
                    "points": [
                        {
                            "id": "p1",
                            "x": 0.0,
                            "y": 0.0,
                        }
                    ],
                }
            )

    def test_rejects_invalid_ranges(self) -> None:
        with self.assertRaises(ValueError):
            measurement_mission_from_dict(
                {
                    "name": "scan-1",
                    "wait_after_arrival_s": -0.1,
                    "points": [
                        {
                            "id": "p1",
                            "x": 0.0,
                            "y": 0.0,
                            "yaw": 0.0,
                        }
                    ],
                }
            )

        with self.assertRaises(ValueError):
            measurement_mission_from_dict(
                {
                    "name": "scan-1",
                    "points": [
                        {
                            "id": "p1",
                            "x": 0.0,
                            "y": 0.0,
                            "yaw": 4.0,
                        }
                    ],
                }
            )

    def test_accepts_numeric_strings_for_coordinates(self) -> None:
        mission = measurement_mission_from_dict(
            {
                "name": "scan-1",
                "points": [
                    {
                        "id": "p1",
                        "x": "1.5",
                        "y": "2,25",
                        "z": "0",
                        "yaw": "0.0",
                    }
                ],
            }
        )

        self.assertAlmostEqual(mission.points[0].x, 1.5)
        self.assertAlmostEqual(mission.points[0].y, 2.25)
        self.assertAlmostEqual(mission.points[0].z, 0.0)


if __name__ == "__main__":
    unittest.main()
