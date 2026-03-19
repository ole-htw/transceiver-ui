# Ettus Transceiver Project

This repository contains tools for generating, transmitting and
receiving signals with Ettus USRP devices. The GUI application
`transceiver` is the central entry point. All other Python scripts are
located in the `transceiver/helpers` package and serve as utilities for
converting or analysing recorded data. Pre‑built binaries for RFNoC
streaming are kept in `bin/`.

## Installation

Clone the repository and install the Python dependencies:

```bash
pip install -r requirements.txt
```

The binaries in `bin/` are pre‑compiled for RFNoC streaming. Rebuild them
only if you need different functionality or platform support.

## UHD Python bindings (TX controller)

The transmit path now uses the UHD Python bindings in-process. Ensure that
`uhd` is importable in your Python environment (e.g. by installing the UHD
Python package that matches your driver version). The controller uses the
TX streamer API and sends an end-of-burst marker on stop to avoid dirty
shutdowns.

## Structure

```
bin/                C++ helper binaries used by the GUI
signals/            directory for generated TX/RX files
transceiver/
    __main__.py     main UI (`python -m transceiver`)
    helpers/        various helper modules (rx_to_file, tx_generator, ...)
```


## Measurement mission format

A measurement mission is represented as a `MeasurementMission` object with these
fields:

- `name` (required): mission name
- `points` (required): non-empty list of mission points
- `repeat` (optional): integer `>= 1`
- `wait_after_arrival_s` (optional): float `>= 0`, default `0.0`

Each point requires:

- `id` **or** `name` (at least one required)
- `x`, `y` (required finite numbers)
- `z` (optional finite number, default `0.0`)
- orientation as exactly one of:
  - `yaw` in radians in range `[-pi, pi]`, or
  - quaternion `qx`, `qy`, `qz`, `qw` (all required, normalized)
- optional metadata fields:
  - `notes` (string)
  - `measurement_profile` (string)

Strict validation rejects:

- missing required fields
- invalid numeric ranges
- points that define both `yaw` and quaternion (or neither)
- duplicate point IDs (`id`)

Use `examples/measurement-mission.yaml` as a reference mission definition.
Programmatic parsing/validation lives in `transceiver/measurement_mission.py`
(`measurement_mission_from_dict` and `load_measurement_mission`).

## Map & Live Position

Zusätzlich zur Missionsdefinition kann eine Mission ein optionales
`map_config`-Objekt enthalten, um Karte und Live-Position in der UI zu
visualisieren.

Schema von `map_config`:

- `image` (required): Pfad zur Kartenbild-Datei (z. B. `.pgm`)
- `resolution` (required): Kartenauflösung in Metern pro Pixel
- `origin` (required): Ursprung der Karte als `[x, y, yaw]` im Karten-Frame

Eine vollständige Dummy-Beispieldatei findest du unter
`examples/measurement-mission-with-map.yaml`.

Um von der Dummy-Konfiguration auf die reale Karte zu wechseln, ersetze in
dieser Datei den Wert von `map_config.image` (aktuell
`maps/DUMMY_MAP_PLACEHOLDER.pgm`) durch den Pfad zu deiner tatsächlichen
Map-Datei.

Wichtig: Für die Positionsdarstellung wird der Koordinatenbezug
`frame_id: map` erwartet.

## Running the UI

Activate your Python environment with the dependencies installed and run

```bash
python -m transceiver
```

The helper modules can also be executed individually, for example:

```bash
python -m transceiver.helpers.rx_to_file --help
```

The GUI shows basic signal statistics (minimum/maximum frequency, maximum
amplitude and 3\ dB bandwidth) for both the generated and the received
signals.

In the TX generate column, the optional shaping uses
**frequency-domain zeroing** with a **hard edge / harte Kante** and the input
field **Bandwidth [Hz]**.

The receive view still offers optional oversampling after capture. This RX
option is unchanged and can improve channel-impulse-response accuracy when
using the built-in cross-correlation tools.

## Mission workflow runtime configuration

The mission workflow can be configured centrally through environment variables:

Create a local `.env` file (for example by copying `.env.example`) to keep
these values persistent across terminal sessions.

- `TRANSCEIVER_ROBOT_HOST` (default `ole@192.168.10.10`)
- `TRANSCEIVER_ROS2_NAMESPACE` (default empty)
- `TRANSCEIVER_ROS2_ACTION_NAME` (default `/navigate_to_pose`)
- `TRANSCEIVER_NAV_GOAL_ACCEPT_TIMEOUT_S` (default `8.0`)
- `TRANSCEIVER_NAV_GOAL_REACHED_TIMEOUT_S` (default `120.0`)
- `TRANSCEIVER_NAV_RETRY_ATTEMPTS` (default `0`)

SSH based ROS2 execution is configured for non-interactive operation (`BatchMode`,
no password prompts, host key auto-accept for new hosts).


### Fast-DDS runtime policy for Nav2 actions

For stable `/navigate_to_pose` actions across environments, set a shared Fast-DDS XML profile
on the robot and export it for all ROS 2 processes (Nav2 server/client and CLI tools).

1. Check the active configuration on the remote host:

```bash
ssh <robot-host> 'bash -lc "echo FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE; echo FASTDDS_DEFAULT_PROFILES_FILE=$FASTDDS_DEFAULT_PROFILES_FILE"'
```

If both variables are empty, Fast-DDS default QoS/memory policy is used.

2. Configure this project to force the profile when the mission workflow dispatches goals:

- `TRANSCEIVER_FASTDDS_PROFILES_FILE` (absolute path to the XML on the remote host)

The app exports **both** `FASTDDS_DEFAULT_PROFILES_FILE` and
`FASTRTPS_DEFAULT_PROFILES_FILE` before invoking `ros2 action send_goal`, so behavior is
consistent on systems using either variable name.

3. In the XML profile, ensure DataReaders used by Nav2 action topics (e.g.
`/navigate_to_pose/_action/*`) have sufficient history/payload allocation (history memory
policy and payload-related settings) for payloads above 32 bytes.

4. Verify end-to-end with:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose '{"pose":{"header":{"frame_id":"map"},"pose":{"position":{"x":0.0,"y":0.0,"z":0.0},"orientation":{"x":0.0,"y":0.0,"z":0.0,"w":1.0}}}}' --feedback
```

The command should stream feedback/result without `RTPS_READER_HISTORY` payload errors.
