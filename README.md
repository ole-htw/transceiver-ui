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
