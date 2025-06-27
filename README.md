# Ettus Transceiver Project

This repository contains tools for generating, transmitting and
receiving signals with Ettus USRP devices.  The GUI application
`transceiver` is the central entry point.  All other Python scripts are
located in the `transceiver/helpers` package and serve as utilities for
converting or analysing recorded data.  Pre‑built binaries for RFNoC
streaming are kept in `bin/`.

## Structure

```
bin/                C++ helper binaries used by the GUI
signals/            directory for generated TX/RX files
transceiver/
    __main__.py     main UI (`python -m transceiver`)
    helpers/        various helper modules (rx_to_file, tx_generator, ...)
```

## Running the UI

Activate your Python environment with the requirements in
`requirements.txt` installed and simply run

```bash
python -m transceiver
```

The helper modules can also be executed individually, e.g.

```bash
python -m transceiver.helpers.rx_to_file --help
```

