"""Shared plot colors to keep previews and PyQtGraph in sync."""

PLOT_COLORS: dict[str, str] = {
    "real": "#0288D1",
    "imag": "#26C6DA",
    "freq": "#388E3C",
    "autocorr": "#7B1FA2",
    "crosscorr": "#1976D2",
    "compare": "#C2185B",
    "los": "#D32F2F",
    "echo": "#00796B",
    "text": "#E0E0E0",
}

MULTI_CHANNEL_COLORS: tuple[str, ...] = (
    "#0288D1",
    "#1976D2",
    "#388E3C",
    "#7B1FA2",
)
