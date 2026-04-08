from __future__ import annotations

import tkinter as tk
from collections.abc import Callable


def get_main_window(widget: tk.Misc) -> tk.Misc:
    """Resolve the main/root window for any widget."""
    try:
        return widget._root()
    except Exception:
        return widget.winfo_toplevel()


def restore_main_window_focus(main_window: tk.Misc) -> None:
    """Bring the main window to front and restore focus."""
    if not main_window.winfo_exists():
        return
    main_window.deiconify()
    main_window.lift()
    main_window.after_idle(main_window.focus_force)


def close_child_window(
    child_window: tk.Misc,
    *,
    main_window: tk.Misc | None = None,
    on_close: Callable[[], None] | None = None,
) -> None:
    """Close a child window and return focus to the main window."""
    main = main_window or get_main_window(child_window)
    if on_close is not None:
        on_close()
    elif child_window.winfo_exists():
        child_window.destroy()
    restore_main_window_focus(main)


def configure_child_window(
    child_window: tk.Misc,
    *,
    parent: tk.Misc,
    modal: bool = False,
    focus: bool = True,
    on_close: Callable[[], None] | None = None,
) -> tk.Misc:
    """Attach a child window to the main window and install a close handler."""
    main_window = get_main_window(parent)
    child_window.transient(main_window)
    if modal:
        child_window.grab_set()
    if focus:
        child_window.after_idle(child_window.focus_set)
    child_window.protocol(
        "WM_DELETE_WINDOW",
        lambda: close_child_window(
            child_window,
            main_window=main_window,
            on_close=on_close,
        ),
    )
    return main_window
