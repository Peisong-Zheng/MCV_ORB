"""Shared panel lettering for the current manuscript and its source figures."""

PANEL_FONT_SIZE = 9
PANEL_FONT_FAMILY = "DejaVu Sans"


def add_panel_label(axis, label, x=-0.12, y=1.04):
    """Keep letters separate from titles; positions can follow the panel layout."""
    return axis.text(
        x, y, f"({label})", transform=axis.transAxes,
        fontsize=PANEL_FONT_SIZE, fontfamily=PANEL_FONT_FAMILY,
        fontweight="bold", color="black", ha="left", va="bottom",
        clip_on=False, zorder=10,
    )
