"""Label fitted phase responses separately from the orbital index itself."""


def format_phase_response_axis(axis, fontsize=7):
    """The extrema named below the ticks belong to the precession index."""
    axis.set_xlim(0, 360)
    axis.set_xticks([0, 90, 180, 270, 360],
                   ["0°\nMinimum", "90°", "180°\nMaximum", "270°", "360°\nMinimum"])
    axis.tick_params(axis="x", labelsize=fontsize)
    # Keep the longer endpoint labels inside narrow publication panels.
    axis.get_xticklabels()[0].set_ha("left")
    axis.get_xticklabels()[-1].set_ha("right")
    axis.set_xlabel("Precession-index phase", labelpad=3)
    axis.set_ylabel("Rate multiplier")


def mark_preferred_phase(axis, phase_deg, max_min_ratio, color="black"):
    """For exp(a sin(phi) + b cos(phi)), the peak multiplier is sqrt(max/min)."""
    axis.plot(phase_deg, max_min_ratio ** 0.5, "o", color=color,
              markersize=3.5, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
