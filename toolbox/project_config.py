"""Project paths and shared constants for the retained MCV analyses."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BIN_WIDTH_KA = 0.2

LR04_XLSX = PROJECT_ROOT / "data/raw/lr04.xlsx"
CO2_XLSX = PROJECT_ROOT / "data/raw/composite_co2.xlsx"
PRE_TXT = PROJECT_ROOT / "data/raw/pre_1000_60_inter100.txt"
OBL_TXT = PROJECT_ROOT / "data/raw/obl_1000_60_inter100.txt"

ORBITAL_DRIVER_SETTINGS = {
    "pre": {
        "label": "Precession index",
        "path": PRE_TXT,
        "source": str(PRE_TXT.relative_to(PROJECT_ROOT)),
        "color": "#0072B2",
    },
    "obl": {
        "label": "Obliquity",
        "path": OBL_TXT,
        "source": str(OBL_TXT.relative_to(PROJECT_ROOT)),
        "color": "#009E73",
    },
}
