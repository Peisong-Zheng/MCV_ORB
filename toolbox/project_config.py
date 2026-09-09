"""Project paths and shared constants for the retained MCV analyses."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BIN_WIDTH_KA = 0.2

LR04_XLSX = PROJECT_ROOT / "data/raw/lr04.xlsx"
CO2_XLSX = PROJECT_ROOT / "data/raw/composite_co2.xlsx"
PRE_TXT = PROJECT_ROOT / "data/raw/pre_1000_60_inter100.txt"
OBL_TXT = PROJECT_ROOT / "data/raw/obl_1000_60_inter100.txt"

# Calendar reference years, not age-model synchronization corrections.
AGE_EPOCH = "BP1950"
BP1950_REFERENCE_YEAR = 1950.0
B2K_REFERENCE_YEAR = 2000.0
B2K_TO_BP1950_KA = (BP1950_REFERENCE_YEAR - B2K_REFERENCE_YEAR) / 1000.0

ORBITAL_SOLUTION = "La2004"
ORBITAL_SOURCE_EPOCH = "J2000.0"
ORBITAL_AGE_OFFSET_TO_BP1950_KA = B2K_TO_BP1950_KA
ORBITAL_EPOCH_SOURCE_URL = (
    "https://ssp.imcce.fr/insola/earth/online/earth/La2004/README.TXT"
)
# Official README: time from J2000 in 1000 years. On 2026-09-05, both
# local drivers matched the official nominal solution at all 1001 shared
# integer-kyr points from -1000 to 0 (within six-decimal rounding).
# See data/curated/age_epoch_audit.csv and docs/age_epoch_audit.md.
LR04_EPOCH_SOURCE_URL = "https://www.ncei.noaa.gov/access/paleo-search/study/5847"
CO2_EPOCH_SOURCE_URL = (
    "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/"
    "antarctica2015co2composite.txt"
)
ORBITAL_REFERENCE = "Laskar et al. (2004), doi:10.1051/0004-6361:20041335"

ORBITAL_DRIVER_SETTINGS = {
    "pre": {
        "label": "Precession index",
        "path": PRE_TXT,
        "source": str(PRE_TXT.relative_to(PROJECT_ROOT)),
        "age_offset_ka": ORBITAL_AGE_OFFSET_TO_BP1950_KA,
        "color": "#0072B2",
    },
    "obl": {
        "label": "Obliquity",
        "path": OBL_TXT,
        "source": str(OBL_TXT.relative_to(PROJECT_ROOT)),
        "age_offset_ka": ORBITAL_AGE_OFFSET_TO_BP1950_KA,
        "color": "#009E73",
    },
}
