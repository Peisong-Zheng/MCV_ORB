"""Project paths and shared constants for the retained MCV analyses."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


LR04_XLSX = PROJECT_ROOT / "data/raw/lr04.xlsx"
CO2_XLSX = PROJECT_ROOT / "data/raw/composite_co2.xlsx"
PRE_TXT = PROJECT_ROOT / "data/raw/pre_1000_60_inter100.txt"
OBL_TXT = PROJECT_ROOT / "data/raw/obl_1000_60_inter100.txt"

BARKER_EVENT_CSVS = {
    definition: PROJECT_ROOT / f"Barker2011/data/processed/barker_events_{definition}.csv"
    for definition in ("variable_threshold", "fixed_threshold")
}

# Calendar reference years
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
