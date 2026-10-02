"""Project paths and shared constants for the retained MCV analyses."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "continuous_exponential_inhibition_2026-09-12"
EVENT_CATALOGUE_CSV = PROJECT_ROOT / "data/curated/ngrip_mis6_warming_events.csv"
OBSERVATION_SEGMENTS_CSV = PROJECT_ROOT / "data/curated/observation_segments.csv"


LR04_XLSX = PROJECT_ROOT / "data/raw/lr04.xlsx"
CO2_XLSX = PROJECT_ROOT / "data/raw/composite_co2.xlsx"
PRE_TXT = PROJECT_ROOT / "data/raw/pre_1000_60_inter100.txt"
OBL_TXT = PROJECT_ROOT / "data/raw/obl_1000_60_inter100.txt"
ECC_TXT = PROJECT_ROOT / "data/raw/ecc_1000_60_inter100.txt"
INSOLATION_NC = PROJECT_ROOT / "data/raw/solstice_insolation_NH.nc"

FORCING_DIR = PROJECT_ROOT / "data/processed/forcings"
LR04_CSV = FORCING_DIR / "lr04.csv"
CO2_CSV = FORCING_DIR / "co2.csv"
ORBITAL_CSV = FORCING_DIR / "orbital.csv"
INSOLATION_65N_CSV = FORCING_DIR / "insolation_65n.csv"
PRECESSION_PHASE_CSV = FORCING_DIR / "precession_phase_anchors.csv"

BARKER_EVENT_CSVS = {
    definition: PROJECT_ROOT / f"Barker2011/data/processed/barker_events_{definition}.csv"
    for definition in ("variable_threshold", "fixed_threshold")
}

CATALOGUE_COLORS = {
    "primary": "#3E6C8E",  # NGRIP–MIS6: blue.
    "variable": "#CC6677",  # Barker varying threshold: rose.
    "fixed": "#228833",  # Barker fixed threshold: green.
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


def generated_notes_dir(output_root):
    """Keep notes outside research outputs, including in isolated review runs.

    Callers supply their catalogue output directory. Source-catalogue notes
    share the parent workspace's agent_work folder with primary-catalogue notes.
    """
    root = Path(output_root)
    scope = "primary"
    if root.name in {"NGRIP", "MIS6", "Barker2011"}:
        scope, root = root.name, root.parent
    return root / "agent_work/scratch/experiment_note" / scope
