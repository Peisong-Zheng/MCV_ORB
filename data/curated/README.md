# Curated analysis inputs

These tables are fixed, reader-facing inputs. Analysis scripts may read them
but must not recreate or overwrite them.

- `ngrip_mis6_warming_events.csv` is the paper catalogue: 34 NGRIP GI starts
  from Rasmussen et al. (2014), followed by 21 MIS 6 speleothem transitions.
  The MIS 6 sequence contains MF 6.1--6.16 from Fohlmeister et al. (2023) and
  Sofular 6.9--6.13 from Held et al. (2024), renumbered MIS 6.17--6.21 after
  concatenation. Each row retains its source record, published label, timing
  method, and source table.
- `observation_segments.csv` defines the two disjoint intervals observed by
  the pooled analysis. NGRIP spans 12--123 kyr BP, based on the approximately
  123-kyr continuous, undisturbed climate-record coverage reported by North
  Greenland Ice Core Project Members (2004); its main 1.5-kyr-history response
  interval is 12--121.5 kyr BP. MIS 6 spans 132.5--204.5 kyr BP, with main
  response exposure to 203 kyr BP. The 123--132.5 kyr BP gap is not event-free
  exposure.
- `age_epoch_audit.csv` records source age conventions and conversions.

The MIS 6 label anchors, MF--Sofular correspondences and legacy U--Th display
subset now live under `MIS6/data/curated/`. The source data, source analyses and
their outputs moved together on 2026-09-08; the pooled catalogue stays here.

All `*_kyr_bp` ages are thousands of years before AD 1950. Label anchors are
annotation positions, not newly defined event ages.

The NGRIP source workbook now lives under `NGRIP/data/raw/`. Its derived
`ngrip_warming_cooling_starts.csv` lives under `NGRIP/data/processed/` and is
rebuilt by `NGRIP/ngrip_data_preparation.ipynb`. The previous copies in root
`data/raw/` and `data/curated/` were removed on 2026-09-07. This relocation does
not change the fixed pooled `ngrip_mis6_warming_events.csv` catalogue.

The 204.5 kyr MIS 6 endpoint lies within Held et al.'s (2024) approximately
205-kyr continuous Sofular record and source-series coverage; the source series
reaches 204.784 kyr BP, and their Figure 3b has no additional published label
beyond 6.13 in the older direction. This boundary supplies history support and
does not add an unpublished event.

North Greenland Ice Core Project Members (2004), *Nature* 431, 147--151,
https://doi.org/10.1038/nature02805.
