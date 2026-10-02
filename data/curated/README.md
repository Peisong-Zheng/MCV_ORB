# Curated analysis inputs

These tables define the inputs used by the combined NGRIP–MIS6 analysis.

| File | Contents |
|---|---|
| [ngrip_mis6_warming_events.csv](ngrip_mis6_warming_events.csv) | NGRIP interstadial starts and the MF–Sofular warming composite, with event identities, ages and source labels |
| [observation_segments.csv](observation_segments.csv) | The two separate intervals covered by the records |
| [age_epoch_audit.csv](age_epoch_audit.csv) | Source age conventions and conversions to BP1950 |
| [orbital_driver_input_audit.csv](orbital_driver_input_audit.csv) | Orbital input sources and coordinate conventions |

Ages named `*_kyr_bp` are thousands of years before AD 1950. Observation bounds
specify record coverage. The fitted interval starts at the oldest event in
each segment and continues to its younger observation boundary; the starting
event initializes history rather than contributing to the response count.
The gap between records contributes neither exposure nor event history.

See [NGRIP](../../NGRIP/README.md) and [MIS6](../../MIS6/README.md) for source
preparation, and the [main README](../../README.md) for the shared model.
