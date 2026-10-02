# Sofular source data

`held2024-so-4.txt` and `held2024-so-57.txt` contain individual-stalagmite isotope
records and U–Th dating tables from [Held et al. (2024)](https://doi.org/10.1038/s41467-024-45507-5),
archived by [NOAA](https://doi.org/10.25921/b84y-cm81).

Both proxy tables give depth in mm and published model ages in kyr before
1950 CE. The comment-prefixed U–Th tables use **kyr for So-4 but years for
So-57**, including their errors; So-57 proxy ages already use kyr.

[The chronology code](../../../sofular_chronology.py) converts the dating units,
uses half the reported 2σ error as the working standard deviation, and
interpolates control offsets without crossing a hiatus or extrapolating.
This transfers analytical errors through the published curves; it does not
reconstruct StalAge or iscam posterior chronologies.
