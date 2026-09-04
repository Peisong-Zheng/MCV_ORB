#!/usr/bin/env python3
"""Compatibility entry point for the audited Barker et al. (2011) analysis.

The earlier implementation duplicated the full analysis and wrote a second
set of result tables. It is kept only so old commands still work; all current
calculations and outputs come from the audited script.
"""

from Barker2011_do_predictive_information_audited import main


if __name__ == "__main__":
    main()
