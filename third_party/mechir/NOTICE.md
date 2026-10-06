Portions of the code in this directory (`mechir/`) are derived from the MechIR
library, licensed under the Apache License, Version 2.0 (see `LICENSE` in this
directory). The upstream project's packaging metadata and README have been
omitted from this vendored copy to preserve double-blind review; full
attribution will be added upon de-anonymization.

This vendored copy includes a modification, not present in the latest PyPI
release of MechIR (`mechir==0.0.4`), that adds path-patching support
(`path_patch` / `get_path_patch` in `mechir/modelling/{dot,patched}.py`),
used by `src/circuit_analysis/experiment_patching.py` to produce Fig. 2.
