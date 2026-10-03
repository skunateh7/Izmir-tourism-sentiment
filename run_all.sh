#!/usr/bin/env bash
# Regenerates every result of the study (CPU; ~1.5 h). DistilBERT (step 5) is run separately on a GPU.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/01_audit_and_freeze.py
python scripts/04_main_experiment.py
python scripts/07_statistics.py
python scripts/06_generalisation_cv.py
python scripts/07b_cv_statistics.py
python scripts/08_ablation.py
python scripts/09_error_analysis.py
python scripts/10_tourism_and_topics.py
python scripts/12_population_reweighting.py
(cd scripts && python 11_figures_tables.py)
echo "All results regenerated in results/"
