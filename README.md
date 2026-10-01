# Selective HVDC Offset Correction

Selective correction of stable HVDC offsets in 1 Hz geomagnetic Z records.
Software v1.0.0 implements the manuscript workflow v2.1.

## Usage

Python 3.10 or later; tested with Python 3.12.7 and NumPy 2.5.3.

```text
python -m pip install -r requirements.txt
python examples/synthetic_demo.py
```

Entry: [`correct_step_d1p1`](src/analysis/funnel_2024_2025_ge0p5/step_corrector_d1p1/step_corrector_d1p1.py).
Configuration: [`workflow_v2_1.json`](configs/workflow_v2_1.json).

The demo assesses a synthetic candidate with `candidate_audit=True`; the returned
observations remain unchanged. It does not reproduce the manuscript results.
Labels denote unchanged samples (0), corrected plateaus (1) and constructed
transitions (2).

## Data and licence

Original observations and full event catalogues are subject to institutional
access policies and are not included. The [MIT licence](LICENSE) covers the
software only.

## Citation

See [CITATION.cff](CITATION.cff). The accompanying manuscript is unpublished.
