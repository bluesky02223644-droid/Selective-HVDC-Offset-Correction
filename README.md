# Selective HVDC Offset Correction

A selective workflow for HVDC-induced stable offsets in 1 Hz geomagnetic
Z-component records. This local release candidate corresponds to manuscript
workflow v2.1 and is distributed under the [MIT licence](LICENSE).

**The synthetic demo checks API behaviour and configuration integrity. It does
not reproduce the manuscript results or write a corrected observation file.**

## Method and entry point

Paired ON/OFF responses support offset estimation and amplitude closure. The
offset is locked before modification boundaries are refined. Final QC evaluates
the resulting candidate. Labels distinguish unchanged observations (0), corrected
plateaus (1) and constructed transitions (2).

The sole entry is `correct_step_d1p1` in
`src/analysis/funnel_2024_2025_ge0p5/step_corrector_d1p1/step_corrector_d1p1.py`.
The sibling modules contain its numerical dependencies. They are not separate
publication workflows. The explicit configuration selects the manuscript
revisions; no new default policy or correction wrapper is introduced.

## Run the synthetic demo

From this folder, with Python 3.10 or later:

```text
python -m pip install -r requirements.txt
python examples/synthetic_demo.py
```

Dependencies are declared as NumPy >=1.26 and <3. This preparation was tested
with Python 3.12.7 and NumPy 2.5.3.

The example generates one artificial day in memory. It reports candidate
availability, the estimated offset and the candidate QC result. Its input and
returned final array remain identical, and its returned flags remain zero.
This is expected: the supplied configuration uses `candidate_audit=True`, which
keeps candidate assessment separate from final observation writeback.

## Version and configuration

`VERSION` gives the package version, `2.1.0-rc.1`.
`configs/workflow_v2_1.json` specifies the unchanged scientific criteria and
revision arguments under the public identifier `v2.1-public-release`.
Catalogue edge ends must be supplied for each record; the demo supplies its
synthetic edge ends. [Version notes](docs/version_notes.md) explain the preserved
calling contract and the presentation-only changes. `SOURCE_MANIFEST.json`
records original and public-source hashes.

## Data access and scope

Institutional raw observations, complete event catalogues, observatory metadata,
manual review labels, internal batch scripts and manuscript result arrays are
not included. Access to observations and catalogues is subject to the relevant
institutional policies. The MIT licence applies to the supplied software and
does not redistribute or license these excluded data.

The example is an API check rather than the original 384-injection experiment,
the 192-injection low-amplitude extension or a real-record cohort reproduction.
This folder is a publication copy and does not replace the active research
workspace. It has not yet been uploaded to GitHub.

## Citation

Software citation metadata are provided in [CITATION.cff](CITATION.cff).
The accompanying manuscript is:

Huo, Q., Wang, X., Ma, X., Guo, Y., and Zhang, S.
*Selective correction of HVDC-induced offsets in geomagnetic observations using
paired-response evidence*. Unpublished manuscript.

No publication DOI or remote repository URL has been assigned to this package.
