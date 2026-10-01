# Version notes

This release candidate corresponds to the manuscript v2.1 workflow. The package
version is `2.1.0-rc.1`; the display and policy identifier is `v2.1-public-release`.
These identify the software package and do not introduce a new scientific method.

The public copy has updated comments and docstrings, removed unused module-status
flags and replaced two descriptive metadata identifiers. Numerical criteria,
function signatures, revision selectors, algorithm branches and the unset default
policy are preserved. The identifiers used to select algorithm revisions remain
unchanged because they specify the evaluated method.

`SOURCE_MANIFEST.json` records both original and public-source hashes. Source
hashes differ where presentation metadata were cleaned. The local release check
compares executable syntax after excluding those declared metadata changes.

The supplied configuration keeps `candidate_audit=True`. It evaluates candidates
without replacing the returned final observation array. A candidate QC pass is
reported in the evidence; the returned array remains raw and the returned flags
remain zero. This setting is part of the preserved calling contract.

Declared dependencies: Python 3.10 or later; NumPy >=1.26 and <3.
Tested environment: Python 3.12.7 and NumPy 2.5.3. Other versions in the declared
range have not been checked in this release preparation.

The manuscript is unpublished. Its title and author list are recorded in
`CITATION.cff`, which also identifies the public GitHub repository. No journal
assignment, DOI or tagged release is claimed. The MIT licence covers this
software package. It does not grant access
to, or distribution rights for, institutional observations or event catalogues.
