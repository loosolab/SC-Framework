# Checklist for new version release

## Before merge:
- [ ] Update date in Changes.md
- [ ] Update sctoolbox/_version.py
- [ ] Update Notebook versions by running `python scripts/change_notebook_versions.py <path_to_notebooks> "<version>"`
- [ ] Check the authors: they are maintained in `CITATION.cff` (incl. ORCID and affiliation) and `pyproject.toml`; add new contributors in both. Zenodo takes the authors from `CITATION.cff`.

## After merge:
### Gitlab (pipeline takes ~1.5 hrs)
- [ ] Create new release (automated)
- [ ] Add new version tag (automated)
- [ ] Add milestone to release and close it (if one is available)
### Github (https://github.com/loosolab/SC-Framework)
The main and dev branch are automatically mirrored.
- [ ] check if the repository updated (may take a few minutes)
- [ ] Create a new release (copy from Gitlab)
### Zenodo (https://zenodo.org/records/11065517)
A Github release automatically triggers a Zenodo release.
- [ ] check if the Github release triggered Zenodo (may take a few minutes)
- [ ] check that Zenodo shows the authors from `CITATION.cff` (fix them there, not in Zenodo)
### PyPI
The final step in the CI/CD pipeline creates a new PyPI release (after ~1.5 hrs).
- [ ] check if the release was successful
