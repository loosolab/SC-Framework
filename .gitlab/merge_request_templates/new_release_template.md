# Checklist for new version release

## Before merge:
- [ ] Update date in Changes.rst
- [ ] Update sctoolbox/_version.py
- [ ] Update Notebook versions by running ./scripts/change_notebook_versions.py

## After merge:
### Gitlab (automated pipeline takes ~1.5 hrs)
- [ ] Create new release
- [ ] Add new version tag
### Github (https://github.com/loosolab/SC-Framework)
The main and dev branch are automatically mirrored.
- [ ] check if the repository updated (may take a few minutes)
- [ ] Create a new release (copy from Gitlab)
### Zenodo (https://zenodo.org/records/14056105)
A Github release automatically triggers a Zenodo release.
- [ ] check if the Github release triggered Zenodo (may take a few minuts)
- [ ] adjust the authors in Zenodo (see the prior release)
