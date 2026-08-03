---
name: sys-changelog
description: Shared CHANGES.md + _version.py protocol — the in-progress section, the b0 beta-suffix invariant, bullet format, and the immutable dated-section rule. Append procedure used by /implement, finalise procedure used by /release. Reference doc, referenced by path, not invoked.
---

# sys-changelog

Shared protocol for the changelog (`CHANGES.md`) and the package version
(`src/sctoolbox/_version.py`). The canonical spec for the changelog *format* is
the Changelog section of `docs/source/development.rst`; the digest is in
`CLAUDE.md`. This file holds the *workflow* invariants the two stages share so
they cannot drift:

- `/implement` **appends** a work item to the active section (and creates it
  after a release) — see **Append**.
- `/release` **finalises** the active section into a dated, released version —
  see **Finalise**.

Referenced by path, not invoked.

## Version lifecycle (the invariant)

- The changelog's **dated** sections (e.g. `## 0.15.1 (15-05-2026)`) record
  **released** versions. They are immutable — never append to or edit them.
- Everything committed since the last release accumulates under a single
  **`## X.Y.Z (in progress)`** section at the top of the changelog.
- While a version is in progress, `src/sctoolbox/_version.py` carries that
  version with a non-release **beta suffix**: `__version__ = "X.Y.Zb0"`. The
  working tree therefore **never** holds a clean released version number.
- `/release` is the only stage that produces a clean number: it strips the
  `b0` suffix in `_version.py` and replaces `(in progress)` with the date.

## Locating the active section

The active section is the **topmost `## … (in progress)` section**. Always
locate it by the `(in progress)` marker — **never** by string-matching a
`## X.Y.Z` header derived from `_version.py`. The dated sections below it are
released and off-limits.

## Section structure

- Package- and docs-scope bullets go directly under the
  `## X.Y.Z (in progress)` header.
- Notebook-scope bullets go under a `### Changes to notebooks` subsection of
  that section (create the subsection if absent).
- A bullet is a short **imperative phrase** describing the user-visible change;
  append ` (#<N>)` **only** if the design or plan cites a GitLab issue.

## Append (used by /implement)

`/implement` performs this with `scripts/add_change.py` (one call per bullet); the
steps below are the spec it implements and the reference for any manual edit.

1. Locate the active section (above).
2. **An `(in progress)` section exists** — append the bullet(s) in the right
   place per *Section structure*. Confirm `_version.py` already carries that
   version with the `b0` suffix (`X.Y.Zb0`); if it still shows the last released
   version, set it now.
3. **No `(in progress)` section exists** (fresh state right after a release) —
   choose the next version `X.Y.Z` above the latest released section (bump the
   patch unless the design/plan calls for a minor/major bump), prepend a new
   `## X.Y.Z (in progress)` section before the first existing `## ` line (add a
   `### Changes to notebooks` subsection if scope includes notebooks), **and**
   set `_version.py` to `__version__ = "X.Y.Zb0"`.

## Finalise (used by /release)

`/release` performs steps 2–3 with `scripts/finalize_release.py`; the steps below
are the spec it implements.

1. The user chooses the release version `X.Y.Z` — the active section's
   in-progress version, or a deliberate bump of it.
2. Set `_version.py` to the clean `__version__ = "X.Y.Z"`; this strips the `b0`
   suffix the working tree was carrying.
3. In the active section header, replace `(in progress)` with today's date in
   `(DD-MM-YYYY)` format. If the chosen release number differs from the
   in-progress one, update the header version to match. If no `(in progress)`
   section exists, warn the user — the changelog may need a manual entry.

## Notes

- CI's `check_changes.py` (`.gitlab-ci.yml`) only verifies `CHANGES.md`
  changed, not its structure — so this protocol, not CI, is what keeps the
  format right.
