---
name: sys-env-check
description: Shared conda-env verification: probes per-scope tooling and offers to install what's missing. Invoked by /design and /implement.
---

# sys-env-check

The shared conda-environment verification procedure for the agentic
workflow. Centralises the existence check, the per-scope dependency probes,
and the dev-setup install command so they live in **one** place. The
canonical dev setup (tooling and the exact install commands) is specified in
`docs/source/development.rst` (Setup section); this procedure just applies
it. Invoked by `/design` (before any `design.md` is written) and by
`/implement` (preflight, in case the env changed since `/design`).

## Inputs (from the caller)

- The **conda environment** name to verify.
- The **scope** — any combination of `package` / `notebooks` / `docs`.

## Procedure

1. **Environment exists?** Run `conda env list`.
   - **Missing.** Offer to create it from `sctoolbox_env.yml`
     (`conda env create -f sctoolbox_env.yml` — note this creates an env named
     `sctoolbox`). If the user meant a different name, ask which existing env to
     use instead. Create only after the user confirms.
2. **Probe dependencies by scope** (run each with `conda run <env-spec> …`):
   - **always** (the binding command starts with `ruff check`): `ruff --version`
     and `python -c "import sctoolbox"` (package importable in editable mode).
   - **package scope** adds: `pytest --version` and `codespell --version`.
   - **notebooks scope** adds: `jupyter --version` (the nbconvert gate needs it).
   - **docs scope** adds: `python -c "import sphinx"`. The docs build also needs
     a system `pandoc` on `PATH` (see `development.rst`); flag it if absent.
3. **Something missing → offer to install it.** Don't just stop — offer to run
   the dev-setup install from `docs/source/development.rst` yourself, and run it
   only after the user confirms:
   `conda run <env-spec> pip install -e '.[all]' --group dev` (the `dev` group
   bundles the test + lint + spellcheck + docs tooling plus dev-only extras such
   as the GraphQL client used by `scripts/gitlab_query.py`).
   Alternatively let the user point to an env that already has everything.
4. **Outcome.** Return one of:
   - **verified** — env exists and every probe for the scope passed.
   - **installed-then-verified** — something was missing, the user confirmed the
     install, and the re-probe now passes.
   - **proceeding-regardless** — the user explicitly chose to continue with a
     gap. The caller records this (`/design` notes it in **Open questions**).
   - **blocked** — env absent and the user does not want it created, or a needed
     install was declined. The caller stops and surfaces the gap.

Do not advance the calling stage until the result is `verified`,
`installed-then-verified`, or an explicit `proceeding-regardless`.

## Hard constraints

- **Run installs only after explicit user confirmation.** Probing
  (`--version`, `import`) is read-only and needs no confirmation; `pip install`
  and `conda env create` mutate the environment and must be confirmed first.
- **Never run a denied git op** (see `.claude/settings.json`).
