---
name: design
description: First stage of the sc-framework dev workflow (design → plan → implement). Interactive, runs in the main loop with no subagent. Reads context, asks 1-4 clarifying questions (always including conda environment and change scope), verifies the chosen conda env has the dependencies the scope needs (offering to install them per docs/source/development.rst), proposes a .work/<YYYY-MM-DD>-<slug>/ directory and waits for confirmation, writes design.md (gitignored, local-only), then offers to advance to /plan.
---

# design

First stage of the sc-framework development workflow
(design → plan → implement). Interactive, **main loop, no subagent** —
subagents can't talk to the user, and this stage is all conversation.
Produces a `design.md` that `/plan` turns into a concrete plan.

## Inputs (combinable, all optional)

1. **Inline prompt** — free text after `/design`.
2. **Topic file** — a markdown path with seed thoughts; read it.

If neither is given, ask the user what they want to design.

## Process

1. **Read context.** Any topic file, plus `CLAUDE.md` (stack, module map,
   conventions) and `CHANGES.md` (what shipped recently).
2. **Think before drafting.** Reason about the goal, which modules it
   touches, and what could go wrong, before writing anything.
3. **Ask clarifying questions.** Before any file is written, surface 1–4
   sharp questions. Always include:
   - **Conda environment** — which env should the binding test command use?
   - **Change scope** — any combination of package (`src/sctoolbox/` + `tests/`), notebooks, and docs (`docs/`)?
   Use `AskUserQuestion` when choices enumerate cleanly; free-text otherwise.
4. **Verify the conda environment.** As soon as the env name is confirmed in
   step 3, check — before any `design.md` is written — that it exists and
   carries the tooling the chosen scope needs. The dev setup (tooling and the
   exact install commands) is specified in `docs/source/development.rst`
   (Setup section); this step just applies it.
   - **Env missing.** If `conda env list` shows no such env, offer to create it
     from `sctoolbox_env.yml` (`mamba env create -f sctoolbox_env.yml` — note
     this creates an env named `sctoolbox`). If the user meant a different
     name, ask which existing env to use instead.
   - **Check dependencies by scope** (run each with `conda run -n <env> …`):
     - always (the binding command starts with `ruff check`): `ruff --version`
       and `python -c "import sctoolbox"` (package importable in editable mode).
     - package scope: also `pytest --version` and `codespell --version`.
     - docs scope: also `python -c "import sphinx"`.
   - **Something missing → offer to install it.** Don't just stop — offer to
     run the dev-setup install from `docs/source/development.rst` yourself,
     and run it only after the user confirms:
     `conda run -n <env> pip install -e '.[all]' --group test --group lint --group spellcheck`
     plus, for docs scope, `conda run -n <env> pip install --group docs`.
     Alternatively let the user point to an env that already has everything.
   - Do not advance until the env verifies, or the user explicitly chooses to
     proceed regardless (note that choice in **Open questions**).

5. **Propose slug + date, then confirm.** Short kebab-case, descriptive
   (e.g. `qc-filter-fix`, `embedding-plot`). Present the full directory
   `.work/<YYYY-MM-DD>-<slug>/` and **wait for confirmation** before creating
   anything. Use today's date from the environment.
6. **Write `.work/<YYYY-MM-DD>-<slug>/design.md`** using
   `.claude/skills/design/design-template.md` as the skeleton. Fill every
   section; do not add or remove sections. Guidance per section:
   - **Problem** — what and why now; link related prior work if relevant.
   - **Approach** — high-level idea, modules touched, what will NOT be done.
   - **Scope** — `Type` is one or more of `package`, `notebooks`, `docs`
     (combine with `+`, e.g. `package + docs`); `Conda environment` is the
     name confirmed in the questions above.
   - **Success criteria** — concrete, observable signals, numeric where possible.
   - **Open questions** — decisions the user must make before planning; empty if
     all resolved during the conversation.

7. **No commit.** `design.md` lives under `.work/`, which is gitignored
   (local-only audit trail) — there is nothing to commit at this stage.
8. **Offer to advance.** Ask the user whether to proceed to planning now.
   - If yes: invoke the `plan` skill via the Skill tool, passing
     `.work/<YYYY-MM-DD>-<slug>/design.md` as args.
   - If no: tell the user to run
     `/plan .work/<YYYY-MM-DD>-<slug>/design.md` when ready.

## Out of scope

- Writing `plan.md`, tests, or any code — those belong to `/plan` and `/implement`.
- Moving, renaming, or refactoring files; running analysis.

If the user wants any of those during design, note it in Open questions.
