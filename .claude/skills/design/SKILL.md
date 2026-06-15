---
name: design
description: Stage 1 of the dev workflow (design → plan → implement). Interactive main-loop skill: clarifies change scope + conda env, verifies the env, writes design.md under .work/, offers to advance to /plan.
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
   step 3, follow the shared `sys-env-check` procedure — before any `design.md`
   is written — to confirm the env exists and carries the tooling the chosen
   scope needs, offering to install anything missing. Pass the confirmed env
   name and the chosen scope. If it returns `proceeding-regardless`, note that
   choice in **Open questions**; if it returns `blocked`, stop and surface the
   gap. Do not advance otherwise.

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
