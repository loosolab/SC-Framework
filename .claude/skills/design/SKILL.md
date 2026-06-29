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
3. **Ask clarifying questions.** Before any file is written, resolve the five
   items below. `AskUserQuestion` caps at four questions per call, so batch the
   four enumerable choices — **conda environment**, **change scope**,
   **commit mode**, **autonomy** — into one `AskUserQuestion` call (each option's
   "Other" handles a prefix env, a custom name/email, etc.), and handle
   **related issues/MRs** as a follow-up free-text prompt (it is open-ended, not
   a clean enumeration). Always resolve all five:
   - **Conda environment** — which env runs the gate commands (via
     `conda run <env-spec>`, `-n <name>` or `-p <prefix>`)? Record the exact
     form in `design.md` so the binding command matches an allow rule. Recommend
     `sctoolbox` (allow-listed in `settings.json`); a prefix or differently-named
     env needs its `settings.local.json` mirror — regenerate it with
     `python3 scripts/sync_local_allowlist.py` (don't hand-edit).
   - **Change scope** — any combination of package (`src/sctoolbox/` + `tests/`), notebooks, and docs (`docs/`)?
   - **Commit mode** — who commits? **manual** (user stages + commits;
     `/implement` never touches git) or **claude** (Claude commits per the
     workflow convention — ask for **name and email**, record as
     `claude (Name <email>)`, which sets author + committer per-commit with no
     `git config`). Default **manual** if unsure (local setup can have issues
     with auto-commits).
   - **Autonomy** — how much should `/implement` run unattended? **per-task**
     (`/implement` pauses after each task for review/commit) or **end** (all
     tasks run, then one review). **No default — always ask.**
   - **Related issues/MRs** — `#`/`!` numbers, a request to **scan**, or none.
     (The lookup runs in step 6 via `scripts/gitlab_query.py`, once the env is
     verified.)
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
6. **Pull related issues/MRs (if any), then write `design.md`.**

   First, if the user gave `#`/`!` numbers or asked to scan in step 3, use the
   read-only `scripts/gitlab_query.py` in the verified env: run
   `conda run <env-spec> python scripts/gitlab_query.py search "<keywords>"` to
   discover candidates (present them, confirm with the user), then
   `… gitlab_query.py fetch issue|mr <n>` for each confirmed item. Save the
   combined markdown to `.work/<YYYY-MM-DD>-<slug>/related.md`. Skip entirely if
   the user said none.

   Then write `.work/<YYYY-MM-DD>-<slug>/design.md` using
   `.claude/skills/design/design-template.md` as the skeleton. Fill every
   section; do not add or remove sections. Guidance per section:
   - **Problem** — what and why now; draw on `related.md` (the issue/MR
     requirements and discussion) where relevant.
   - **Approach** — high-level idea, modules touched, what will NOT be done.
   - **Scope** — fill `Type` (package/notebooks/docs, combine with `+`),
     `Conda environment`, `Commit mode`, `Autonomy`, and `Related` from the
     answers above (the template explains each field).
   - **Success criteria** — concrete, observable signals, numeric where possible
     (derive from the issue/MR acceptance criteria in `related.md` where present).
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
