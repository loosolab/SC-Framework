---
name: implement
description: Stage 3 of the dev workflow. Main-loop orchestrator: spawns the implementer (TDD, per-task commits) and code-reviewer sub-agents, runs the regression + codespell gates, updates CHANGES.md, then offers to advance to /retro.
---

# implement

Third stage of the sc-framework development workflow
(design → plan → implement → retro). Runs in the **main loop** and orchestrates
two sub-agents. Executes an approved `plan.md` under its TDD (test-driven
development) gate, reviews the result, and records the work item in
`CHANGES.md`.

## Input

A path to a `plan.md`, at `.work/<YYYY-MM-DD>-<slug>/plan.md`. If invoked
without a path, ask for one.

## Preflight

1. Read `plan.md`. Extract the **binding test command** (the line beginning
   `**Test command for this plan:**`). If absent, stop — tell the user the
   plan is missing its gate and to re-run `/plan`.
2. Confirm the command is invocable (the env may have changed since `/design`
   verified it): follow `sys-env-check` with the env name and scope from
   `plan.md` / `design.md`. If it returns `blocked`, stop and surface the gap.
3. **Resolve the commit mode.** Read `**Commit mode:**` from `plan.md` (fall
   back to `design.md` `## Scope`). `manual` → no step here commits.
   `claude (Name <email>)` → every `sys-commit` call passes that author. Missing
   or `unspecified` → **ask the user now** (manual, or claude + name/email).
   Carry the resolved mode through every `sys-commit` below.
4. **Resolve the autonomy level.** Read `**Autonomy:**` from `plan.md` (fall
   back to `design.md` `## Scope`). `per-task` → drive the loop from this skill,
   pausing after each task (Process step 1, per-task branch). `end` → spawn the
   implementer once for the whole plan (end branch). Missing or `unspecified` →
   **ask the user now**.

## Process

1. **Run the implementation, per the autonomy level resolved in preflight.**

   **`end` (autonomous run).** Spawn the `implementer` once (Agent tool,
   `subagent_type: implementer`), passing the `plan.md` path and the resolved
   commit mode. It runs `sys-implement`'s full TDD loop (one task at a time,
   per-task commit in `claude` mode, 3-attempt cap) and returns a summary.

   **`per-task` (pause after each task).** Drive the loop from this skill, one
   task at a time. For each unchecked `- [ ] T<N>` in `plan.md`, in order:
   1. **Spawn the `implementer` for that one task** (Agent tool,
      `subagent_type: implementer`) in **single-task mode** (`sys-implement`),
      passing the `plan.md` path, the task id `T<N>`, and the commit mode. It
      returns a summary **without** ticking the checkbox.
   2. **Confirm the task's own tests pass** — the tests related to that task, not
      the full binding gate. The binding gate and the full-suite regression run
      once at the end (steps 3 and 7), so re-running everything after each task
      pays for the same verification ten times over. Then show the user the change
      (`git status --short` + `git diff`) and **pause for review**. In `manual`
      mode the user commits the task; in `claude` mode it is already committed and
      the user reviews (and may amend/revert) before continuing.
   3. **On the user's go-ahead, tick `- [x] T<N>`** in `plan.md` (a local-only
      edit — never staged) and move to the next task. If the user wants changes,
      address them (re-spawning the implementer if needed) before ticking.
      If the user flags friction at this pause — or you hit a correction, a
      denial, or a rule broken — append it to `.work/<date>-<slug>/retro-notes.md`
      then and there (see the `retro` skill). Offer the note, not a whole retro;
      `/retro` reads the file later, whenever it runs.

   The pause happens after **every** task, in both commit modes. A retry-cap or
   blocker handoff from any single-task spawn is handled by step 2 below.
2. **Handle a retry-cap or blocker handoff.** If the implementer returns a
   structured failure summary, **stop and ask the user** how to proceed:
   - Continue trying (resets the counter, explicit consent) — re-spawn the
     implementer with that instruction.
   - Re-plan (`/plan`).
   - Re-design (`/design`).
   Do not invent tasks or force past the cap yourself.
3. **Confirm the binding gate.** After all tasks are done, run the binding test
   command once yourself and confirm exit 0. If not, hand the still-failing task
   back to the implementer.

   **Render-check changed `.. exec_code::` examples.** Only if `git diff` shows
   an added/modified `.. exec_code::` docstring example (skip otherwise — the
   binding command does not execute these). When it does, follow the single-page
   `dummy`-build **Render-check procedure** in `.claude/docs/sys-examples.md`; a
   non-zero exit means an example raised — hand it back to the implementer.
4. **Spawn the `code-reviewer`.** Use the Agent tool with
   `subagent_type: code-reviewer`, passing `plan.md` and the changed files /
   work-item directory. It follows `sys-code-review` and **always runs**.
5. **Write the review** verbatim to `.work/<date>-<slug>/review-code.md`.
6. **Address findings.** Fix every **blocker**, re-running the binding test
   command after each fix. If the reviewer flags a structural mismatch you
   cannot resolve without user input, stop and hand off.

   **Spellcheck (codespell).** CI runs `codespell --toml pyproject.toml`
   (`.gitlab-ci.yml` `spellcheck` job). Run it here **read-only** (never `-w`)
   over only this work item's changed files (`git diff --name-only`, not the
   whole repo):
   `conda run <env-spec> codespell --toml pyproject.toml <changed files>`.
   Resolve each report **by hand**:
   - **Clear typo** (one obvious correction, in a comment/docstring/string, not
     behaviour-changing code) → fix with `Edit`.
   - **Ambiguous** (multiple candidates) or a **domain term** codespell doesn't
     know → **stop and ask the user**: apply a specific fix, or add the term to
     `uri-ignore-words-list` / `ignore-words-list` in
     `pyproject.toml [tool.codespell]`. Do NOT guess.

   After any fix, re-run codespell to confirm clean, and re-run the binding test
   command (files changed).

   If addressing findings or the spellcheck changed any code, invoke
   `sys-commit` (step: review, slug: `<slug>`, commit mode, intended files: the
   fixes + any spellcheck typo fixes — **never** `review-code.md`, which lives
   under the gitignored `.work/`) → message `review: <slug>`. If no code changed
   (no blockers, no typo fixes), there is nothing to commit at this step —
   `review-code.md` stays local-only.
7. **Run the full-suite regression — the end-of-workflow gate.** Now that all
   code is final (tasks *and* any review/spellcheck fixes from step 6), run the
   plan's `**Full-suite regression command:**` **once**, **verbatim** as the plan
   declares it — do not reconstruct it from the binding command. This is the
   workflow's only full-suite run: it catches cross-file regressions the per-file
   binding command cannot see, including any a review fix just introduced. It
   MUST exit 0; if it fails, a change regressed another module — hand the failure
   back to the implementer, then re-run this step. (Notebook-only and docs-only
   scope declare it `n/a` — skip.)
8. **Update `CHANGES.md` and `_version.py`.** Run `scripts/add_change.py` once per
   user-visible bullet — it applies the **Append** procedure from
   `.claude/docs/sys-changelog.md` deterministically:
   `python3 scripts/add_change.py "<imperative phrase>" --scope package|docs|notebook [--issue <N>]`.
   Choose `--scope` by where the change lands (`notebook` → the
   `### Changes to notebooks` subsection); pass `--issue <N>` from the plan's
   `**Related:**` line when it cites an issue. Use `--dry-run` first if unsure.
   (Format digest: `CLAUDE.md`; canonical spec: `development.rst`, Changelog
   section.)

   Then invoke `sys-commit` (step: changes, slug: `<slug>`, commit mode, intended
   files: `CHANGES.md`, plus `src/sctoolbox/_version.py` if you changed it) →
   message `changes: <slug>`.
9. **Done.** Report what shipped and the final test result. In `manual` commit
   mode, also list every changed file (`git status --short`) and remind the user
   that nothing was staged or committed — they stage and commit it themselves.
10. **Offer to advance to `/retro`.** Ask the user whether to run the
    retrospective now, while the session still holds the evidence it needs (the
    reviewer findings, the deviations, and the friction from this conversation).
    - If yes: invoke the `retro` skill via the Skill tool, passing
      `.work/<date>-<slug>/plan.md` as args.
    - If no: tell the user to run `/retro .work/<date>-<slug>/` when ready, and
      note that its session-friction input (Process step 2) is weaker once this
      conversation is gone.

## Writing sub-agent briefs

The agent reads `plan.md` itself. A brief that re-explains the task doubles its
cost for nothing.

- **Name the task and stop.** "Execute only T6 of `<plan path>`, single-task
  mode." Do not restate the task's content, its rationale, or its test cases.
- **Include only what is not in the plan:** the env spec, the commit mode, the
  binding command verbatim, corrections agreed after the plan was written, and
  anything the *previous* task changed that this one must respect.
- **Point, don't quote.** "Read T6 in full — it is long and every bullet is
  load-bearing" beats reproducing the bullets.
- **Downgrade the model for mechanical tasks.** The Agent tool takes a `model`
  override. Docstring passes, changelog edits and other low-judgement tasks do not
  need the same model as a risky refactor.
- **Summarise a returned review in a few lines**, not in full: the verbatim text
  is already written to `.work/`, so quoting it back costs twice.

## Scope changes arriving mid-implementation

If the user adds scope after `/plan` (a new requirement, not a correction):

1. **Amend `design.md` first, then `plan.md`, in the same pass** — a new Approach
   item, the success criteria it changes, then the task and its test cases.
   Amending one and then the other doubles the review rounds.
2. Insert the task in dependency order and renumber; **grep for stale
   cross-references** afterwards (`T<n>`, "guard for T…", "re-checked in T…") —
   they are the literal instructions executed at per-task boundaries.
3. Re-run the `plan-reviewer` on the amendment before the new task runs.
4. Never carry an environment claim forward without re-checking it — a recorded
   "this failed last time" (a `PermissionError`, a missing tool) may already be
   fixed.

## Constraints

- **Reviewers are read-only and return text** — this skill writes
  `review-code.md`, not the agent.
- **No new tasks beyond the plan.** If the plan is short a task, route back
  to `/plan` — or, for scope the user adds mid-flight, follow the section above.
- **Commit only intended files** via `sys-commit`; never blind `git add -A`,
  never a git op denied by `.claude/settings.json`.
- **Honour the commit mode.** In `manual` mode, no step here commits or stages —
  leave the working tree for the user. In `claude` mode, pass the author through
  every `sys-commit` call.
