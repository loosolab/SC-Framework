---
name: implement
description: Stage 3 of the dev workflow. Main-loop orchestrator: spawns the implementer (TDD, per-task commits) and code-reviewer sub-agents, runs the regression + codespell gates, updates CHANGES.md.
---

# implement

Third and final stage of the sc-framework development workflow
(design → plan → implement). Runs in the **main loop** and orchestrates two
sub-agents. Executes an approved `plan.md` under its TDD (test-driven
development) gate, reviews the result, and records the work item in
`CHANGES.md`.

## Input

A path to a `plan.md`, at `.work/<YYYY-MM-DD>-<slug>/plan.md`. If invoked
without a path, ask for one.

## Preflight

1. Read `plan.md`. Extract the **binding test command** (the line beginning
   `**Test command for this plan:**`). If absent, stop — tell the user the
   plan is missing its gate and to re-run `/plan`.
2. Confirm the command is invocable. `/design` already verifies this up front,
   but re-check here in case the env changed since: follow the shared
   `sys-env-check` procedure, passing the env name and scope from `plan.md` /
   `design.md`. It probes the per-scope tooling and offers to install anything
   missing (running it only after you confirm). If it returns `blocked` (env
   absent and not wanted, or a needed install declined), stop and surface the
   gap.
3. **Resolve the commit mode.** Read the `**Commit mode:**` line from `plan.md`
   (falling back to `design.md` `## Scope`). If it is `manual`, no step in this
   skill commits — the user stages and commits everything themselves. If it is
   `claude (Name <email>)`, every `sys-commit` call passes that author. If it is
   missing or `unspecified` (e.g. a plan predating this field, or `/implement`
   run without `/design`), **ask the user now**: manual, or claude — and if
   claude, their name and email. Carry the resolved mode through every
   `sys-commit` call below.

## Process

1. **Spawn the `implementer`.** Use the Agent tool with
   `subagent_type: implementer`, passing the `plan.md` path and the resolved
   commit mode. It follows `sys-implement`: writes tests first (red), implements
   tasks one at a time under the binding command, marks `- [x] T<N>` only on
   green with no regression, and — in `claude` commit mode — **commits per
   completed task** via `sys-commit` (`impl(<slug>): T<N> <desc>`); in `manual`
   mode it leaves changes unstaged. It retries up to 3 attempts per task.
2. **Handle a retry-cap or blocker handoff.** If the implementer returns a
   structured failure summary, **stop and ask the user** how to proceed:
   - Continue trying (resets the counter, explicit consent) — re-spawn the
     implementer with that instruction.
   - Re-plan (`/plan`).
   - Re-design (`/design`).
   Do not invent tasks or force past the cap yourself.
3. **Confirm the gate.** When the implementer reports done, run the binding
   test command once yourself and confirm exit 0. If not, hand the
   still-failing task back to the implementer.

   **Then run the plan's `**Full-suite regression command:**` once** to catch
   cross-file regressions the per-file binding command cannot see. Run it
   **verbatim** as the plan declares it — do not reconstruct it from the binding
   command. It MUST exit 0; if it fails, a change regressed another module —
   hand the failure back to the implementer before proceeding. (Notebook-only
   and docs-only scope declare it `n/a` — skip this step.)

   **Render-check changed `.. exec_code::` examples.** If `git diff` shows an
   added/modified doc-string `.. exec_code::` example, the binding command did
   not execute it. Build that module's API page with the **`dummy`** builder
   (executes `.. exec_code::`, skips HTML finalisation — a single-page `-b html`
   fails on nbsphinx's notebook collection, so do **not** use it):
   `conda run -n <env> sphinx-build -b dummy docs/source /tmp/scdocs docs/source/API/<m>.rst`
   (`<m>` = `tools`, `utils`, …). A **non-zero exit** means an example raised (a
   traceback, a `NameError` from a missing pre-code variable) — hand it back to
   the implementer to fix the example or extend `utils_pre_code.py`. The
   `toctree`/cross-reference warnings about the unbuilt rest of the docs are
   expected and do **not** fail it. Per `.claude/skills/sys-examples/SKILL.md`,
   `.. plot::` examples are only parsed here, not executed (CI's full
   `make -C docs html` verifies those); skip this step entirely when no
   `.. exec_code::` example changed.
4. **Spawn the `code-reviewer`.** Use the Agent tool with
   `subagent_type: code-reviewer`, passing `plan.md` and the changed files /
   work-item directory. It follows `sys-code-review` and **always runs**.
5. **Write the review** verbatim to `.work/<date>-<slug>/review-code.md`.
6. **Address findings.** Fix every **blocker**, re-running the binding test
   command after each fix. If the reviewer flags a structural mismatch you
   cannot resolve without user input, stop and hand off.

   **Spellcheck (codespell).** CI runs `codespell --toml pyproject.toml`
   (`.gitlab-ci.yml` `spellcheck` job). Run it here **read-only** (detection
   only — never `-w`) over the files this work item changed (the same set
   `git diff --name-only` reports for the work — never the whole repo, to honour
   minimal-diff):
   `conda run -n <env> codespell --toml pyproject.toml <changed files>`.
   Resolve each thing it reports **by hand** — codespell never edits files here:
   - **Clear typo** (a single obvious correction, in a comment/docstring/string,
     not in code that would change behaviour) → fix it directly with `Edit`.
   - **Ambiguous** (codespell lists multiple candidate fixes) or a **domain term**
     codespell doesn't know → **stop and ask the user** how to resolve each:
     apply a specific fix, or add the term to `uri-ignore-words-list` /
     `ignore-words-list` in `pyproject.toml [tool.codespell]`. Do NOT guess a
     correction.

   After any fix, re-run `codespell --toml pyproject.toml <changed files>` to
   confirm it is clean, and re-run the binding test command since files changed.

   If addressing findings or the spellcheck changed any code, invoke
   `sys-commit` (step: review, slug: `<slug>`, commit mode, intended files: the
   fixes + any spellcheck typo fixes — **never** `review-code.md`, which lives
   under the gitignored `.work/`) → message `review: <slug>`. In `manual`
   commit mode, skip the commit and leave the fixes unstaged. If no code changed
   (no blockers, no typo fixes), there is nothing to commit at this step —
   `review-code.md` stays local-only.
7. **Update `CHANGES.md` and `_version.py`.** Use the format in `CLAUDE.md`
   (canonical spec: `development.rst`, Changelog section); CI's
   `check_changes.py` only verifies the file changed, not its structure.

   The changelog's dated sections record **released** versions. Everything
   committed since the last release accumulates under a single
   `## X.Y.Z (in progress)` section, and `src/sctoolbox/_version.py` mirrors
   that in-progress version with a non-release **beta suffix**
   (`__version__ = "X.Y.Zb0"`) so the working tree never carries a clean
   released version number. `/release` later strips the suffix and dates the
   section.
   a. Locate the active section: the **topmost `## … (in progress)` section**.
      The dated sections below it (e.g. `## 0.15.1 (15-05-2026)`) are released
      and immutable — **never append to them.** Do **not** derive the target
      from `_version.py` by string-matching a `## X.Y.Z` header; use the
      `(in progress)` marker.
   b. Apply by case:
      - **An `(in progress)` section exists** — append bullet(s): package/docs
        changes under its main header, notebook changes under its
        `### Changes to notebooks` subsection (create the subsection if absent).
        Confirm `_version.py` already carries that version with the `b0` suffix
        (`X.Y.Zb0`); if it still shows the last released version, set it now.
      - **No `(in progress)` section exists** (fresh state right after a
        release) — choose the next version `X.Y.Z` above the latest released
        section (bump the patch unless the design or plan calls for a
        minor/major bump), prepend a new `## X.Y.Z (in progress)` section before
        the first existing `## ` line (add a `### Changes to notebooks`
        subsection if scope includes notebooks), **and** set
        `src/sctoolbox/_version.py` to `__version__ = "X.Y.Zb0"`.
   c. Bullet = short imperative phrase for the user-visible change; append
      ` (#<N>)` only if the design or plan cites a GitLab issue.
   d. Invoke `sys-commit` (step: changes, slug: `<slug>`, commit mode, intended
      files: `CHANGES.md`, plus `src/sctoolbox/_version.py` if you changed it) →
      message `changes: <slug>`. In `manual` commit mode, skip the commit and
      leave the files unstaged.
8. **Done.** Report what shipped and the final test result. In `manual` commit
   mode, also list every changed file (`git status --short`) and remind the user
   that nothing was staged or committed — they stage and commit it themselves.

## Constraints

- **Reviewers are read-only and return text** — this skill writes
  `review-code.md`, not the agent.
- **No new tasks beyond the plan.** If the plan is short a task, route back
  to `/plan`.
- **Commit only intended files** via `sys-commit`; never blind `git add -A`,
  never a git op denied by `.claude/settings.json`.
- **Honour the commit mode.** In `manual` mode, no step here commits or stages —
  leave the working tree for the user. In `claude` mode, pass the author through
  every `sys-commit` call.
