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

## Process

1. **Spawn the `implementer`.** Use the Agent tool with
   `subagent_type: implementer`, passing the `plan.md` path. It follows
   `sys-implement`: writes tests first (red), implements tasks one at a time
   under the binding command, marks `- [x] T<N>` only on green with no
   regression, and **commits per completed task** via `sys-commit`
   (`impl(<slug>): T<N> <desc>`). It retries up to 3 attempts per task.
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

   **Then run the full test suite once** to catch cross-file regressions the
   per-file binding command cannot see. When the binding command contains
   `pytest` (any scope that includes package), run the whole suite with the same conda
   env prefix the binding command uses — i.e. replace its
   `python -m pytest tests/<target>.py -v` segment with
   `conda run -n <env> python -m pytest tests`. It MUST exit 0; if it fails, a
   change regressed another module — hand the failure back to the implementer
   before proceeding. (Notebook-only and docs-only scope have no pytest
   segment; skip this.)

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
   (`.gitlab-ci.yml` `spellcheck` job). Run it here over the files this work
   item changed (the same set `git diff --name-only` reports for the work —
   never the whole repo, to honour minimal-diff):
   `conda run -n <env> codespell --toml pyproject.toml <changed files>`.
   - **Auto-fix the unambiguous cases.** Run
     `conda run -n <env> codespell -w --toml pyproject.toml <changed files>`.
     With `-w`, codespell rewrites only single-candidate corrections (a clear
     typo → one obvious fix) and leaves multi-candidate ones untouched. Verify
     each rewrite landed in a comment/docstring/string, not in code that would
     change behaviour; re-run the binding test command after fixing.
   - **Block on the unclear cases.** Re-run
     `codespell --toml pyproject.toml <changed files>`. Anything it still
     reports is ambiguous (multiple candidate fixes) or a domain term codespell
     doesn't know. **Stop and ask the user** how to resolve each — apply a
     specific fix, or add the term to `uri-ignore-words-list` /
     `ignore-words-list` in `pyproject.toml [tool.codespell]`. Do NOT guess a
     correction.

   If addressing findings or the spellcheck changed any code, invoke
   `sys-commit` (step: review, slug: `<slug>`, intended files: the fixes + any
   codespell auto-fixes — **never** `review-code.md`, which lives under the
   gitignored `.work/`) → message `review: <slug>`. If no code changed (no
   blockers, no typo fixes), there is nothing to commit at this step —
   `review-code.md` stays local-only.
7. **Update `CHANGES.md`.** Use the format in `CLAUDE.md` (canonical spec:
   `development.rst`, Changelog section); CI's `check_changes.py` only verifies
   the file changed, not its structure.
   a. Read the current version `X.Y.Z` from `src/sctoolbox/_version.py`.
   b. Find the `## X.Y.Z` section in `CHANGES.md`:
      - **Exists** — append bullet(s): package/docs changes under the main
        header, notebook changes under `### Changes to notebooks` (create it if
        absent).
      - **Absent** — prepend a new `## X.Y.Z (in progress)` section (plus a
        `### Changes to notebooks` subsection if scope includes notebooks)
        before the first existing `## ` line.
   c. Bullet = short imperative phrase for the user-visible change; append
      ` (#<N>)` only if the design or plan cites a GitLab issue.
   d. Invoke `sys-commit` (step: changes, slug: `<slug>`, intended files:
      `CHANGES.md`) → message `changes: <slug>`.
8. **Done.** Report what shipped and the final test result.

## Constraints

- **Reviewers are read-only and return text** — this skill writes
  `review-code.md`, not the agent.
- **No new tasks beyond the plan.** If the plan is short a task, route back
  to `/plan`.
- **Commit only intended files** via `sys-commit`; never blind `git add -A`,
  never a git op denied by `.claude/settings.json`.
