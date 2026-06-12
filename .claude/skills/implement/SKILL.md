---
name: implement
description: Third stage of the sc-framework dev workflow (design → plan → implement). Main-loop orchestrator. Preflight-checks the binding test command, spawns the implementer sub-agent (sys-implement, TDD (test-driven development), per-task commits), spawns the code-reviewer sub-agent (sys-code-review), writes review-code.md, addresses findings, runs codespell (auto-fixing unambiguous typos and blocking on unclear ones), updates CHANGES.md, and commits.
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
2. Confirm the command is invocable (conda env present, ruff and pytest
   reachable). If not, stop and surface the gap.

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
7. **Update `CHANGES.md`.** The canonical changelog format is specified in
   `docs/source/development.rst` (Changelog section) — follow it; the steps
   below restate it, and CI's `check_changes.py` only verifies the file was
   updated, not its structure.
   a. Read the current version from `src/sctoolbox/_version.py`
      (first line: `__version__ = "X.Y.Z"`).
   b. Open `CHANGES.md` and search for a line starting with `## X.Y.Z`
      (the version string, no surrounding text required to match):
      - **Section exists** — append new bullet(s) to it.  For package- and
        docs-scope changes add under the main section header; for
        notebook-scope changes add under the `### Changes to notebooks`
        subsection (create it if absent).
      - **Section absent** — prepend a new section immediately before the
        first existing `## ` line:
        ```markdown
        ## X.Y.Z (in progress)
        - <one-line description of what shipped>
        ```
        If the design scope includes notebooks also append:
        ```markdown
        ### Changes to notebooks
        - <one-line description>
        ```
   c. Bullet text: a short imperative phrase describing the user-visible
      change. Omit the issue number if there is no associated GitLab issue;
      append ` (#<N>)` if the design or plan references one.
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
