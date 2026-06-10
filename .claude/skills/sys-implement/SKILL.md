---
name: sys-implement
description: Agent-facing procedure followed by the implementer sub-agent. Writes tests first (red), implements plan tasks one at a time under the binding test command, marks - [x] T<N> only on green with no regression, retries up to a 3-attempt cap per task, commits each completed task via sys-commit, and refuses to invent tasks. On cap-hit, returns a structured failure summary. Not user-facing — driven by /implement.
---

# sys-implement

The **implementer** sub-agent's procedure. Reads an approved `plan.md` and
executes its tasks under the plan's declared **binding test command** as a
TDD (test-driven development) gate. Commits per completed task via
`sys-commit`. Not user-facing — the `/implement` skill spawns the
`implementer` agent, which follows this.

## Input

A path to a `plan.md`, at `.work/<YYYY-MM-DD>-<slug>/plan.md`. The calling
skill passes it. If absent, return a note that the path is required.

## Preflight

Before touching any code:

1. Read `plan.md`. Locate the `## Tasks` checkbox list and the `## Tests`
   section. Read sibling `design.md` and `review-plan.md` for context.
2. Extract the **binding test command** (the line beginning
   `**Test command for this plan:**`). If absent, return a note that the
   plan is missing its gate — the user should re-run `/plan`.
3. Confirm the command is invokable (conda env present, ruff and pytest or
   jupyter reachable, target files exist or are part of T1). If not, return
   the gap; do not start.

## Tests-first run (expect red)

1. Translate the plan's `TC<N>` (test case N) cases into concrete test code
   BEFORE any implementation, placed where the binding command finds them
   (`tests/<target>.py` for package scope).
2. Run the test command. Expect non-zero exit — the TDD red state. If it is
   already green before any implementation, investigate before continuing.

## Task loop (one at a time, with retry cap)

For each unchecked `- [ ] T<N>. ...` in order:

1. **Implement only what this task describes.** No scope creep into the next
   task. Follow all conventions in `CLAUDE.md`:
   - `@beartype` on all new public functions
   - `@log_anndata` on top-level functions that receive AnnData/MuData
     (decorator order: `@log_anndata` outermost, `@beartype` directly above
     function)
   - New submodules registered in parent `__init__.py`
   - Numpy-style docstrings on all functions
   - Notebook outputs cleared if scope includes notebooks
2. **Run the test command.**
3. **Decide:**
   - This task's target checks pass AND nothing previously-passing regressed
     → mark `- [x] T<N>` in `plan.md`; **commit via `sys-commit`** with
     message `impl(<slug>): T<N> <short desc>`; continue.
   - Target checks still failing → diagnose, fix, retry. **Up to 3 attempts
     total per task.** The counter resets when a task is marked done.
   - A previously-passing check regressed → revert the regression first; a
     regression repair does not consume a retry.
4. **If the 3-attempt cap is hit:** stop. Do NOT mark the task done, do NOT
   commit it. Return a **structured failure summary** to the calling skill —
   what was tried across the attempts and the last test output.

## After all tasks pass

1. Run the test command once more. It MUST exit 0.
2. Stop and return a structured summary (tasks completed, commits made, final
   test result) to the calling skill.

## Constraints

- **No new tasks.** If the plan needs a task it lacks, stop and return a
  note so the user can re-run `/plan`.
- **Only edit `plan.md`'s checkbox state and `Status` field.**
- **Commit only the intended files** via `sys-commit` — never blind
  `git add -A`, never a denied git op (see `.claude/settings.json`).
- **Deprecations:** if removing or replacing existing functionality, use the
  `deprecation` package targeting removal in 2 minor versions, and add
  `@deprecation.fail_if_not_removed` to the function's test.
