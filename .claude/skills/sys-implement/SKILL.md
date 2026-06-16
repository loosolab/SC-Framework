---
name: sys-implement
description: Procedure for the implementer sub-agent: TDD task loop under the binding test command, per-task commits, 3-attempt cap. Driven by /implement.
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
   Also read the **commit mode** (the `**Commit mode:**` line). In
   **`manual`** mode you do **not** commit at any point — complete tasks and run
   the gate, but leave all changes unstaged for the user. In
   **`claude (Name <email>)`** mode, commit per task via `sys-commit`, passing
   the mode so the author is set. If the line is missing or `unspecified`,
   return a note — the calling skill resolves it before you start.
3. Confirm the command is invocable (conda env present, ruff and pytest or
   jupyter reachable, target files exist or are part of T1). If not, return
   the gap; do not start.

## Tests-first run (expect red)

1. Translate the plan's `TC<N>` (test case N) cases into concrete test code
   BEFORE any implementation, placed where the binding command finds them
   (`tests/<target>.py` for package scope).
2. Run the test command. Expect non-zero exit — the TDD red state. If it is
   already green before any implementation, investigate before continuing.

Docs-only and notebook-only plans usually declare **no `TC<N>` cases** — the
Sphinx build (`make -C docs html`) or notebook execution is the whole gate, so
there is no red state to author. Skip the tests-first run and go straight to
the task loop; the binding command still runs after every task.

## Task loop (one at a time, with retry cap)

For each unchecked `- [ ] T<N>. ...` in order:

1. **Implement only what this task describes.** No scope creep into the next
   task. Follow all conventions in `CLAUDE.md` (already in your context).
   Task-specific items the plan may invoke:
   - **Renderable examples** when the plan has an example task: write it per
     `.claude/skills/sys-examples/SKILL.md` (directive choice, reuse pre-code
     variables, never reassign shared ones such as `adata`, extend the pre-code
     script minimally if new input data is needed).
   - Notebook outputs cleared if scope includes notebooks
2. **Run the test command.**
3. **Decide:**
   - This task's target checks pass AND nothing previously-passing regressed
     → mark `- [x] T<N>` in `plan.md` (a local-only edit — `plan.md` is under
     the gitignored `.work/` and is **never staged**). In **`claude`** commit
     mode, **commit via `sys-commit`** with message
     `impl(<slug>): T<N> <short desc>`, staging only this task's code/test files.
     In **`manual`** mode, skip the commit and leave the task's changes
     unstaged. Either way, continue to the next task.
   - Target checks still failing → diagnose, fix, retry. **Up to 3 attempts
     total per task.** The counter resets when a task is marked done.
   - A previously-passing check regressed → revert the regression first; a
     regression repair does not consume a retry.
4. **If the 3-attempt cap is hit:** stop. Do NOT mark the task done, do NOT
   commit it. Return a **structured failure summary** to the calling skill —
   what was tried across the attempts and the last test output.

## After all tasks pass

1. Run the test command once more. It MUST exit 0.
2. Stop and return a structured summary (tasks completed, commits made — or
   "none, manual commit mode" — and final test result) to the calling skill.
   In manual mode, list the changed files so the user can stage them.

## Constraints

- **No new tasks.** If the plan needs a task it lacks, stop and return a
  note so the user can re-run `/plan`.
- **Only edit `plan.md`'s checkbox state and `Status` field.**
- **Commit only the intended files** via `sys-commit` — never blind
  `git add -A`, never a denied git op (see `.claude/settings.json`).
- **Deprecations:** per `CLAUDE.md`'s rule — `deprecation` package, removal in
  2 minor versions, `@deprecation.fail_if_not_removed` on the function's test.
