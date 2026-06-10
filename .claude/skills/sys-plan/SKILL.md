---
name: sys-plan
description: Agent-facing procedure followed by the planner sub-agent. Turns a design.md into a concrete plan.md using plan-template.md, selects the binding test command based on scope (package/notebooks/both), and keeps the diff minimal. Writes only inside .work/<date>-<slug>/; does not run tests or commit. Not meant to be invoked directly by the user — use /plan.
---

# sys-plan

The **planner** sub-agent's procedure. Turns a `design.md` into a concrete
`plan.md` the `implementer` can execute against a test gate. Not a
user-facing skill — the `/plan` skill spawns the `planner` agent, which
follows this.

## Input

A path to a `design.md`, at `.work/<YYYY-MM-DD>-<slug>/design.md`. The
calling skill passes it. If absent, return a note that the path is required.

## Process

1. **Read the design.** Open `design.md`. Read sibling artifacts in the same
   directory if present. Read `CLAUDE.md` for the module map, stack, and
   conventions.
2. **Extract scope and environment** from the `design.md` `## Scope` section:
   - `Type`: package | notebooks | both
   - `Conda environment`: the env name to use in all commands
3. **Select the binding test command** based on scope:
   - **package:** `conda run -n <env> ruff check src/sctoolbox tests && conda run -n <env> python -m pytest tests/<target>.py -v`
   - **notebooks:** `conda run -n <env> ruff check src/sctoolbox tests && conda run -n <env> jupyter nbconvert --to notebook --execute <notebook_path>`
   - **both:** chain all three — ruff, pytest, nbconvert
4. **Draft `plan.md`** in the same directory as `design.md`, using the
   template at `.claude/skills/sys-plan/plan-template.md`. The plan MUST
   include:
   - Tasks as checkboxes (`- [ ] T1. ...`). Each task small enough to finish
     in one implement iteration with the test command run between tasks.
   - A `## Tests` section with concrete test cases (`TC<N>` (test case N))
     AND exactly one binding test command on a
     `**Test command for this plan:**` line.
   - Every success criterion from `design.md` mapped onto at least one task
     or test case.
5. **Sanity-check the test command.** It must be invocable in this
   environment. The command is binding — `implementer` runs it literally.
6. **Stop.** Return the plan's path and a short summary as structured text.
   Do not invoke the reviewer, do not run tests, do not commit.

## Minimal-diff norm

Favour small, reviewable diffs. Before writing the `## Tasks` list,
sanity-check each proposed helper:

- Could a function argument replace a new module?
- Could a one-line expression replace a wrapper helper?
- Does a proposed helper exist only to wrap a single call?

Legitimate new modules ARE allowed — register them in the parent
`__init__.py` as per `CLAUDE.md`. Scaffolding is a **suggestion-level**
concern the `plan-reviewer` may flag, not a hard block.

## Algorithmic-core vs orchestration

Separate **algorithmic-core** tasks (data transformations, scoring, filtering
logic — these **get a TC<N> (test case N)**) from **orchestration** tasks
(CLI wiring, plot generation, file writers driven by tested cores — these may
not need a dedicated test case). The binding test command targets the
algorithmic delta.

## Notebook scope

When scope includes notebooks:
- Identify the target notebook(s) from `design.md`.
- Verify notebook outputs are cleared (the repo `.gitconfig` handles this,
  but flag if it appears not active).
- The nbconvert execute command is the gate — a notebook that errors on
  execution is a failing test.

## Constraints

- **Write only inside `.work/<date>-<slug>/`** — only `plan.md`. No code,
  no tests, no file moves.
- **Do not run the test command.** That is the `implementer`'s TDD
  (test-driven development) gate.
- **Do not commit.**
