---
name: sys-plan
description: Procedure for the planner sub-agent: turns a design.md into a plan.md and selects the binding test command from scope. Driven by /plan.
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
   directory if present — in particular `related.md` (the fetched issue/MR
   requirements and discussion), which is prime planning context. `CLAUDE.md`
   (module map, stack, conventions) is already in your context — no need to
   re-read it.
2. **Extract scope and environment** from `design.md` `## Scope`. Copy
   `Commit mode`, `Autonomy`, and `Related` **verbatim** onto the matching
   `**…:**` line of `plan.md` so `/implement` and the implementer needn't
   re-read `design.md`:
   - `Type`: one or more of package | notebooks | docs (combinable) — drives the
     binding command (step 3).
   - `Conda environment`: the env selector used in all commands.
   - `Commit mode`: `manual` or `claude (Name <email>)`. If omitted, write
     `unspecified` and note it in Open questions.
   - `Autonomy`: `per-task` or `end` (`/design` always asks, so a well-formed
     design records it; this default only guards a malformed `design.md`). If
     omitted, default `end` and note it in Open questions.
   - `Related`: `#`/`!` numbers or `none` — lets `/implement` cite the issue in
     the changelog (`add_change.py --issue <N>`).
3. **Select the binding test command** based on scope. It opens with the shared
   ruff step, then chains the gate(s) for each scope touched with `&&`.
   `<env-spec>` is the env selector from `design.md` `## Scope` (`-n <name>` or
   `-p <prefix>`) — use exactly the form the design records so the command
   matches an allow rule:
   - **ruff (always, leads the command):** `conda run <env-spec> ruff check --preview .`
   - **package → pytest:** `conda run <env-spec> python -m pytest tests/<target>.py -v`
   - **notebooks → nbconvert:** `conda run <env-spec> jupyter nbconvert --to notebook --execute <notebook_path>`
   - **docs → Sphinx build:** `conda run <env-spec> make -C docs html`

   Multi-area scope chains each gate after ruff (e.g. package + notebooks →
   `ruff … && pytest … && nbconvert …`).

   The ruff step is `ruff check --preview .` **verbatim** — identical to the CI
   `lint` job and relying on `[tool.ruff].include` in `pyproject.toml` (covers
   `src/`, `scripts/`, `tests/`, notebooks). Do NOT narrow it to e.g.
   `ruff check src/sctoolbox tests` — that skips files CI lints, so the gate
   passes while CI fails.

   The per-file `pytest tests/<target>.py` keeps the TDD loop fast. Separately
   declare the whole-suite run on a `**Full-suite regression command:**` line
   (`conda run <env-spec> python -m pytest tests`, same env) — `/implement` runs
   it **verbatim** at final confirmation to catch cross-file regressions. Scope
   with no pytest segment (notebook-only, docs-only) → `n/a`.
4. **Draft `plan.md`** in the same directory as `design.md`, using the
   template at `.claude/docs/plan-template.md`. The plan MUST
   include:
   - Tasks as checkboxes (`- [ ] T1. ...`). Each task small enough to finish
     in one implement iteration with the test command run between tasks.
   - A `## Tests` section with concrete test cases (`TC<N>` (test case N))
     AND exactly one binding test command on a
     `**Test command for this plan:**` line.
   - A `**Full-suite regression command:**` line — the whole-suite pytest run
     for package scope, or `n/a` if scope excludes package.
   - A `**Commit mode:**` line copied verbatim from `design.md` `## Scope`.
   - An `**Autonomy:**` line copied verbatim from `design.md` `## Scope`.
   - A `**Related:**` line copied verbatim from `design.md` `## Scope`.
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
logic — these **get a TC<N>**) from **orchestration** tasks
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

## Docs scope

When scope includes docs (changes under `docs/`):
- The Sphinx build (`make -C docs html`) is the gate — a malformed `.rst`,
  broken cross-reference, or failing directive makes the build error and is a
  failing test. This mirrors CI's `build-pages` job.
- The build needs the `docs` dependency-group and a system `pandoc`
  installed in the conda env (see `development.rst`). Flag in the plan if the
  env may lack them so the user can install before `/implement`.
- Docs-only changes usually have **no `TC<N>` cases** — the build is the
  whole gate. Tasks are the concrete `.rst`/doc edits, in order.
- Plain `.rst` edits are not linted by ruff, but the `ruff check --preview .`
  step still runs first as the universal gate (and catches any `.py` docs
  helpers such as `docs/source/*.py`).

## Doc-string examples

Policy in `.claude/docs/sys-examples.md`. When a package-scope change
adds or modifies public functions, weigh planning an example task — `.. plot::`
for plotting functions (highly encouraged), `.. exec_code::` for other functions
whose output is illustrative (where it adds value). Pair a pre-code fixture task
if the example needs new input data. These are orchestration tasks: no `TC<N>`,
never in the binding test command, and not a docs gate on their own — do **not**
add `make -C docs html` to the binding command for examples alone.

## Constraints

- **Write only inside `.work/<date>-<slug>/`** — only `plan.md`. No code,
  no tests, no file moves.
- **Do not run the test command.** That is the `implementer`'s TDD
  (test-driven development) gate.
- **Do not commit.**
