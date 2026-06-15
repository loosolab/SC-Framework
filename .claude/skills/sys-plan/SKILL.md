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
   directory if present. `CLAUDE.md` (module map, stack, conventions) is already
   in your context — no need to re-read it.
2. **Extract scope and environment** from the `design.md` `## Scope` section:
   - `Type`: one or more of package | notebooks | docs (combinable)
   - `Conda environment`: the env name to use in all commands
3. **Select the binding test command** based on scope. Every command opens
   with the single shared ruff step; append the gate(s) for each scope the
   change touches, chained with `&&`:
   - **ruff (always, leads the command):** `conda run -n <env> ruff check --preview .`
   - **package → pytest:** `conda run -n <env> python -m pytest tests/<target>.py -v`
   - **notebooks → nbconvert:** `conda run -n <env> jupyter nbconvert --to notebook --execute <notebook_path>`
   - **docs → Sphinx build:** `conda run -n <env> make -C docs html`

   A multi-area change chains the relevant gates after the shared ruff step —
   e.g. package + notebooks is `ruff … && pytest … && nbconvert …`, and
   package + docs is `ruff … && pytest … && make -C docs html`.

   The ruff step is `ruff check --preview .` **verbatim** — identical to the CI
   `lint` job (`.gitlab-ci.yml`). It relies on the `[tool.ruff].include` list in
   `pyproject.toml` (which already covers `src/`, `scripts/`, `tests/`, and the
   notebooks), so a notebook- or script-scope change is actually linted. Do NOT
   substitute a narrower path such as `ruff check src/sctoolbox tests` — that
   skips the very files a notebook plan changes and lets CI fail on a
   green-locally gate.

   The per-file `pytest tests/<target>.py` keeps the TDD loop fast; `/implement`
   runs the full `pytest tests` once at final confirmation to catch cross-file
   regressions (see `implement/SKILL.md`).
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

Renderable `Examples` sections are best practice but never mandatory (see the
"Example code and results" section of `docs/source/development.rst`). When a
package-scope change adds or modifies public functions, weigh whether to plan
an example task:

- **New/modified public plotting function:** plan a task to add a
  `.. plot:: :context: close-figs` example. Highly encouraged — omit only with a
  noted reason.
- **New/modified public non-plotting function** whose output is illustrative
  (a table, a printed summary): plan a task to add an `.. exec_code::` example.
  Encouraged where it adds value; skip otherwise.
- **Pre-code fixtures.** If such an example needs input data not already
  prepared in the relevant pre-code script (`docs/source/plot_pre_code.py` for
  plotting, `docs/source/utils_pre_code.py` for utils), plan a paired task to
  extend that script minimally so the variable exists.

These example tasks are **orchestration**, not algorithmic-core — they get no
`TC<N>`, and they never go in the binding test command. Their
verification differs by directive: `.. exec_code::` examples are smoke-checked
locally by the single-page `dummy` build `/implement` runs; `.. plot::` example
*execution* is only verified by the full `make -C docs html` on CI. Do **not**
add a docs gate to the binding command for examples alone; reserve
`make -C docs html` for changes that declare docs scope.

## Constraints

- **Write only inside `.work/<date>-<slug>/`** — only `plan.md`. No code,
  no tests, no file moves.
- **Do not run the test command.** That is the `implementer`'s TDD
  (test-driven development) gate.
- **Do not commit.**
