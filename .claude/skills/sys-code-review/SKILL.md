---
name: sys-code-review
description: Agent-facing procedure followed by the code-reviewer sub-agent. Read-only audit after the binding test command exits 0 — correctness against the plan, sctoolbox conventions (decorator order, AnnData mutations, docstrings, __init__.py registration), TDD (test-driven development) gate integrity (reruns the command, confirms each TC<N> (test case N) is implemented), and scope discipline. Returns findings; writes nothing. Not user-facing — driven by /implement.
---

# sys-code-review

The **code-reviewer** sub-agent's procedure. Read-only audit after all plan
tasks are checked and the binding test command exits 0. Return findings as
structured text; the `/implement` skill writes them to `review-code.md`. You
write nothing.

## Inputs

The calling skill passes:
- Path to `plan.md` (`.work/<YYYY-MM-DD>-<slug>/plan.md`).
- Optional: relevant changed files or the work-item directory. If absent,
  derive scope from `git diff --name-only`.

## Checks

1. **Correctness against the plan.** Read the plan's tasks and design
   decisions; verify the code matches. sctoolbox failure modes to watch:
   - AnnData modified in-place when the design says return (or vice versa).
   - Missing or incorrect column/key names in `.obs`, `.var`, `.uns`.
   - Functions that accept AnnData but also return a modified copy — never
     both.
   - Filter or threshold values contradicting plan numbers.
2. **sctoolbox conventions.** Check against `CLAUDE.md`:
   - Decorator order: `@log_anndata` (if present) outermost, `@beartype`
     directly above function. `@beartype` missing on a public function is a
     blocker.
   - New submodules registered in the parent `__init__.py` `__all__`.
   - Numpy-style docstrings present on all new/modified public functions.
   - Plotting functions accept `ax` and return it.
   - Deprecated code uses the `deprecation` package with `fail_if_not_removed`
     in tests.
3. **TDD (test-driven development) gate integrity.**
   - Run the declared binding test command; confirm exit 0.
   - Confirm each `TC<N>` (test case N) in the plan's `## Tests` is actually
     implemented as a concrete test — a green gate does not help if cases are
     missing.
4. **Notebook conventions** (if scope includes notebooks):
   - Notebook outputs are cleared.
   - Naming prefix follows the project convention (numeric or letter-based).
   - Input cells use `bgcolor()` and are marked blue.
5. **Scope discipline.** Flag changes outside the plan: edits to modules not
   listed as tasks, missing `__init__.py` registrations.

## Self-judgement (trivial vs substantive)

Read the diff before drafting findings. The review **always runs**, but
depth scales with the change:

- **Substantive:** run all checks and emit real findings. Default for any new
  function, data transformation, or multi-file change.
- **Trivial:** if the change is genuinely too small to warrant findings
  (a one-line fix, a single parameter default, a typo), return the literal
  phrase `trivial — no findings`.

## Output

Return a structured response in this shape:

```markdown
# Code review

**Plan:** .work/<date>-<slug>/plan.md
**Verdict:** ready | needs revision

## Correctness
- <observation or issue>

## Conventions
- <observation or issue>

## TDD gate
- Test command: `<verbatim>`
- Last exit code: 0 | non-zero
- Test cases verified: TC1, TC2, ... | not all (list missing)
- Issues: <none | ...>

## Scope discipline
- <files changed outside the plan, or "none">

## Findings
- F1. <issue> — severity: blocker | suggestion

## Other notes
- ...
```

Set `Verdict: needs revision` if any **blocker** is present.

## Out of scope

- Editing code or fixing issues — this is a review.
- Judging scientific validity of analysis choices.
- Modifying `design.md`, `plan.md`, or upstream artifacts.
