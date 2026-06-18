---
name: sys-code-review
description: Procedure for the code-reviewer sub-agent: read-only audit of the implementation against the plan and sctoolbox conventions. Driven by /implement.
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
2. **sctoolbox conventions.** Audit against the `CLAUDE.md` conventions
   (already in your context): decorators, `__init__.py` registration, numpy
   docstrings, plotting `ax`, deprecation. A missing `@beartype` on a public
   function is a **blocker**.
   - **Renderable examples** — policy in
     `.claude/docs/sys-examples.md`. A *missing* example is only ever a
     suggestion (never a blocker). A *present* one must be correct — right
     pre-code variables, no reassigned `adata`, `.. plot::` carries
     `:context: close-figs`, any new fixture added to the pre-code script — and
     a present-but-broken example is a **real finding** (it fails the docs
     build).
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
6. **Docs build safety.** The docs build (`make html`, run on the `dev`
   pipeline with `allow_failure: false`) regenerates the API reference via
   `docs/source/build_api.py`, which auto-discovers modules and emits an
   `.. automodule::` for each. So Sphinx **imports every module** — and
   `autodoc_mock_imports = []` means **nothing is mocked**. Module wiring is
   automatic (no manual API edit needed for a new submodule under
   `tools/`/`plotting/`/`utils/`), but **import-time failures break the docs
   build** even when tests pass. For any new or modified module, check:
   - **Optional/heavy deps must be imported lazily** (inside the function
     that uses them), not at module top level — matching existing sctoolbox
     practice. A top-level `import` of a dependency from an optional dep group
     (`atac`, `receptor_ligand`, `pseudotime`, etc.) or any package not
     guaranteed by `pip install .[all]` is a **blocker**: it will raise at
     autodoc import and fail `make html` on `dev`. (Alternatively the dep is
     added to `autodoc_mock_imports` in `conf.py`, but lazy import is
     preferred.)
   - **A genuinely new top-level module** directly under `src/sctoolbox/`
     (not a submodule of `tools`/`plotting`/`utils`) must be added to the
     `docs/source/API/index.rst` toctree, or its generated page is orphaned
     from the API nav. Submodules of the existing parents need no toctree edit.

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
