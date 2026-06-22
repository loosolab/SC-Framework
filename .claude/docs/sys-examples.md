---
name: sys-examples
description: Shared policy for renderable docstring examples (.. plot:: / .. exec_code::) across the plan/implement/review stages. Referenced by path, not invoked.
---

# sys-examples

Shared policy for renderable `Examples` sections in docstrings, used by
`sys-plan` (decide whether to plan one), `sys-implement` (write it),
`sys-code-review` (audit it), and `/implement` (render-check it). The authoring
mechanics — the directives and the per-module pre-code fixtures — are the
"Example code and results" section of `docs/source/development.rst` (canonical);
the encouraged-not-required policy is in `CLAUDE.md`. This file holds only the
*workflow* policy the stages share.

## Policy

- **Encouraged, never mandatory.** A *missing* example is at most a suggestion,
  never a blocker.
- **Which directive:**
  - New/modified public **plotting** function → `.. plot:: :context: close-figs`
    (highly encouraged).
  - Other public function whose output is illustrative (a table, a printed
    summary) → `.. exec_code::` (encouraged where it adds value; skip otherwise).
- **Pre-code fixtures.** Reuse variables from the page's pre-code script
  (`docs/source/plot_pre_code.py` for plotting, `utils_pre_code.py` for utils);
  **never reassign shared ones** such as `adata`. If an example needs input data
  not already prepared there, extend that script minimally (the plan pairs such
  a task).
- **Not algorithmic-core.** Example tasks are orchestration — they get **no
  `TC<N>`**, never enter the binding test command, and are not a docs gate on
  their own.
- **Verification differs by directive:**
  - `.. exec_code::` — smoke-checked locally by the single-page `dummy` build
    `/implement` runs (see **Render-check procedure** below); an example that
    raises fails it.
  - `.. plot::` — execution verified only by the full `make -C docs html` on CI;
    locally it is parsed, not executed.

## Render-check procedure (used by /implement)

Run this **only** when `git diff` shows an added/modified `.. exec_code::`
docstring example (the binding command does not execute these; skip otherwise).
Build that module's API page with the **`dummy`** builder — it executes
`.. exec_code::` but skips HTML finalisation (a single-page `-b html` fails on
nbsphinx's notebook collection — do **not** use it):

`conda run <env-spec> sphinx-build -b dummy docs/source /tmp/scdocs docs/source/API/<m>.rst`

(`<m>` = `tools`, `utils`, …). **Non-zero exit** = an example raised (a
traceback, or a `NameError` from a missing pre-code variable) → hand back to the
implementer to fix the example or extend `utils_pre_code.py`. The
`toctree`/cross-reference warnings about the unbuilt rest of the docs are
expected and do **not** fail it. `.. plot::` examples are only parsed by this
build, not executed — CI's `make -C docs html` verifies those.
