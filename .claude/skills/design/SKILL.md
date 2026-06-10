---
name: design
description: First stage of the sc-framework dev workflow (design → plan → implement). Interactive, runs in the main loop with no subagent. Reads context, asks 1-4 clarifying questions (always including conda environment and change scope), proposes a .work/<YYYY-MM-DD>-<slug>/ directory and waits for confirmation, writes design.md, commits via sys-commit, and hands off to /plan.
---

# design

First stage of the sc-framework development workflow
(design → plan → implement). Interactive, **main loop, no subagent** —
subagents can't talk to the user, and this stage is all conversation.
Produces a `design.md` that `/plan` turns into a concrete plan.

## Inputs (combinable, all optional)

1. **Inline prompt** — free text after `/design`.
2. **Topic file** — a markdown path with seed thoughts; read it.

If neither is given, ask the user what they want to design.

## Process

1. **Read context.** Any topic file, plus `CLAUDE.md` (stack, module map,
   conventions) and `CHANGES.md` (what shipped recently).
2. **Think before drafting.** Reason about the goal, which modules it
   touches, and what could go wrong, before writing anything.
3. **Ask clarifying questions.** Before any file is written, surface 1–4
   sharp questions. Always include:
   - **Conda environment** — which env should the binding test command use?
   - **Change scope** — package (`src/sctoolbox/` + `tests/`), notebooks, or both?
   Use `AskUserQuestion` when choices enumerate cleanly; free-text otherwise.
4. **Propose slug + date, then confirm.** Short kebab-case, descriptive
   (e.g. `qc-filter-fix`, `embedding-plot`). Present the full directory
   `.work/<YYYY-MM-DD>-<slug>/` and **wait for confirmation** before creating
   anything. Use today's date from the environment.
5. **Write `.work/<YYYY-MM-DD>-<slug>/design.md`** with exactly these
   sections in order:

   ```markdown
   # <Title>

   ## Problem
   What are we trying to do, and why now? Reference related prior work if
   relevant.

   ## Approach
   High-level idea, not implementation. Which sctoolbox modules / data flow
   it touches, the angle, and what we deliberately will NOT do.

   ## Scope
   - Type: package | notebooks | both
   - Conda environment: <name>

   ## Success criteria
   Concrete, observable signals — numeric where possible.

   ## Open questions
   What the user must decide before planning. Empty if all resolved.
   ```

6. **Commit.** Invoke `sys-commit` (step: design, slug: `<slug>`, intended
   files: the new `design.md`) → message `design: <slug>`.
7. **Hand off and stop.** Tell the user the next step is
   `/plan .work/<YYYY-MM-DD>-<slug>/design.md`. Do not continue into planning.

## Out of scope

- Writing `plan.md`, tests, or any code — those belong to `/plan` and `/implement`.
- Moving, renaming, or refactoring files; running analysis.

If the user wants any of those during design, note it in Open questions.
