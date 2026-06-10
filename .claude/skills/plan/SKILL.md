---
name: plan
description: Second stage of the sc-framework dev workflow (design → plan → implement). Main-loop orchestrator. Spawns the planner sub-agent (sys-plan) to draft plan.md, spawns the plan-reviewer sub-agent (sys-plan-review), writes review-plan.md from the reviewer's findings, addresses blockers, commits via sys-commit, and hands off to /implement.
---

# plan

Second stage of the sc-framework development workflow
(design → plan → implement). Runs in the **main loop** and orchestrates two
sub-agents. Turns a `design.md` into a reviewed, executable `plan.md`.

## Input

A path to a `design.md`, at `.work/<YYYY-MM-DD>-<slug>/design.md`. If
invoked without a path, ask for one.

## Process

1. **Spawn the `planner`.** Use the Agent tool with
   `subagent_type: planner`, passing the `design.md` path. The planner
   follows `sys-plan`, drafts `plan.md` in the work-item directory, and
   returns its path + summary. It writes only inside `.work/`; it does not
   run tests or commit.
2. **Soft code-protection backstop.** Run `git status --porcelain` and
   confirm the planner touched **only** files under
   `.work/<date>-<slug>/`. If it wrote anything under `src/`, `tests/`,
   or notebooks directories, stop and surface it to the user.
3. **Spawn the `plan-reviewer`.** Use the Agent tool with
   `subagent_type: plan-reviewer`, passing the `design.md` and `plan.md`
   paths. It follows `sys-plan-review` and returns a structured verdict +
   findings.
4. **Write the review.** Write the reviewer's response verbatim to
   `.work/<date>-<slug>/review-plan.md`.
5. **Address findings.** Close every **blocker** the reviewer flagged —
   re-spawn the `planner` with the specific gaps if a redraft is needed.
   Scaffolding suggestions are the user's call, not auto-blockers. If the
   reviewer reports a design-vs-plan divergence you cannot resolve without
   user input, **stop and ask the user**.
6. **Commit.** Once the verdict is `ready` (no open blockers), set
   `Status: ready` in the plan frontmatter and invoke `sys-commit` (step:
   plan, slug: `<slug>`, intended files: `plan.md` and `review-plan.md`)
   → message `plan: <slug>`.
7. **Hand off and stop.** Tell the user the next step is
   `/implement .work/<date>-<slug>/plan.md`. Do not start implementing.

## Constraints

- **Do not write code or tests.** This stage produces `plan.md` and
  `review-plan.md` only.
- **Reviewers are read-only and return text** — this skill writes
  `review-plan.md`, not the agent.
