---
name: implement
description: Third stage of the sc-framework dev workflow (design → plan → implement). Main-loop orchestrator. Preflight-checks the binding test command, spawns the implementer sub-agent (sys-implement, TDD (test-driven development), per-task commits), spawns the code-reviewer sub-agent (sys-code-review), writes review-code.md, addresses findings, updates CHANGES.md, and commits.
---

# implement

Third and final stage of the sc-framework development workflow
(design → plan → implement). Runs in the **main loop** and orchestrates two
sub-agents. Executes an approved `plan.md` under its TDD (test-driven
development) gate, reviews the result, and records the work item in
`CHANGES.md`.

## Input

A path to a `plan.md`, at `.work/<YYYY-MM-DD>-<slug>/plan.md`. If invoked
without a path, ask for one.

## Preflight

1. Read `plan.md`. Extract the **binding test command** (the line beginning
   `**Test command for this plan:**`). If absent, stop — tell the user the
   plan is missing its gate and to re-run `/plan`.
2. Confirm the command is invokable (conda env present, ruff and pytest
   reachable). If not, stop and surface the gap.

## Process

1. **Spawn the `implementer`.** Use the Agent tool with
   `subagent_type: implementer`, passing the `plan.md` path. It follows
   `sys-implement`: writes tests first (red), implements tasks one at a time
   under the binding command, marks `- [x] T<N>` only on green with no
   regression, and **commits per completed task** via `sys-commit`
   (`impl(<slug>): T<N> <desc>`). It retries up to 3 attempts per task.
2. **Handle a retry-cap or blocker handoff.** If the implementer returns a
   structured failure summary, **stop and ask the user** how to proceed:
   - Continue trying (resets the counter, explicit consent) — re-spawn the
     implementer with that instruction.
   - Re-plan (`/plan`).
   - Re-design (`/design`).
   Do not invent tasks or force past the cap yourself.
3. **Confirm the gate.** When the implementer reports done, run the binding
   test command once yourself and confirm exit 0. If not, hand the
   still-failing task back to the implementer.
4. **Spawn the `code-reviewer`.** Use the Agent tool with
   `subagent_type: code-reviewer`, passing `plan.md` and the changed files /
   work-item directory. It follows `sys-code-review` and **always runs**.
5. **Write the review** verbatim to `.work/<date>-<slug>/review-code.md`.
6. **Address findings.** Fix every **blocker**, re-running the binding test
   command after each fix. If the reviewer flags a structural mismatch you
   cannot resolve without user input, stop and hand off. Then invoke
   `sys-commit` (step: review, slug: `<slug>`, intended files: the fixes +
   `review-code.md`) → message `review: <slug>`. If there were no findings,
   still commit `review-code.md`.
7. **Update `CHANGES.md`.** Gather the work item's commits with
   `git log --grep=<slug> --oneline` and prepend a new entry at the top of
   `CHANGES.md` (newest first):

   ```markdown
   ## <slug> — <YYYY-MM-DD>
   <one-line summary of what shipped>

   Commits:
   - <sha> design: <slug>
   - <sha> plan: <slug>
   - <sha> impl(<slug>): T1 <desc>
   - <sha> review: <slug>
   ```

   Then invoke `sys-commit` (step: changes, slug: `<slug>`, intended files:
   `CHANGES.md`) → message `changes: <slug>`.
8. **Done.** Report what shipped and the final test result.

## Constraints

- **Reviewers are read-only and return text** — this skill writes
  `review-code.md`, not the agent.
- **No new tasks beyond the plan.** If the plan is short a task, route back
  to `/plan`.
- **Commit only intended files** via `sys-commit`; never blind `git add -A`,
  never a git op denied by `.claude/settings.json`.
