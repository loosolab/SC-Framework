---
name: sys-commit
description: Agent-facing shared commit procedure. Stages only the intended files (never blind git add -A), writes a tagged commit message following the workflow convention, and never runs a git op denied by .claude/settings.json. Invoked by the main-loop skills at checkpoints and by the implementer per completed task. Not user-facing.
---

# sys-commit

The shared commit procedure for the agentic workflow. Centralises the
tagged-message convention and safe staging. Invoked by `/design`, `/plan`,
`/implement`, and by the `implementer` agent per completed task.

## Inputs (from the caller)

- The **step** being committed (design / plan / impl-task / review / changes).
- The **slug** for the work item.
- For an impl-task commit: the task number `T<N>` and a short description.
- The set of **intended files** to stage (explicit paths).

## Procedure

1. **Inspect first.** Run `git status` and `git diff --stat` to see what
   changed. Confirm the changes match the step being committed.
2. **Stage only intended files.** `git add <explicit paths>`. **Never** use
   `git add -A` / `git add .` / `git commit -a` — blind staging risks
   including unrelated work or sensitive files.
3. **Commit with a tagged message** from the table below.
4. **Confirm.** Run `git log --oneline -1` so the SHA is visible to the
   caller.

## Tagged-message convention

| Step | Commit message |
|---|---|
| design done | `design: <slug>` |
| plan + review done | `plan: <slug>` |
| each implement task | `impl(<slug>): T<N> <desc>` |
| code review addressed | `review: <slug>` |
| CHANGES.md updated | `changes: <slug>` |

Keep the subject line short. The `<slug>` must appear in every message so
`git log --grep=<slug>` recovers the full work-item history.

## Hard constraints

- **Never run a denied git op.** `.claude/settings.json` denies `git merge`,
  `checkout`, `switch`, `rebase`, `reset`, `cherry-pick`, and `push`. Do not
  attempt them.
- **Never discard work.** No `git checkout -- <file>`, no `git reset`.
- **Never blind-stage.** Explicit paths only.
- If staging or committing fails (hook rejection, nothing staged), surface
  the error to the caller rather than forcing through.
