---
name: sys-commit
description: Shared commit procedure: stages only intended files, tagged-message convention. Invoked by the workflow skills and the implementer.
---

# sys-commit

The shared commit procedure for the agentic workflow. Centralises the
tagged-message convention and safe staging. Invoked by `/design`, `/plan`,
`/implement`, and by the `implementer` agent per completed task.

## Inputs (from the caller)

- The **step** being committed (design / plan / impl-task / review / changes).
- The **slug** for the work item.
- The **commit mode** for the work item (`manual` or `claude (Name <email>)`),
  read from the plan's `**Commit mode:**` line (or `design.md` `## Scope`).
- For an impl-task commit: the task number `T<N>` and a short description.
- The set of **intended files** to stage (explicit paths).

## Commit mode

- **`manual`** — the user stages and commits everything themselves. This skill
  is a **no-op**: do **not** `git add`, do **not** `git commit`. Return a note
  to the caller that the change is left unstaged for the user. (Callers should
  not invoke this skill at all in manual mode; this is a defensive guard.)
- **`claude (Name <email>)`** — proceed with the procedure below, setting both
  the **author** and the **committer** to that person via **per-invocation
  `-c` flags** (not `git config`):
  `git -c user.name="Name" -c user.email="email" commit --author="Name <email>" -m "<message>"`.
  This scopes the identity to the single commit — it writes nothing to global,
  user, or `.git/config`, so it cannot leak to other commits, repos, or
  folders. **Never** run `git config` (local or global) to set the identity.

## Procedure

1. **Inspect first.** Run `git status` and `git diff --stat` to see what
   changed. Confirm the changes match the step being committed.
2. **Stage only intended files.** `git add <explicit paths>`. **Never** use
   `git add -A` / `git add .` / `git commit -a` — blind staging risks
   including unrelated work or sensitive files.
3. **Commit with a tagged message** from the table below, setting author and
   committer from the commit mode via per-invocation `-c` flags (e.g.
   `git -c user.name="Name" -c user.email="email" commit --author="Name <email>" -m "<message>"`).
4. **Confirm.** Run `git log --oneline -1` so the SHA is visible to the
   caller.

## Tagged-message convention

| Step | Commit message |
|---|---|
| each implement task | `impl(<slug>): T<N> <desc>` |
| code review addressed | `review: <slug>` |
| CHANGES.md updated | `changes: <slug>` |

Keep the subject line short. The `<slug>` must appear in every message so
`git log --grep=<slug>` recovers the full work-item history.

**The `.work/` artifacts are never committed.** `design.md`, `plan.md`, and
`review-*.md` live under the gitignored `.work/` directory (local-only audit
trail) — the `/design` and `/plan` stages therefore make **no commit**, and
implement-stage commits stage only code, tests, `CHANGES.md`, and (when a task
changes it) `pyproject.toml`. Never
`git add` a path under `.work/` (it would require `-f` to override the ignore;
do not do this).

## Hard constraints

- **Never run a denied git op.** `.claude/settings.json` denies `git merge`,
  `checkout`, `switch`, `rebase`, `reset`, `cherry-pick`, and `push`. Do not
  attempt them — and never use the `-c` prefix to wrap any subcommand other
  than `commit`. The `-c user.name`/`-c user.email` flags are **only** for
  setting the per-commit identity; they must never carry a denied op past the
  deny list.
- **Never write git identity to config.** No `git config` (global, `--local`,
  or otherwise). Identity is set per-invocation via `-c` so it stays scoped to
  the single commit and never leaks to other commits, repos, or folders.
- **Never discard work.** No `git checkout -- <file>`, no `git reset`.
- **Never blind-stage.** Explicit paths only.
- If staging or committing fails (hook rejection, nothing staged), surface
  the error to the caller rather than forcing through.
