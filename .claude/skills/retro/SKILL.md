---
name: retro
description: Stage 4 of the dev workflow. Interactive main-loop skill: reviews how the finished work item actually went, mines the session for friction, and turns the lessons into durable repo changes (CLAUDE.md / skills / scripts). Proposes first, applies only what the user approves.
---

# retro

Fourth and final stage of the sc-framework development workflow
(design → plan → implement → retro). Interactive **main-loop** skill, no
subagent (this stage is all conversation). Turns what the finished work item
taught us into durable changes in **the repository**.

This stage exists because this project keeps all project knowledge in the repo
and never in agent memory (see `CLAUDE.md`, "Working agreements"). Without a
step like this, every lesson from a work item is lost when the session ends, or
— worse — survives only in one installation's private memory, so two Claude
installations end up working from different ground.

## Input

A path to a work-item directory or its `plan.md`
(`.work/<YYYY-MM-DD>-<slug>/`). If invoked without one, use the most recently
modified `.work/<date>-<slug>/` and say which you picked. Runs standalone on any
past work item, not only straight after `/implement`.

## Process

1. **Gather the artifact evidence.** Read the work item's `design.md`,
   `plan.md` (including which task checkboxes are ticked), `review-plan.md` and
   `review-code.md`. Note specifically: blockers a reviewer caught, deviations
   recorded under `## Design decisions`, open questions that had to go back to the
   user, retry-cap or blocker handoffs, and any success criterion that ended up
   unverifiable as written.
2. **Mine the current session — the highest-signal input.** The user's own
   messages are evidence. Look for: a question the user had to ask twice; a
   correction they had to make; a steer they had to repeat; a decision that was
   reopened because a stage recorded it ambiguously; work that was wasted or
   redone; permission prompts or denials; and any command that failed or prompted
   because a helper in `scripts/` or a skill already covered it and was not used.
   Treat each of these as a defect in the workflow documentation, not as a defect
   in the user. Also look for gates that fired **late**: a problem a later stage
   caught that an earlier stage should have.
3. **Sweep for stray memories.** Project knowledge belongs in the repo. Check
   this session's memory directory; if any memory carries sc_framework knowledge
   (a convention, a gotcha, a tooling detail), propose folding its content into
   the right repo file and then deleting the memory file **and** its `MEMORY.md`
   index line. Never write a new memory in this project.
4. **Ask the user what should improve.** Before presenting your own list, ask
   them directly — they saw things the artifacts do not record: where the workflow
   felt slow or heavy, what they had to repeat, what they expected a stage to do
   and it did not, anything they found themselves working around. Ask it as an
   open question, wait for the answer, and treat what comes back as the primary
   input to step 5 rather than as a footnote to your own findings.
5. **Propose, small and specific.** Present a short numbered list — the user's
   items from step 4 first, then yours — each naming the target file and the exact
   change, ordered by value. Sort each into:
   - a durable convention → `CLAUDE.md`
   - workflow mechanics → the relevant `.claude/skills/*/SKILL.md` or
     `.claude/docs/sys-*.md`
   - a tooling gap → a `scripts/` helper
   - a code/test change → **not this stage**; offer `/design` instead
   If your own list is empty, say so plainly — a retro that invents work is worse
   than a short one. An empty list from *you* is still a retro if step 4 produced
   something.
6. **Discuss, then apply.** Wait for the user's agreement per item. Apply only
   approved items, with `Edit`/`Write`. Never apply an item the user did not
   accept, and never bundle an unapproved change into an approved one.
7. **Record.** Write `.work/<date>-<slug>/retro.md`: what the work item taught,
   what the user raised in step 4, what changed (file + one line each), what was
   declined and why, and any follow-up work items worth their own `/design`.
   Local-only — `.work/` is gitignored.
8. **Commit.** Honour the work item's commit mode from `plan.md`. `CLAUDE.md` and
   `.claude/**` edits are workflow config, not sc-framework changes: they need
   **no** `CHANGES.md` entry. In `manual` mode, list the changed files and leave
   them staged-free for the user; in `claude` mode, invoke `sys-commit`
   (step: retro, slug: `<slug>`) → message `retro: <slug>`.

## Constraints

- **Discussion first.** Never edit a workflow file before the user has agreed to
  that specific item. This is a discussion step, not an autonomous rewrite.
- **Never write to the memory store** in this project; the repo is the only home
  for project knowledge.
- **Prefer correcting an existing section over adding a new one.** `CLAUDE.md` is
  checked-in team documentation and keep-by-default: fix a stale rule in place
  rather than appending a competing one, and do not propose trimming it for
  "derivable" content.
- **Keep it to a handful of items.** A rule someone can follow, not an essay.
- **No code, tests, or notebook edits** — those go through
  `/design → /plan → /implement`.
