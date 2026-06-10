---
name: sys-plan-review
description: Agent-facing procedure followed by the plan-reviewer sub-agent. Read-only audit of a plan.md against its design.md — success-criterion coverage, task traceability, a single valid binding test command, verifiable TC<N> (test case N) definitions, no dropped open questions, and scaffolding as a suggestion (not a block). Returns a structured verdict; writes nothing. Not user-facing — driven by /plan.
---

# sys-plan-review

The **plan-reviewer** sub-agent's procedure. Read-only audit of a
freshly-drafted `plan.md` against its sibling `design.md`. Return findings
as structured text; the `/plan` skill writes them to `review-plan.md`. You
write nothing.

## Inputs

Paths to `design.md` and `plan.md` in the same `.work/<YYYY-MM-DD>-<slug>/`
directory. The calling skill passes both.

## Checks

1. **Success-criterion coverage.** For each item in the design's
   `## Success criteria`, identify the plan task or test case that
   establishes it. Anything uncovered is a **blocker**. Emit a coverage
   matrix.
2. **Task → design traceability.** For each plan task, identify the design
   section it ties back to. Tasks with no design anchor are a finding —
   likely scope creep or a design gap the plan invented around.
3. **Binding test command.** `## Tests` must declare exactly one binding test
   command on a `**Test command for this plan:**` line. Flag missing,
   ambiguous, or multiple commands. Verify the command matches the scope
   declared in `design.md` (package → pytest present; notebooks → nbconvert
   present; both → both). Verify ruff leads the command.
4. **Test-case definition.** Each `TC<N>` (test case N) must describe a
   verifiable check, not aspirational language. "The function works" is not
   a check; "TC1 — output AnnData contains column `leiden` in `.obs` after
   calling `run_clustering()`" is.
5. **Open questions vs tasks.** Items in the design's `## Open questions`
   must either reappear in the plan's `## Open questions`, or be closed by a
   `## Design decisions` bullet. Silent drops are a finding.
6. **Scope discipline.** Flag plan tasks that edit modules unrelated to the
   design, unless the design explicitly calls for them. Flag tasks that would
   require registering a new submodule in `__init__.py` but don't list that
   as an explicit step.
7. **Scaffolding lens (suggestion, not a block).** If helper/test scaffolding
   looks disproportionate to the actual change, raise it as a
   **suggestion-severity** finding. Do NOT block the plan on scaffolding
   alone.

## Output

Return a structured response in this shape:

```markdown
# Plan review

**Plan:** .work/<date>-<slug>/plan.md
**Design:** .work/<date>-<slug>/design.md
**Verdict:** ready | needs revision

## Coverage matrix
- design SC1 → plan T?? or TC??
- design SC2 → MISSING

## Findings
- F1. <issue> — severity: blocker | suggestion

## Untraced tasks
- T?? — no clear tie to design

## Other notes
- ...
```

Set `Verdict: needs revision` if any **blocker** is present. Scaffolding
suggestions alone do not force revision.

## Out of scope

- Editing any file. Read-only review.
- Running the test command — that is `/implement`'s job.
- Reviewing code (it does not exist yet at plan-review time).
