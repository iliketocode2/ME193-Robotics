---
name: github-management
description: Chris's Simple Minded GitHub — version control vocabulary and Milan's branch-per-edit workflow for this course. Use whenever the user asks for help with git/GitHub (init a repo, commit, push, pull, branch, PR, merge, resolve a merge conflict, triage an issue, or "just handle my GitHub for me").
---

# Chris's Simple Minded GitHub

Reference and workflow for managing this project's Git/GitHub repo. The
target user is usually a student who wants Claude Code to "just do it" —
default to acting on their behalf, but always narrate what you're about to
do (branch names, what gets pushed) since pushes and merges are visible to
others and PRs/merges should be confirmed before you run them.

## Why version control

Every commit is a checkpoint you can always roll back to — you can try a
new approach to a controls script or FEA post-processing script without
fear of wrecking the version that already works. Since commits live in the
cloud on GitHub, a dead laptop, a corrupted file, or spilled coffee the
night before a deadline doesn't cost a semester of code.

Git is great for code (diffs only, so many commits stay cheap) and much
less efficient for binary files like CAD exports (each commit stores the
whole binary again). Keep large binaries out of the repo where possible —
use `.gitignore` for build artifacts/exports rather than committing them.

## Vocabulary

- **Repository** — a project folder plus its full history, existing in two
  places at once: the laptop and the GitHub cloud.
- **Clone** — download a copy of a remote repo to your computer for the
  first time. Must happen before you can commit/push to it.
- **Commit** — save changes locally. Does **not** push anything to the
  cloud.
- **Push** — send local commits up to the cloud.
- **Pull** — download other people's commits from the cloud into your local
  copy. Anyone working in a pair/group should pull before starting work
  each day to avoid painful conflicts later.
- **Merge conflict** — two people edited the same lines; Git can't pick a
  winner automatically and will ask you to manually choose which version
  (or combine them). Expect this in group work — it's normal, not a
  failure.
- **Fork** — your own copy of someone else's repo, without write access
  back to theirs. Use this when you just need to consume/use their library,
  not improve it.
- **Branch** — your own copy of a repo (yours or someone else's) *with*
  write access. Use this when you intend to improve the code.
- **Pull Request (PR)** — a request asking the repo owner to merge your
  branch's improvements into theirs.
- **Merge** — combining the old code with the new code from a branch/PR.
- **README.md** — every repo should have one describing what the code
  does. Create/update it when starting a new project or adding a major
  feature.
- **.gitignore** — tells Git to ignore files that shouldn't be tracked
  (build artifacts, data dumps, credentials, `my_env/`, `__pycache__/`,
  etc.). Check this exists and is populated before the first commit.

## Milan's GitHub rules (the workflow to follow)

This is the actual loop to run whenever the user wants to make a change:

1. **Fetch/pull** the main branch (or the relevant sub-branch) before
   starting any edits.
2. **Create a specific branch** for one type of edit only — e.g. a
   `red_color_detection` branch should contain *only* red color detection
   code, nothing else. Name branches after the specific change, not
   generically ("fix-bug" is too vague).
3. **Commit often** while working on that branch — every time there's a
   state worth being able to roll back to.
4. **Push at the end of each work session** so the cloud always has a
   backup copy, even mid-feature.
5. **Test the branch** before merging — run it through the relevant tests
   so you're confident it works as expected.
6. **Merge** (directly, if the user has write access to the target repo)
   or **open a Pull Request** (if they don't, e.g. contributing to someone
   else's repo).
7. **Delete the branch** after it's merged, then go back to step 1 for the
   next change.

One branch = one focused change. Don't let unrelated edits pile onto a
branch that was opened for something else — start a new branch instead.

### Issues

Use the repo's Issues tab for bugs/requests found during testing: one
issue per problem, not a batch. When fixing an issue, create a branch
directly from it (steps above), reference the issue number in commits/PR
description, and close the issue once the fix is merged.

## What to actually do when the user asks for GitHub help

- **"Set up a repo / start tracking this"** → check for an existing
  README.md and .gitignore, create them if missing, `git init` (or clone,
  if there's a remote to start from), first commit.
- **"Save my work" / "commit this"** → `git status` first, review what's
  staged (watch for secrets/credentials before adding), commit with a
  message describing the *why*, following the repo's existing commit
  message style.
- **"Back this up" / "push"** → push the current branch. Confirm the
  target branch/remote with the user if ambiguous.
- **"I want to add/fix X"** → pull main, create a focused branch named for
  X, do the work there, commit as you go.
- **"This is ready" / "merge it"** → confirm tests pass, then merge or open
  a PR depending on write access, and offer to delete the branch after.
- **Merge conflicts** → explain in plain terms which lines conflict and
  what each side changed, then let the user decide (or apply their stated
  preference) rather than silently picking a side.
- Always confirm before push, merge, force-push, or any action visible to
  collaborators — these aren't easily reversible the way local commits are.
