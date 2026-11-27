# CNN-in-automatic-drive

## Working Guidelines

1. Use `reference/` for supporting materials and examples.
2. Keep library and application packages under `src/`, tests in `tests/`, and notebooks in `examples/`.
3. This repository supports both macOS and Windows. Use portable paths and document platform-specific setup.
4. When a change is requested, first analyze it and provide code suggestions before editing files.
5. Keep changes and suggestions as simple as possible, and think twice before adding complexity.
6. Keep changes narrowly scoped and avoid broad structural changes.
7. When starting a task, continue on the current branch if it is related to the task and is not `master`. If the current branch is `master` or is unrelated to the task, create a new branch for the task from the current branch. Continue review fixes on the same branch.

## MyDeZero

- Keep implementations readable and close to the local DeZero reference. Document
  expected inputs and use NumPy's normal behavior unless a check fixes a concrete
  bug or the task requires it.
- Keep docstrings concise. Put detailed examples, gradient conventions and
  implementation comparisons in `doc/mydezero/`. Explain substantial documentation
  additions before writing them.
- Test behavior and correctness. When commits are requested, separate functionality,
  tests, documentation and etc.

## Testing Guidelines

1. After every task, proactively ask whether additional tests are needed and suggest relevant checks.
2. Run checks appropriate to the change and report results, skips, or reasons for not testing.
3. Keep tests focused on behavior and correctness.

## Documentation Guidelines

1. After every task, proactively ask whether documentation needs to be added or updated and suggest the relevant files.
2. Write new and updated documentation in concise English, except the root `README.md` and original course materials in `request/`.
3. Keep documentation suitable for Git: use shared setup instructions and relative paths; omit personal machine details and local environment status.
4. Keep documentation aligned with the implementation and avoid repeating information.
