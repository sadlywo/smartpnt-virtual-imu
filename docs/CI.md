# Continuous Integration

The repository includes a ready-to-enable GitHub Actions workflow template at
[`ci/github-actions.yml`](../ci/github-actions.yml).

The release account used for this upload has GitHub token scopes `gist`,
`read:org`, and `repo`, but not `workflow`. GitHub therefore rejects adding an
active workflow file until that scope is granted. After authorization, copy the
template to `.github/workflows/ci.yml` and commit it:

```bash
mkdir -p .github/workflows
cp ci/github-actions.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "ci: enable reproducible test workflow"
```

The same checks can always be run locally:

```bash
uv sync --frozen --extra dev
uv run python scripts/check_repository.py
uv run ruff check src scripts tests
uv run pytest -m "not integration"
```
