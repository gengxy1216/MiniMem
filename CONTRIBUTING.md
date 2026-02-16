# Contributing to MiniMem

Thanks for your interest in improving MiniMem.

## Development Setup

```bash
pip install -e .
pytest -q
```

Run locally:

```bash
python main.py
```

UI: `http://127.0.0.1:20195/ui`

## Pull Request Rules

- Keep PRs focused and small.
- Add or update tests for behavior changes.
- Do not submit mock-based core logic for chat or memory write paths.
- Keep UI changes responsive on desktop and mobile.
- Follow `CODE_OF_CONDUCT.md`.

## Commit and Review

- Use clear commit messages with intent and scope.
- Include reproduction and verification steps in PR description.
- Link related issues when applicable.

## Release Notes

If your PR changes user-visible behavior, add notes in `CHANGELOG.md` under `Unreleased`.

## Code Style

- Prefer simple and explicit code.
- Keep comments concise and only where needed.
- Preserve existing API behavior unless change is intentional and documented.
