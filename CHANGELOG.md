# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

## [0.1.3] - 2026-03-06

### Changed

- Bumped and unified Python package version and OpenClaw plugin version to `0.1.3`.

## [0.1.1] - 2026-03-05

### Added

- Independent extractor provider configuration in runtime settings.
- Extractor runtime hot-reload via `PUT /api/v1/model-config`.
- Extractor connectivity checks in `/api/v1/model-config/test`.
- Atomic-fact-aware retrieval for better memory recall.
- Open-source community files: `LICENSE`, `CODE_OF_CONDUCT.md`.

### Fixed

- Admin-sensitive config routes now support bearer-based admin protection.
- Sensitive config fields are redacted in API responses.
- Config raw update flow preserves existing secrets when redacted placeholders are submitted.
- Unified auth header handling across OpenAI-compatible chat/extractor/retrieval clients.
- Wheel package now includes `flockmem/ui/index.html`.

### Changed

- Memory extraction pipeline now supports dedicated provider credentials.
- Memory relevance filtering includes `atomic_fact_text` for prompt context.

## [0.1.0] - 2026-02-16

### Added

- Local-first memory runtime (`SQLite + LanceDB + optional Kuzu graph`).
- Memory ingest/manage/chat/graph web console.
- Runtime-switchable chat and embedding providers.
- Memory citation and retrieval trace in chat responses.
- Basic CI (`pytest -q`) and issue/PR templates.

