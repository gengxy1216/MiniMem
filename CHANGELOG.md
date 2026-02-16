# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Added

- Independent extractor provider configuration in runtime settings.
- Extractor runtime hot-reload via `PUT /api/v1/model-config`.
- Extractor connectivity checks in `/api/v1/model-config/test`.
- Atomic-fact-aware retrieval for better memory recall.
- Open-source community files: `LICENSE`, `CODE_OF_CONDUCT.md`.

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
