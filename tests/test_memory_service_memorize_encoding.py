from __future__ import annotations

import unittest
from pathlib import Path
from flockmem.testing.writable_tempdir import WritableTempDir

from flockmem.infra.sqlite.db import SQLiteEngine
from flockmem.infra.sqlite.init_schema import init_schema
from flockmem.service.extractor import ExtractedMemory
from flockmem.service.memory_service import MemorizeInput, MemoryService


class _StubVectorStore:
    enabled = False
    vector_dim = 4

    def search(self, **kwargs):  # pragma: no cover
        return []

    def upsert(self, *args, **kwargs):  # pragma: no cover
        return None


class _StubEmbeddingProvider:
    def embed(self, text: str) -> list[float]:  # pragma: no cover
        return [0.1, 0.2, 0.3, 0.4]


class _StubGraphStore:
    enabled = False

    def upsert_triples(self, *args, **kwargs):  # pragma: no cover
        return None


class _BrokenEpisodeExtractor:
    def extract(self, content: str, sender: str, group_id: str | None):
        return ExtractedMemory(
            episode="??????:???????????",
            summary="乱码摘要",
            subject=sender,
            importance_score=0.7,
            atomic_facts=[],
            foresights=[],
            profile_patch={},
        )


class _MetadataMutatingExtractor:
    def extract(self, content: str, sender: str, group_id: str | None):
        return ExtractedMemory(
            episode='[DIALOGUE] 回写内容\n[metadata] {"agentid":"agent-a"}',
            summary="metadata-mutation",
            subject=sender,
            importance_score=0.7,
            atomic_facts=[],
            foresights=[],
            profile_patch={},
        )


class MemoryServiceMemorizeEncodingTests(unittest.TestCase):
    def test_memorize_prefers_raw_payload_when_extracted_episode_is_broken(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            engine = SQLiteEngine(Path(tmp) / "lite.db")
            init_schema(engine)
            service = MemoryService(
                engine=engine,
                vector_store=_StubVectorStore(),
                embedding_provider=_StubEmbeddingProvider(),
                extractor=_BrokenEpisodeExtractor(),
                graph_store=_StubGraphStore(),
            )
            out = service.memorize(
                MemorizeInput(
                    message_id="enc-case-1",
                    create_time=1735603201,
                    sender="u1",
                    content="飞书",
                    group_id="default:u1",
                    group_name=None,
                    sender_name=None,
                    role="user",
                ),
                request_id="req-enc-case-1",
            )
            memory = out.get("memory", {})
            self.assertEqual("飞书", str(memory.get("episode", "")))

    def test_memorize_preserves_raw_metadata_keys_when_extractor_mutates_them(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            engine = SQLiteEngine(Path(tmp) / "lite.db")
            init_schema(engine)
            service = MemoryService(
                engine=engine,
                vector_store=_StubVectorStore(),
                embedding_provider=_StubEmbeddingProvider(),
                extractor=_MetadataMutatingExtractor(),
                graph_store=_StubGraphStore(),
            )
            raw_content = (
                '[DIALOGUE] 回写内容\n'
                '[metadata] {"agent_id":"agent-a","route_acl_fallback":true}'
            )
            out = service.memorize(
                MemorizeInput(
                    message_id="enc-case-2",
                    create_time=1735603201,
                    sender="u1",
                    content=raw_content,
                    group_id="default:u1",
                    group_name=None,
                    sender_name=None,
                    role="user",
                ),
                request_id="req-enc-case-2",
            )
            memory = out.get("memory", {})
            self.assertEqual(raw_content, str(memory.get("episode", "")))


if __name__ == "__main__":
    unittest.main()

