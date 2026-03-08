from __future__ import annotations

import unittest

from tests.test_collective_contracts import (
    build_collective_client,
    ingest_success,
    skip_if_collective_routes_missing,
)


class CollectiveScopeAclBoundaryTests(unittest.TestCase):
    def test_write_acl_blocks_non_member_updates(self) -> None:
        runtime = build_collective_client(prefix="qa-acl-write")
        self.addCleanup(runtime["tmp"].cleanup)
        skip_if_collective_routes_missing(self, runtime["routes"], suite="collective-scope-acl")
        client = runtime["client"]

        ingest_success(
            self,
            client=client,
            knowledge_id="k-write-acl-1",
            scope_type="personal",
            scope_id="u-acl-write",
            actor_id="writer-a",
            write_acl=["writer-a"],
            read_acl=[],
        )

        blocked = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-write-acl-1",
                "scope_type": "personal",
                "scope_id": "u-acl-write",
                "content": {"text": "unauthorized update"},
                "change_type": "update",
                "changed_by": "agent",
                "actor_id": "writer-b",
            },
        )
        self.assertEqual(403, blocked.status_code, msg=blocked.text)

        allowed = client.post(
            "/api/v1/collective/ingest",
            json={
                "knowledge_id": "k-write-acl-1",
                "scope_type": "personal",
                "scope_id": "u-acl-write",
                "content": {"text": "authorized update"},
                "change_type": "update",
                "changed_by": "agent",
                "actor_id": "writer-a",
            },
        )
        self.assertEqual(200, allowed.status_code, msg=allowed.text)
        client.close()

    def test_read_acl_filters_context_by_actor(self) -> None:
        runtime = build_collective_client(prefix="qa-acl-read")
        self.addCleanup(runtime["tmp"].cleanup)
        skip_if_collective_routes_missing(self, runtime["routes"], suite="collective-scope-acl")
        client = runtime["client"]

        ingest_success(
            self,
            client=client,
            knowledge_id="k-read-acl-1",
            scope_type="personal",
            scope_id="u-acl-read",
            actor_id="writer-a",
            read_acl=["reader-a"],
            write_acl=["writer-a"],
        )

        allowed_ctx = client.post(
            "/api/v1/collective/context",
            json={
                "query": "acl read allowed",
                "actor_id": "reader-a",
                "personal_scope_id": "u-acl-read",
                "include_global": False,
            },
        )
        self.assertEqual(200, allowed_ctx.status_code, msg=allowed_ctx.text)
        allowed_items = allowed_ctx.json()["result"]["items"]
        self.assertEqual(1, len(allowed_items))
        self.assertEqual("k-read-acl-1", allowed_items[0]["knowledge_id"])

        denied_ctx = client.post(
            "/api/v1/collective/context",
            json={
                "query": "acl read denied",
                "actor_id": "reader-b",
                "personal_scope_id": "u-acl-read",
                "include_global": False,
            },
        )
        self.assertEqual(200, denied_ctx.status_code, msg=denied_ctx.text)
        self.assertEqual(0, denied_ctx.json()["result"]["count"])

        anonymous_ctx = client.post(
            "/api/v1/collective/context",
            json={
                "query": "acl no actor",
                "personal_scope_id": "u-acl-read",
                "include_global": False,
            },
        )
        self.assertEqual(200, anonymous_ctx.status_code, msg=anonymous_ctx.text)
        self.assertEqual(0, anonymous_ctx.json()["result"]["count"])
        client.close()

    def test_scope_priority_keeps_personal_team_global_order(self) -> None:
        runtime = build_collective_client(prefix="qa-scope-order")
        self.addCleanup(runtime["tmp"].cleanup)
        skip_if_collective_routes_missing(self, runtime["routes"], suite="collective-scope-acl")
        client = runtime["client"]

        ingest_success(
            self,
            client=client,
            knowledge_id="k-scope-personal",
            scope_type="personal",
            scope_id="u-priority",
            actor_id="scope-writer",
        )
        ingest_success(
            self,
            client=client,
            knowledge_id="k-scope-team",
            scope_type="team",
            scope_id="team-priority",
            actor_id="scope-writer",
        )
        ingest_success(
            self,
            client=client,
            knowledge_id="k-scope-global",
            scope_type="global",
            scope_id=None,
            actor_id="scope-writer",
        )

        context_response = client.post(
            "/api/v1/collective/context",
            json={
                "query": "scope order check",
                "actor_id": "scope-writer",
                "personal_scope_id": "u-priority",
                "team_scope_id": "team-priority",
                "include_global": True,
                "top_k": 10,
            },
        )
        self.assertEqual(200, context_response.status_code, msg=context_response.text)
        context_result = context_response.json()["result"]
        self.assertEqual(["personal", "team", "global"], context_result["scope_order"])
        self.assertEqual(3, context_result["count"])

        rank = {"personal": 0, "team": 1, "global": 2}
        scope_ranks = [rank[item["scope_type"]] for item in context_result["items"]]
        self.assertEqual(sorted(scope_ranks), scope_ranks)
        client.close()


if __name__ == "__main__":
    unittest.main()
