from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from flockmem.config.openclaw_primary_sync import (
    _default_minimem_config_path,
    detect_primary_model_snapshot,
    sync_openclaw_primary_to_minimem_config,
    to_public_primary_snapshot,
)
from flockmem.testing.writable_tempdir import WritableTempDir


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class OpenClawPrimarySyncTests(unittest.TestCase):
    def test_default_minimem_config_path_uses_user_config_dir(self) -> None:
        env = {}
        if os.name == "nt":
            env["USERPROFILE"] = r"C:\Users\tester"
            expected = Path(r"C:\Users\tester\.minimem\config.json").resolve()
        else:
            env["HOME"] = "/home/tester"
            expected = Path("/home/tester/.minimem/config.json").resolve()
        with patch.dict(os.environ, env, clear=True):
            actual = _default_minimem_config_path().resolve()
        self.assertEqual(expected, actual)

    def test_detect_primary_from_models_reference(self) -> None:
        cfg = {
            "models": {
                "primary": "main",
                "main": {
                    "provider": "siliconflow",
                    "base_url": "https://api.example/v1",
                    "api_key": "k-main",
                    "model": "Qwen/Qwen3-32B",
                },
            }
        }
        snap = detect_primary_model_snapshot(cfg)
        self.assertEqual("siliconflow", snap.get("provider"))
        self.assertEqual("https://api.example/v1", snap.get("base_url"))
        self.assertEqual("Qwen/Qwen3-32B", snap.get("model"))
        public_snap = to_public_primary_snapshot(snap)
        self.assertNotIn("api_key", public_snap)
        self.assertEqual("Qwen/Qwen3-32B", public_snap.get("model"))

    def test_sync_applies_snapshot_to_minimem_config(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            openclaw_path = root / "openclaw.json"
            minimem_path = root / "config.json"
            _write_json(
                openclaw_path,
                {
                    "model": {
                        "primary": {
                            "provider": "openai",
                            "base_url": "https://chat.example/v1",
                            "api_key": "chat-key",
                            "model": "gpt-4.1-mini",
                        }
                    }
                },
            )

            result = sync_openclaw_primary_to_minimem_config(
                openclaw_config_path=openclaw_path,
                minimem_config_path=minimem_path,
            )

            self.assertTrue(result["applied"])
            self.assertEqual("applied", result["status"])
            payload = _read_json(minimem_path)
            models = payload.get("models", {})
            self.assertEqual("openai", models.get("chat", {}).get("provider"))
            self.assertEqual("https://chat.example/v1", models.get("chat", {}).get("base_url"))
            self.assertEqual("gpt-4.1-mini", models.get("chat", {}).get("model"))
            self.assertEqual("chat_model", models.get("extractor", {}).get("provider"))
            self.assertTrue(str(models.get("embedding", {}).get("provider", "")).strip())
            self.assertTrue(str(models.get("embedding", {}).get("model", "")).strip())
            self.assertIsInstance(models.get("embedding"), dict)
            self.assertIsInstance(models.get("chat"), dict)
            self.assertIsInstance(models.get("extractor"), dict)
            self.assertIsInstance(models.get("rerank"), dict)
            self.assertNotIn("chat_provider", models)
            self.assertNotIn("chat_model", models)
            meta = payload.get("integration", {}).get("openclaw", {})
            self.assertEqual("applied", meta.get("last_sync_status"))
            self.assertIn("last_applied_models", meta)
            self.assertIn("runtime_model_snapshot", meta)

    def test_sync_skips_when_manual_override_detected(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            openclaw_path = root / "openclaw.json"
            minimem_path = root / "config.json"
            _write_json(
                openclaw_path,
                {
                    "model": {
                        "primary": {
                            "provider": "openai",
                            "base_url": "https://chat.example/v1",
                            "api_key": "chat-key",
                            "model": "model-a",
                        }
                    }
                },
            )
            first = sync_openclaw_primary_to_minimem_config(
                openclaw_config_path=openclaw_path,
                minimem_config_path=minimem_path,
            )
            self.assertEqual("applied", first["status"])

            payload = _read_json(minimem_path)
            payload["models"]["chat"]["model"] = "manual-model-x"
            _write_json(minimem_path, payload)

            _write_json(
                openclaw_path,
                {
                    "model": {
                        "primary": {
                            "provider": "openai",
                            "base_url": "https://chat.example/v1",
                            "api_key": "chat-key",
                            "model": "model-b",
                        }
                    }
                },
            )

            second = sync_openclaw_primary_to_minimem_config(
                openclaw_config_path=openclaw_path,
                minimem_config_path=minimem_path,
            )
            self.assertEqual("skipped_manual_override", second["status"])
            current = _read_json(minimem_path)
            self.assertEqual("manual-model-x", current.get("models", {}).get("chat", {}).get("model"))

            third = sync_openclaw_primary_to_minimem_config(
                openclaw_config_path=openclaw_path,
                minimem_config_path=minimem_path,
                force=True,
            )
            self.assertEqual("applied", third["status"])
            forced = _read_json(minimem_path)
            self.assertEqual("model-b", forced.get("models", {}).get("chat", {}).get("model"))

    def test_sync_respects_inherit_disabled(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            openclaw_path = root / "openclaw.json"
            minimem_path = root / "config.json"
            _write_json(
                openclaw_path,
                {
                    "model": {
                        "primary": {
                            "provider": "openai",
                            "base_url": "https://chat.example/v1",
                            "api_key": "chat-key",
                            "model": "model-a",
                        }
                    }
                },
            )
            result = sync_openclaw_primary_to_minimem_config(
                openclaw_config_path=openclaw_path,
                minimem_config_path=minimem_path,
                inherit_primary_model=False,
            )
            self.assertEqual("inherit_disabled", result["status"])
            payload = _read_json(minimem_path)
            models = payload.get("models", {})
            self.assertNotEqual("model-a", models.get("chat", {}).get("model"))

    def test_detect_primary_from_agents_defaults_model_object(self) -> None:
        cfg = {
            "agents": {
                "defaults": {
                    "model": {
                        "primary": "custom-provider/model-x",
                    }
                }
            }
        }
        snap = detect_primary_model_snapshot(cfg)
        self.assertEqual("custom-provider", snap.get("provider"))
        self.assertEqual("custom-provider/model-x", snap.get("model"))

    def test_detect_primary_enriches_credentials_from_models_providers(self) -> None:
        cfg = {
            "models": {
                "providers": {
                    "custom-provider": {
                        "baseUrl": "https://api.custom.example/v1",
                        "apiKey": "custom-key-1",
                    }
                }
            },
            "agents": {
                "defaults": {
                    "model": {
                        "primary": "custom-provider/model-x",
                    }
                }
            },
        }
        snap = detect_primary_model_snapshot(cfg)
        self.assertEqual("custom-provider", snap.get("provider"))
        self.assertEqual("https://api.custom.example/v1", snap.get("base_url"))
        self.assertEqual("custom-key-1", snap.get("api_key"))
        self.assertEqual("custom-provider/model-x", snap.get("model"))

    def test_sync_applies_credentials_from_models_providers(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            openclaw_path = root / "openclaw.json"
            minimem_path = root / "config.json"
            _write_json(
                openclaw_path,
                {
                    "models": {
                        "providers": {
                            "custom-provider": {
                                "baseUrl": "https://api.custom.example/v1",
                                "apiKey": "custom-key-1",
                            }
                        }
                    },
                    "agents": {
                        "defaults": {
                            "model": {
                                "primary": "custom-provider/model-x",
                            }
                        }
                    },
                },
            )

            result = sync_openclaw_primary_to_minimem_config(
                openclaw_config_path=openclaw_path,
                minimem_config_path=minimem_path,
            )

            self.assertTrue(result["applied"])
            payload = _read_json(minimem_path)
            models = payload.get("models", {})
            self.assertEqual("custom-provider", models.get("chat", {}).get("provider"))
            self.assertEqual("https://api.custom.example/v1", models.get("chat", {}).get("base_url"))
            self.assertEqual("custom-key-1", models.get("chat", {}).get("api_key"))
            self.assertEqual("custom-provider/model-x", models.get("chat", {}).get("model"))

    def test_sync_does_not_override_existing_rerank_fields(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            openclaw_path = root / "openclaw.json"
            minimem_path = root / "config.json"
            _write_json(
                openclaw_path,
                {
                    "model": {
                        "primary": {
                            "provider": "openai",
                            "base_url": "https://chat.example/v1",
                            "api_key": "chat-key",
                            "model": "gpt-4.1-mini",
                        }
                    }
                },
            )
            _write_json(
                minimem_path,
                {
                    "version": 1,
                    "settings": {},
                    "models": {
                        "chat": {
                            "provider": "openai",
                            "base_url": "https://chat.old/v1",
                            "api_key": "old-chat-key",
                            "model": "old-chat-model",
                        },
                        "embedding": {
                            "provider": "openai",
                            "base_url": "https://embed.example/v1",
                            "api_key": "embed-key",
                            "model": "embed-model",
                        },
                        "extractor": {
                            "provider": "chat_model",
                            "base_url": "https://chat.old/v1",
                            "api_key": "old-chat-key",
                            "model": "old-chat-model",
                        },
                        "rerank": {
                            "provider": "siliconflow",
                            "base_url": "https://rerank.example/v1",
                            "api_key": "rerank-key",
                            "model": "rerank-model",
                        },
                    },
                },
            )

            result = sync_openclaw_primary_to_minimem_config(
                openclaw_config_path=openclaw_path,
                minimem_config_path=minimem_path,
                force=True,
            )
            self.assertEqual("applied", result["status"])
            payload = _read_json(minimem_path)
            rerank = payload.get("models", {}).get("rerank", {})
            self.assertEqual("siliconflow", rerank.get("provider"))
            self.assertEqual("https://rerank.example/v1", rerank.get("base_url"))
            self.assertEqual("rerank-key", rerank.get("api_key"))
            self.assertEqual("rerank-model", rerank.get("model"))

    def test_runtime_snapshot_strips_provider_prefix_to_model_id(self) -> None:
        with WritableTempDir(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            openclaw_path = root / "openclaw.json"
            minimem_path = root / "config.json"
            _write_json(
                openclaw_path,
                {
                    "models": {
                        "providers": {
                            "custom-provider": {
                                "baseUrl": "https://api.custom.example/v2",
                                "apiKey": "custom-key-1",
                                "models": [{"id": "model-x"}],
                            }
                        }
                    },
                    "agents": {
                        "defaults": {
                            "model": {
                                "primary": "custom-provider/model-x",
                            }
                        }
                    },
                },
            )

            result = sync_openclaw_primary_to_minimem_config(
                openclaw_config_path=openclaw_path,
                minimem_config_path=minimem_path,
                force=True,
            )
            self.assertEqual("applied", result["status"])
            runtime_snapshot = result.get("runtime_model_snapshot", {})
            self.assertEqual("model-x", runtime_snapshot.get("model"))
            payload = _read_json(minimem_path)
            self.assertEqual("model-x", payload.get("models", {}).get("chat", {}).get("model"))


if __name__ == "__main__":
    unittest.main()

