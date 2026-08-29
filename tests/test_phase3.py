from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.prompt_builder import PromptBuilder
from core.repository import SqliteRepository


class FakeEmbeddingClient:
    def __init__(self, vector: list[float], available: bool = True):
        self.vector = vector
        self.is_available = available

    def available(self) -> bool:
        return self.is_available

    def embed(self, texto: str) -> list[float]:
        return self.vector


class Phase3Tests(unittest.TestCase):
    def setUp(self) -> None:
        # ignore_cleanup_errors: no Windows o handle do sqlite pode demorar a
        # ser liberado pelo GC; a limpeza do tempdir não deve mascarar o teste.
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "app.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_migrates_old_database_without_losing_rows(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE style_samples (
                    id INTEGER PRIMARY KEY, texto TEXT NOT NULL,
                    contexto TEXT, tags TEXT, criado_em TEXT
                );
                CREATE TABLE rewrite_pairs (
                    id INTEGER PRIMARY KEY, antes TEXT NOT NULL,
                    depois TEXT NOT NULL, contexto TEXT, criado_em TEXT
                );
                CREATE TABLE rules (
                    id INTEGER PRIMARY KEY, regra TEXT NOT NULL,
                    categoria TEXT, peso INTEGER DEFAULT 1, ativa INTEGER DEFAULT 1
                );
                CREATE TABLE generations (
                    id INTEGER PRIMARY KEY, prompt_usado TEXT, entrada TEXT,
                    saida TEXT, nota INTEGER, editada TEXT, criado_em TEXT
                );
                INSERT INTO style_samples (texto) VALUES ('legado');
                """
            )

        repo = SqliteRepository(self.db_path)
        with repo._session() as conn:
            sample_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(style_samples)")
            }
            pair_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(rewrite_pairs)")
            }
        self.assertIn("embedding", sample_columns)
        self.assertIn("embedding", pair_columns)
        self.assertEqual(repo.get_samples(limit=1)[0]["texto"], "legado")

    def test_cosine_similarity_and_context_filter(self) -> None:
        repo = SqliteRepository(self.db_path)
        repo.add_sample("culinária", contexto="blog", embedding=[1.0, 0.0])
        repo.add_sample("programação", contexto="blog", embedding=[0.0, 1.0])
        repo.add_sample("outro contexto", contexto="email", embedding=[1.0, 0.0])
        repo.add_pair("antes cozinha", "depois cozinha", embedding=[1.0, 0.0])
        repo.add_pair("antes código", "depois código", embedding=[0.0, 1.0])

        samples = repo.get_samples_by_similarity([0.9, 0.1], "blog", limit=1)
        pairs = repo.get_pairs_by_similarity([0.9, 0.1], limit=1)

        self.assertEqual(samples[0]["texto"], "culinária")
        self.assertEqual(pairs[0]["depois"], "depois cozinha")

    def test_prompt_uses_similarity_and_falls_back_when_unavailable(self) -> None:
        repo = SqliteRepository(self.db_path)
        repo.add_sample("amostra relevante", embedding=[1.0, 0.0])
        repo.add_pair("antes relevante", "depois relevante", embedding=[1.0, 0.0])

        semantic_prompt = PromptBuilder(
            repo, FakeEmbeddingClient([1.0, 0.0])
        ).build(texto_alvo="consulta")
        fallback_prompt = PromptBuilder(
            repo, FakeEmbeddingClient([1.0, 0.0], available=False)
        ).build(texto_alvo="consulta")

        self.assertIn("amostra relevante", semantic_prompt)
        self.assertIn("depois relevante", semantic_prompt)
        self.assertIn("amostra relevante", fallback_prompt)

    def test_reindex_storage_is_idempotent(self) -> None:
        repo = SqliteRepository(self.db_path)
        sample_id = repo.add_sample("sem vetor")
        pair_id = repo.add_pair("antes", "depois")

        pending = repo.iter_rows_without_embedding()
        self.assertEqual(len(pending), 2)
        for row in pending:
            repo.save_embedding(row["row_type"], row["id"], [1.0, 0.0])

        self.assertEqual(repo.iter_rows_without_embedding(), [])
        self.assertEqual(
            repo.get_samples_by_similarity([1.0, 0.0], limit=1)[0]["id"],
            sample_id,
        )
        self.assertEqual(
            repo.get_pairs_by_similarity([1.0, 0.0], limit=1)[0]["id"],
            pair_id,
        )


if __name__ == "__main__":
    unittest.main()
