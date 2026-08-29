"""Camada de dados. Hoje SQLite; a interface permite trocar por Firebase depois."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class Repository(ABC):
    """Interface de persistência. O resto do app só depende destes métodos."""

    @abstractmethod
    def add_sample(
        self,
        texto: str,
        contexto: str = "",
        tags: str = "",
        embedding: list[float] | None = None,
    ) -> int: ...

    @abstractmethod
    def get_samples(self, contexto: str | None = None, limit: int = 3) -> list[dict]: ...

    @abstractmethod
    def get_samples_by_similarity(
        self,
        query_embedding: list[float],
        contexto: str | None = None,
        limit: int = 3,
    ) -> list[dict]: ...

    @abstractmethod
    def add_rule(self, regra: str, categoria: str = "", peso: int = 1) -> int: ...

    @abstractmethod
    def get_rules(self) -> list[dict]: ...

    @abstractmethod
    def add_pair(
        self,
        antes: str,
        depois: str,
        contexto: str = "",
        embedding: list[float] | None = None,
    ) -> int: ...

    @abstractmethod
    def get_pairs(self, limit: int = 2) -> list[dict]: ...

    @abstractmethod
    def get_pairs_by_similarity(
        self, query_embedding: list[float], limit: int = 2
    ) -> list[dict]: ...

    @abstractmethod
    def iter_rows_without_embedding(self) -> list[dict]: ...

    @abstractmethod
    def save_embedding(
        self, row_type: str, row_id: int, embedding: list[float]
    ) -> None: ...

    @abstractmethod
    def save_generation(self, prompt_usado: str, entrada: str, saida: str) -> int: ...

    @abstractmethod
    def rate_generation(self, gen_id: int, nota: int, editada: str = "") -> None: ...

    @abstractmethod
    def get_generation(self, gen_id: int) -> dict | None: ...


class SqliteRepository(Repository):
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        schema = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
        with self._session() as c:
            c.executescript(schema.read_text(encoding="utf-8"))
            self._migrate_embeddings(c)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _session(self):
        """Abre uma conexão que faz commit E fecha (libera o handle no Windows)."""
        conn = self._conn()
        try:
            with conn:  # commit em sucesso, rollback em exceção
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _migrate_embeddings(conn: sqlite3.Connection) -> None:
        """Adiciona as colunas da Fase 3 sem recriar bancos antigos."""
        for table in ("style_samples", "rewrite_pairs"):
            columns = {
                row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
            }
            if "embedding" not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN embedding TEXT")

    @staticmethod
    def _serialize_embedding(embedding: list[float] | None) -> str | None:
        if embedding is None:
            return None
        return json.dumps(embedding, separators=(",", ":"))

    @staticmethod
    def _rank_by_similarity(
        rows: list[sqlite3.Row], query_embedding: list[float], limit: int
    ) -> list[dict]:
        if limit <= 0:
            return []

        query = np.asarray(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query)
        if query.ndim != 1 or query.size == 0 or query_norm == 0:
            return []

        ranked: list[tuple[float, dict]] = []
        for row in rows:
            try:
                vector = np.asarray(json.loads(row["embedding"]), dtype=float)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            vector_norm = np.linalg.norm(vector)
            if vector.shape != query.shape or vector_norm == 0:
                continue
            score = float(np.dot(query, vector) / (query_norm * vector_norm))
            ranked.append((score, dict(row)))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in ranked[:limit]]

    # ---- amostras de estilo ----
    def add_sample(
        self,
        texto: str,
        contexto: str = "",
        tags: str = "",
        embedding: list[float] | None = None,
    ) -> int:
        with self._session() as c:
            cur = c.execute(
                "INSERT INTO style_samples (texto, contexto, tags, embedding) "
                "VALUES (?, ?, ?, ?)",
                (texto, contexto, tags, self._serialize_embedding(embedding)),
            )
            return cur.lastrowid

    def get_samples(self, contexto: str | None = None, limit: int = 3) -> list[dict]:
        with self._session() as c:
            if contexto:
                rows = c.execute(
                    "SELECT * FROM style_samples WHERE contexto = ? "
                    "ORDER BY criado_em DESC LIMIT ?",
                    (contexto, limit),
                ).fetchall()
                if rows:
                    return [dict(r) for r in rows]
            rows = c.execute(
                "SELECT * FROM style_samples ORDER BY criado_em DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_samples_by_similarity(
        self,
        query_embedding: list[float],
        contexto: str | None = None,
        limit: int = 3,
    ) -> list[dict]:
        with self._session() as c:
            rows: list[sqlite3.Row] = []
            if contexto:
                rows = c.execute(
                    "SELECT * FROM style_samples "
                    "WHERE contexto = ? AND embedding IS NOT NULL "
                    "ORDER BY criado_em DESC, id DESC",
                    (contexto,),
                ).fetchall()
            if not rows:
                rows = c.execute(
                    "SELECT * FROM style_samples WHERE embedding IS NOT NULL "
                    "ORDER BY criado_em DESC, id DESC"
                ).fetchall()
        return self._rank_by_similarity(rows, query_embedding, limit)

    # ---- regras ----
    def add_rule(self, regra: str, categoria: str = "", peso: int = 1) -> int:
        with self._session() as c:
            cur = c.execute(
                "INSERT INTO rules (regra, categoria, peso) VALUES (?, ?, ?)",
                (regra, categoria, peso),
            )
            return cur.lastrowid

    def get_rules(self) -> list[dict]:
        with self._session() as c:
            rows = c.execute(
                "SELECT * FROM rules WHERE ativa = 1 ORDER BY peso DESC, id ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- pares antes/depois ----
    def add_pair(
        self,
        antes: str,
        depois: str,
        contexto: str = "",
        embedding: list[float] | None = None,
    ) -> int:
        with self._session() as c:
            cur = c.execute(
                "INSERT INTO rewrite_pairs (antes, depois, contexto, embedding) "
                "VALUES (?, ?, ?, ?)",
                (antes, depois, contexto, self._serialize_embedding(embedding)),
            )
            return cur.lastrowid

    def get_pairs(self, limit: int = 2) -> list[dict]:
        with self._session() as c:
            rows = c.execute(
                "SELECT * FROM rewrite_pairs ORDER BY criado_em DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_pairs_by_similarity(
        self, query_embedding: list[float], limit: int = 2
    ) -> list[dict]:
        with self._session() as c:
            rows = c.execute(
                "SELECT * FROM rewrite_pairs WHERE embedding IS NOT NULL "
                "ORDER BY criado_em DESC, id DESC"
            ).fetchall()
        return self._rank_by_similarity(rows, query_embedding, limit)

    # ---- indexação semântica ----
    def iter_rows_without_embedding(self) -> list[dict]:
        with self._session() as c:
            samples = c.execute(
                "SELECT id, texto FROM style_samples WHERE embedding IS NULL"
            ).fetchall()
            pairs = c.execute(
                "SELECT id, depois AS texto FROM rewrite_pairs WHERE embedding IS NULL"
            ).fetchall()
        return [
            {"row_type": "sample", "id": row["id"], "texto": row["texto"]}
            for row in samples
        ] + [
            {"row_type": "pair", "id": row["id"], "texto": row["texto"]}
            for row in pairs
        ]

    def save_embedding(
        self, row_type: str, row_id: int, embedding: list[float]
    ) -> None:
        tables = {"sample": "style_samples", "pair": "rewrite_pairs"}
        try:
            table = tables[row_type]
        except KeyError as exc:
            raise ValueError(f"Tipo de linha inválido: {row_type}") from exc
        with self._session() as c:
            c.execute(
                f"UPDATE {table} SET embedding = ? WHERE id = ?",
                (self._serialize_embedding(embedding), row_id),
            )

    # ---- gerações (loop de aprendizado) ----
    def save_generation(self, prompt_usado: str, entrada: str, saida: str) -> int:
        with self._session() as c:
            cur = c.execute(
                "INSERT INTO generations (prompt_usado, entrada, saida) VALUES (?, ?, ?)",
                (prompt_usado, entrada, saida),
            )
            return cur.lastrowid

    def rate_generation(self, gen_id: int, nota: int, editada: str = "") -> None:
        with self._session() as c:
            c.execute(
                "UPDATE generations SET nota = ?, editada = ? WHERE id = ?",
                (nota, editada, gen_id),
            )

    def get_generation(self, gen_id: int) -> dict | None:
        with self._session() as c:
            row = c.execute(
                "SELECT * FROM generations WHERE id = ?", (gen_id,)
            ).fetchone()
            return dict(row) if row else None
