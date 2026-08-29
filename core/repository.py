"""Camada de dados. Hoje SQLite; a interface permite trocar por Firebase depois."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path


class Repository(ABC):
    """Interface de persistência. O resto do app só depende destes métodos."""

    @abstractmethod
    def add_sample(self, texto: str, contexto: str = "", tags: str = "") -> int: ...

    @abstractmethod
    def get_samples(self, contexto: str | None = None, limit: int = 3) -> list[dict]: ...

    @abstractmethod
    def add_rule(self, regra: str, categoria: str = "", peso: int = 1) -> int: ...

    @abstractmethod
    def get_rules(self) -> list[dict]: ...

    @abstractmethod
    def add_pair(self, antes: str, depois: str, contexto: str = "") -> int: ...

    @abstractmethod
    def get_pairs(self, limit: int = 2) -> list[dict]: ...

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
        with self._conn() as c:
            c.executescript(schema.read_text(encoding="utf-8"))

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- amostras de estilo ----
    def add_sample(self, texto: str, contexto: str = "", tags: str = "") -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO style_samples (texto, contexto, tags) VALUES (?, ?, ?)",
                (texto, contexto, tags),
            )
            return cur.lastrowid

    def get_samples(self, contexto: str | None = None, limit: int = 3) -> list[dict]:
        with self._conn() as c:
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

    # ---- regras ----
    def add_rule(self, regra: str, categoria: str = "", peso: int = 1) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO rules (regra, categoria, peso) VALUES (?, ?, ?)",
                (regra, categoria, peso),
            )
            return cur.lastrowid

    def get_rules(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM rules WHERE ativa = 1 ORDER BY peso DESC, id ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- pares antes/depois ----
    def add_pair(self, antes: str, depois: str, contexto: str = "") -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO rewrite_pairs (antes, depois, contexto) VALUES (?, ?, ?)",
                (antes, depois, contexto),
            )
            return cur.lastrowid

    def get_pairs(self, limit: int = 2) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM rewrite_pairs ORDER BY criado_em DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- gerações (loop de aprendizado) ----
    def save_generation(self, prompt_usado: str, entrada: str, saida: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO generations (prompt_usado, entrada, saida) VALUES (?, ?, ?)",
                (prompt_usado, entrada, saida),
            )
            return cur.lastrowid

    def rate_generation(self, gen_id: int, nota: int, editada: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE generations SET nota = ?, editada = ? WHERE id = ?",
                (nota, editada, gen_id),
            )

    def get_generation(self, gen_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM generations WHERE id = ?", (gen_id,)
            ).fetchone()
            return dict(row) if row else None
