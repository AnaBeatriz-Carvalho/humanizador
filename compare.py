"""Compara dois (ou mais) modelos do LM Studio no MESMO texto e MESMO prompt.

Carrega um modelo por vez (VRAM de 8 GB não comporta os dois juntos),
cronometra a geração e imprime os resultados lado a lado.

Uso:
  python compare.py "texto com cheiro de IA..."
  python compare.py --contexto blog "texto..."
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from core.llm_client import LLMClient
from core.prompt_builder import PromptBuilder
from core.repository import SqliteRepository

DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"
LMS = Path.home() / ".lmstudio" / "bin" / "lms.exe"

# Modelos a comparar (identificadores exatos do 'lms ls').
MODELOS = ["gemma-2-9b-it", "meta-llama-3.1-8b-instruct"]


def lms(*args: str) -> None:
    # encoding/errors evita UnicodeDecodeError com a barra de progresso do lms.
    subprocess.run(
        [str(LMS), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )


def rodar_modelo(modelo: str, prompt: str, texto: str) -> dict:
    lms("unload", "--all")
    t_load = time.perf_counter()
    lms("load", modelo, "--gpu", "max")
    load_s = time.perf_counter() - t_load

    client = LLMClient(model=modelo)
    t_gen = time.perf_counter()
    saida = client.humanize(prompt, texto)
    gen_s = time.perf_counter() - t_gen

    return {"modelo": modelo, "saida": saida, "load_s": load_s, "gen_s": gen_s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("texto")
    ap.add_argument("--contexto", default=None)
    args = ap.parse_args()

    repo = SqliteRepository(DB_PATH)
    prompt = PromptBuilder(repo).build(contexto=args.contexto)

    resultados = []
    for modelo in MODELOS:
        print(f"\n>>> Carregando e testando: {modelo} ...")
        try:
            resultados.append(rodar_modelo(modelo, prompt, args.texto))
        except Exception as e:
            resultados.append({"modelo": modelo, "saida": f"[ERRO: {e}]",
                               "load_s": 0, "gen_s": 0})

    print("\n" + "=" * 70)
    print(f"TEXTO ORIGINAL:\n{args.texto}")
    for r in resultados:
        print("\n" + "=" * 70)
        print(f"MODELO: {r['modelo']}   "
              f"(load {r['load_s']:.1f}s | geração {r['gen_s']:.1f}s)")
        print("-" * 70)
        print(r["saida"])
    print("\n" + "=" * 70)
    lms("unload", "--all")


if __name__ == "__main__":
    main()
