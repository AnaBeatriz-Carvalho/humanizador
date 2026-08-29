"""CLI do humanizador.

Uso:
  python main.py humanizar "texto..."         reescreve um texto
  python main.py humanizar --contexto blog "..."
  python main.py add-sample "meu texto..."    guarda uma amostra da sua escrita
  python main.py add-rule "não use gerúndio"  adiciona uma regra ao prompt
  python main.py add-pair "antes" "depois"    guarda um par de reescrita
  python main.py avaliar 12 5 --editada "..." dá nota (1-5) a uma geração
  python main.py preview                       mostra o prompt montado
  python main.py ping                          testa a conexão com o LM Studio
  python main.py reindex                       indexa exemplos antigos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.embeddings import EmbeddingClient
from core.llm_client import LLMClient
from core.prompt_builder import PromptBuilder
from core.repository import SqliteRepository

DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"


def _print(titulo: str, corpo: str) -> None:
    print(f"\n=== {titulo} ===\n{corpo}\n")


def _embedding_or_none(texto: str) -> list[float] | None:
    """Gera um vetor sem transformar indisponibilidade em erro da CLI."""
    try:
        return EmbeddingClient().embed(texto)
    except Exception:
        print(
            "Aviso: embeddings indisponíveis; item salvo sem vetor "
            "(rode 'python main.py reindex' depois)."
        )
        return None


def cmd_humanizar(repo, args) -> None:
    llm = LLMClient()
    if not llm.ping():
        sys.exit(
            "Erro: LM Studio não respondeu em http://localhost:1234/v1\n"
            "Abra o LM Studio, carregue o modelo Llama 3.1 8B Instruct e "
            "clique em 'Start Server' na aba Developer."
        )

    embedding_client = EmbeddingClient()
    if embedding_client.available():
        builder = PromptBuilder(repo, embedding_client=embedding_client)
    else:
        print(
            "Aviso: modelo de embeddings indisponível; "
            "usando exemplos mais recentes."
        )
        builder = PromptBuilder(repo, embedding_client=embedding_client)

    prompt = builder.build(contexto=args.contexto, texto_alvo=args.texto)
    saida = llm.humanize(prompt, args.texto)
    gen_id = repo.save_generation(prompt, args.texto, saida)
    _print(f"RESULTADO (geração #{gen_id})", saida)
    print(f"Gostou? Avalie com:  python main.py avaliar {gen_id} <nota 1-5>")
    print(f'Corrigiu à mão? Ensine o sistema:  python main.py avaliar {gen_id} 5 --editada "sua versão"')


def cmd_add_sample(repo, args) -> None:
    embedding = _embedding_or_none(args.texto)
    sid = repo.add_sample(
        args.texto,
        contexto=args.contexto or "",
        tags=args.tags or "",
        embedding=embedding,
    )
    print(f"Amostra de estilo #{sid} salva.")


def cmd_add_rule(repo, args) -> None:
    rid = repo.add_rule(args.regra, categoria=args.categoria or "", peso=args.peso)
    print(f"Regra #{rid} salva.")


def cmd_add_pair(repo, args) -> None:
    embedding = _embedding_or_none(args.depois)
    pid = repo.add_pair(
        args.antes,
        args.depois,
        contexto=args.contexto or "",
        embedding=embedding,
    )
    print(f"Par de reescrita #{pid} salvo.")


def cmd_avaliar(repo, args) -> None:
    gen = repo.get_generation(args.id)
    if not gen:
        sys.exit(f"Geração #{args.id} não encontrada.")
    repo.rate_generation(args.id, args.nota, editada=args.editada or "")
    # Loop de aprendizado: sua versão editada vira exemplo de estilo E par.
    if args.editada:
        embedding = _embedding_or_none(args.editada)
        repo.add_sample(
            args.editada,
            contexto="editado",
            tags="feedback",
            embedding=embedding,
        )
        repo.add_pair(
            gen["saida"],
            args.editada,
            contexto="editado",
            embedding=embedding,
        )
        print("Nota salva. Sua versão editada virou nova amostra + par de reescrita.")
        print("O prompt já melhorou para as próximas gerações.")
    else:
        print("Nota salva.")


def cmd_preview(repo, args) -> None:
    _print("PROMPT MONTADO", PromptBuilder(repo).build(contexto=args.contexto))


def cmd_ping(repo, args) -> None:
    ok = LLMClient().ping()
    print("LM Studio OK" if ok else "LM Studio não respondeu (inicie o servidor).")


def cmd_reindex(repo, args) -> None:
    embedding_client = EmbeddingClient()
    if not embedding_client.available():
        sys.exit(
            "Erro: modelo de embeddings indisponível. Carregue "
            "text-embedding-nomic-embed-text-v1.5 no LM Studio e tente novamente."
        )

    rows = repo.iter_rows_without_embedding()
    indexed = 0
    try:
        for start in range(0, len(rows), 64):
            batch = rows[start : start + 64]
            vectors = embedding_client.embed_many([row["texto"] for row in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("quantidade inesperada de embeddings na resposta")
            for row, vector in zip(batch, vectors):
                repo.save_embedding(row["row_type"], row["id"], vector)
                indexed += 1
    except Exception as exc:
        sys.exit(f"Erro durante a reindexação: {exc}")

    print(f"{indexed} linha(s) indexada(s).")


def _parse_blocos(texto: str) -> list[tuple[str, str]]:
    """Divide o arquivo em (contexto, amostra).

    Blocos separados por uma linha só com '==='. Cada bloco pode começar com
    uma linha '@contexto' que define o contexto daquela amostra.
    """
    brutos: list[list[str]] = [[]]
    for linha in texto.splitlines():
        if linha.strip() == "===":
            brutos.append([])
        else:
            brutos[-1].append(linha)

    itens: list[tuple[str, str]] = []
    for linhas in brutos:
        if not any(l.strip() for l in linhas):
            continue
        contexto = ""
        corpo = linhas
        if corpo and corpo[0].strip().startswith("@"):
            contexto = corpo[0].strip()[1:].strip()
            corpo = corpo[1:]
        amostra = "\n".join(corpo).strip()
        if amostra:
            itens.append((contexto, amostra))
    return itens


def cmd_importar(repo, args) -> None:
    caminho = Path(args.arquivo)
    if not caminho.exists():
        sys.exit(f"Arquivo não encontrado: {caminho}")
    itens = _parse_blocos(caminho.read_text(encoding="utf-8"))
    if not itens:
        sys.exit("Nenhuma amostra encontrada (use '===' entre os blocos).")

    ec = EmbeddingClient()
    vetores: list[list[float] | None] = [None] * len(itens)
    if ec.available():
        try:
            vetores = ec.embed_many([texto for _, texto in itens])
        except Exception:
            print("Aviso: falha nos embeddings; importando sem vetor (rode 'reindex' depois).")
            vetores = [None] * len(itens)
    else:
        print("Aviso: embeddings indisponíveis; importando sem vetor (rode 'reindex' depois).")

    n = 0
    for (contexto, texto), vetor in zip(itens, vetores):
        repo.add_sample(texto, contexto=contexto, tags="import", embedding=vetor)
        n += 1
    print(f"{n} amostra(s) importada(s).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Humanizador de textos (anti-IA).")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("humanizar", help="reescreve um texto")
    h.add_argument("texto")
    h.add_argument("--contexto", default=None)
    h.set_defaults(func=cmd_humanizar)

    s = sub.add_parser("add-sample", help="guarda uma amostra da sua escrita")
    s.add_argument("texto")
    s.add_argument("--contexto", default=None)
    s.add_argument("--tags", default=None)
    s.set_defaults(func=cmd_add_sample)

    r = sub.add_parser("add-rule", help="adiciona uma regra ao prompt")
    r.add_argument("regra")
    r.add_argument("--categoria", default=None)
    r.add_argument("--peso", type=int, default=1)
    r.set_defaults(func=cmd_add_rule)

    pr = sub.add_parser("add-pair", help="guarda um par antes/depois")
    pr.add_argument("antes")
    pr.add_argument("depois")
    pr.add_argument("--contexto", default=None)
    pr.set_defaults(func=cmd_add_pair)

    a = sub.add_parser("avaliar", help="dá nota a uma geração")
    a.add_argument("id", type=int)
    a.add_argument("nota", type=int, choices=range(1, 6))
    a.add_argument("--editada", default=None)
    a.set_defaults(func=cmd_avaliar)

    pv = sub.add_parser("preview", help="mostra o prompt montado")
    pv.add_argument("--contexto", default=None)
    pv.set_defaults(func=cmd_preview)

    pg = sub.add_parser("ping", help="testa conexão com o LM Studio")
    pg.set_defaults(func=cmd_ping)

    ri = sub.add_parser("reindex", help="indexa amostras e pares sem embedding")
    ri.set_defaults(func=cmd_reindex)

    im = sub.add_parser(
        "importar", help="importa várias amostras de um arquivo (blocos separados por ===)"
    )
    im.add_argument("arquivo")
    im.set_defaults(func=cmd_importar)

    return p


def main() -> None:
    args = build_parser().parse_args()
    repo = SqliteRepository(DB_PATH)
    args.func(repo, args)


if __name__ == "__main__":
    main()
