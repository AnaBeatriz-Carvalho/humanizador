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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.llm_client import LLMClient
from core.prompt_builder import PromptBuilder
from core.repository import SqliteRepository

DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"


def _print(titulo: str, corpo: str) -> None:
    print(f"\n=== {titulo} ===\n{corpo}\n")


def cmd_humanizar(repo, args) -> None:
    llm = LLMClient()
    if not llm.ping():
        sys.exit(
            "Erro: LM Studio não respondeu em http://localhost:1234/v1\n"
            "Abra o LM Studio, carregue o modelo Llama 3.1 8B Instruct e "
            "clique em 'Start Server' na aba Developer."
        )
    prompt = PromptBuilder(repo).build(contexto=args.contexto)
    saida = llm.humanize(prompt, args.texto)
    gen_id = repo.save_generation(prompt, args.texto, saida)
    _print(f"RESULTADO (geração #{gen_id})", saida)
    print(f"Gostou? Avalie com:  python main.py avaliar {gen_id} <nota 1-5>")
    print(f'Corrigiu à mão? Ensine o sistema:  python main.py avaliar {gen_id} 5 --editada "sua versão"')


def cmd_add_sample(repo, args) -> None:
    sid = repo.add_sample(args.texto, contexto=args.contexto or "", tags=args.tags or "")
    print(f"Amostra de estilo #{sid} salva.")


def cmd_add_rule(repo, args) -> None:
    rid = repo.add_rule(args.regra, categoria=args.categoria or "", peso=args.peso)
    print(f"Regra #{rid} salva.")


def cmd_add_pair(repo, args) -> None:
    pid = repo.add_pair(args.antes, args.depois, contexto=args.contexto or "")
    print(f"Par de reescrita #{pid} salvo.")


def cmd_avaliar(repo, args) -> None:
    gen = repo.get_generation(args.id)
    if not gen:
        sys.exit(f"Geração #{args.id} não encontrada.")
    repo.rate_generation(args.id, args.nota, editada=args.editada or "")
    # Loop de aprendizado: sua versão editada vira exemplo de estilo E par.
    if args.editada:
        repo.add_sample(args.editada, contexto="editado", tags="feedback")
        repo.add_pair(gen["saida"], args.editada, contexto="editado")
        print("Nota salva. Sua versão editada virou nova amostra + par de reescrita.")
        print("O prompt já melhorou para as próximas gerações.")
    else:
        print("Nota salva.")


def cmd_preview(repo, args) -> None:
    _print("PROMPT MONTADO", PromptBuilder(repo).build(contexto=args.contexto))


def cmd_ping(repo, args) -> None:
    ok = LLMClient().ping()
    print("LM Studio OK" if ok else "LM Studio não respondeu (inicie o servidor).")


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

    return p


def main() -> None:
    args = build_parser().parse_args()
    repo = SqliteRepository(DB_PATH)
    args.func(repo, args)


if __name__ == "__main__":
    main()
