"""Monta o system prompt anti-IA dinamicamente a partir do que há no banco."""

from __future__ import annotations

from .embeddings import EmbeddingClient
from .repository import Repository

# Base fixa. As regras/exemplos do banco são ANEXADOS a isto.
BASE = """Você reescreve textos para soarem como uma pessoa específica escrevendo, \
não como um assistente genérico. Responda SOMENTE com o texto reescrito, sem \
comentários, sem título, sem aspas ao redor.

DIRETRIZES DE VOZ
- Tenha posição. Não fique em cima do muro nem encha de ressalvas vazias.
- Varie o ritmo das frases. Curtas. Depois uma mais longa, que respira e \
desenvolve a ideia. Curta de novo.
- Escreva em prosa. Só use listas quando for de fato uma lista de itens.

EVITE (o que denuncia IA)
- Aberturas de moldura ("Num cenário cada vez mais...", "É importante ressaltar que...").
- Conclusões que só resumem ("Em suma", "Portanto, fica claro que").
- Hedging vazio ("pode-se dizer que", "de certa forma") sem incerteza real.
- Transições decorativas no início de todo parágrafo ("Além disso", "Ademais").
- Agrupar sempre em três. Deixe as listas assimétricas.
- Adjetivos ocos ("robusto", "eficaz", "significativo", "fundamental").

CONCRETUDE
- Prefira o exemplo específico ao princípio genérico.
- Se uma frase pudesse abrir qualquer texto sobre qualquer assunto, reescreva-a."""


class PromptBuilder:
    def __init__(
        self,
        repo: Repository,
        embedding_client: EmbeddingClient | None = None,
    ):
        self.repo = repo
        self.embedding_client = embedding_client or EmbeddingClient()

    def build(
        self,
        contexto: str | None = None,
        texto_alvo: str | None = None,
    ) -> str:
        partes = [BASE]

        regras = self.repo.get_rules()
        if regras:
            linhas = [f"- {r['regra']}" for r in regras]
            partes.append("REGRAS ADICIONAIS (do seu perfil):\n" + "\n".join(linhas))

        amostras = None
        pares = None
        if texto_alvo and self.embedding_client is not None:
            try:
                if self.embedding_client.available():
                    query_embedding = self.embedding_client.embed(texto_alvo)
                    amostras = self.repo.get_samples_by_similarity(
                        query_embedding, contexto=contexto, limit=3
                    )
                    pares = self.repo.get_pairs_by_similarity(query_embedding, limit=2)
            except Exception:
                # Falhas de rede, modelo ou vetores inválidos nunca impedem o prompt.
                amostras = None
                pares = None

        if not amostras:
            amostras = self.repo.get_samples(contexto=contexto, limit=3)
        if not pares:
            pares = self.repo.get_pairs(limit=2)

        if amostras:
            blocos = [f'"""{a["texto"]}"""' for a in amostras]
            partes.append(
                "EXEMPLOS DA SUA ESCRITA (imite este tom, ritmo e vocabulário — "
                "NÃO copie o conteúdo):\n" + "\n\n".join(blocos)
            )

        if pares:
            blocos = [
                f"ANTES (cheiro de IA):\n{p['antes']}\n\nDEPOIS (humanizado):\n{p['depois']}"
                for p in pares
            ]
            partes.append(
                "EXEMPLOS DE REESCRITA (siga este tipo de transformação):\n\n"
                + "\n\n---\n\n".join(blocos)
            )

        return "\n\n".join(partes)
