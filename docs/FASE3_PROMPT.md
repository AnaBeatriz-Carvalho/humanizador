# Prompt para implementar a Fase 3 — Relevância por Embeddings

> Cole este documento inteiro para a IA que vai implementar. Ele é
> autossuficiente: descreve o código que já existe, o objetivo e os
> critérios de aceite.

---

## Contexto do projeto

`humanizador` é um app Python que reescreve textos para soarem naturais (não
como IA). O system prompt anti-IA é montado **dinamicamente** a partir de um
banco SQLite com: regras (`rules`), amostras da escrita do usuário
(`style_samples`), pares antes/depois (`rewrite_pairs`) e histórico
(`generations`). O LLM roda local no **LM Studio** (servidor compatível com a
API da OpenAI em `http://localhost:1234/v1`), acessado via a lib `openai`.

Estrutura atual:

```
humanizador/
├─ main.py               # CLI (argparse)
├─ compare.py            # compara modelos
├─ requirements.txt      # openai>=1.0
├─ db/schema.sql         # CREATE TABLE IF NOT EXISTS ...
└─ core/
   ├─ repository.py      # interface Repository + SqliteRepository
   ├─ prompt_builder.py  # PromptBuilder.build(contexto) monta o prompt
   └─ llm_client.py      # LLMClient.humanize(system_prompt, texto)
```

**Comportamento atual (Fase 1 e 2):** `PromptBuilder.build()` pega as regras
ativas, as **3 amostras mais recentes** e os **2 pares mais recentes** e cola
tudo num prompt. É recência pura — não olha se o exemplo tem a ver com o texto
sendo reescrito.

## Objetivo da Fase 3

Trocar "mais recentes" por **"mais relevantes"**: dado o texto que o usuário quer
humanizar, buscar as amostras e os pares **semanticamente mais parecidos** usando
embeddings, e injetar só esses no prompt. Isso torna os exemplos few-shot
específicos ao assunto/tom do texto atual, melhorando muito o resultado.

O LM Studio já serve um modelo de embeddings local:
`text-embedding-nomic-embed-text-v1.5` (768 dimensões).

## Tarefas

### 1. `core/embeddings.py` (novo)
- Classe `EmbeddingClient` com `embed(texto: str) -> list[float]` e
  `embed_many(textos: list[str]) -> list[list[float]]`.
- Usa a lib `openai` apontada para `http://localhost:1234/v1`
  (`client.embeddings.create(model=..., input=...)`,
  vetor em `resp.data[i].embedding`).
- Modelo padrão: `"text-embedding-nomic-embed-text-v1.5"` (configurável).
- Método `available() -> bool` que testa se o servidor/modelo respondem
  (para permitir fallback gracioso).

### 2. Persistência dos vetores (`core/repository.py` + migração)
- Guardar o embedding de cada `style_samples` e cada `rewrite_pairs`.
  Sugestão: coluna nova `embedding TEXT` (o vetor serializado como JSON) em
  ambas as tabelas. Como `schema.sql` usa `CREATE TABLE IF NOT EXISTS`, faça a
  migração no `__init__` do `SqliteRepository`: cheque com
  `PRAGMA table_info(<tabela>)` e rode `ALTER TABLE ... ADD COLUMN embedding TEXT`
  se a coluna não existir. Não quebre bancos já existentes.
- Ao inserir amostra/par, calcular e salvar o embedding (do `texto` da amostra;
  para o par, use o campo `depois`, que é o alvo humanizado). Injete o
  `EmbeddingClient` no repositório ou passe o vetor pronto — escolha o desenho
  mais limpo, mas mantenha o `Repository` uma interface trocável (não acople o
  SQLite à rede desnecessariamente).
- Novos métodos na interface e na implementação:
  - `get_samples_by_similarity(query_embedding: list[float], contexto, limit) -> list[dict]`
  - `get_pairs_by_similarity(query_embedding: list[float], limit) -> list[dict]`
  - `iter_rows_without_embedding()` / algo para o reindex (ver tarefa 5).
- Similaridade: **cosseno**. Pode usar `numpy` (adicione a `requirements.txt`).
  O N é pequeno (dezenas/centenas de linhas), então calcular em memória tudo bem.

### 3. `core/prompt_builder.py`
- `build(contexto, texto_alvo=None)`: se `texto_alvo` for dado **e** os
  embeddings estiverem disponíveis, calcule o embedding do `texto_alvo` e use os
  métodos `*_by_similarity` para escolher amostras e pares. Caso contrário,
  **mantenha o comportamento atual** (mais recentes) como fallback.
- Não mude o formato do prompt nem os textos-base — só a **seleção** dos exemplos.

### 4. `main.py`
- No comando `humanizar`, passar o próprio texto do usuário como `texto_alvo`
  para o `PromptBuilder.build(...)`.
- Se o servidor de embeddings estiver fora, avisar em uma linha e seguir com o
  fallback por recência (não travar).

### 5. Comando `reindex` (novo, em `main.py`)
- `python main.py reindex` calcula e grava o embedding de todas as amostras e
  pares que ainda não têm um (inclui as criadas nas Fases 1 e 2). Idempotente.
- Imprime quantas linhas foram indexadas.

## Detalhes da API (LM Studio)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
resp = client.embeddings.create(
    model="text-embedding-nomic-embed-text-v1.5",
    input="texto aqui",           # ou uma lista de textos
)
vetor = resp.data[0].embedding    # list[float], 768 dims
```

O modelo de embeddings precisa estar **carregado** no LM Studio
(`lms load text-embedding-nomic-embed-text-v1.5`) — o de chat e o de embeddings
podem ficar carregados ao mesmo tempo (o de embeddings é minúsculo, ~84 MB).

## Critérios de aceite

1. `pip install -r requirements.txt` funciona (openai + numpy).
2. `python main.py reindex` popula embeddings das linhas existentes sem erro e
   é idempotente (rodar de novo não duplica trabalho nem quebra).
3. Com o servidor de embeddings ligado, `python main.py preview` **não** muda,
   mas `humanizar` passa a selecionar exemplos por similaridade — dá para
   comprovar criando amostras de dois assuntos bem distintos (ex.: culinária e
   programação) e verificando que, ao humanizar um texto de culinária, as
   amostras escolhidas são as de culinária.
4. Com o servidor de embeddings **desligado**, tudo continua funcionando pelo
   fallback de recência, com um aviso claro.
5. Bancos criados nas Fases 1/2 continuam abrindo (migração não-destrutiva).

## Cuidados (ambiente Windows)

- O console é cp1252; ao capturar saída de subprocess do `lms`, use
  `encoding="utf-8", errors="ignore"` (já feito em `compare.py`) para evitar
  `UnicodeDecodeError`.
- Não introduza dependências pesadas (nada de sentence-transformers/torch —
  os embeddings vêm do LM Studio via HTTP). Só `openai` e `numpy`.
- Mantenha a interface `Repository` abstrata intacta para não fechar a porta ao
  backend Firebase da Fase 5.
- Não versione `data/` nem `*.db` (já está no `.gitignore`).
