# humanizador

Aplicação em Python que reescreve textos para soarem naturais — não como IA.
O diferencial: o *system prompt* anti-IA **não é fixo**. Ele é montado
dinamicamente a partir de dados que você acumula (suas amostras de escrita,
regras, pares antes/depois e feedback). Quanto mais você usa, melhor a voz fica.

Roda 100% local via [LM Studio](https://lmstudio.ai/) (offline, sem custo por uso,
seus textos nunca saem da máquina).

## Arquitetura

```
Interface (CLI)                 main.py
   │
PromptBuilder                   core/prompt_builder.py   monta o prompt do banco
LLMClient                       core/llm_client.py       fala com o LM Studio
Repository (SQLite, trocável)   core/repository.py       persistência
```

Tabelas: `style_samples`, `rules`, `rewrite_pairs`, `generations`.
A tabela `generations` + a sua versão editada formam o **loop de aprendizado**:
ao avaliar uma geração com `--editada`, sua correção vira nova amostra + par.

## Requisitos

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) com um modelo instruct carregado e o
  servidor local ligado (aba *Developer* → *Start Server*, porta 1234).
  Padrão do projeto: `meta-llama-3.1-8b-instruct`.
- `pip install -r requirements.txt`

## Uso

```bash
python main.py ping                                  # testa a conexão
python main.py add-rule "sem gerundismo" --peso 5    # regra do prompt
python main.py add-sample "um texto seu..."          # amostra da sua voz
python main.py humanizar "texto com cara de IA"      # reescreve
python main.py avaliar 3 5 --editada "sua versão"    # ensina o sistema
python main.py preview                               # vê o prompt montado
```

## Roteiro

- [x] Fase 1 — núcleo (SQLite + prompt dinâmico + LM Studio + CLI)
- [x] Fase 2 — loop de aprendizado (feedback vira exemplo)
- [ ] Fase 3 — relevância por embeddings (ver `docs/FASE3_PROMPT.md`)
- [ ] Fase 4 — interface Streamlit
- [ ] Fase 5 (opcional) — sincronização em nuvem (Firebase)
