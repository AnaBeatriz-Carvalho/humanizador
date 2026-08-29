# Como rodar o humanizador

Guia passo a passo. Pode ser seguido por uma pessoa ou entregue a uma IA
assistente.

## Antes de tudo: o que este projeto é (e o que NÃO é)

- **NÃO treina** nenhum modelo de IA. O modelo do LM Studio nunca muda.
- O "aprendizado" é um **banco de dados que cresce**: as amostras da sua escrita
  e as suas correções ficam salvas e são **injetadas no prompt** na hora de
  reescrever. Melhora = melhores exemplos no prompt, não pesos ajustados.
- Tudo roda **local**, offline. Seus textos não saem da máquina.

## Pré-requisitos (só na primeira vez)

1. **Python 3.11+** instalado.
2. Dependências do projeto:
   ```
   pip install -r requirements.txt
   ```
3. **LM Studio** instalado, com dois modelos baixados:
   - `meta-llama-3.1-8b-instruct` (reescreve o texto)
   - `text-embedding-nomic-embed-text-v1.5` (busca de exemplos por similaridade)

## Toda vez que for usar: ligar o LM Studio

1. Abrir o LM Studio.
2. Carregar os **dois** modelos (cabem juntos em 8 GB de VRAM):
   - Pela interface: *My Models* → carregar cada um.
   - Ou pelo terminal:
     ```
     lms load meta-llama-3.1-8b-instruct --gpu max
     lms load text-embedding-nomic-embed-text-v1.5
     ```
3. Ligar o servidor: aba **Developer** → **Start Server** (porta 1234).
4. Conferir que respondeu:
   ```
   python main.py ping
   ```
   Deve imprimir `LM Studio OK`.

## Comandos do dia a dia

Sempre a partir da pasta do projeto (`humanizador/`).

**Reescrever um texto** (o principal):
```
python main.py humanizar "seu texto com cara de IA aqui"
python main.py humanizar --contexto academico "um parágrafo acadêmico..."
```
Use `--contexto` para o sistema preferir amostras daquele tipo
(ex.: `academico`, `pessoal`, `mensagem`, `reflexivo`).

**Ensinar o sistema** (o loop de aprendizado): depois de humanizar, se você
corrigir a saída à mão, salve sua versão — ela vira nova amostra + par:
```
python main.py avaliar <id> <nota 1-5> --editada "sua versão corrigida"
```
(O `<id>` aparece no resultado do `humanizar`, ex.: "geração #7".)

**Adicionar amostras em massa** (arquivo com blocos separados por `===`,
`@contexto` opcional no topo de cada bloco):
```
python main.py importar data/meu_lote.txt
```
Cuidado: não re-rode o mesmo arquivo — ele insere de novo (duplica).

**Adicionar uma amostra ou regra avulsa:**
```
python main.py add-sample "um texto seu" --contexto blog
python main.py add-rule "sem gerundismo" --peso 5
```

## Como VER o que está acontecendo (ninguém "treina" às cegas)

**Ver o prompt exato que o modelo recebe**, com suas amostras dentro:
```
python main.py preview --contexto academico
```

**Ver quantas amostras já existem e seus contextos:**
```
python -c "from core.repository import SqliteRepository; from pathlib import Path; r=SqliteRepository(Path('data/app.db')); c=r._conn(); [print(f\"#{x['id']} [{x['contexto']}] {x['texto'][:50]}\") for x in c.execute('SELECT id,contexto,texto FROM style_samples ORDER BY id')]; c.close()"
```

**Se você adicionou algo com o servidor de embeddings desligado**, indexe depois:
```
python main.py reindex
```

## Rodar os testes (conferir que está tudo são)
```
python -m unittest tests.test_phase3 -v
```

## O ciclo que faz a voz melhorar

1. `humanizar` um texto.
2. Não gostou de algo? Corrige à mão e roda `avaliar <id> 5 --editada "..."`.
3. Sua correção entra no banco → o próximo `humanizar` já usa esse exemplo.
4. De tempos em tempos, `importar` um lote novo de textos seus.

Quanto mais você usa e corrige, mais o resultado soa como você.
