-- Esquema do humanizador. Rodado uma vez na inicialização (idempotente).

-- Amostras da SUA escrita — a âncora de voz do prompt.
CREATE TABLE IF NOT EXISTS style_samples (
  id        INTEGER PRIMARY KEY,
  texto     TEXT NOT NULL,
  contexto  TEXT,                       -- "email", "blog", "academico"...
  tags      TEXT,                       -- separadas por vírgula
  criado_em TEXT DEFAULT (datetime('now'))
);

-- Regras/heurísticas que compõem o prompt anti-IA.
CREATE TABLE IF NOT EXISTS rules (
  id        INTEGER PRIMARY KEY,
  regra     TEXT NOT NULL,
  categoria TEXT,                        -- "proibido", "voz", "concretude"
  peso      INTEGER DEFAULT 1,           -- maior = aparece primeiro
  ativa     INTEGER DEFAULT 1
);

-- Pares antes/depois: texto com "cheiro de IA" -> versão humanizada.
CREATE TABLE IF NOT EXISTS rewrite_pairs (
  id        INTEGER PRIMARY KEY,
  antes     TEXT NOT NULL,
  depois    TEXT NOT NULL,
  contexto  TEXT,
  criado_em TEXT DEFAULT (datetime('now'))
);

-- Histórico de gerações — o coração do loop de aprendizado.
CREATE TABLE IF NOT EXISTS generations (
  id           INTEGER PRIMARY KEY,
  prompt_usado TEXT,
  entrada      TEXT,
  saida        TEXT,
  nota         INTEGER,                  -- 1..5 dado por você
  editada      TEXT,                     -- sua versão final corrigida
  criado_em    TEXT DEFAULT (datetime('now'))
);
