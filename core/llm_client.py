"""Conversa com o LM Studio (servidor local compatível com a API da OpenAI)."""

from __future__ import annotations

from openai import OpenAI

# LM Studio -> aba Developer -> Start Server. Porta padrão 1234.
DEFAULT_BASE_URL = "http://localhost:1234/v1"
# Nome do modelo carregado no LM Studio (identificador exato do 'lms ls').
# Troque aqui se quiser usar outro (ex.: "qwen2.5-7b-instruct-1m").
DEFAULT_MODEL = "meta-llama-3.1-8b-instruct"


class LLMClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.8,
        max_tokens: int = 600,
    ):
        # api_key é ignorada pelo LM Studio, mas o SDK exige algum valor.
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def humanize(self, system_prompt: str, texto: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Reescreva o texto a seguir:\n\n{texto}"},
            ],
        )
        return resp.choices[0].message.content.strip()

    def ping(self) -> bool:
        """Verifica se o servidor do LM Studio está no ar."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
