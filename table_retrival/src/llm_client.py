"""LLM client wrapper using OpenAI-compatible API."""

import json
import re


class LLMClient:

    def __init__(self, api_key: str = "", base_url: str = "", model: str = "gpt-3.5-turbo",
                 max_tokens: int = 4096, temperature: float = 0.3,
                 dry_run: bool = False):
        self.dry_run = dry_run
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None

        if not dry_run:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, system_prompt: str, user_message: str) -> str:
        if self.dry_run:
            print("\n" + "=" * 60)
            print("[DRY RUN]")
            print("=" * 60)
            print(f"System:\n{system_prompt}\n")
            print(f"User:\n{user_message}\n")
            print("=" * 60)
            return "{}"

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content

    @staticmethod
    def extract_json(text: str) -> dict:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            return json.loads(m.group(1))
        return json.loads(text)
