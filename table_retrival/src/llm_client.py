"""LLM client wrapper. Supports both Anthropic native and OpenAI-compatible APIs."""

import json
import os
import re
import sys

from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_API_BASE, LLM_API_KEY


class LLMClient:

    def __init__(self, model: str | None = None, dry_run: bool = False):
        self.dry_run = dry_run
        self.model = model or LLM_MODEL
        self.client = None
        self._backend = self._detect_backend()

        if not dry_run:
            self._init_client()

    def _detect_backend(self) -> str:
        if LLM_API_BASE:
            return "openai"
        return "anthropic"

    def _init_client(self):
        if self._backend == "openai":
            import openai

            api_key = LLM_API_KEY or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("[ERROR] LLM_API_KEY not set. Export it or use --dry-run.")
                sys.exit(1)
            self.client = openai.OpenAI(api_key=api_key, base_url=LLM_API_BASE)
        else:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("[ERROR] ANTHROPIC_API_KEY not set. Export it or use --dry-run.")
                sys.exit(1)
            self.client = anthropic.Anthropic(api_key=api_key)

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a single-turn chat and return the text response."""
        if self.dry_run:
            print("\n" + "=" * 60)
            print(f"[DRY RUN — {self._backend}]")
            print("=" * 60)
            print(f"System:\n{system_prompt}\n")
            print(f"User:\n{user_message}\n")
            print("=" * 60)
            return "{}"

        if self._backend == "openai":
            return self._chat_openai(system_prompt, user_message)
        return self._chat_anthropic(system_prompt, user_message)

    def _chat_openai(self, system_prompt: str, user_message: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content

    def _chat_anthropic(self, system_prompt: str, user_message: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text

    @staticmethod
    def extract_json(text: str) -> dict:
        """Extract a JSON object from model response, handling markdown code fences."""
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            return json.loads(m.group(1))
        return json.loads(text)
