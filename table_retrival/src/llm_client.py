"""LLM client wrapper for calling the Anthropic API."""

import json
import os
import re
import sys

from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE


class LLMClient:
    """Wraps Anthropic API calls with json-mode extraction."""

    def __init__(self, model: str | None = None, dry_run: bool = False):
        self.dry_run = dry_run
        self.model = model or LLM_MODEL
        self.client = None
        if not dry_run:
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
            print("[DRY RUN — no API call made]")
            print("=" * 60)
            print(f"System:\n{system_prompt}\n")
            print(f"User:\n{user_message}\n")
            print("=" * 60)
            return "{}"

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
        # Try to find JSON in ```json ... ``` fences first
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            return json.loads(m.group(1))
        # Otherwise, try parsing the raw text
        return json.loads(text)
