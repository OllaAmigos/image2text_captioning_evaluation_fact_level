"""Thin client for any OpenAI-compatible chat endpoint (e.g. a self-hosted vLLM server).

Configuration comes from environment variables, never from code:

    CAPTION_EVAL_BASE_URL   e.g. http://localhost:8000
    CAPTION_EVAL_MODEL      e.g. my-vlm
    CAPTION_EVAL_API_KEY    optional
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

import requests

_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def parse_json(text: str):
    """Parse a JSON reply, tolerating think-blocks and ```json fences."""
    text = strip_think(text)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


class Judge:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, max_retries: int = 5, timeout: int = 180):
        self.base_url = (base_url or os.environ["CAPTION_EVAL_BASE_URL"]).rstrip("/")
        self.model = model or os.environ["CAPTION_EVAL_MODEL"]
        self.api_key = api_key or os.environ.get("CAPTION_EVAL_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout

    def _post(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{self.base_url}/v1/chat/completions", json=payload,
                                  headers=headers, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503):
                    raise requests.HTTPError(str(r.status_code))
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # exponential backoff
        raise RuntimeError("unreachable")

    def ask(self, prompt: str, image: str | Path | None = None, *,
            thinking: bool = False, as_json: bool = True):
        """Send a text (+ optional image path or http URL) prompt; temperature 0."""
        content: list[dict] = []
        if image is not None:
            url = str(image)
            if not url.startswith("http"):
                b64 = base64.b64encode(Path(url).read_bytes()).decode()
                url = f"data:image/png;base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.model,
            "temperature": 0.0,
            # servers that support it: switch the reasoning block off so JSON parses cleanly
            "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": content}],
        }
        out = self._post(payload)["choices"][0]["message"]["content"]
        return parse_json(out) if as_json else strip_think(out)
