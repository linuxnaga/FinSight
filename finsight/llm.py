"""Wrapper around the course's OpenAI-compatible endpoint.

The interesting part is `structured()`: small models regularly return
malformed or schema-violating JSON, so every structured call goes through
  1. constrained decoding when the endpoint supports it (json_schema / json_object)
  2. tolerant JSON extraction (code fences, trailing prose, single quotes)
  3. Pydantic validation
  4. retry with the validation error fed back to the model
and the number of repairs is recorded so the report can show the effect.

Run `python -m finsight.llm --probe` to see which JSON modes the endpoint accepts.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from finsight import tracing
from finsight.config import settings

T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    pass


class EndpointError(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    def __init__(self, msg: str, raw: str):
        super().__init__(msg)
        self.raw = raw


# Counters the eval report reads (reset per eval run).
STATS = {"calls": 0, "json_repairs": 0, "json_failures": 0, "mode_downgrades": 0}


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model response."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("no JSON object found in model output")
    # Walk to the matching close bracket, respecting strings.
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # common small-model slips: trailing commas, single quotes
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    if '"' not in fixed:
                        fixed = fixed.replace("'", '"')
                    return json.loads(fixed)
    raise ValueError("unterminated JSON in model output")


class LLM:
    def __init__(self, client: Any = None, model: str | None = None, json_mode: str | None = None):
        s = settings()
        if client is None:
            from openai import OpenAI

            if not s.llm_base_url:  # never fall through to api.openai.com by accident
                raise RuntimeError("LLM_BASE_URL is not set: copy .env.example to .env and add the course endpoint")
            client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=60, max_retries=2)
        self.client = client
        self.model = model or s.llm_model
        self.json_mode = json_mode or s.llm_json_mode
        self.max_tokens_per_query = s.max_tokens_per_query

    # ------------------------------------------------------------------ raw
    def chat(self, messages: list[dict], *, name: str = "chat", temperature: float = 0.0,
             max_tokens: int = 700, response_format: dict | None = None, n: int = 1) -> list[str]:
        t = tracing.current()
        if t is not None and t.tokens > self.max_tokens_per_query:
            raise BudgetExceeded(f"token budget {self.max_tokens_per_query} exceeded ({t.tokens})")
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, temperature=temperature,
                                      max_tokens=max_tokens)
        if n > 1:
            kwargs["n"] = n
        if response_format:
            kwargs["response_format"] = response_format
        with tracing.span(name, "llm", input=messages[-1]["content"][-2000:], temperature=temperature) as sp:
            STATS["calls"] += 1
            resp = self.client.chat.completions.create(**kwargs)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                tracing.add_usage(usage.prompt_tokens, usage.completion_tokens)
                sp.meta["tokens"] = usage.prompt_tokens + usage.completion_tokens
            if not getattr(resp, "choices", None):
                # some gateways return errors as HTTP 200 with {"error": {...}} instead of choices
                err = (getattr(resp, "model_extra", None) or {}).get("error") or getattr(resp, "error", None)
                msg = err.get("message") if isinstance(err, dict) else str(err or "no choices in response")
                hint = " Check LLM_MODEL in .env." if "does not exist" in msg else ""
                raise EndpointError(f"endpoint returned an error instead of a completion: {msg}.{hint}")
            outs = [c.message.content or "" for c in resp.choices]
            sp.output = outs[0][:2000]
        return outs

    def text(self, system: str, user: str, **kw) -> str:
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)[0]

    # ----------------------------------------------------------- structured
    def _response_format(self, schema: type[BaseModel]) -> dict | None:
        if self.json_mode == "json_schema":
            return {"type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
        if self.json_mode == "json_object":
            return {"type": "json_object"}
        return None

    def structured(self, system: str, user: str, schema: type[T], *, name: str = "structured",
                   retries: int = 2, temperature: float = 0.0, max_tokens: int = 700) -> T:
        schema_hint = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        messages = [
            {"role": "system", "content": f"{system}\n\nRespond with ONLY a JSON object matching this JSON schema:\n{schema_hint}"},
            {"role": "user", "content": user},
        ]
        raw = ""
        for attempt in range(retries + 1):
            try:
                raw = self.chat(messages, name=f"{name}#{attempt}" if attempt else name, temperature=temperature,
                                max_tokens=max_tokens, response_format=self._response_format(schema))[0]
            except Exception as e:  # endpoint rejected response_format -> downgrade and retry
                msg = str(e).lower()
                if self.json_mode != "none" and ("response_format" in msg or "400" in msg):
                    STATS["mode_downgrades"] += 1
                    self.json_mode = "json_object" if self.json_mode == "json_schema" else "none"
                    continue
                raise
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as e:
                STATS["json_repairs"] += 1
                sp = tracing.current().spans[-1] if tracing.current() and tracing.current().spans else None
                if sp is not None:
                    sp.failure = "tool_call_error:malformed_json"
                err = str(e)[:600]
                messages = messages + [
                    {"role": "assistant", "content": raw[:1500]},
                    {"role": "user", "content": f"That output was invalid: {err}\nReturn ONLY the corrected JSON object."},
                ]
        STATS["json_failures"] += 1
        raise StructuredOutputError(f"{name}: could not get valid {schema.__name__} after {retries + 1} attempts", raw)


def probe() -> None:
    from pydantic import BaseModel as BM

    class Ping(BM):
        ok: bool
        n: int

    s = settings()
    print(f"endpoint={s.llm_base_url} model={s.llm_model}")
    llm = LLM()
    try:
        print("plain:", llm.text("You are terse.", "Say hi in 3 words.")[:80])
    except EndpointError as e:
        print(f"FAIL: {e}")
        # the gateway answers with its default model when none is named: report it
        raw = llm.client.chat.completions.create(model="", messages=[{"role": "user", "content": "hi"}], max_tokens=1)
        if getattr(raw, "model", None):
            print(f"The endpoint's default model is '{raw.model}'. Set LLM_MODEL={raw.model} in .env")
        return
    for mode in ("json_schema", "json_object", "none"):
        llm.json_mode = mode
        try:
            out = llm.structured("Return ok=true and n=7.", "go", Ping, retries=0)
            print(f"{mode:12s} OK  {out}")
        except Exception as e:
            print(f"{mode:12s} FAIL {type(e).__name__}: {str(e)[:120]}")
    print("Set LLM_JSON_MODE in .env to the first mode that says OK.")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
