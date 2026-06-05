"""FastAPI OpenAI-compatible HTTP server: /v1/chat/completions with SSE streaming."""

import json
import time
import uuid
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="LLM Inference Engine")

_model      = None
_tokenizer  = None


@app.on_event("startup")
def _load_model():
    global _model, _tokenizer
    from transformers import AutoTokenizer

    from engine.loader import load_config, load_weights
    from engine.model import LlamaModel

    weights_path = "weights"
    config       = load_config(weights_path)
    weights      = load_weights(weights_path, config)
    _model       = LlamaModel(weights, config)
    _tokenizer   = AutoTokenizer.from_pretrained(weights_path)


class _Message(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    messages:    list[_Message]
    max_tokens:  int   = 256
    temperature: float = 1.0
    top_p:       float = 1.0
    top_k:       int   = 0
    seed:        int | None = None
    stream:      bool  = False


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    from engine.sampler import get_sampler
    from engine.scheduler import generate

    messages   = [{"role": m.role, "content": m.content} for m in request.messages]
    token_ids  = _tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    sampler_fn = get_sampler(
        temp  = request.temperature,
        top_k = request.top_k,
        top_p = request.top_p,
        seed  = request.seed,
    )

    model_id      = "llama-3.2-1b-instruct"
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created       = int(time.time())

    if request.stream:
        async def _event_stream() -> AsyncGenerator[str, None]:
            for token_id in generate(_model, token_ids, sampler_fn, max_tokens=request.max_tokens):
                text  = _tokenizer.decode([token_id], skip_special_tokens=True)
                chunk = {
                    "id":      completion_id,
                    "object":  "chat.completion.chunk",
                    "created": created,
                    "model":   model_id,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            final = {
                "id":      completion_id,
                "object":  "chat.completion.chunk",
                "created": created,
                "model":   model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    # Non-streaming: collect all tokens then return
    tokens = list(generate(_model, token_ids, sampler_fn, max_tokens=request.max_tokens))
    text   = _tokenizer.decode(tokens, skip_special_tokens=True)
    return {
        "id":      completion_id,
        "object":  "chat.completion",
        "created": created,
        "model":   model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage":   {
            "prompt_tokens":     len(token_ids),
            "completion_tokens": len(tokens),
            "total_tokens":      len(token_ids) + len(tokens),
        },
    }
