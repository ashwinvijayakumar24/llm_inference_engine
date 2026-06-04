"""FastAPI OpenAI-compatible HTTP server: /v1/chat/completions with SSE streaming."""

from fastapi import FastAPI

app = FastAPI(title="LLM Inference Engine")


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    raise NotImplementedError
