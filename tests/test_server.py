"""
HTTP server smoke tests — marked slow (model load + generation).
Run with: pytest -m slow tests/test_server.py

Starts uvicorn in a subprocess, waits for ready, fires requests, checks responses.
"""

import json
import subprocess
import sys
import time

import pytest

pytest.importorskip("httpx")
import httpx  # noqa: E402

BASE_URL = "http://127.0.0.1:8765"
PAYLOAD  = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}


@pytest.fixture(scope="module")
def server():
    """Start uvicorn server subprocess, yield, then terminate."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "engine.server:app",
            "--host", "127.0.0.1",
            "--port", "8765",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Wait up to 120s for the server to be ready (model load is slow)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            httpx.get(f"{BASE_URL}/docs", timeout=2)
            break
        except Exception:
            time.sleep(2)
    else:
        proc.terminate()
        pytest.fail("Server did not start within 120s")

    yield proc

    proc.terminate()
    proc.wait()


@pytest.mark.slow
def test_non_stream_response(server):
    resp = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={**PAYLOAD, "stream": False},
        timeout=120,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(data["choices"][0]["message"]["content"], str)
    assert len(data["choices"][0]["message"]["content"]) > 0


@pytest.mark.slow
def test_stream_response(server):
    chunks = []
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        json={**PAYLOAD, "stream": True},
        timeout=120,
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                chunks.append(chunk)

    assert len(chunks) >= 2  # at least one content chunk + final stop chunk
    # Last chunk should have finish_reason=stop
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # Content chunks should have delta.content
    content_chunks = [c for c in chunks if c["choices"][0]["delta"].get("content")]
    assert len(content_chunks) >= 1
