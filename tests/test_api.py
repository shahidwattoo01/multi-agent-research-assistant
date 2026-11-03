import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from research_assistant.api import create_app  # noqa: E402
from research_assistant.graph import ResearchOrchestrator  # noqa: E402
from tests.fakes import FakeFetcher, FakeLLM, FakeSearch  # noqa: E402


@pytest.fixture
def client(settings):
    def factory(backend, **overrides):
        s = settings.model_copy(update={k: v for k, v in overrides.items() if v is not None})
        return ResearchOrchestrator(s, FakeLLM(), FakeSearch(), FakeFetcher())

    return TestClient(create_app(factory))


def _wait(client, run_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/research/{run_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_submit_poll_and_fetch_report(client):
    resp = client.post("/research", json={"question": "How have renewable costs changed?"})
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    body = _wait(client, run_id)
    assert body["status"] == "completed", body.get("error")
    assert body["report"]["title"] == "Renewable Energy Cost Trends"

    md = client.get(f"/research/{run_id}/report.md")
    assert md.status_code == 200 and md.text.startswith("# Renewable Energy Cost Trends")

    with client.stream("GET", f"/research/{run_id}/events") as stream:
        payload = "".join(stream.iter_text())
    assert "event: plan" in payload and "event: end" in payload

    assert any(r["run_id"] == run_id for r in client.get("/research").json())


def test_validation_and_404(client):
    assert client.post("/research", json={"question": "hi"}).status_code == 422
    assert client.get("/research/run_missing").status_code == 404
