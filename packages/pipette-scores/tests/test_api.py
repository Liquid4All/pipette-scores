from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pipette_scores.api.app import _APP_VERSION, app
from pipette_scores.types import (
    EvalSample,
    ChatMessage,
    IFStructSample,
)


def _ifstruct_sample(id="1"):
    return IFStructSample(
        id=id,
        seed=1,
        prompt="Generate JSON",
        json_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        top_level_count=None,
        top_level_key=None,
        require_wrapper_key=False,
        require_code_block=False,
        require_no_commentary=False,
        output_format="json",
        entity_type="json",
    )


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET / and /favicon.ico
# ---------------------------------------------------------------------------


class TestIndex:
    def test_renders_with_version_and_no_template_marker(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        body = resp.text
        assert "Welcome to pipette-scores" in body
        assert _APP_VERSION in body
        # If the marker is renamed on one side but not the other, the literal
        # `{version}` would leak through into the rendered page.
        assert "{version}" not in body

    def test_favicon_returns_204(self, client):
        resp = client.get("/favicon.ico")
        assert resp.status_code == 204
        assert resp.content == b""


# ---------------------------------------------------------------------------
# GET /evals/{eval_id}/datasets/{dataset_name}/samples
# ---------------------------------------------------------------------------


class TestGetSamples:
    def test_unknown_eval_returns_404(self, client):
        resp = client.get("/evals/nonexistent/datasets/test/samples")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "Eval not found: nonexistent"}

    def test_unknown_dataset_returns_404(self, client):
        # Valid eval, bogus dataset — loader raises FileNotFoundError → 404.
        resp = client.get("/evals/ifstruct/datasets/does-not-exist/samples")
        assert resp.status_code == 404

    def test_success(self, client):
        prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
        with patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts):
            resp = client.get("/evals/ifstruct/datasets/test/samples")
        assert resp.status_code == 200
        body = resp.json()
        assert "generation_params" not in body
        assert len(body["samples"]) == 1
        assert body["samples"][0]["messages"][0]["content"] == "Q?"


# ---------------------------------------------------------------------------
# POST /score
# ---------------------------------------------------------------------------


class TestScoreEndpoint:
    def test_unknown_eval_returns_404(self, client):
        resp = client.post(
            "/score",
            json={"eval_id": "nonexistent", "dataset_name": "test", "completions": []},
        )
        assert resp.status_code == 404

    def test_missing_eval_id_returns_422(self, client):
        resp = client.post("/score", json={"dataset_name": "test", "completions": []})
        assert resp.status_code == 422

    def test_missing_dataset_name_returns_422(self, client):
        resp = client.post("/score", json={"eval_id": "ifstruct", "completions": []})
        assert resp.status_code == 422

    def test_score_round_trip(self, client):
        samples = [_ifstruct_sample()]
        prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
        with (
            patch("pipette_scores.api.app._load_dataset_cached", return_value=samples),
            patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts),
        ):
            resp = client.post(
                "/score",
                json={
                    "eval_id": "ifstruct",
                    "dataset_name": "test",
                    "completions": [{"id": "1", "completion": '{"name": "Alice"}'}],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["runtime_version"] == "dev"
        assert len(body["scored_samples"]) == 1
        sample = body["scored_samples"][0]
        assert sample["id"] == "1"
        assert sample["is_correct"] is True
        assert sample["completion"] == '{"name": "Alice"}'
        assert sample["messages"][0]["content"] == "Q?"

    def test_unknown_completion_id_returns_400(self, client):
        samples = [_ifstruct_sample()]
        prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
        with (
            patch("pipette_scores.api.app._load_dataset_cached", return_value=samples),
            patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts),
        ):
            resp = client.post(
                "/score",
                json={
                    "eval_id": "ifstruct",
                    "dataset_name": "test",
                    "completions": [{"id": "bogus", "completion": '{"name": "Alice"}'}],
                },
            )
        assert resp.status_code == 400
        assert "bogus" in resp.json()["detail"]

    def test_duplicate_completion_ids_return_400(self, client):
        samples = [_ifstruct_sample()]
        prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
        with (
            patch("pipette_scores.api.app._load_dataset_cached", return_value=samples),
            patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts),
        ):
            resp = client.post(
                "/score",
                json={
                    "eval_id": "ifstruct",
                    "dataset_name": "test",
                    "completions": [
                        {"id": "1", "completion": '{"name": "Alice"}'},
                        {"id": "1", "completion": '{"name": "Bob"}'},
                    ],
                },
            )
        assert resp.status_code == 400
        assert "duplicate" in resp.json()["detail"].lower()

    def test_score_wrong_answer(self, client):
        samples = [_ifstruct_sample()]
        prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
        with (
            patch("pipette_scores.api.app._load_dataset_cached", return_value=samples),
            patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts),
        ):
            resp = client.post(
                "/score",
                json={
                    "eval_id": "ifstruct",
                    "dataset_name": "test",
                    "completions": [{"id": "1", "completion": "This is not JSON at all."}],
                },
            )
        body = resp.json()
        assert body["scored_samples"][0]["is_correct"] is False
        assert body["scored_samples"][0]["completion"] == "This is not JSON at all."
