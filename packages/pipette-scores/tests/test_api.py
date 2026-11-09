import hashlib
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import asyncio
import threading
import time

from pipette_scores import dataset_catalog, packages
from pipette_scores.api.app import _APP_VERSION, _DIGESTS, _SERVED, _samples_digest_or_error, app
from pipette_scores.hashing import samples_digest
from pipette_scores.types import (
    EvalSample,
    ChatMessage,
    IFStructSample,
    SampleScore,
    VlmSample,
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


@pytest.fixture(autouse=True)
def _fresh_digests():
    # Digests are memoized per (eval, dataset, profile); the tests patch the
    # loader under the same keys with different samples.
    _DIGESTS.clear()
    _SERVED.clear()
    yield
    _DIGESTS.clear()
    _SERVED.clear()


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

    def test_prompt_profile_is_passed_to_the_loader(self, client):
        prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
        with patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts) as load:
            resp = client.get("/evals/mmstar/datasets/test/samples?prompt_profile=internvl")
        assert resp.status_code == 200
        load.assert_called_once_with("mmstar", "test", "internvl")

    @pytest.mark.parametrize("profile", ["glm_v", "kimi_vl", "nemotron_think", "phi4_rv_think", "mimo_vl"])
    def test_the_new_vlm_prompt_profiles_are_served(self, client, profile):
        prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
        with patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts) as load:
            resp = client.get(f"/evals/mmstar/datasets/test/samples?prompt_profile={profile}")
        assert resp.status_code == 200
        load.assert_called_once_with("mmstar", "test", profile)

    def test_unknown_prompt_profile_returns_400(self, client):
        resp = client.get("/evals/mmstar/datasets/test/samples?prompt_profile=nope")
        assert resp.status_code == 400
        assert "Unknown prompt profile: nope" in resp.json()["detail"]

    def test_prompt_profile_on_a_text_eval_returns_400(self, client):
        resp = client.get("/evals/ifstruct/datasets/test/samples?prompt_profile=internvl")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /evals/{eval_id}/datasets/{dataset_name}/samples/digest
# ---------------------------------------------------------------------------

VLM_SAMPLES = [
    EvalSample(
        id="a1",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                    {"type": "text", "text": "Which mood? 心情\nA. Sad\nB. Happy"},
                ],
            )
        ],
    ),
    EvalSample(id="b2", messages=[ChatMessage(role="user", content="Is there a dog?")]),
]


class TestSamplesDigest:
    def test_the_digest_is_that_of_the_served_body(self, client):
        with (
            patch("pipette_scores.api.app._load_prompts_cached", return_value=VLM_SAMPLES),
            patch("pipette_scores.api.app._build_prompts", return_value=VLM_SAMPLES),
        ):
            digest = client.get("/evals/mmstar/datasets/test/samples/digest")
            served = client.get("/evals/mmstar/datasets/test/samples")
        assert digest.status_code == 200
        # Recomputed from the HTTP body alone, as a client holding a copy could.
        lines = "".join(
            json.dumps(s, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            for s in served.json()["samples"]
        )
        expected = f"sha256:{hashlib.sha256(lines.encode()).hexdigest()}"
        assert digest.json() == {"digest": expected, "sample_count": 2}
        assert served.headers["X-Samples-Digest"] == expected

    def test_the_header_is_hashed_from_the_samples_served(self, client):
        # /samples hashes the list it serves; nothing is built a second time.
        with (
            patch("pipette_scores.api.app._load_prompts_cached", return_value=VLM_SAMPLES),
            patch("pipette_scores.api.app._build_prompts") as build,
        ):
            served = client.get("/evals/mmstar/datasets/test/samples")
        assert served.headers["X-Samples-Digest"] == samples_digest(VLM_SAMPLES)
        build.assert_not_called()

    def test_the_digest_route_keeps_no_samples(self, client):
        # A probe asks for every pinned set every minute: the route builds the
        # samples to hash them and holds on to the digest alone.
        with (
            patch("pipette_scores.api.app._load_prompts_cached") as cached,
            patch("pipette_scores.api.app._build_prompts", return_value=VLM_SAMPLES) as build,
        ):
            resp = client.get("/evals/mmstar/datasets/test/samples/digest?prompt_profile=qwen3_5")
        assert resp.status_code == 200
        build.assert_called_once_with("mmstar", "test", "qwen3_5")
        cached.assert_not_called()

    def test_the_digest_is_computed_once(self, client):
        with (
            patch("pipette_scores.api.app._build_prompts", return_value=VLM_SAMPLES) as build,
            patch("pipette_scores.api.app.samples_digest", wraps=samples_digest) as digest,
        ):
            first = client.get("/evals/mmstar/datasets/test/samples/digest").json()
            second = client.get("/evals/mmstar/datasets/test/samples/digest").json()
        assert first == second
        assert build.call_count == 1
        assert digest.call_count == 1

    def test_concurrent_requests_build_a_cold_set_once(self):
        # After a deploy every pinned set is cold, and a cache probes them at
        # once: one build per set, and one cold build at a time in the process.
        builds, running, peak = [], [0], [0]
        guard = threading.Lock()

        def slow_build(eval_id, dataset_name, prompt_profile):
            with guard:
                builds.append(prompt_profile)
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with guard:
                running[0] -= 1
            return VLM_SAMPLES

        async def probe_all():
            return await asyncio.gather(
                *(_samples_digest_or_error("mmstar", "test", profile) for profile in [None] * 4 + ["qwen3_5"] * 4)
            )

        with patch("pipette_scores.api.app._build_prompts", side_effect=slow_build):
            digests = asyncio.run(probe_all())
        assert sorted(builds, key=str) == [None, "qwen3_5"]
        assert peak[0] == 1
        assert len({d for d, _ in digests}) == 1

    def test_a_served_list_is_hashed_once(self, client):
        # The cached list /samples serves is the same object every time.
        with (
            patch("pipette_scores.api.app._load_prompts_cached", return_value=VLM_SAMPLES),
            patch("pipette_scores.api.app.samples_digest", wraps=samples_digest) as digest,
        ):
            first = client.get("/evals/mmstar/datasets/test/samples")
            second = client.get("/evals/mmstar/datasets/test/samples")
        assert first.headers["X-Samples-Digest"] == second.headers["X-Samples-Digest"]
        assert digest.call_count == 1

    def test_a_malformed_samples_digest_is_a_bad_request(self, client):
        with patch("pipette_scores.api.app._build_prompts") as build:
            resp = client.post(
                "/score",
                json={"eval_id": "mmstar", "dataset_name": "test", "completions": [], "samples_digest": "garbage"},
            )
        assert resp.status_code == 422
        build.assert_not_called()

    def test_the_header_is_always_the_bodys(self, client):
        # A digest cached for another body never rides on this one.
        _DIGESTS[("mmstar", "test", None)] = ("sha256:" + "0" * 64, 2)
        with patch("pipette_scores.api.app._load_prompts_cached", return_value=VLM_SAMPLES):
            served = client.get("/evals/mmstar/datasets/test/samples")
            digest = client.get("/evals/mmstar/datasets/test/samples/digest")
        assert served.headers["X-Samples-Digest"] == samples_digest(VLM_SAMPLES)
        assert digest.json()["digest"] == samples_digest(VLM_SAMPLES)

    def test_a_subclasss_own_fields_are_hashed_as_served(self, client):
        # The response writes samples through the declared EvalSample, so a
        # loader's subclass field never reaches the body, nor the digest.
        class Extended(EvalSample):
            generation_params: dict = {"temperature": 0.0}

        extended = [Extended(id=s.id, messages=s.messages) for s in VLM_SAMPLES]
        with patch("pipette_scores.api.app._load_prompts_cached", return_value=extended):
            served = client.get("/evals/mmstar/datasets/test/samples")
        assert "generation_params" not in served.text
        assert served.headers["X-Samples-Digest"] == samples_digest(VLM_SAMPLES)

    @pytest.mark.parametrize(
        ("path", "status"),
        [
            ("/evals/nonexistent/datasets/test/samples/digest", 404),
            ("/evals/ifstruct/datasets/does-not-exist/samples/digest", 404),
            ("/evals/mmstar/datasets/test/samples/digest?prompt_profile=nope", 400),
            ("/evals/ifstruct/datasets/test/samples/digest?prompt_profile=internvl", 400),
        ],
    )
    def test_it_refuses_what_samples_refuses(self, client, path, status):
        assert client.get(path).status_code == status


class TestSamplesDigestFromThePackagesIndex:
    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dataset_catalog, "_DEFAULT_ROOT", tmp_path)
        for eval_id in ("mmstar", "asr_fleurs_nl_nl"):
            (tmp_path / "datasets" / eval_id / "test").mkdir(parents=True)
        monkeypatch.setattr("pipette_scores.api.app._PACKAGES", packages.LocalStore(tmp_path / "packages"))
        return tmp_path / "packages"

    def _index(self, store, eval_id: str, digests: dict[str | None, str], dataset: str = "test") -> None:
        path = store / packages.index_key(_APP_VERSION, eval_id, dataset)
        path.parent.mkdir(parents=True)
        profiles = [{"prompt_profile": p, "samples_digest": d, "sample_count": 2} for p, d in digests.items()]
        path.write_text(json.dumps({"eval_id": eval_id, "dataset_version": dataset, "profiles": profiles}))

    def test_a_set_the_index_lists_is_never_built(self, client, store):
        digests = {None: "sha256:" + "1" * 64, "qwen3_5": "sha256:" + "2" * 64}
        self._index(store, "mmstar", digests)
        self._index(store, "asr_fleurs_nl_nl", {"voxtral": "sha256:" + "3" * 64})
        with (
            patch("pipette_scores.api.app._build_prompts", side_effect=AssertionError("built")),
            patch("pipette_scores.api.app.dataset_catalog.asr_prompt_digests", side_effect=AssertionError("built")),
        ):
            for profile, digest in digests.items():
                query = f"?prompt_profile={profile}" if profile else ""
                resp = client.get(f"/evals/mmstar/datasets/test/samples/digest{query}")
                assert resp.json() == {"digest": digest, "sample_count": 2}
            resp = client.get("/evals/asr_fleurs_nl_nl/datasets/test/samples/digest?prompt_profile=voxtral")
            assert resp.json() == {"digest": "sha256:" + "3" * 64, "sample_count": 2}

    def test_an_indexed_set_this_server_lacks_is_not_found(self, client, store):
        self._index(store, "mmstar", {None: "sha256:" + "1" * 64}, "not-synced")
        self._index(store, "asr_fleurs_nl_nl", {"voxtral": "sha256:" + "3" * 64}, "not-synced")
        assert client.get("/evals/mmstar/datasets/not-synced/samples/digest").status_code == 404
        resp = client.get("/evals/asr_fleurs_nl_nl/datasets/not-synced/samples/digest?prompt_profile=voxtral")
        assert resp.status_code == 404

    def test_a_set_without_an_index_is_built(self, client, store):
        with patch("pipette_scores.api.app._build_prompts", return_value=VLM_SAMPLES) as build:
            resp = client.get("/evals/mmstar/datasets/test/samples/digest")
        assert resp.json() == {"digest": samples_digest(VLM_SAMPLES), "sample_count": 2}
        build.assert_called_once_with("mmstar", "test", None)

    def test_an_unreadable_index_is_an_error_not_a_build(self, client, store):
        path = store / packages.index_key(_APP_VERSION, "mmstar", "test")
        path.parent.mkdir(parents=True)
        path.write_text("{")
        with patch("pipette_scores.api.app._build_prompts", return_value=VLM_SAMPLES) as build:
            resp = client.get("/evals/mmstar/datasets/test/samples/digest")
        assert resp.status_code == 500
        build.assert_not_called()


# ---------------------------------------------------------------------------
# POST /score
# ---------------------------------------------------------------------------


class TestScoreBindsToTheSamplesGiven:
    def test_scoring_is_refused_when_the_samples_moved(self, client):
        with patch("pipette_scores.api.app._build_prompts", return_value=VLM_SAMPLES):
            resp = client.post(
                "/score",
                json={
                    "eval_id": "mmstar",
                    "dataset_name": "test",
                    "completions": [],
                    "samples_digest": "sha256:" + "0" * 64,
                },
            )
        assert resp.status_code == 409
        assert "not to these prompts" in resp.json()["detail"]

    def test_scoring_goes_on_when_the_samples_are_the_ones_given(self, client):
        with (
            patch("pipette_scores.api.app._build_prompts", return_value=VLM_SAMPLES),
            patch("pipette_scores.api.app._load_dataset_or_error", return_value=[]),
            patch("pipette_scores.api.app._load_prompts_cached", return_value=VLM_SAMPLES),
            patch("pipette_scores.api.app.scoring.score", return_value=([], {})),
        ):
            resp = client.post(
                "/score",
                json={
                    "eval_id": "mmstar",
                    "dataset_name": "test",
                    "completions": [],
                    "samples_digest": samples_digest(VLM_SAMPLES),
                },
            )
        assert resp.status_code == 200, resp.text


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

    def test_scored_samples_carry_the_profiles_prompts(self, client):
        prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
        with (
            patch("pipette_scores.api.app._load_dataset_cached", return_value=[_ifstruct_sample()]),
            patch("pipette_scores.api.app.scoring.score", return_value=([], {})),
            patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts) as load,
        ):
            resp = client.post(
                "/score",
                json={
                    "eval_id": "mmstar",
                    "dataset_name": "test",
                    "completions": [{"id": "1", "completion": "A"}],
                    "prompt_profile": "internvl",
                },
            )
        assert resp.status_code == 200
        load.assert_called_once_with("mmstar", "test", "internvl")

    @pytest.mark.parametrize(
        "completions, names",
        [
            ([{"id": "a1", "completion": "B"}], "unanswered: b2"),
            (
                [{"id": "a1", "completion": "B"}, {"id": "b2", "completion": "", "stop_reason": "failure"}],
                "failed: b2",
            ),
        ],
    )
    def test_a_vlm_eval_is_scored_only_with_an_answer_to_every_sample(self, client, completions, names):
        dataset = [VlmSample(id=id, prompt="", answer="A", raw_doc={}) for id in ("a1", "b2")]
        with (
            patch("pipette_scores.api.app._load_dataset_cached", return_value=dataset),
            patch("pipette_scores.api.app.scoring.score") as score,
        ):
            resp = client.post("/score", json={"eval_id": "pope", "dataset_name": "test", "completions": completions})
        assert resp.status_code == 400
        assert names in resp.json()["detail"]
        score.assert_not_called()

    def test_scored_samples_keep_their_text_but_not_their_images(self, client):
        scores = [SampleScore(id="a1", is_correct=True), SampleScore(id="b2", is_correct=False)]
        dataset = [VlmSample(id=id, prompt="", answer="A", raw_doc={}) for id in ("a1", "b2")]
        with (
            patch("pipette_scores.api.app._load_dataset_cached", return_value=dataset),
            patch("pipette_scores.api.app.scoring.score", return_value=(scores, {})),
            patch("pipette_scores.api.app._load_prompts_cached", return_value=VLM_SAMPLES),
        ):
            resp = client.post(
                "/score",
                json={
                    "eval_id": "mmstar",
                    "dataset_name": "test",
                    "completions": [{"id": "a1", "completion": "B"}, {"id": "b2", "completion": "no"}],
                },
            )
        assert resp.status_code == 200
        messages = [s["messages"] for s in resp.json()["scored_samples"]]
        assert messages == [
            [{"role": "user", "content": [{"type": "text", "text": "Which mood? 心情\nA. Sad\nB. Happy"}]}],
            [{"role": "user", "content": "Is there a dog?"}],
        ]
        # What the samples route serves is unchanged.
        assert VLM_SAMPLES[0].messages[0].content[0]["type"] == "image_url"

    @pytest.mark.parametrize("eval_id, profile", [("mmstar", "nope"), ("ifstruct", "internvl")])
    def test_score_refuses_a_prompt_profile_it_cannot_apply(self, client, eval_id, profile):
        resp = client.post(
            "/score",
            json={"eval_id": eval_id, "dataset_name": "test", "completions": [], "prompt_profile": profile},
        )
        assert resp.status_code == 400

    @pytest.mark.parametrize("enable_thinking, blanked", [(True, 1), (False, 0), (None, 0)])
    def test_score_reads_stop_reason_and_enable_thinking(self, client, enable_thinking, blanked):
        samples = [_ifstruct_sample()]
        prompts = [EvalSample(id="1", messages=[ChatMessage(role="user", content="Q?")])]
        completion = 'Draft:\n```json\n{"name": "Alice"}\n```\nNow check whether the'
        body = {
            "eval_id": "ifstruct",
            "dataset_name": "test",
            # Keys the scorer has no use for, such as stop_detail, are ignored.
            "completions": [{"id": "1", "completion": completion, "stop_reason": "truncated", "stop_detail": "limit"}],
        }
        if enable_thinking is not None:
            body["enable_thinking"] = enable_thinking
        with (
            patch("pipette_scores.api.app._load_dataset_cached", return_value=samples),
            patch("pipette_scores.api.app._load_prompts_cached", return_value=prompts),
        ):
            resp = client.post("/score", json=body)
        assert resp.status_code == 200
        result = resp.json()
        # The cut draft is graded as written; only the strict view blanks it.
        assert result["scored_samples"][0]["is_correct"] is True
        # The completion is echoed as received, not as graded.
        assert result["scored_samples"][0]["completion"] == completion
        assert result["context"]["ifstruct_blanked_truncated"] == blanked
        assert result["context"]["ifstruct_pass_at_1_blanked"] == (0.0 if blanked else 1.0)

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


def test_a_worker_keeps_a_bounded_number_of_prompt_sets():
    """Each (eval, dataset, profile) is a set of its own, images included: a
    worker keeps the few most recent, outside the catalog's own cache."""
    from pipette_scores.api import app as app_module
    from pipette_scores import dataset_catalog

    assert app_module._load_prompts_cached.cache_info().maxsize == app_module._PROMPT_SETS
    app_module._load_prompts_cached.cache_clear()
    before = dataset_catalog.load_prompt_samples.cache_info().currsize
    with patch.object(dataset_catalog.load_prompt_samples, "__wrapped__", return_value=[]) as built:
        for profile in ("a", "b", "c", "d", "e", "a"):
            app_module._load_prompts_cached("mmstar", "v", profile)
    # "a" was evicted by the fifth set and built again.
    assert built.call_count == 6
    assert dataset_catalog.load_prompt_samples.cache_info().currsize == before
    app_module._load_prompts_cached.cache_clear()


def test_openapi_describes_the_samples_and_scored_samples():
    """The wrap serializers that omit unset fields keep the published schemas."""
    schemas = app.openapi()["components"]["schemas"]
    assert {"id", "messages", "chat_template_kwargs"} <= set(schemas["EvalSample"]["properties"])
    assert {"id", "completion", "is_correct", "measures"} <= set(schemas["ScoredSample"]["properties"])
