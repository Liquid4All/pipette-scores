"""samples_digest: the content digest a client pins a copy of /samples with."""

import hashlib

from pipette_scores.hashing import samples_digest
from pipette_scores.types import ChatMessage, EvalSample

SAMPLES = [
    EvalSample(
        id="a1",
        messages=[
            ChatMessage(role="system", content="Think first."),
            ChatMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                    {"type": "text", "text": "Which mood? 心情\nA. Sad\nB. Happy"},
                ],
            ),
        ],
    ),
    EvalSample(id="b2", messages=[ChatMessage(role="user", content='Is there a dog? "Yes" or no.')]),
]

# The canonical form written out by hand: keys sorted, no whitespace, UTF-8, one
# line per sample. A client may recompute the digest from a /samples body this way.
CANONICAL = (
    '{"id":"a1","messages":[{"content":"Think first.","role":"system"},{"content":['
    '{"image_url":{"url":"data:image/png;base64,iVBORw0KGgo="},"type":"image_url"},'
    '{"text":"Which mood? 心情\\nA. Sad\\nB. Happy","type":"text"}],"role":"user"}]}\n'
    '{"id":"b2","messages":[{"content":"Is there a dog? \\"Yes\\" or no.","role":"user"}]}\n'
)


def test_the_digest_is_the_sha256_of_the_canonical_form():
    assert samples_digest(SAMPLES) == f"sha256:{hashlib.sha256(CANONICAL.encode()).hexdigest()}"


def test_the_digest_is_pinned():
    # Digests are pinned downstream (a corpus records the one it was built from),
    # so a change to the canonical form must be deliberate: it moves every pin.
    # mgmt's Rust digest pins this same value.
    assert samples_digest(SAMPLES) == "sha256:5654ef883c378c4325cfe70594545d4a19c094defdf21b32487f9de94a6b4849"


def test_awkward_text_is_pinned():
    # Escapes another language could write differently: control characters,
    # DEL, quotes, backslashes, an astral character, U+2028. mgmt recomputes
    # this digest in Rust (pipette-plan-types `vlm_eval_corpus`) and pins the
    # same value, so a corpus it builds is checked against the scorer's.
    sample = EvalSample(
        id="c3",
        messages=[ChatMessage(role="user", content='tab\there\u0001\u001f\u007f \\ "q" / é 😀 \u2028\r\n\u0008\u000c')],
    )
    assert samples_digest([sample]) == "sha256:398cc6ebdc6f48b76c7bc670f047e4f82df20b1443fa0a30b2926808c29238f6"


def test_the_digest_follows_what_is_served():
    reworded = [SAMPLES[0], EvalSample(id="b2", messages=[ChatMessage(role="user", content="Is there a cat?")])]
    assert len({samples_digest(SAMPLES), samples_digest(reworded), samples_digest(SAMPLES[::-1])}) == 3


def test_an_empty_set_has_a_digest():
    assert samples_digest([]) == f"sha256:{hashlib.sha256(b'').hexdigest()}"


_AUDIO = {"type": "input_audio", "input_audio": {"data": "UklGRiQAAABXQVZF", "format": "wav"}}
# An ASR sample's shape: an input_audio part (base64 WAV) and the optional
# chat_template_kwargs, present only when a profile sets them.
ASR_SAMPLES = [
    EvalSample(
        id="d4",
        messages=[ChatMessage(role="user", content=[_AUDIO])],
        chat_template_kwargs={"transcription": True, "language": "nl"},
    ),
    EvalSample(
        id="e5",
        messages=[
            ChatMessage(role="system", content="Perform ASR."),
            ChatMessage(role="user", content=[{"type": "text", "text": "Transcribe."}, _AUDIO]),
        ],
    ),
]


def test_an_asr_sample_is_pinned():
    # mgmt's Rust digest pins this value too, so an ASR corpus it packs is
    # checked against the scorer's.
    canonical = (
        '{"chat_template_kwargs":{"language":"nl","transcription":true},"id":"d4","messages":[{"content":['
        '{"input_audio":{"data":"UklGRiQAAABXQVZF","format":"wav"},"type":"input_audio"}],"role":"user"}]}\n'
        '{"id":"e5","messages":[{"content":"Perform ASR.","role":"system"},{"content":[{"text":"Transcribe.",'
        '"type":"text"},{"input_audio":{"data":"UklGRiQAAABXQVZF","format":"wav"},"type":"input_audio"}],'
        '"role":"user"}]}\n'
    )
    digest = samples_digest(ASR_SAMPLES)
    assert digest == f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
    assert digest == "sha256:b49fae68dfd0d735e59dc3df621d8b6189ec7977fc5830b54b368bd10f396ef7"
