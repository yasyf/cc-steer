from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import httpx
import pytest

from cc_steer.rendering import DRAFT_CHAR_CAP, tail_messages
from cc_steer.retrain import sentinel as retrain_sentinel
from cc_steer.watcher.cascade import DRAFT_SYSTEM, flattened
from cc_steer.watcher.drafter_http import COMPLETIONS_PATH, DrafterResponseError, HttpDrafter, sentinel_prob
from cc_steer.watcher.types import Draft

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.anyio

ENDPOINT = "https://watcher.modal.run"
MODEL = "watcher-9b"
RECENT_PROMPT = [{"role": "user", "content": "delete the prod table"}, {"role": "assistant", "content": "on it"}]
TAIL = flattened(tail_messages(RECENT_PROMPT, DRAFT_CHAR_CAP))
SENTINEL = ord("N")


class StubTokenizer:
    """A byte-per-char stand-in for the base tokenizer: chat template is ``role:content`` lines, encode is ``ord``."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        parts = [f"{m['role']}:{m['content']}\n" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


def expected_prompt(tail: str) -> list[int]:
    """The teacher-forced ``prefix + [sentinel]`` the stub renders for a tail: ids up to the answer, then ``N``."""
    return [ord(char) for char in f"system:{DRAFT_SYSTEM}\nuser:{tail}\nassistant:"] + [SENTINEL]


def completions_payload(*, sentinel: int = SENTINEL, logprob: float) -> dict[str, object]:
    """A vLLM ``prompt_logprobs`` body: a leading null, then per-position dicts, the last keyed by the sentinel."""
    return {
        "choices": [
            {
                "prompt_logprobs": [
                    None,
                    {"32": {"logprob": -4.2}},
                    {str(sentinel): {"logprob": logprob}, "999": {"logprob": -9.0}},
                ]
            }
        ]
    }


def generate_payload(*, content: str) -> dict[str, object]:
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


def drafter(
    handler: Callable[[httpx.Request], httpx.Response], *, threshold: float = 0.5, timeout: float = 30.0
) -> HttpDrafter:
    return HttpDrafter(
        endpoint=ENDPOINT,
        model=MODEL,
        threshold=threshold,
        timeout=timeout,
        api_key="sk-test",
        transport=httpx.MockTransport(handler),
        tokenizer=StubTokenizer(),
    )


def test_client_follows_the_scale_to_zero_wake_redirect() -> None:
    assert drafter(lambda request: httpx.Response(200)).client.follow_redirects is True


@pytest.mark.parametrize(
    ("logprob", "expected"),
    [
        pytest.param(math.log(0.8), 0.8, id="high-confidence"),
        pytest.param(math.log(0.05), 0.05, id="fire-signal"),
        pytest.param(0.0, 1.0, id="certain"),
    ],
)
def test_sentinel_prob_reads_the_exact_logprob_from_the_last_position(logprob: float, expected: float) -> None:
    assert sentinel_prob(completions_payload(logprob=logprob), SENTINEL) == pytest.approx(expected)


def test_sentinel_prob_raises_when_the_payload_carries_no_prompt_logprobs() -> None:
    with pytest.raises(DrafterResponseError, match="no prompt_logprobs"):
        sentinel_prob({"choices": [{"text": ""}]}, SENTINEL)


def test_sentinel_prob_raises_when_the_sentinel_is_absent_at_the_answer_position() -> None:
    payload = {"choices": [{"prompt_logprobs": [None, {"999": {"logprob": -1.0}}]}]}
    with pytest.raises(DrafterResponseError, match="carries no logprob"):
        sentinel_prob(payload, SENTINEL)


def test_prefix_and_sentinel_matches_the_canonical_sentinel_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    import athome.train.data as adata

    stub = StubTokenizer()
    monkeypatch.setattr(adata, "tokenizer", lambda mlx_id: stub)
    made = HttpDrafter(endpoint=ENDPOINT, model=MODEL, threshold=0.5, timeout=5.0, tokenizer=stub)
    assert made._prefix_and_sentinel(TAIL) == retrain_sentinel.prefix_and_sentinel(DRAFT_SYSTEM, TAIL, "stub")


async def test_scoring_request_posts_teacher_forced_ids_to_the_completions_path() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        return httpx.Response(200, json=completions_payload(logprob=math.log(0.9)))

    await drafter(handler).draft(RECENT_PROMPT)
    assert seen["url"] == f"{ENDPOINT}{COMPLETIONS_PATH}"
    assert seen["model"] == MODEL
    assert seen["prompt"] == expected_prompt(TAIL)
    assert seen["prompt"][-1] == SENTINEL
    assert seen["max_tokens"] == 1
    assert seen["temperature"] == 0.0
    assert seen["prompt_logprobs"] == 0
    assert "messages" not in seen


async def test_draft_abstains_and_skips_generation_when_prob_meets_threshold() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json=completions_payload(logprob=math.log(0.9)))

    result = await drafter(handler, threshold=0.5).draft(RECENT_PROMPT)
    assert result == Draft("NO_STEER", pytest.approx(0.9))
    assert calls == [COMPLETIONS_PATH]


async def test_draft_fires_and_generates_when_below_threshold() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == COMPLETIONS_PATH:
            return httpx.Response(200, json=completions_payload(logprob=math.log(0.1)))
        return httpx.Response(200, json=generate_payload(content="<think>\n\n</think>\n\ndon't touch prod"))

    result = await drafter(handler, threshold=0.5).draft(RECENT_PROMPT)
    assert result == Draft("don't touch prod", pytest.approx(0.1))
    assert calls == [COMPLETIONS_PATH, "/v1/chat/completions"]


async def test_generation_carries_the_training_contract_messages() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == COMPLETIONS_PATH:
            return httpx.Response(200, json=completions_payload(logprob=math.log(0.1)))
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=generate_payload(content="reconsider that migration"))

    await drafter(handler, threshold=0.5).draft(RECENT_PROMPT)
    assert seen["model"] == MODEL
    assert seen["max_tokens"] == 256
    assert seen["temperature"] == 0.0
    assert seen["messages"] == [
        {"role": "system", "content": DRAFT_SYSTEM},
        {"role": "user", "content": TAIL},
    ]


async def test_timeout_from_config_reaches_every_request() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json=completions_payload(logprob=math.log(0.9)))

    made = drafter(handler, timeout=3.5)
    assert made.client.timeout.read == 3.5
    await made.draft(RECENT_PROMPT)
    assert seen["timeout"] == {"connect": 3.5, "read": 3.5, "write": 3.5, "pool": 3.5}


async def test_scoring_request_forwards_the_api_key_as_a_bearer_header() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json=completions_payload(logprob=math.log(0.9)))

    await drafter(handler).draft(RECENT_PROMPT)
    assert seen["auth"] == "Bearer sk-test"


async def test_fail_open_on_transport_error(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("endpoint unreachable")

    result = await drafter(handler).draft(RECENT_PROMPT)
    assert result == Draft("NO_STEER", None)
    err = capsys.readouterr().err
    assert "failed open to NO_STEER" in err
    assert "ConnectError" in err


async def test_fail_open_on_timeout(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow endpoint")

    result = await drafter(handler).draft(RECENT_PROMPT)
    assert result == Draft("NO_STEER", None)
    assert "ReadTimeout" in capsys.readouterr().err


async def test_fail_open_on_http_error_status(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "scaling up"})

    result = await drafter(handler).draft(RECENT_PROMPT)
    assert result == Draft("NO_STEER", None)
    assert "failed open to NO_STEER" in capsys.readouterr().err


async def test_fail_open_on_malformed_payload(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{}]})

    result = await drafter(handler).draft(RECENT_PROMPT)
    assert result == Draft("NO_STEER", None)
    assert "DrafterResponseError" in capsys.readouterr().err


async def test_fail_open_on_non_json_body(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>502 Bad Gateway</html>", headers={"content-type": "text/html"})

    result = await drafter(handler).draft(RECENT_PROMPT)
    assert result == Draft("NO_STEER", None)
    assert "failed open to NO_STEER" in capsys.readouterr().err


async def test_fail_open_when_generation_faults_below_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == COMPLETIONS_PATH:
            return httpx.Response(200, json=completions_payload(logprob=math.log(0.1)))
        raise httpx.ReadTimeout("slow generation")

    result = await drafter(handler, threshold=0.5).draft(RECENT_PROMPT)
    assert result == Draft("NO_STEER", None)
    err = capsys.readouterr().err
    assert "failed open to NO_STEER" in err
    assert "ReadTimeout" in err
