from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import httpx
import pytest

from cc_steer.rendering import DRAFT_CHAR_CAP, tail_messages
from cc_steer.watcher.cascade import DRAFT_SYSTEM, flattened
from cc_steer.watcher.drafter_http import TOP_LOGPROBS, DrafterResponseError, HttpDrafter, sentinel_prob
from cc_steer.watcher.types import Draft

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.anyio

ENDPOINT = "https://watcher.modal.run"
MODEL = "watcher-9b"
RECENT_PROMPT = [{"role": "user", "content": "delete the prod table"}, {"role": "assistant", "content": "on it"}]


def score_payload(*, top_logprobs: list[dict[str, object]]) -> dict[str, object]:
    """A realistic ``max_tokens=1`` scoring body: ``content[0]`` is the answer position reached after
    the ``<think>\\n\\n</think>\\n\\n`` prefill, so its token is the sentinel's ``NO``, not the scaffold opener."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "NO_STEER"},
                "logprobs": {"content": [{"token": "NO", "logprob": -0.01, "top_logprobs": top_logprobs}]},
                "finish_reason": "length",
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
    )


def test_sentinel_prob_from_realistic_payload() -> None:
    payload = score_payload(
        top_logprobs=[
            {"token": "NO", "logprob": math.log(0.8)},
            {"token": " Actually", "logprob": -3.5},
            {"token": "Stop", "logprob": -4.0},
        ]
    )
    assert sentinel_prob(payload) == pytest.approx(0.8)


def test_sentinel_prob_prefers_the_highest_probability_prefix() -> None:
    payload = score_payload(
        top_logprobs=[
            {"token": "N", "logprob": math.log(0.1)},
            {"token": "NO_STE", "logprob": math.log(0.2)},
            {"token": "NO", "logprob": math.log(0.7)},
        ]
    )
    assert sentinel_prob(payload) == pytest.approx(0.7)


def test_sentinel_prob_is_none_at_an_unprefilled_think_scaffold_position() -> None:
    payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "<think>"},
                "logprobs": {
                    "content": [
                        {
                            "token": "<think>",
                            "logprob": -0.001,
                            "top_logprobs": [
                                {"token": "<think>", "logprob": -0.001},
                                {"token": "\n\n", "logprob": -6.5},
                                {"token": "<tool", "logprob": -7.2},
                            ],
                        }
                    ]
                },
            }
        ]
    }
    assert sentinel_prob(payload) is None


def test_sentinel_prob_is_none_when_sentinel_below_top_k() -> None:
    payload = score_payload(top_logprobs=[{"token": "Actually", "logprob": -0.1}, {"token": "Wait", "logprob": -1.2}])
    assert sentinel_prob(payload) is None


def test_sentinel_prob_raises_on_a_shapeless_payload() -> None:
    with pytest.raises(DrafterResponseError, match="no first-token logprobs"):
        sentinel_prob({"error": {"message": "model not found"}})


async def test_draft_abstains_and_skips_generation_when_prob_meets_threshold() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=score_payload(top_logprobs=[{"token": "NO", "logprob": math.log(0.9)}]))

    result = await drafter(handler, threshold=0.5).draft(RECENT_PROMPT)
    assert result == Draft("NO_STEER", pytest.approx(0.9))
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 1


async def test_draft_fires_and_generates_when_below_threshold() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if body["max_tokens"] == 1:
            return httpx.Response(200, json=score_payload(top_logprobs=[{"token": "NO", "logprob": math.log(0.1)}]))
        return httpx.Response(200, json=generate_payload(content="<think>\n\n</think>\n\ndon't touch prod"))

    result = await drafter(handler, threshold=0.5).draft(RECENT_PROMPT)
    assert result == Draft("don't touch prod", pytest.approx(0.1))
    assert [call["max_tokens"] for call in calls] == [1, 256]


async def test_draft_fires_with_none_prob_when_sentinel_absent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["max_tokens"] == 1:
            return httpx.Response(200, json=score_payload(top_logprobs=[{"token": "Wait", "logprob": -0.1}]))
        return httpx.Response(200, json=generate_payload(content="reconsider that migration"))

    result = await drafter(handler, threshold=0.5).draft(RECENT_PROMPT)
    assert result == Draft("reconsider that migration", None)


async def test_scoring_request_carries_the_training_contract_and_logprob_knobs() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        return httpx.Response(200, json=score_payload(top_logprobs=[{"token": "NO", "logprob": math.log(0.9)}]))

    await drafter(handler).draft(RECENT_PROMPT)
    assert seen["url"] == f"{ENDPOINT}/v1/chat/completions"
    assert seen["model"] == MODEL
    assert seen["max_tokens"] == 1
    assert seen["logprobs"] is True
    assert seen["top_logprobs"] == TOP_LOGPROBS
    assert TOP_LOGPROBS <= 40
    assert seen["add_generation_prompt"] is False
    assert seen["continue_final_message"] is True
    assert seen["messages"] == [
        {"role": "system", "content": DRAFT_SYSTEM},
        {"role": "user", "content": flattened(tail_messages(RECENT_PROMPT, DRAFT_CHAR_CAP))},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
    ]


async def test_timeout_from_config_reaches_every_request() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json=score_payload(top_logprobs=[{"token": "NO", "logprob": math.log(0.9)}]))

    made = drafter(handler, timeout=3.5)
    assert made.client.timeout.read == 3.5
    await made.draft(RECENT_PROMPT)
    assert seen["timeout"] == {"connect": 3.5, "read": 3.5, "write": 3.5, "pool": 3.5}


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
        return httpx.Response(200, json={"choices": []})

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
        if json.loads(request.content)["max_tokens"] == 1:
            return httpx.Response(200, json=score_payload(top_logprobs=[{"token": "NO", "logprob": math.log(0.1)}]))
        raise httpx.ReadTimeout("slow generation")

    result = await drafter(handler, threshold=0.5).draft(RECENT_PROMPT)
    assert result == Draft("NO_STEER", None)
    err = capsys.readouterr().err
    assert "failed open to NO_STEER" in err
    assert "ReadTimeout" in err
