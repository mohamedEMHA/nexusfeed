"""Local unit-test for the scoring extraction/normalization changes.

Run from the nexusfeed directory:
    python test_scoring_fixes.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch

# Ensure local module is imported.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import news_bot
from news_bot import (
    _extract_chat_completion_content,
    normalize_scoring_result,
)


def assert_eq(label, got, expected):
    if got == expected:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}: got={got!r} expected={expected!r}")
        raise SystemExit(1)


def test_extract_chat_completion_content():
    print("test_extract_chat_completion_content")
    # OpenAI/Mistral standard shape
    assert_eq(
        "choices.message.content",
        _extract_chat_completion_content(
            {"choices": [{"message": {"content": "hello world"}}]}
        ),
        "hello world",
    )
    # Nested JSON string content
    assert_eq(
        "choices.text fallback",
        _extract_chat_completion_content(
            {"choices": [{"text": "fallback text"}]}
        ),
        "fallback text",
    )
    # Wrapped output
    assert_eq(
        "output.content wrapped",
        _extract_chat_completion_content({"output": {"content": "wrapped"}}),
        "wrapped",
    )
    # Empty
    assert_eq(
        "empty dict",
        _extract_chat_completion_content({}),
        None,
    )
    # Non-dict
    assert_eq(
        "non-dict",
        _extract_chat_completion_content("plain string"),
        None,
    )


def test_normalize_scoring_result_valid():
    print("test_normalize_scoring_result_valid")
    payload = {
        "articles": [
            {
                "index": 0,
                "title": "AI breakthrough",
                "novelty_score": 4.0,
                "impact_score": 3.0,
                "freshness_score": 2.0,
                "source_score": 1.0,
                "total_score": 10.0,
                "red_flag": False,
                "red_flag_reason": None,
                "reason": "ok",
            }
        ],
        "best_index": 0,
        "best_score": 10.0,
        "recommendation": "POST_NOW",
    }
    normalized = normalize_scoring_result(payload, "TestProvider")
    assert normalized is not None
    assert_eq("article count", len(normalized["articles"]), 1)
    assert_eq("total score", normalized["articles"][0]["total_score"], 10.0)
    assert_eq("recommendation", normalized["recommendation"], "POST_NOW")


def test_normalize_scoring_result_alias():
    print("test_normalize_scoring_result_alias")
    # Provider renames "articles" to "scores"
    payload = {
        "scores": [
            {
                "index": 1,
                "title": "renamed field",
                "novelty_score": 1.0,
                "impact_score": 1.0,
                "freshness_score": 1.0,
                "source_score": 1.0,
                "total_score": 4.0,
                "red_flag": False,
                "reason": "ok",
            }
        ],
        "best_index": 1,
        "best_score": 4.0,
        "recommendation": "SKIP",
    }
    normalized = normalize_scoring_result(payload, "TestProvider")
    assert normalized is not None
    assert_eq("alias coerced", len(normalized["articles"]), 1)


def test_normalize_scoring_result_invalid():
    print("test_normalize_scoring_result_invalid")
    # Capture the body that would be sent to Telegram.
    sent = []
    with patch.object(
        news_bot.requests,
        "post",
        side_effect=lambda url, json, timeout: sent.append(json["text"]) or type("R", (), {"raise_for_status": lambda self: None})(),
    ):
        result = normalize_scoring_result({"foo": "bar"}, "TestProvider")
    assert result is None
    assert len(sent) == 1, f"expected 1 telegram body, got {len(sent)}"
    body = sent[0]
    assert "TestProvider Scoring" in body, body
    assert "InvalidResponse" in body, body
    assert "normalize_scoring_result" in body, body
    assert body.rstrip().endswith(f"Repo: {news_bot.REPO_NAME}"), body
    print("  [PASS] invalid payload reported + repo footer present")


def test_telegram_message_has_repo_footer():
    print("test_telegram_message_has_repo_footer")
    # Disable real network calls, but allow the function to run.
    captured = {}
    fake_response = type("Resp", (), {"raise_for_status": lambda self: None})()

    def fake_post(url, json, timeout):
        captured["text"] = json["text"]
        return fake_response

    os.environ["TELEGRAM_BOT_TOKEN"] = "x"
    os.environ["TELEGRAM_CHAT_ID"] = "y"
    with patch.object(news_bot.requests, "post", side_effect=fake_post):
        news_bot.send_telegram_error("hello world", {"foo": "bar"})
    assert captured.get("text", "").endswith(f"Repo: {news_bot.REPO_NAME}"), captured.get("text")
    print("  [PASS] repo footer appended to Telegram body")


if __name__ == "__main__":
    test_extract_chat_completion_content()
    test_normalize_scoring_result_valid()
    test_normalize_scoring_result_alias()
    test_normalize_scoring_result_invalid()
    test_telegram_message_has_repo_footer()
    print("ALL TESTS PASSED")
