"""End-to-end smoke test for the Cerebras batch pipeline.

Run from the nexusfeed directory:
    python test_scoring_pipeline.py
"""
from __future__ import annotations

import os
import sys
import json as json_module
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import news_bot
from news_bot import (
    Article,
    score_candidates_with_cerebras_batches,
)


def make_article(idx: int, title: str = "Sample") -> Article:
    url = f"https://example.com/{idx}"
    return Article(
        index=idx,
        title=f"{title} #{idx}",
        summary="summary " * 30,
        url=url,
        canonical_url=url,
        url_hash=f"hash{idx}",
        cleaned_title=f"{title} {idx}",
        source="Example",
        tier="tier1",
        published_at="2026-01-01T00:00:00+00:00",
        published_ts=0.0,
        source_rank=1,
    )


def make_cerebras_response(articles):
    return {
        "choices": [
            {
                "message": {
                    "content": json_module.dumps(
                        {
                            "articles": articles,
                            "best_index": articles[0]["index"] if articles else -1,
                            "best_score": max((a["total_score"] for a in articles), default=0.0),
                            "recommendation": "POST_NOW",
                        }
                    )
                }
            }
        ]
    }


def fake_post_factory(responses):
    """Return a fake requests.post that cycles through the provided responses."""
    import itertools
    iterator = itertools.cycle(responses)

    def fake_post(url, json=None, headers=None, timeout=None):
        data = next(iterator)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = data
        resp.raise_for_status = lambda: None
        resp.text = json_module.dumps(data) if not isinstance(data, str) else data
        return resp

    return fake_post


def test_pipeline_happy_path():
    print("test_pipeline_happy_path")
    candidates = [make_article(i) for i in range(1, 6)]
    ai_cache = {"entries": {}}
    now = datetime.now(timezone.utc)

    # Two batches, each returning a valid response.
    response1 = make_cerebras_response(
        [
            {"index": 1, "title": "A1", "novelty_score": 4, "impact_score": 3, "freshness_score": 2, "source_score": 1, "total_score": 10, "red_flag": False, "reason": "ok"},
            {"index": 2, "title": "A2", "novelty_score": 1, "impact_score": 1, "freshness_score": 1, "source_score": 1, "total_score": 4, "red_flag": False, "reason": "ok"},
        ]
    )
    response2 = make_cerebras_response(
        [
            {"index": 3, "title": "A3", "novelty_score": 2, "impact_score": 2, "freshness_score": 2, "source_score": 1, "total_score": 7, "red_flag": False, "reason": "ok"},
        ]
    )

    with patch.object(news_bot.requests, "post", side_effect=fake_post_factory([response1, response2])):
        result = score_candidates_with_cerebras_batches(candidates, now, "fake-key", ai_cache)

    assert result is not None, "expected a result"
    assert result["best_index"] == 1, f"expected best=1, got {result['best_index']}"
    assert result["recommendation"] == "POST_NOW"
    print("  [PASS] pipeline merged two batches and picked best article")


def test_pipeline_recovery_via_alias():
    print("test_pipeline_recovery_via_alias")
    """The model renamed 'articles' to 'scores' but the rest is valid.
    The pipeline should still recover and produce a result."""
    candidates = [make_article(i) for i in range(1, 4)]
    ai_cache = {"entries": {}}
    now = datetime.now(timezone.utc)

    response = {
        "choices": [
            {
                "message": {
                    "content": json_module.dumps(
                        {
                            "scores": [
                                {"index": 1, "title": "A1", "novelty_score": 3, "impact_score": 2, "freshness_score": 2, "source_score": 1, "total_score": 8, "red_flag": False, "reason": "ok"},
                            ],
                            "best_index": 1,
                            "best_score": 8,
                            "recommendation": "POST_NOW",
                        }
                    )
                }
            }
        ]
    }

    sent_to_telegram = []
    def fake_post(url, json=None, headers=None, timeout=None):
        if "api.telegram.org" in url:
            sent_to_telegram.append(json["text"])
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = response
        resp.raise_for_status = lambda: None
        resp.text = json_module.dumps(response)
        return resp

    os.environ["TELEGRAM_BOT_TOKEN"] = "x"
    os.environ["TELEGRAM_CHAT_ID"] = "y"
    with patch.object(news_bot.requests, "post", side_effect=fake_post):
        result = score_candidates_with_cerebras_batches(candidates, now, "fake-key", ai_cache)

    assert result is not None, "expected a result via alias recovery"
    assert result["best_index"] == 1
    assert not any("InvalidResponse" in m for m in sent_to_telegram), sent_to_telegram
    print("  [PASS] pipeline recovered from 'scores' alias without invalid-response notification")


def test_pipeline_handles_non_dict_payload():
    print("test_pipeline_handles_non_dict_payload")
    """The model returns a string instead of JSON. We should fall back
    to a SKIP result and report a single invalid-response notification
    per batch (not duplicate noise)."""
    candidates = [make_article(i) for i in range(1, 3)]
    ai_cache = {"entries": {}}
    now = datetime.now(timezone.utc)

    sent_to_telegram = []
    def fake_post(url, json=None, headers=None, timeout=None):
        if "api.telegram.org" in url:
            sent_to_telegram.append(json["text"])
            return MagicMock(status_code=200, raise_for_status=lambda: None)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "not valid json"}}]}
        resp.raise_for_status = lambda: None
        resp.text = "{}"
        return resp

    os.environ["TELEGRAM_BOT_TOKEN"] = "x"
    os.environ["TELEGRAM_CHAT_ID"] = "y"
    with patch.object(news_bot.requests, "post", side_effect=fake_post):
        result = score_candidates_with_cerebras_batches(candidates, now, "fake-key", ai_cache)

    # Should fall back to skip result with authoritative_score=0
    assert result is not None
    assert result["recommendation"] == "SKIP"
    assert result["authoritative_score"] == 0.0
    # Should include Repo footer in messages
    assert all(m.rstrip().endswith(f"Repo: {news_bot.REPO_NAME}") for m in sent_to_telegram), sent_to_telegram
    # Should report AllBatchesFailed at minimum
    assert any("AllBatchesFailed" in m for m in sent_to_telegram), sent_to_telegram
    print(f"  [PASS] pipeline fallback to SKIP, {len(sent_to_telegram)} telegram msgs all with repo footer")


if __name__ == "__main__":
    test_pipeline_happy_path()
    test_pipeline_recovery_via_alias()
    test_pipeline_handles_non_dict_payload()
    print("PIPELINE SMOKE TESTS PASSED")
