from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser
import gspread
import requests
from dateutil.parser import isoparse


ROOT = Path(__file__).resolve().parent
POSTED_PATH = ROOT / "posted_articles.json"
STATE_PATH = ROOT / "daily_state.json"
AI_CACHE_PATH = ROOT / "ai_cache.json"
DEBUG_CEREBRAS_RESPONSE_PATH = ROOT / "debug-cerebras-response.txt"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
DEFAULT_CEREBRAS_MODEL = "gpt-oss-120b"
CEREBRAS_SUPPORTED_MODELS = {"gpt-oss-120b", "zai-glm-4.7"}
TELEGRAM_API_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"
MAX_CANDIDATES = 25
MAX_ARTICLE_AGE_HOURS = 24
TITLE_SIMILARITY_THRESHOLD = 0.80
POSTED_RETENTION_DAYS = 7
RECENT_TITLE_LOOKBACK_HOURS = 24
REQUIRED_SECRETS = [
    "GROQ_API_KEY",
    "CEREBRAS_API_KEY",
    "GOOGLE_CREDENTIALS",
    "GOOGLE_SHEET_ID",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
]
DOTENV_OVERRIDE_KEYS = {"CEREBRAS_API_KEY"}
TRACKING_QUERY_PREFIXES = (
    "utm_",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "ref",
    "ref_src",
    "ref_url",
    "source",
)
RED_FLAG_PATTERNS = [
    "top 10",
    "top 5",
    "top 7",
    "top 15",
    "top 20",
    "best tools",
    "best practices",
    "best ai tools",
    "what is",
    "guide to",
    "how to",
    "tutorial",
    "weekly recap",
    "weekly roundup",
    "last week",
    "last month",
    "ultimate guide",
    "everything you need to know",
    "in 2023",
    "in 2024",
    "previously announced",
    "announced last",
    "looking back",
    "retrospective",
    "history of",
    "beginners guide",
    "beginner's guide",
    "getting started",
    "step by step",
    "step-by-step",
    "cheat sheet",
    "roundup",
]
STOP_WORDS = {
    "a",
    "an",
    "and",
    "announces",
    "announcing",
    "for",
    "from",
    "in",
    "launches",
    "new",
    "of",
    "on",
    "releases",
    "the",
    "to",
    "with",
}
AI_TOPIC_KEYWORDS = (
    "agent",
    "agents",
    "ai",
    "artificial intelligence",
    "benchmark",
    "chatgpt",
    "claude",
    "copilot",
    "deepmind",
    "embedding",
    "foundation model",
    "gemini",
    "generative",
    "gpu",
    "inference",
    "llm",
    "machine learning",
    "mcp",
    "model",
    "multimodal",
    "neural",
    "openai",
    "reasoning",
    "robot",
    "safety",
    "transformer",
)
SOFTWARE_TOPIC_KEYWORDS = (
    "api",
    "compiler",
    "copilot",
    "developer",
    "engineering",
    "framework",
    "github",
    "ide",
    "inference",
    "llm",
    "mcp",
    "model",
    "open source",
    "runtime",
    "sdk",
    "software",
    "tooling",
    "vscode",
)
GENERAL_TOPIC_KEYWORDS = tuple(sorted(set(AI_TOPIC_KEYWORDS + SOFTWARE_TOPIC_KEYWORDS)))
TIER_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3}
SOCIAL_POST_PLATFORMS = ("twitter", "facebook", "threads", "telegram")
FORCE_APPEND_BEST_FOR_TEST = True
GROQ_MAX_FAILURES_PER_RUN = 2
GROQ_MAX_RETRY_AFTER_SECONDS = 5.0
GROQ_IMMEDIATE_RETRY_DELAY_SECONDS = 1.0
AI_CACHE_VERSION = 1
CEREBRAS_SCORING_PROMPT_VERSION = "cerebras-scoring-v1"
GROQ_SOCIAL_PROMPT_VERSION = "groq-social-v1"
CEREBRAS_SCORING_CACHE_TTL = timedelta(hours=24)
GROQ_SOCIAL_CACHE_TTL = timedelta(days=7)
SEEN_ARTICLE_LINK_CACHE_TTL = timedelta(days=7)
CEREBRAS_BATCH_COUNT = 8
CEREBRAS_MAX_BATCH_PROMPT_CHARS = 12000

SCORING_SYSTEM_PROMPT = """
You are a strict AI news curator targeting software engineers and AI researchers.

Your job per run:
1. Score every article using the 4-criteria system (max 10.00)
2. Apply red flags (score = 0.00 if ANY red flag matches)
3. Select the single best article
4. Decide: POST_NOW / SKIP
5. If decision != SKIP: write 4 ready-to-post social media messages

SCORING CRITERIA:
- Novelty (0.00-3.00): new release=3, research=2.5, industry move=2,
  update=1.5, opinion=0.5, tutorial=0.2, recap=0
- Impact (0.00-3.00): everyone=3, all devs=2.5, niche=1.5, one company=0.5
- Freshness (0.00-2.00): <1h=2, 1-3h=1.5, 3-6h=1, 6-12h=0.5, >12h=0
- Source (0.00-2.00): Tier S=2, A=1.5, B=1, C=0.5

RED FLAGS (auto 0.00): "top 10", "top 5", "best tools", "what is",
"guide to", "how to", "tutorial", "weekly recap", "roundup",
"last week", "last month", "ultimate guide", "in 2023", "in 2024",
"retrospective", "looking back", "everything you need to know"

DECISION RULES:
- best_score >= 8.5 -> POST_NOW
- best_score < 8.5  -> SKIP

SOCIAL MEDIA FORMAT RULES:
- Twitter/X:
  - Max 280 characters including the URL.
  - Write like a sharp, informed tech journalist. One punchy sentence that makes someone stop scrolling.
  - Add the URL on a new line at the end.
  - NO emoji. NO hashtags.
  - Example tone: "OpenAI just shipped a new reasoning model that beats o1 on AIME. Early benchmarks show it halves hallucination rates on math tasks.\n\nhttps://..."
- Facebook:
  - Max 500 characters excluding URL.
  - Write like a knowledgeable person sharing something with their professional network. Use 2-3 sentences.
  - The first sentence is the hook, the rest adds context that makes it worth clicking.
  - Add "Link: [URL]" at the very end on a new line.
  - NO emoji. NO hashtags. No "Share this with..." style calls to action.
  - Conversational but substantive. Sound like a person who read the article, not a bot summarizing it.
- Threads:
  - Max 500 characters including URL.
  - Same voice as Twitter but 1-2 sentences with slightly more room for one extra piece of context.
  - Add the URL on a new line at the end.
  - NO emoji. NO hashtags.
  - Sound like a developer or researcher casually sharing something interesting, not a social media manager posting content.
- Telegram:
  - Keep the structured format with score, source, tier, and emoji separators.
  - Format:
    ━━━━━━━━━━━━━━━━━━━━━
    🔥 *[HEADLINE]*

    [1-2 sentence factual summary, no marketing language]

    🏛️ Source: [Name] ([Tier])
    ⏰ Published: [X hours ago]
    ⭐ Score: [score]/10

    🔗 [URL]
    ━━━━━━━━━━━━━━━━━━━━━

UNIVERSAL RULES FOR ALL PLATFORMS:
- Never start a post with the article title verbatim; rephrase it as a statement or observation.
- Never use words like: "exciting", "groundbreaking", "revolutionary", "game-changing", "thrilled", "delighted", "proud to announce".
- Never use passive voice if active voice is possible.
- Write as if a senior software engineer or AI researcher is sharing this with peers, not as a social media manager.
- The post must pass a "would a human write this?" check. If it sounds robotic or templated, rewrite it.
- Match the source article's language (English article -> English post).
- If recommendation is POST_NOW, you MUST return all four keys in social_posts: twitter, facebook, threads, telegram. These fields are REQUIRED when POST_NOW. Never return empty strings for these fields.

Respond ONLY with valid JSON. No markdown, no explanation outside JSON.
""".strip()

SOCIAL_POST_SYSTEM_PROMPT = """
You write four ready-to-post social media messages for one selected AI or software engineering news article.

Your job:
1. Read the single selected article payload
2. Generate four social media posts in `social_posts`
3. Return all four keys: twitter, facebook, threads, telegram
4. Never return empty strings for these fields

SOCIAL MEDIA FORMAT RULES:
- Twitter/X:
  - Max 280 characters.
  - Write like a sharp, informed tech journalist. One punchy sentence that makes someone stop scrolling.
  - Do NOT include any URL or any "Link:" text.
  - Do NOT use list formatting, bullet points, or leading hyphens (`-`).
  - Include exactly 1 relevant `@tag` and 1-2 relevant `#hashtags`.
  - NO emoji.
- Facebook:
  - Max 500 characters.
  - Write like a knowledgeable person sharing something with their professional network. Use 2-3 sentences.
  - The first sentence is the hook, the rest adds context that makes it worth clicking.
  - Do NOT include any URL or any "Link:" text.
  - Do NOT use list formatting, bullet points, or leading hyphens (`-`).
  - Include 2-3 relevant `#hashtags`.
  - Do not use `@tags` unless the tag is clearly justified by the article.
  - NO emoji.
- Threads:
  - Max 500 characters.
  - Same voice as Twitter but 1-2 sentences with slightly more room for one extra piece of context.
  - Do NOT include any URL or any "Link:" text.
  - Do NOT use list formatting, bullet points, or leading hyphens (`-`).
  - Include exactly 1 relevant `#hashtags` and 1 relevant `@tag`.
  - NO emoji.
- Telegram:
  - Keep the structured format with score, source, tier, and emoji separators.
  - Do NOT include any URL or any `🔗` link line.
  - Do NOT use list formatting, bullet points, or leading hyphens (`-`).
  - Add 2-3 relevant `#hashtags` on the final line.
  - Format:
    ━━━━━━━━━━━━━━━━━━━━━
    🔥 *[HEADLINE]*

    [1-2 sentence factual summary, no marketing language]

    🏛️ Source: [Name] ([Tier])
    ⏰ Published: [X hours ago]
    ⭐ Score: [score]/10
    
    #HashtagOne #HashtagTwo
    ━━━━━━━━━━━━━━━━━━━━━

UNIVERSAL RULES:
- Never start a post with the article title verbatim; rephrase it as a statement or observation.
- Never use words like: "exciting", "groundbreaking", "revolutionary", "game-changing", "thrilled", "delighted", "proud to announce".
- Write as if a senior software engineer or AI researcher is sharing this with peers.
- The post must sound human and specific to the article.
- Never include raw URLs anywhere in the post.
- Never use `Link:` labels.
- Never use bullet-list style or lines that start with `-`.

Respond ONLY with valid JSON. No markdown, no explanation outside JSON.
""".strip()

SCORING_RESPONSE_SCHEMA = {
    "articles": [
        {
            "index": 0,
            "title": "string",
            "novelty_score": 0.0,
            "impact_score": 0.0,
            "freshness_score": 0.0,
            "source_score": 0.0,
            "total_score": 0.0,
            "red_flag": False,
            "red_flag_reason": "string or null",
            "reason": "string (1 sentence why this score)",
        }
    ],
    "best_index": 0,
    "best_score": 0.0,
    "recommendation": "POST_NOW | SKIP",
    "social_posts": {
        "twitter": "string (max 280 chars including URL, no emoji, no hashtags)",
        "facebook": "string (max 500 chars excluding URL, no emoji, no hashtags)",
        "threads": "string (max 500 chars including URL, no emoji, no hashtags)",
        "telegram": "string (Telegram Markdown format)",
    },
}

SOCIAL_POST_RESPONSE_SCHEMA = {
    "social_posts": {
        "twitter": "string (max 280 chars, no URL, no leading '-', include 1 @tag and 1-2 hashtags)",
        "facebook": "string (max 500 chars, no URL, no leading '-', include 2-3 hashtags)",
        "threads": "string (max 500 chars, no URL, no leading '-', include 1 @tag and 2-3 hashtags)",
        "telegram": "string (Telegram Markdown format, no URL, no leading '-', end with 2-3 hashtags)",
    }
}


@dataclass(frozen=True)
class FeedSource:
    name: str
    tier: str
    urls: tuple[str, ...]
    topic_keywords: tuple[str, ...] = ()


@dataclass
class Article:
    index: int
    title: str
    summary: str
    url: str
    canonical_url: str
    url_hash: str
    cleaned_title: str
    source: str
    tier: str
    published_at: str
    published_ts: float
    source_rank: int


FEEDS: tuple[FeedSource, ...] = (
    FeedSource("OpenAI", "S", ("https://openai.com/news/rss.xml",)),
    # community-maintained scraper, updated hourly
    FeedSource(
        "Anthropic",
        "S",
        (
            "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml",
        ),
    ),
    FeedSource(
        "Google AI",
        "S",
        (
            "https://blog.google/technology/ai/rss/",
            "https://blog.research.google/feeds/posts/default?alt=rss",
        ),
    ),
    FeedSource("HuggingFace", "S", ("https://huggingface.co/blog/feed.xml",)),
    FeedSource(
        "Microsoft AI",
        "S",
        (
            "https://news.microsoft.com/source/topics/ai/feed/",
            "https://blogs.microsoft.com/ai/feed/",
            "https://blogs.microsoft.com/feed/",
        ),
        topic_keywords=GENERAL_TOPIC_KEYWORDS,
    ),
    FeedSource(
        "TechCrunch",
        "A",
        (
            "https://techcrunch.com/category/artificial-intelligence/feed/",
            "https://techcrunch.com/tag/artificial-intelligence/feed/",
        ),
    ),
    FeedSource(
        "The Verge",
        "A",
        ("https://www.theverge.com/rss/index.xml",),
        topic_keywords=GENERAL_TOPIC_KEYWORDS,
    ),
    FeedSource("Ars Technica", "A", ("https://arstechnica.com/ai/feed/",)),
    FeedSource("MarkTechPost", "A", ("https://www.marktechpost.com/feed/",)),
    FeedSource("Wired AI", "A", ("https://www.wired.com/feed/tag/ai/latest/rss",)),
    FeedSource(
        "MIT News AI",
        "B",
        ("https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",),
    ),
    FeedSource(
        "InfoQ AI/ML",
        "B",
        ("https://feed.infoq.com/",),
        topic_keywords=GENERAL_TOPIC_KEYWORDS,
    ),
    FeedSource("AI News", "B", ("https://artificialintelligence-news.com/feed/",)),
    # SSL error in local env only - works fine on GitHub Actions runner
    FeedSource("arXiv cs.AI", "C", ("https://rss.arxiv.org/rss/cs.AI",)),
    FeedSource(
        "Hacker News",
        "C",
        ("https://news.ycombinator.com/rss",),
        topic_keywords=GENERAL_TOPIC_KEYWORDS,
    ),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return isoparse(value).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    if not key:
        return None
    return key, value


def load_local_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    loaded_keys = 0
    placeholder_values = {"dummy", "@dummy"}
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_dotenv_line(line)
            if not parsed:
                continue
            key, value = parsed
            if key in DOTENV_OVERRIDE_KEYS:
                if value:
                    os.environ[key] = value
                    loaded_keys += 1
                continue
            current = clean_whitespace(os.environ.get(key, ""))
            if current and current not in placeholder_values:
                continue
            if value:
                os.environ[key] = value
                loaded_keys += 1
    except OSError as exc:
        logging.warning("Failed to read local .env file: %s", exc)
        return

    if loaded_keys:
        logging.info("Loaded %s secret(s) from local .env.", loaded_keys)


def require_env() -> dict[str, str]:
    load_local_env_file()
    values: dict[str, str] = {}
    for key in REQUIRED_SECRETS:
        value = os.environ.get(key)
        if not value:
            raise EnvironmentError(f"Missing required secret: {key}")
        values[key] = value
    return values


def get_cerebras_model() -> str:
    requested_model = clean_whitespace(os.environ.get("CEREBRAS_MODEL", DEFAULT_CEREBRAS_MODEL))
    if requested_model not in CEREBRAS_SUPPORTED_MODELS:
        raise EnvironmentError(f"Unsupported Cerebras model for this organization: {requested_model}")
    return requested_model


def default_state(now: datetime | None = None) -> dict[str, Any]:
    current = now or utc_now()
    return {
        "date": current.date().isoformat(),
        "posts_today": 0,
        "last_post_time": None,
    }


def sanitize_state(state: dict[str, Any], now: datetime) -> dict[str, Any]:
    sanitized = default_state(now)
    if isinstance(state, dict):
        if isinstance(state.get("date"), str):
            sanitized["date"] = state["date"]
        try:
            sanitized["posts_today"] = int(state.get("posts_today", 0))
        except (TypeError, ValueError):
            sanitized["posts_today"] = 0
        last_post_time = parse_iso_datetime(state.get("last_post_time"))
        sanitized["last_post_time"] = isoformat_utc(last_post_time) if last_post_time else None
    return sanitized


def default_posted() -> dict[str, Any]:
    return {
        "hashes": [],
        "recent_titles": [],
        "hash_records": [],
    }


def load_json(path: Path, default_factory) -> dict[str, Any]:
    if not path.exists():
        return default_factory()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Failed to load %s: %s. Recreating with defaults.", path.name, exc)
        notify_component_error(
            "JSON Parsing",
            type(exc).__name__,
            f"Failed to load {path.name}; recreating with defaults.",
            {"path": path.name, "run_context": "load_json"},
        )
    return default_factory()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def save_debug_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def default_skip_scoring_result() -> dict[str, Any]:
    return {
        "articles": [],
        "best_index": -1,
        "best_score": 0.0,
        "recommendation": "SKIP",
        "social_posts": normalize_social_posts({}),
    }


def default_ai_cache() -> dict[str, Any]:
    return {
        "version": AI_CACHE_VERSION,
        "cerebras_scoring": {},
        "groq_social_posts": {},
        "seen_article_links": {},
    }


def sanitize_ai_cache(data: dict[str, Any]) -> dict[str, Any]:
    sanitized = default_ai_cache()
    if not isinstance(data, dict):
        return sanitized
    for bucket_name in ("cerebras_scoring", "groq_social_posts", "seen_article_links"):
        bucket = data.get(bucket_name)
        if isinstance(bucket, dict):
            sanitized[bucket_name] = bucket
    return sanitized


def format_cache_age(created_at: datetime, now: datetime) -> str:
    age_seconds = max(0, int((now - created_at).total_seconds()))
    hours, remainder = divmod(age_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def cleanup_ai_cache(ai_cache: dict[str, Any], now: datetime) -> bool:
    changed = False
    bucket_ttls = {
        "cerebras_scoring": CEREBRAS_SCORING_CACHE_TTL,
        "groq_social_posts": GROQ_SOCIAL_CACHE_TTL,
        "seen_article_links": SEEN_ARTICLE_LINK_CACHE_TTL,
    }
    for bucket_name, ttl in bucket_ttls.items():
        bucket = ai_cache.get(bucket_name, {})
        if not isinstance(bucket, dict):
            ai_cache[bucket_name] = {}
            changed = True
            continue
        expired_keys: list[str] = []
        for cache_key, entry in bucket.items():
            if not isinstance(entry, dict):
                expired_keys.append(cache_key)
                continue
            created_at = parse_iso_datetime(entry.get("created_at"))
            if not created_at or now - created_at > ttl:
                expired_keys.append(cache_key)
        for cache_key in expired_keys:
            del bucket[cache_key]
            changed = True
    return changed


def load_ai_cache(now: datetime) -> dict[str, Any]:
    ai_cache = sanitize_ai_cache(load_json(AI_CACHE_PATH, default_ai_cache))
    if cleanup_ai_cache(ai_cache, now):
        save_json(AI_CACHE_PATH, ai_cache)
    return ai_cache


def persist_ai_cache(ai_cache: dict[str, Any]) -> None:
    save_json(AI_CACHE_PATH, ai_cache)


def build_cache_key(parts: dict[str, Any]) -> str:
    return stable_hash(json.dumps(parts, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def build_cerebras_scoring_cache_key(candidates: list["Article"]) -> str:
    cerebras_model = get_cerebras_model()
    key_payload = {
        "cache_type": "cerebras_scoring",
        "model": cerebras_model,
        "prompt_version": CEREBRAS_SCORING_PROMPT_VERSION,
        "candidates": [
            {
                "canonical_url": article.canonical_url,
                "title": article.title,
                "summary": article.summary,
            }
            for article in candidates
        ],
    }
    return build_cache_key(key_payload)


def build_groq_social_cache_key(article: "Article") -> str:
    key_payload = {
        "cache_type": "groq_social_posts",
        "model": GROQ_MODEL,
        "prompt_version": GROQ_SOCIAL_PROMPT_VERSION,
        "article": {
            "canonical_url": article.canonical_url,
            "title": article.title,
            "summary": article.summary,
        },
    }
    return build_cache_key(key_payload)


def get_cached_ai_result(
    ai_cache: dict[str, Any],
    bucket_name: str,
    cache_key: str,
    now: datetime,
    ttl: timedelta,
    cache_label: str,
) -> Any | None:
    bucket = ai_cache.get(bucket_name, {})
    if not isinstance(bucket, dict):
        ai_cache[bucket_name] = {}
        bucket = ai_cache[bucket_name]
    entry = bucket.get(cache_key)
    if not isinstance(entry, dict):
        logging.info("%s cache miss: key=%s. Calling API.", cache_label, cache_key)
        return None
    created_at = parse_iso_datetime(entry.get("created_at"))
    if not created_at or now - created_at > ttl:
        logging.info("%s cache miss: key=%s expired. Calling API.", cache_label, cache_key)
        del bucket[cache_key]
        persist_ai_cache(ai_cache)
        return None
    logging.info("%s cache hit: key=%s age=%s", cache_label, cache_key, format_cache_age(created_at, now))
    return entry.get("payload")


def set_cached_ai_result(
    ai_cache: dict[str, Any],
    bucket_name: str,
    cache_key: str,
    payload: Any,
    now: datetime,
) -> None:
    bucket = ai_cache.setdefault(bucket_name, {})
    if not isinstance(bucket, dict):
        ai_cache[bucket_name] = {}
        bucket = ai_cache[bucket_name]
    bucket[cache_key] = {
        "created_at": isoformat_utc(now),
        "payload": payload,
    }
    persist_ai_cache(ai_cache)


def mark_article_links_seen(
    ai_cache: dict[str, Any],
    articles: list["Article"],
    now: datetime,
    status: str,
) -> None:
    bucket = ai_cache.setdefault("seen_article_links", {})
    if not isinstance(bucket, dict):
        ai_cache["seen_article_links"] = {}
        bucket = ai_cache["seen_article_links"]
    timestamp = isoformat_utc(now)
    changed = False
    for article in articles:
        canonical_url = clean_whitespace(article.canonical_url)
        if not canonical_url:
            continue
        bucket[canonical_url] = {
            "created_at": timestamp,
            "canonical_url": canonical_url,
            "title": article.title,
            "status": status,
        }
        changed = True
    if changed:
        persist_ai_cache(ai_cache)


def filter_unseen_candidates(candidates: list["Article"], ai_cache: dict[str, Any], now: datetime) -> list["Article"]:
    bucket = ai_cache.get("seen_article_links", {})
    if not isinstance(bucket, dict):
        ai_cache["seen_article_links"] = {}
        bucket = ai_cache["seen_article_links"]

    unseen_candidates: list[Article] = []
    skipped_count = 0
    for article in candidates:
        canonical_url = clean_whitespace(article.canonical_url)
        entry = bucket.get(canonical_url)
        if not isinstance(entry, dict):
            unseen_candidates.append(article)
            continue
        created_at = parse_iso_datetime(entry.get("created_at"))
        if not created_at or now - created_at > SEEN_ARTICLE_LINK_CACHE_TTL:
            unseen_candidates.append(article)
            continue
        skipped_count += 1

    if skipped_count:
        logging.info(
            "Seen-link cache filtered out %s already-processed candidate(s) before Cerebras scoring.",
            skipped_count,
        )
    return unseen_candidates


def sanitize_posted(data: dict[str, Any]) -> dict[str, Any]:
    hashes = data.get("hashes")
    recent_titles = data.get("recent_titles")
    hash_records = data.get("hash_records")
    return {
        "hashes": hashes if isinstance(hashes, list) else [],
        "recent_titles": recent_titles if isinstance(recent_titles, list) else [],
        "hash_records": hash_records if isinstance(hash_records, list) else [],
    }


def cleanup_posted_history(posted: dict[str, Any], now: datetime) -> None:
    cutoff = now - timedelta(days=POSTED_RETENTION_DAYS)
    recent_titles: list[dict[str, Any]] = []
    for item in posted.get("recent_titles", []):
        if not isinstance(item, dict):
            continue
        posted_at = parse_iso_datetime(item.get("posted_at"))
        if posted_at and posted_at >= cutoff:
            recent_titles.append(item)

    hash_records: list[dict[str, Any]] = []
    for item in posted.get("hash_records", []):
        if not isinstance(item, dict):
            continue
        posted_at = parse_iso_datetime(item.get("posted_at"))
        if posted_at and posted_at >= cutoff and isinstance(item.get("hash"), str):
            hash_records.append(item)

    posted["recent_titles"] = recent_titles
    posted["hash_records"] = hash_records
    posted["hashes"] = sorted({item["hash"] for item in hash_records})


def reset_state_if_needed(state: dict[str, Any], now: datetime) -> tuple[dict[str, Any], bool]:
    if state.get("date") == now.date().isoformat():
        return state, False
    logging.info("Resetting daily state for new UTC day.")
    return default_state(now), True


def clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_multiline_text(value: str) -> str:
    lines = [line.rstrip() for line in str(value).replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def trim_error_text(value: Any, limit: int = 400) -> str:
    cleaned = clean_multiline_text(str(value))
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def mask_secret_preview(value: str) -> str:
    cleaned = clean_whitespace(value)
    if not cleaned:
        return "<missing>"
    if len(cleaned) <= 10:
        return f"{cleaned[:2]}...{cleaned[-2:]}"
    return f"{cleaned[:6]}...{cleaned[-4:]}"


def strip_markdown_code_fences(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped, count=1)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
    return stripped.strip()


def extract_outermost_json_object(value: str) -> str | None:
    start = value.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return None


def parse_llm_json_object(content: str) -> dict[str, Any]:
    stripped = clean_multiline_text(content)
    candidates = [stripped]

    fenced = strip_markdown_code_fences(stripped)
    if fenced != stripped:
        candidates.append(fenced)

    extracted = extract_outermost_json_object(fenced)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"{exc.msg} at line {exc.lineno} column {exc.colno} (char {exc.pos})")
            continue
        if isinstance(parsed, dict):
            return parsed
        errors.append("Parsed content was not a JSON object.")

    if stripped.startswith("{") and not stripped.rstrip().endswith("}"):
        errors.append("Detected truncated JSON object before closing brace.")
    raise ValueError("Unable to recover valid JSON object from model output. " + " | ".join(errors))


def send_telegram_error(message: str, context: dict | None = None) -> bool:
    bot_token = clean_whitespace(os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    chat_id = clean_whitespace(os.environ.get("TELEGRAM_CHAT_ID", ""))
    if not bot_token or not chat_id:
        logging.warning("Telegram error notifier is not configured.")
        return False

    lines = [trim_error_text(message, 3000)]
    if context:
        serialized_context = json.dumps(context, ensure_ascii=True, default=str, separators=(",", ":"))
        lines.append(f"Context: {trim_error_text(serialized_context, 800)}")
    payload = {
        "chat_id": chat_id,
        "text": "\n".join(lines),
        "disable_web_page_preview": True,
    }
    url = TELEGRAM_API_URL_TEMPLATE.format(token=bot_token)
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logging.error("Telegram error notifier failed: %s", exc)
        return False


def notify_component_error(component: str, error_type: str, explanation: str, context: dict | None = None) -> bool:
    lines = [
        f"Component: {component}",
        f"Error: {error_type}",
        f"Details: {trim_error_text(explanation, 500)}",
    ]
    if context:
        article_title = clean_whitespace(str(context.get("article_title", "") or context.get("title", "")))
        run_context = clean_whitespace(str(context.get("run_context", "")))
        if article_title:
            lines.append(f"Article: {trim_error_text(article_title, 200)}")
        elif run_context:
            lines.append(f"Run: {trim_error_text(run_context, 200)}")
    return send_telegram_error("\n".join(lines), context)


def normalize_social_posts(raw_social_posts: Any) -> dict[str, str]:
    if not isinstance(raw_social_posts, dict):
        raw_social_posts = {}
    return {
        platform: clean_multiline_text(str(raw_social_posts.get(platform, "")))
        for platform in SOCIAL_POST_PLATFORMS
    }


def sanitize_groq_social_post_text(platform: str, text: str) -> str:
    cleaned = clean_multiline_text(text)
    cleaned = re.sub(r"https?://\S+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?im)^\s*link:\s*.*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*-\s*", "", cleaned)
    if platform == "telegram":
        cleaned = re.sub(r"(?im)^\s*🔗\s*.*$", "", cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def sanitize_groq_social_posts(social_posts: dict[str, str]) -> dict[str, str]:
    normalized = normalize_social_posts(social_posts)
    return {
        platform: sanitize_groq_social_post_text(platform, normalized.get(platform, ""))
        for platform in SOCIAL_POST_PLATFORMS
    }


def missing_social_post_platforms(social_posts: dict[str, str]) -> list[str]:
    return [platform for platform in SOCIAL_POST_PLATFORMS if not social_posts.get(platform, "").strip()]


def sanitize_telegram_markdown_text(value: str) -> str:
    cleaned = clean_whitespace(str(value))
    return re.sub(r"[_*\[\]`]", "", cleaned)


def strip_html(value: str) -> str:
    value = re.sub(r"<script[\s\S]*?</script>", " ", value or "", flags=re.IGNORECASE)
    value = re.sub(r"<style[\s\S]*?</style>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    return clean_whitespace(html.unescape(value))


def truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    query_items = []
    for key, val in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in TRACKING_QUERY_PREFIXES:
            continue
        query_items.append((key, val))
    cleaned = parsed._replace(
        scheme=parsed.scheme.lower() or "https",
        netloc=parsed.netloc.lower(),
        query=urlencode(query_items, doseq=True),
        fragment="",
    )
    return urlunparse(cleaned).rstrip("/")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_title(title: str) -> str:
    normalized = title.lower()
    normalized = re.sub(r"https?://\S+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    words = [word for word in normalized.split() if word and word not in STOP_WORDS]
    return " ".join(words)


def word_overlap_ratio(left: str, right: str) -> float:
    left_words = set(left.split())
    right_words = set(right.split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / max(len(left_words), len(right_words))


def title_is_duplicate(cleaned_title: str, other_title: str) -> bool:
    return word_overlap_ratio(cleaned_title, other_title) >= TITLE_SIMILARITY_THRESHOLD


def _get_article_field(article: Any, field: str) -> str:
    if isinstance(article, dict):
        return str(article.get(field) or "")
    return str(getattr(article, field, "") or "")


def find_local_red_flag_pattern(article: Any) -> str | None:
    text = ((_get_article_field(article, "title") + " " + _get_article_field(article, "summary")[:300]).lower())
    for pattern in RED_FLAG_PATTERNS:
        if pattern in text:
            return pattern
    return None


def has_local_red_flag(article: Any) -> bool:
    """
    Hard local check BEFORE sending to Groq.
    Returns True if article should be auto-rejected.
    Check both title and first 300 chars of summary.
    Case-insensitive. If ANY pattern matches -> True.
    """
    return find_local_red_flag_pattern(article) is not None


def contains_topic_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    haystack = f" {text.lower()} "
    return any(keyword.lower() in haystack for keyword in keywords)


def parse_entry_datetime(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        value = getattr(entry, key, None)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    for key in ("published", "updated"):
        value = getattr(entry, key, None)
        if value:
            try:
                return parsedate_to_datetime(value).astimezone(timezone.utc)
            except (TypeError, ValueError, IndexError):
                continue
    return None


def extract_entry_text(entry: Any) -> tuple[str, str]:
    title = clean_whitespace(getattr(entry, "title", ""))
    summary_raw = getattr(entry, "summary", "") or getattr(entry, "description", "")
    if not summary_raw and getattr(entry, "content", None):
        try:
            summary_raw = entry.content[0].value
        except (IndexError, AttributeError, KeyError, TypeError):
            summary_raw = ""
    summary = truncate(strip_html(summary_raw), 300)
    return title, summary


def fetch_feed(feed: FeedSource, now: datetime) -> list[Article]:
    cutoff = now - timedelta(hours=MAX_ARTICLE_AGE_HOURS)
    headers = {
        "User-Agent": "NexusFeedBot/1.0 (+https://github.com/mohabdelkarim/NexusFeed)",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
    }
    response_content = None

    for url in feed.urls:
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            response_content = response.content
            break
        except requests.RequestException as exc:
            logging.warning("Feed fetch failed for %s via %s: %s", feed.name, url, exc)
            notify_component_error(
                "Feed Fetch",
                type(exc).__name__,
                f"Feed fetch failed for {feed.name}.",
                {"feed": feed.name, "url": url, "run_context": "fetch_feed"},
            )

    if response_content is None:
        logging.warning("Skipping %s after all feed URLs failed.", feed.name)
        notify_component_error(
            "Feed Fetch",
            "AllSourcesFailed",
            f"Skipping {feed.name} after all feed URLs failed.",
            {"feed": feed.name, "run_context": "fetch_feed"},
        )
        return []

    parsed = feedparser.parse(response_content)
    articles: list[Article] = []

    for entry in parsed.entries:
        published_dt = parse_entry_datetime(entry)
        if not published_dt or published_dt < cutoff or published_dt > now + timedelta(minutes=5):
            continue

        title, summary = extract_entry_text(entry)
        if not title:
            continue

        entry_text = clean_whitespace(f"{title} {summary}")
        if feed.topic_keywords and not contains_topic_keyword(entry_text, feed.topic_keywords):
            continue

        raw_url = clean_whitespace(getattr(entry, "link", ""))
        if not raw_url:
            continue

        canonical_url = canonicalize_url(raw_url)
        articles.append(
            Article(
                index=-1,
                title=title,
                summary=summary,
                url=raw_url,
                canonical_url=canonical_url,
                url_hash=stable_hash(raw_url),
                cleaned_title=normalize_title(title),
                source=feed.name,
                tier=feed.tier,
                published_at=isoformat_utc(published_dt),
                published_ts=published_dt.timestamp(),
                source_rank=TIER_ORDER[feed.tier],
            )
        )

    return sorted(articles, key=lambda item: item.published_ts, reverse=True)


def fetch_all_feeds(now: datetime) -> list[Article]:
    all_articles: list[Article] = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_map = {executor.submit(fetch_feed, feed, now): feed for feed in FEEDS}
        for future in as_completed(future_map):
            feed = future_map[future]
            try:
                articles = future.result()
                logging.info("Fetched %s article(s) from %s.", len(articles), feed.name)
                all_articles.extend(articles)
            except Exception as exc:  # pragma: no cover
                logging.exception("Unexpected error while processing %s: %s", feed.name, exc)
                notify_component_error(
                    "Feed Parsing",
                    type(exc).__name__,
                    f"Unexpected error while processing {feed.name}.",
                    {"feed": feed.name, "run_context": "fetch_all_feeds"},
                )
    return sorted(all_articles, key=lambda item: -item.published_ts)


def is_duplicate_against_posted(article: Article, posted: dict[str, Any], now: datetime) -> bool:
    if article.url_hash in set(posted.get("hashes", [])):
        return True

    recent_cutoff = now - timedelta(hours=RECENT_TITLE_LOOKBACK_HOURS)
    for item in posted.get("recent_titles", []):
        if not isinstance(item, dict):
            continue
        posted_at = parse_iso_datetime(item.get("posted_at"))
        cleaned_title = clean_whitespace(str(item.get("cleaned_title", "")))
        if posted_at and posted_at >= recent_cutoff and cleaned_title and title_is_duplicate(article.cleaned_title, cleaned_title):
            return True
    return False


def dedupe_candidates(articles: list[Article], posted: dict[str, Any], now: datetime) -> list[Article]:
    chosen: list[Article] = []
    for article in articles:
        matched_pattern = find_local_red_flag_pattern(article)
        if matched_pattern:
            logging.warning('Local red flag: [%s] matched pattern "%s"', article.title, matched_pattern)
            continue

        if is_duplicate_against_posted(article, posted, now):
            continue

        duplicate_index = None
        for idx, existing in enumerate(chosen):
            same_story = article.url_hash == existing.url_hash or title_is_duplicate(article.cleaned_title, existing.cleaned_title)
            if same_story:
                duplicate_index = idx
                break

        if duplicate_index is None:
            chosen.append(article)
            continue

        existing = chosen[duplicate_index]
        current_key = (article.source_rank, -article.published_ts)
        existing_key = (existing.source_rank, -existing.published_ts)
        if current_key < existing_key:
            chosen[duplicate_index] = article

    for index, article in enumerate(sorted(chosen, key=lambda item: -item.published_ts)):
        article.index = index
    return chosen[:MAX_CANDIDATES]


def hours_ago_text(published_ts: float, now: datetime) -> str:
    hours = max(0.0, (now.timestamp() - published_ts) / 3600)
    if hours < 1:
        return "<1 hour ago"
    if hours < 2:
        return "1 hour ago"
    return f"{int(hours)} hours ago"


def parse_retry_after(header_value: str, fallback: float) -> float:
    """
    Parses Groq's retry-after header.
    Supports:
      - Integer seconds: "30"
      - Float seconds: "1.5"
      - HTTP-date: "Fri, 06 Jun 2026 14:00:00 GMT"
    Returns seconds to wait as float. Min=1, Max=120.
    Falls back to fallback if unparseable.
    """
    if not header_value:
        return fallback
    try:
        return max(1.0, min(120.0, float(header_value)))
    except ValueError:
        pass
    try:
        retry_dt = parsedate_to_datetime(header_value)
        now = datetime.now(timezone.utc)
        delta = (retry_dt - now).total_seconds()
        return max(1.0, min(120.0, delta))
    except Exception:
        return fallback


def rebuild_prompt_with_summary_limit(candidates: list[Article], char_limit: int) -> str:
    now = utc_now()
    prompt_payload = {
        "response_schema": SCORING_RESPONSE_SCHEMA,
        "prompt_version": CEREBRAS_SCORING_PROMPT_VERSION,
        "social_media_rules": [
            "If decision is POST_NOW, generate all four keys in social_posts: twitter, facebook, threads, telegram.",
            "If recommendation is POST_NOW, you MUST return all four keys in social_posts: twitter, facebook, threads, telegram. These fields are REQUIRED when POST_NOW. Never return empty strings for these fields.",
            "Use the FULL article title as headline without inventing facts.",
            "Keep summaries factual, direct, and ready to publish with no hashtags or marketing fluff.",
        ],
        "candidates": [
            {
                "index": article.index,
                "title": article.title,
                "summary": truncate(article.summary, char_limit),
                "source": article.source,
                "tier": article.tier,
                "published_time_utc": article.published_at,
                "published_relative": hours_ago_text(article.published_ts, now),
                "article_url": article.url,
            }
            for article in candidates
        ],
    }
    return json.dumps(prompt_payload, ensure_ascii=True, separators=(",", ":"))


def build_scoring_prompt(candidates: list[Article], now: datetime) -> str:
    prompt_payload = {
        "response_schema": SCORING_RESPONSE_SCHEMA,
        "prompt_version": CEREBRAS_SCORING_PROMPT_VERSION,
        "social_media_rules": [
            "If decision is POST_NOW, generate all four keys in social_posts: twitter, facebook, threads, telegram.",
            "If recommendation is POST_NOW, you MUST return all four keys in social_posts: twitter, facebook, threads, telegram. These fields are REQUIRED when POST_NOW. Never return empty strings for these fields.",
            "Use the FULL article title as headline without inventing facts.",
            "Keep summaries factual, direct, and ready to publish with no hashtags or marketing fluff.",
        ],
        "candidates": [
            {
                "index": article.index,
                "title": article.title,
                "summary": article.summary,
                "source": article.source,
                "tier": article.tier,
                "published_time_utc": article.published_at,
                "published_relative": hours_ago_text(article.published_ts, now),
                "article_url": article.url,
            }
            for article in candidates
        ],
    }
    return json.dumps(prompt_payload, ensure_ascii=True, separators=(",", ":"))


def split_candidates_into_cerebras_batches(candidates: list[Article]) -> list[list[Article]]:
    if not candidates:
        return []
    batch_count = min(CEREBRAS_BATCH_COUNT, len(candidates))
    base_size = len(candidates) // batch_count
    remainder = len(candidates) % batch_count
    batches: list[list[Article]] = []
    start = 0
    for batch_index in range(batch_count):
        batch_size = base_size + (1 if batch_index < remainder else 0)
        end = start + batch_size
        if start < end:
            batches.append(candidates[start:end])
        start = end
    return batches


def build_cerebras_batch_prompt(candidates: list[Article], now: datetime) -> str:
    prompt_content = build_scoring_prompt(candidates, now)
    if len(prompt_content) <= CEREBRAS_MAX_BATCH_PROMPT_CHARS:
        return prompt_content

    for char_limit in (200, 150, 100, 80, 60):
        prompt_content = rebuild_prompt_with_summary_limit(candidates, char_limit)
        if len(prompt_content) <= CEREBRAS_MAX_BATCH_PROMPT_CHARS:
            logging.warning(
                "Cerebras batch prompt exceeded safe size; truncating summaries to %s chars (prompt length=%s).",
                char_limit,
                len(prompt_content),
            )
            return prompt_content

    raise ValueError(
        f"Cerebras batch prompt exceeds safe size limit ({CEREBRAS_MAX_BATCH_PROMPT_CHARS} chars) even after truncation."
    )


def build_social_post_prompt(article: Article, score: dict[str, Any], now: datetime) -> str:
    prompt_payload = {
        "response_schema": SOCIAL_POST_RESPONSE_SCHEMA,
        "prompt_version": GROQ_SOCIAL_PROMPT_VERSION,
        "article": {
            "index": article.index,
            "title": article.title,
            "summary": article.summary,
            "source": article.source,
            "tier": article.tier,
            "published_time_utc": article.published_at,
            "published_relative": hours_ago_text(article.published_ts, now),
            "article_url": article.url,
            "score": {
                "novelty_score": round(float(score.get("novelty_score", 0.0)), 2),
                "impact_score": round(float(score.get("impact_score", 0.0)), 2),
                "freshness_score": round(float(score.get("freshness_score", 0.0)), 2),
                "source_score": round(float(score.get("source_score", 0.0)), 2),
                "total_score": round(float(score.get("authoritative_score", score.get("total_score", 0.0))), 2),
                "reason": clean_whitespace(str(score.get("reason", ""))),
            },
        },
    }
    return json.dumps(prompt_payload, ensure_ascii=True, separators=(",", ":"))


def groq_attempt_label(attempt: int) -> str:
    return "first" if attempt == 1 else "second"


def log_groq_fallback_after_second_failure() -> None:
    logging.warning("Groq failed twice; switching to Cerebras fallback.")


def call_groq_social_posts(
    article: Article,
    score: dict[str, Any],
    now: datetime,
    api_key: str,
    ai_cache: dict[str, Any],
) -> dict[str, str] | None:
    cache_key = build_groq_social_cache_key(article)
    cached_result = get_cached_ai_result(
        ai_cache,
        "groq_social_posts",
        cache_key,
        now,
        GROQ_SOCIAL_CACHE_TTL,
        "Groq social-post",
    )
    if isinstance(cached_result, dict):
        return sanitize_groq_social_posts(cached_result)

    prompt_content = build_social_post_prompt(article, score, now)
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.3,
        "max_completion_tokens": 1000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SOCIAL_POST_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, GROQ_MAX_FAILURES_PER_RUN + 1):
        try:
            response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        except requests.RequestException as exc:
            logging.warning("Groq %s failure: request error on attempt %s: %s", groq_attempt_label(attempt), attempt, exc)
            notify_component_error(
                "Groq Social Posts",
                type(exc).__name__,
                f"Groq social-post request failed on attempt {attempt}.",
                {"article_title": article.title, "attempt": attempt, "run_context": "call_groq_social_posts"},
            )
            if attempt == GROQ_MAX_FAILURES_PER_RUN:
                log_groq_fallback_after_second_failure()
                return None
            time.sleep(GROQ_IMMEDIATE_RETRY_DELAY_SECONDS)
            continue

        if response.status_code == 429:
            raw_header = response.headers.get("retry-after", "")
            wait_seconds = parse_retry_after(raw_header, GROQ_IMMEDIATE_RETRY_DELAY_SECONDS)
            logging.warning("Groq %s failure: rate limited (429).", groq_attempt_label(attempt))
            logging.warning("Groq retry-after value received: '%s' (parsed %.1fs).", raw_header, wait_seconds)
            notify_component_error(
                "Groq Social Posts",
                "429",
                "Groq social-post generation was rate limited.",
                {
                    "article_title": article.title,
                    "attempt": attempt,
                    "retry_after": raw_header,
                    "parsed_retry_after_seconds": wait_seconds,
                    "run_context": "call_groq_social_posts",
                },
            )
            if attempt == GROQ_MAX_FAILURES_PER_RUN:
                log_groq_fallback_after_second_failure()
                return None
            if wait_seconds > GROQ_MAX_RETRY_AFTER_SECONDS:
                logging.warning(
                    "Groq retry-after %.1fs is too large for this run; using immediate retry instead.", wait_seconds
                )
                wait_seconds = GROQ_IMMEDIATE_RETRY_DELAY_SECONDS
            time.sleep(wait_seconds)
            continue

        if response.status_code in {500, 502, 503}:
            logging.warning(
                "Groq %s failure: server error %s on attempt %s.",
                groq_attempt_label(attempt),
                response.status_code,
                attempt,
            )
            notify_component_error(
                "Groq Social Posts",
                str(response.status_code),
                "Groq social-post generation returned a server error.",
                {"article_title": article.title, "attempt": attempt, "run_context": "call_groq_social_posts"},
            )
            if attempt == GROQ_MAX_FAILURES_PER_RUN:
                log_groq_fallback_after_second_failure()
                return None
            time.sleep(GROQ_IMMEDIATE_RETRY_DELAY_SECONDS)
            continue

        if response.status_code in {400, 413}:
            logging.warning(
                "Groq %s failure: %s on attempt %s.", groq_attempt_label(attempt), response.status_code, attempt
            )
            notify_component_error(
                "Groq Social Posts",
                str(response.status_code),
                "Groq social-post generation returned a client error.",
                {"article_title": article.title, "attempt": attempt, "run_context": "call_groq_social_posts"},
            )
            if attempt == GROQ_MAX_FAILURES_PER_RUN:
                log_groq_fallback_after_second_failure()
                return None
            time.sleep(GROQ_IMMEDIATE_RETRY_DELAY_SECONDS)
            continue

        if response.status_code == 401:
            logging.error("Groq API key is invalid or missing (401 Unauthorized).")
            logging.error("Check GROQ_API_KEY secret in GitHub -> Settings -> Secrets.")
            notify_component_error(
                "Groq Social Posts",
                "401",
                "Groq API key is invalid or missing.",
                {"article_title": article.title, "run_context": "call_groq_social_posts"},
            )
            return None

        if response.status_code == 403:
            logging.error(
                "Groq API access forbidden (403). Model may not be allowed for this org/project. "
                "Check Groq console -> model permissions."
            )
            notify_component_error(
                "Groq Social Posts",
                "403",
                "Groq API access is forbidden for social-post generation.",
                {"article_title": article.title, "run_context": "call_groq_social_posts"},
            )
            return None

        try:
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            social_posts = sanitize_groq_social_posts(result.get("social_posts") if isinstance(result, dict) else {})
            set_cached_ai_result(ai_cache, "groq_social_posts", cache_key, social_posts, now)
            return social_posts
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as exc:
            logging.error("Groq %s failure: unexpected error on attempt %s: %s", groq_attempt_label(attempt), attempt, exc)
            notify_component_error(
                "Groq Social Posts",
                type(exc).__name__,
                f"Groq social-post response parsing failed on attempt {attempt}.",
                {"article_title": article.title, "attempt": attempt, "run_context": "call_groq_social_posts"},
            )
            if attempt == GROQ_MAX_FAILURES_PER_RUN:
                log_groq_fallback_after_second_failure()
                return None
            time.sleep(GROQ_IMMEDIATE_RETRY_DELAY_SECONDS)
            continue

    return None


def call_cerebras(
    candidates: list[Article],
    now: datetime,
    api_key: str,
    ai_cache: dict[str, Any],
) -> dict[str, Any] | None:
    cerebras_model = get_cerebras_model()
    env_api_key = os.environ.get("CEREBRAS_API_KEY", "")
    logging.info("Cerebras debug: env var name used = CEREBRAS_API_KEY")
    logging.info("Cerebras debug: CEREBRAS_API_KEY exists in os.environ = %s", "CEREBRAS_API_KEY" in os.environ)
    logging.info("Cerebras debug: env key length = %s", len(env_api_key))
    logging.info("Cerebras debug: env key preview = %s", mask_secret_preview(env_api_key))
    logging.info("Cerebras debug: function key length = %s", len(api_key))
    logging.info("Cerebras debug: function key preview = %s", mask_secret_preview(api_key))
    logging.info("Cerebras debug: key matches env value = %s", api_key == env_api_key)
    logging.info("Cerebras debug: base URL = %s", CEREBRAS_URL)
    logging.info("Cerebras debug: model = %s", cerebras_model)

    cache_key = build_cerebras_scoring_cache_key(candidates)
    cached_result = get_cached_ai_result(
        ai_cache,
        "cerebras_scoring",
        cache_key,
        now,
        CEREBRAS_SCORING_CACHE_TTL,
        "Cerebras scoring",
    )
    if isinstance(cached_result, dict):
        mark_article_links_seen(ai_cache, candidates, now, "scoring_cache_hit")
        return cached_result

    prompt_content = build_cerebras_batch_prompt(candidates, now)
    payload = {
        "model": cerebras_model,
        "temperature": 0.3,
        "max_completion_tokens": 2000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    logging.info("Cerebras debug: Authorization header scheme = Bearer")
    mark_article_links_seen(ai_cache, candidates, now, "scoring_attempted")

    backoff = 2.0
    for attempt in range(1, 6):
        try:
            response = requests.post(CEREBRAS_URL, headers=headers, json=payload, timeout=60)
        except requests.RequestException as exc:
            logging.warning("Cerebras request failed on attempt %s: %s", attempt, exc)
            notify_component_error(
                "Cerebras Scoring",
                type(exc).__name__,
                f"Cerebras scoring request failed on attempt {attempt}.",
                {"attempt": attempt, "candidate_count": len(candidates), "run_context": "call_cerebras"},
            )
            if attempt == 5:
                return None
            time.sleep(backoff)
            backoff *= 2
            continue

        if response.status_code == 429:
            raw_header = response.headers.get("retry-after", "")
            wait_seconds = parse_retry_after(raw_header, backoff)
            logging.warning("Cerebras rate limited (429). Waiting %.1fs (retry-after: '%s').", wait_seconds, raw_header)
            notify_component_error(
                "Cerebras Scoring",
                "429",
                "Cerebras scoring was rate limited.",
                {
                    "attempt": attempt,
                    "candidate_count": len(candidates),
                    "retry_after": raw_header,
                    "parsed_retry_after_seconds": wait_seconds,
                    "run_context": "call_cerebras",
                },
            )
            time.sleep(wait_seconds)
            backoff = min(backoff * 2, 120.0)
            continue

        if response.status_code in {500, 502, 503}:
            logging.warning("Cerebras server error %s on attempt %s.", response.status_code, attempt)
            notify_component_error(
                "Cerebras Scoring",
                str(response.status_code),
                "Cerebras scoring returned a server error.",
                {"attempt": attempt, "candidate_count": len(candidates), "run_context": "call_cerebras"},
            )
            if attempt == 5:
                return None
            time.sleep(backoff)
            backoff *= 2
            continue

        if response.status_code == 400:
            logging.warning("Cerebras 400 Bad Request on attempt %s.", attempt)
            notify_component_error(
                "Cerebras Scoring",
                "400",
                "Cerebras scoring returned 400 Bad Request.",
                {"attempt": attempt, "candidate_count": len(candidates), "run_context": "call_cerebras"},
            )
            if attempt == 1:
                logging.warning("Retrying with summary truncated to 150 chars...")
                prompt_content = rebuild_prompt_with_summary_limit(candidates, 150)
                payload["messages"][1]["content"] = prompt_content
                continue
            if attempt == 2:
                logging.warning("Retrying with max 10 candidates...")
                prompt_content = rebuild_prompt_with_summary_limit(candidates[:10], 100)
                payload["messages"][1]["content"] = prompt_content
                continue
            logging.error("Cerebras 400 persists after 2 payload reductions. Skipping run.")
            return None

        if response.status_code == 413:
            notify_component_error(
                "Cerebras Scoring",
                "413",
                "Cerebras scoring payload was too large.",
                {"attempt": attempt, "candidate_count": len(candidates), "run_context": "call_cerebras"},
            )
            if attempt == 1:
                logging.warning("Cerebras 413: payload too large. Retrying with 15 articles / 200-char summaries.")
                prompt_content = rebuild_prompt_with_summary_limit(candidates[:15], 200)
                payload["messages"][1]["content"] = prompt_content
                continue
            if attempt == 2:
                logging.warning("Cerebras 413 again. Retrying with 8 articles / 100-char summaries.")
                prompt_content = rebuild_prompt_with_summary_limit(candidates[:8], 100)
                payload["messages"][1]["content"] = prompt_content
                continue
            logging.error("Cerebras 413 persists after 2 reductions. Skipping run.")
            return None

        if response.status_code == 401:
            logging.error("Cerebras API key is invalid or missing (401 Unauthorized).")
            logging.error("Check CEREBRAS_API_KEY secret in GitHub -> Settings -> Secrets.")
            logging.error("Cerebras debug: 401 response body = %s", trim_error_text(response.text, 1000))
            notify_component_error(
                "Cerebras Scoring",
                "401",
                "Cerebras API key is invalid or missing.",
                {"run_context": "call_cerebras"},
            )
            return None

        if response.status_code == 403:
            logging.error(
                "Cerebras API access forbidden (403). Model may not be allowed for this org/project. "
                "Check Cerebras console -> model permissions."
            )
            notify_component_error(
                "Cerebras Scoring",
                "403",
                "Cerebras API access is forbidden.",
                {"run_context": "call_cerebras"},
            )
            return None

        try:
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if len(content) > 1200:
                save_debug_text(DEBUG_CEREBRAS_RESPONSE_PATH, content)
                logging.error(
                    "Cerebras raw response content saved to %s (length=%s).",
                    DEBUG_CEREBRAS_RESPONSE_PATH.name,
                    len(content),
                )
            else:
                logging.error("Cerebras raw response content: %s", content)
            result = parse_llm_json_object(content)
            set_cached_ai_result(ai_cache, "cerebras_scoring", cache_key, result, now)
            return result
        except (requests.RequestException, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            logging.error("Cerebras unexpected error on attempt %s: %s", attempt, exc)
            notify_component_error(
                "Cerebras Scoring",
                type(exc).__name__,
                f"Cerebras scoring response parsing failed on attempt {attempt}.",
                {"attempt": attempt, "candidate_count": len(candidates), "run_context": "call_cerebras"},
            )
            return default_skip_scoring_result()

    return None


def merge_cerebras_batch_results(
    batch_results: list[dict[str, Any]],
    candidates: list[Article],
) -> dict[str, Any]:
    merged_articles: list[dict[str, Any]] = []
    batch_winners: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for batch_result in batch_results:
        articles = batch_result.get("articles", [])
        if isinstance(articles, list):
            merged_articles.extend(item for item in articles if isinstance(item, dict))
        score_map = score_map_from_result(batch_result)
        batch_best_index = batch_result.get("best_index", -1)
        if isinstance(batch_best_index, int):
            batch_best_score = score_map.get(batch_best_index)
            if batch_best_score:
                batch_winners.append((batch_result, batch_best_score))

    if not merged_articles or not batch_winners:
        fallback = default_skip_scoring_result()
        fallback["authoritative_score"] = 0.0
        return fallback

    merged_articles.sort(key=lambda item: int(item.get("index", -1)))
    candidate_index_map = {article.index: article for article in candidates}

    def winner_sort_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, float]:
        _, winner_score = item
        winner_index = int(winner_score.get("index", -1))
        return (
            float(winner_score.get("total_score", 0.0)),
            -float(winner_index),
        )

    selected_batch_result, selected_winner = sorted(batch_winners, key=winner_sort_key, reverse=True)[0]
    best_index = int(selected_winner.get("index", -1))
    best_article = candidate_index_map.get(best_index)
    authoritative_score = round(float(selected_winner.get("total_score", 0.0)), 2)
    recommendation = "POST_NOW" if authoritative_score >= 8.5 else "SKIP"
    social_posts = normalize_social_posts(selected_batch_result.get("social_posts", {}))

    return {
        "articles": merged_articles,
        "best_index": best_index,
        "best_score": authoritative_score,
        "authoritative_score": authoritative_score,
        "recommendation": recommendation,
        "social_posts": social_posts,
        "selected_title": best_article.title if best_article else "",
    }


def score_candidates_with_cerebras_batches(
    candidates: list[Article],
    now: datetime,
    api_key: str,
    ai_cache: dict[str, Any],
) -> dict[str, Any] | None:
    batch_results: list[dict[str, Any]] = []
    batches = split_candidates_into_cerebras_batches(candidates)
    for batch_number, batch_candidates in enumerate(batches, start=1):
        logging.info(
            "Scoring Cerebras batch %s/%s with %s candidate(s).",
            batch_number,
            len(batches),
            len(batch_candidates),
        )
        try:
            batch_raw_result = call_cerebras(batch_candidates, now, api_key, ai_cache)
        except ValueError as exc:
            logging.error("Cerebras batch %s failed before request: %s", batch_number, exc)
            notify_component_error(
                "Cerebras Scoring",
                "BatchPreparationError",
                f"Cerebras batch {batch_number} failed before request.",
                {"batch_number": batch_number, "candidate_count": len(batch_candidates), "run_context": "score_candidates_with_cerebras_batches"},
            )
            continue

        if batch_raw_result is None:
            logging.warning("Skipping Cerebras batch %s after request failure.", batch_number)
            continue

        normalized_batch = normalize_scoring_result(batch_raw_result, f"Cerebras batch {batch_number}")
        if not normalized_batch or not normalized_batch.get("articles"):
            logging.warning("Skipping Cerebras batch %s after invalid or empty structured output.", batch_number)
            continue

        batch_results.append(normalized_batch)

    if not batch_results:
        logging.warning("All Cerebras scoring batches failed or returned invalid output.")
        notify_component_error(
            "Cerebras Scoring",
            "AllBatchesFailed",
            "All Cerebras scoring batches failed or returned invalid structured output.",
            {"batch_count": len(batches), "run_context": "score_candidates_with_cerebras_batches"},
        )
        fallback = default_skip_scoring_result()
        fallback["authoritative_score"] = 0.0
        return fallback

    merged_result = merge_cerebras_batch_results(batch_results, candidates)
    logging.info(
        "Merged %s successful Cerebras batch result(s); final best_index=%s score=%.2f recommendation=%s.",
        len(batch_results),
        merged_result.get("best_index", -1),
        float(merged_result.get("authoritative_score", 0.0)),
        merged_result.get("recommendation", "SKIP"),
    )
    return merged_result


def normalize_scoring_result(result: dict[str, Any], provider_name: str = "Provider") -> dict[str, Any] | None:
    if not isinstance(result, dict) or not isinstance(result.get("articles"), list):
        notify_component_error(
            f"{provider_name} Scoring",
            "InvalidResponse",
            f"{provider_name} returned a non-dict or missing articles list.",
            {"run_context": "normalize_scoring_result"},
        )
        return None
    recommendation = str(result.get("recommendation", "SKIP")).upper()
    if recommendation not in {"POST_NOW", "SKIP"}:
        recommendation = "SKIP"

    normalized_articles = []
    for item in result.get("articles", []):
        if not isinstance(item, dict):
            continue
        try:
            normalized_item = {
                "index": int(item.get("index", -1)),
                "title": clean_whitespace(str(item.get("title", ""))),
                "summary": clean_whitespace(str(item.get("summary", ""))),
                "novelty_score": float(item.get("novelty_score", 0.0)),
                "impact_score": float(item.get("impact_score", 0.0)),
                "freshness_score": float(item.get("freshness_score", 0.0)),
                "source_score": float(item.get("source_score", 0.0)),
                "total_score": float(item.get("total_score", 0.0)),
                "red_flag": bool(item.get("red_flag", False)),
                "red_flag_reason": item.get("red_flag_reason"),
                "reason": clean_whitespace(str(item.get("reason", ""))),
            }
            matched_pattern = find_local_red_flag_pattern(normalized_item)
            if matched_pattern and not normalized_item["red_flag"]:
                normalized_item["red_flag"] = True
                normalized_item["total_score"] = 0.0
                normalized_item["red_flag_reason"] = "Local enforcement override"
                logging.warning(
                    '%s missed red flag on [%s] - overriding to 0.00',
                    provider_name,
                    normalized_item["title"],
                )
                notify_component_error(
                    f"{provider_name} Scoring",
                    "RedFlagOverride",
                    f"{provider_name} missed a local red-flag pattern; local enforcement applied.",
                    {"article_title": normalized_item["title"], "run_context": "normalize_scoring_result"},
                )
            elif normalized_item["red_flag"]:
                normalized_item["total_score"] = 0.0
            normalized_articles.append(normalized_item)
        except (TypeError, ValueError):
            continue

    try:
        best_index = int(result.get("best_index", -1))
    except (TypeError, ValueError):
        best_index = -1

    try:
        best_score = float(result.get("best_score", 0.0))
    except (TypeError, ValueError):
        best_score = 0.0

    valid_articles = [item for item in normalized_articles if item.get("index", -1) >= 0]
    if best_index < 0 or not any(item["index"] == best_index for item in valid_articles):
        if valid_articles:
            logging.warning("Invalid best_index from %s - using local max", provider_name)
            notify_component_error(
                f"{provider_name} Scoring",
                "InvalidBestIndex",
                f"{provider_name} returned an invalid best_index; local max was used.",
                {"run_context": "normalize_scoring_result"},
            )
            local_max = max(valid_articles, key=lambda item: item["total_score"])
            best_index = int(local_max["index"])
        else:
            best_index = -1

    enforced_best_index = best_index
    enforced_best_score = 0.0
    best_article = next((item for item in normalized_articles if item["index"] == best_index), None)
    if not best_article or best_article["red_flag"]:
        non_red_articles = [item for item in normalized_articles if not item["red_flag"]]
        if non_red_articles:
            best_article = max(non_red_articles, key=lambda item: item["total_score"])
            enforced_best_index = int(best_article["index"])
        else:
            enforced_best_index = -1
            enforced_best_score = 0.0
            recommendation = "SKIP"

    if best_article:
        article_score = float(best_article["total_score"])
        if abs(article_score - best_score) > 0.5:
            logging.warning(
                "Score mismatch: %s best_score=%.2f, article[%s].total_score=%.2f. Using article score.",
                provider_name,
                best_score,
                enforced_best_index,
                article_score,
            )
            notify_component_error(
                f"{provider_name} Scoring",
                "ScoreMismatch",
                f"{provider_name} best_score did not match the selected article score; local article score was used.",
                {"article_title": best_article["title"], "run_context": "normalize_scoring_result"},
            )
        enforced_best_score = article_score

    if enforced_best_score >= 8.5:
        recommendation = "POST_NOW"
    else:
        recommendation = "SKIP"

    social_posts = normalize_social_posts(result.get("social_posts"))

    return {
        "articles": normalized_articles,
        "best_index": enforced_best_index,
        "best_score": enforced_best_score,
        "authoritative_score": enforced_best_score,
        "recommendation": recommendation,
        "social_posts": social_posts,
    }


def score_map_from_result(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {item["index"]: item for item in result.get("articles", []) if item.get("index", -1) >= 0}


def build_candidate_payload(
    article: Article,
    score: dict[str, Any],
    social_posts: dict[str, str],
    saved_at: datetime,
) -> dict[str, Any]:
    authoritative_score = round(float(score.get("authoritative_score", score.get("total_score", 0.0))), 2)
    cleaned_social_posts = normalize_social_posts(social_posts)
    return {
        "index": article.index,
        "title": article.title,
        "summary": article.summary,
        "url": article.url,
        "canonical_url": article.canonical_url,
        "url_hash": article.url_hash,
        "cleaned_title": article.cleaned_title,
        "source": article.source,
        "tier": article.tier,
        "published_at": article.published_at,
        "score": {
            "novelty_score": round(float(score.get("novelty_score", 0.0)), 2),
            "impact_score": round(float(score.get("impact_score", 0.0)), 2),
            "freshness_score": round(float(score.get("freshness_score", 0.0)), 2),
            "source_score": round(float(score.get("source_score", 0.0)), 2),
            "total_score": authoritative_score,
            "red_flag": bool(score.get("red_flag", False)),
            "red_flag_reason": score.get("red_flag_reason"),
            "reason": clean_whitespace(str(score.get("reason", ""))),
        },
        "authoritative_score": authoritative_score,
        "social_posts": cleaned_social_posts,
        "saved_at": isoformat_utc(saved_at),
    }


def init_google_sheets(credentials_json: str) -> gspread.Worksheet:
    credentials = json.loads(credentials_json)
    client = gspread.service_account_from_dict(credentials)
    spreadsheet = client.open_by_key(os.environ["GOOGLE_SHEET_ID"])
    worksheet = spreadsheet.get_worksheet(0)
    if worksheet is None:
        raise RuntimeError("Google Sheet does not contain a first worksheet.")
    return worksheet


def append_to_sheet(
    worksheet: gspread.Worksheet,
    article: Article,
    score: dict[str, Any],
    social_posts: dict[str, str],
    now: datetime,
) -> bool:
    total_score = round(float(score.get("authoritative_score", score.get("total_score", 0.0))), 2)
    row = [
        isoformat_utc(now),
        article.title,
        f"{article.source} ({article.tier})",
        total_score,
        article.canonical_url,
        social_posts.get("twitter", ""),
        social_posts.get("facebook", ""),
        social_posts.get("threads", ""),
        social_posts.get("telegram", ""),
    ]
    try:
        worksheet.append_row(row, value_input_option="RAW")
    except Exception as exc:  # pragma: no cover
        logging.error("Google Sheets append failed: %s", exc)
        notify_component_error(
            "Google Sheets",
            type(exc).__name__,
            "Google Sheets append failed.",
            {"article_title": article.title, "run_context": "append_to_sheet"},
        )
        return False
    return True


def log_append_candidate_details(
    article: Article,
    score: float,
    recommendation: str,
    social_posts: dict[str, str],
    test_force_mode_used: bool,
) -> None:
    logging.info("Append candidate title: %s", article.title)
    logging.info("Append candidate score: %.2f", score)
    logging.info("Append candidate recommendation: %s", recommendation)
    logging.info("Append candidate test-force mode used: %s", test_force_mode_used)
    for platform in SOCIAL_POST_PLATFORMS:
        is_non_empty = bool(social_posts.get(platform, "").strip())
        logging.info(
            "Append candidate social post non-empty [%s]: %s",
            platform,
            is_non_empty,
        )
        if not is_non_empty:
            logging.warning("Append candidate missing social post for platform: %s", platform)


def mark_as_posted(posted: dict[str, Any], payload: dict[str, Any], posted_at: datetime) -> None:
    timestamp = isoformat_utc(posted_at)
    hash_record = {"hash": payload["url_hash"], "posted_at": timestamp}
    title_record = {
        "cleaned_title": payload["cleaned_title"],
        "posted_at": timestamp,
        "source": payload["source"],
        "score": round(float(payload["score"]["total_score"]), 2),
    }
    posted.setdefault("hash_records", []).append(hash_record)
    posted.setdefault("recent_titles", []).append(title_record)
    cleanup_posted_history(posted, posted_at)


def persist_state_files(state: dict[str, Any], posted: dict[str, Any]) -> None:
    save_json(STATE_PATH, state)
    save_json(POSTED_PATH, posted)


def log_final_decision(score: float, recommendation: str, best_index: int) -> None:
    logging.info(
        "Final decision: score=%.2f, recommendation=%s, source=article[%s]",
        score,
        recommendation,
        best_index,
    )


def main() -> int:
    configure_logging()
    try:
        secrets = require_env()
        cerebras_model = get_cerebras_model()
        logging.info("Cerebras model in use: %s", cerebras_model)
        now = utc_now()
        worksheet = init_google_sheets(secrets["GOOGLE_CREDENTIALS"])
        ai_cache = load_ai_cache(now)

        posted = sanitize_posted(load_json(POSTED_PATH, default_posted))
        cleanup_posted_history(posted, now)
        state = sanitize_state(load_json(STATE_PATH, lambda: default_state(now)), now)
        state, _ = reset_state_if_needed(state, now)
        fetched_articles = fetch_all_feeds(now)
        candidates = dedupe_candidates(fetched_articles, posted, now)
        logging.info("Found %s deduplicated fresh candidate(s).", len(candidates))
        candidates = filter_unseen_candidates(candidates, ai_cache, now)
        logging.info("Proceeding with %s unseen candidate(s) after ai_cache link filtering.", len(candidates))

        if candidates:
            normalized = score_candidates_with_cerebras_batches(candidates, now, secrets["CEREBRAS_API_KEY"], ai_cache)
            if normalized:
                score_map = score_map_from_result(normalized)
                best_index = normalized["best_index"]
                best_article = next((article for article in candidates if article.index == best_index), None)
                best_score = score_map.get(best_index)
                authoritative_score = float(normalized.get("authoritative_score", 0.0))

                if best_article and best_score and not best_score["red_flag"]:
                    best_score = dict(best_score)
                    best_score["authoritative_score"] = authoritative_score
                    recommendation = normalized["recommendation"]
                    test_force_mode_used = recommendation != "POST_NOW" and FORCE_APPEND_BEST_FOR_TEST
                    effective_recommendation = "POST_NOW" if test_force_mode_used else recommendation
                    selected_social_posts = normalize_social_posts(normalized.get("social_posts", {}))

                    if effective_recommendation == "POST_NOW":
                        groq_social_posts = call_groq_social_posts(
                            best_article, best_score, now, secrets["GROQ_API_KEY"], ai_cache
                        )
                        missing_platforms = missing_social_post_platforms(groq_social_posts or {})
                        if groq_social_posts and not missing_platforms:
                            selected_social_posts = groq_social_posts
                        else:
                            explanation = "Groq social-post generation failed; using Cerebras social posts fallback."
                            if missing_platforms:
                                explanation = (
                                    "Groq social-post generation returned empty social posts; "
                                    "using Cerebras social posts fallback."
                                )
                            notify_component_error(
                                "Groq Social Posts",
                                "FallbackToCerebras",
                                explanation,
                                {
                                    "article_title": best_article.title,
                                    "missing_platforms": missing_platforms,
                                    "run_context": "main",
                                },
                            )
                            for platform in missing_platforms:
                                logging.warning("Groq returned empty social post for platform: %s", platform)

                    payload = build_candidate_payload(best_article, best_score, selected_social_posts, now)
                    log_append_candidate_details(
                        best_article,
                        authoritative_score,
                        effective_recommendation,
                        payload["social_posts"],
                        test_force_mode_used,
                    )
                    log_final_decision(authoritative_score, effective_recommendation, best_index)

                    if effective_recommendation == "POST_NOW":
                        if test_force_mode_used:
                            logging.warning(
                                "TEST MODE: forcing append of best-scoring article to Google Sheets despite SKIP recommendation."
                            )
                        success = append_to_sheet(worksheet, best_article, best_score, payload["social_posts"], now)
                        if success:
                            state["posts_today"] = int(state.get("posts_today", 0)) + 1
                            state["last_post_time"] = isoformat_utc(now)
                            mark_as_posted(posted, payload, now)
                            persist_state_files(state, posted)
                            logging.info("Article written to Google Sheet: %s", best_article.title)
                            return 0
                    else:
                        persist_state_files(state, posted)
                        return 0
                else:
                    log_final_decision(authoritative_score, normalized["recommendation"], best_index)
            else:
                log_final_decision(0.0, "SKIP", -1)

        if not candidates:
            logging.info("No new articles qualified for scoring in this run.")
        else:
            logging.info("No candidate qualified for Google Sheets logging.")

        if not candidates:
            log_final_decision(0.0, "SKIP", -1)

        persist_state_files(state, posted)
        return 0
    except Exception as exc:
        logging.exception("Unhandled error in main: %s", exc)
        notify_component_error(
            "Main",
            type(exc).__name__,
            "Unhandled runtime failure in main().",
            {"run_context": "main"},
        )
        return 1
