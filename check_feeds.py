from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser
import requests


FEEDS = [
    {"source": "OpenAI", "urls": ["https://openai.com/news/rss.xml"]},
    {
        "source": "Anthropic",
        "urls": ["https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml"],
    },
    {
        "source": "Google AI",
        "urls": [
            "https://blog.google/technology/ai/rss/",
            "https://blog.research.google/feeds/posts/default?alt=rss",
        ],
    },
    {"source": "HuggingFace", "urls": ["https://huggingface.co/blog/feed.xml"]},
    {
        "source": "Microsoft AI",
        "urls": [
            "https://news.microsoft.com/source/topics/ai/feed/",
            "https://blogs.microsoft.com/ai/feed/",
            "https://blogs.microsoft.com/feed/",
        ],
    },
    {
        "source": "TechCrunch",
        "urls": [
            "https://techcrunch.com/category/artificial-intelligence/feed/",
            "https://techcrunch.com/tag/artificial-intelligence/feed/",
        ],
    },
    {"source": "The Verge", "urls": ["https://www.theverge.com/rss/index.xml"]},
    {"source": "Ars Technica", "urls": ["https://arstechnica.com/ai/feed/"]},
    {"source": "MarkTechPost", "urls": ["https://www.marktechpost.com/feed/"]},
    {"source": "Wired AI", "urls": ["https://www.wired.com/feed/tag/ai/latest/rss"]},
    {
        "source": "MIT News AI",
        "urls": ["https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml"],
    },
    {"source": "InfoQ AI/ML", "urls": ["https://feed.infoq.com/"]},
    {"source": "AI News", "urls": ["https://artificialintelligence-news.com/feed/"]},
    {"source": "arXiv cs.AI", "urls": ["https://rss.arxiv.org/rss/cs.AI"]},
    {"source": "Hacker News", "urls": ["https://news.ycombinator.com/rss"]},
]

REQUEST_HEADERS = {
    "User-Agent": "NexusFeedFeedHealth/1.0",
    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_entry_datetime(entry) -> datetime | None:
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


def age_text(delta: timedelta | None) -> str:
    if delta is None:
        return "N/A"
    total_seconds = max(0, int(delta.total_seconds()))
    days, remainder = divmod(total_seconds, 86400)
    hours, _ = divmod(remainder, 3600)
    if days > 0:
        return f"{days}d {hours}h ago"
    if hours > 0:
        return f"{hours}h ago"
    minutes = max(1, total_seconds // 60)
    return f"{minutes}m ago"


def evaluate_url(url: str) -> dict:
    started = time.perf_counter()
    http_status = None
    response_time_ms = None
    http_ok = False
    http_error = ""
    parse_ok = False
    bozo = True
    entries_count = 0
    latest_age = None
    status = "HTTP_FAIL"

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        response_time_ms = int((time.perf_counter() - started) * 1000)
        http_status = response.status_code
        http_ok = response.status_code == 200
        if not http_ok:
            status = "HTTP_FAIL"
    except requests.RequestException as exc:
        response_time_ms = int((time.perf_counter() - started) * 1000)
        http_error = str(exc)
        return {
            "url": url,
            "http_status": "ERR",
            "response_time_ms": response_time_ms,
            "http_ok": False,
            "parse": "FAIL",
            "entries": 0,
            "latest_entry_age": "N/A",
            "status": "HTTP_FAIL",
            "bozo": True,
            "http_error": http_error,
        }

    parsed = feedparser.parse(response.content)
    bozo = bool(getattr(parsed, "bozo", False))
    entries_count = len(parsed.entries)
    parse_ok = not bozo and entries_count > 0

    if not parse_ok:
        status = "PARSE_FAIL"
    else:
        latest_dt = None
        for entry in parsed.entries:
            entry_dt = parse_entry_datetime(entry)
            if entry_dt and (latest_dt is None or entry_dt > latest_dt):
                latest_dt = entry_dt

        if latest_dt is None:
            status = "NO_TIMESTAMP"
        else:
            latest_age = utc_now() - latest_dt
            if latest_age > timedelta(days=7):
                status = "STALE"
            else:
                status = "OK"

    return {
        "url": url,
        "http_status": http_status,
        "response_time_ms": response_time_ms,
        "http_ok": http_ok,
        "parse": "OK" if parse_ok else "FAIL",
        "entries": entries_count,
        "latest_entry_age": age_text(latest_age),
        "status": status,
        "bozo": bozo,
        "http_error": http_error,
    }


def pick_best_result(results: list[dict]) -> dict:
    order = {"OK": 0, "STALE": 1, "NO_TIMESTAMP": 2, "PARSE_FAIL": 3, "HTTP_FAIL": 4}
    return sorted(results, key=lambda item: (order.get(item["status"], 99), 0 if item["http_ok"] else 1))[0]


def print_table(rows: list[dict]) -> None:
    headers = ["SOURCE", "URL", "HTTP", "PARSE", "ENTRIES", "LATEST_ENTRY_AGE", "STATUS"]
    widths = {
        "SOURCE": max(len("SOURCE"), *(len(row["source"]) for row in rows)),
        "URL": max(len("URL"), *(len(row["url"]) for row in rows)),
        "HTTP": len("HTTP"),
        "PARSE": len("PARSE"),
        "ENTRIES": len("ENTRIES"),
        "LATEST_ENTRY_AGE": len("LATEST_ENTRY_AGE"),
        "STATUS": len("STATUS"),
    }
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    divider_line = "-|-".join("-" * widths[header] for header in headers)
    print(header_line)
    print(divider_line)
    for row in rows:
        print(
            " | ".join(
                [
                    row["source"].ljust(widths["SOURCE"]),
                    row["url"].ljust(widths["URL"]),
                    str(row["http_status"]).ljust(widths["HTTP"]),
                    row["parse"].ljust(widths["PARSE"]),
                    str(row["entries"]).ljust(widths["ENTRIES"]),
                    row["latest_entry_age"].ljust(widths["LATEST_ENTRY_AGE"]),
                    row["display_status"].ljust(widths["STATUS"]),
                ]
            )
        )


def main() -> int:
    rows = []
    dead_feeds = []
    stale_feeds = []
    no_timestamp_feeds = []
    working_count = 0

    for feed in FEEDS:
        results = [evaluate_url(url) for url in feed["urls"]]
        best = pick_best_result(results)
        best["source"] = feed["source"]

        if best["status"] == "OK":
            best["display_status"] = "OK"
            working_count += 1
        elif best["status"] == "STALE":
            best["display_status"] = "STALE"
            stale_feeds.append(feed["source"])
        elif best["status"] == "NO_TIMESTAMP":
            best["display_status"] = "NO_TIMESTAMP"
            no_timestamp_feeds.append(feed["source"])
        else:
            best["display_status"] = "DEAD"
            dead_feeds.append(feed["source"])

        rows.append(best)

    print_table(rows)
    print()
    print(f"Working feeds: {working_count} / {len(FEEDS)}")
    print(f"Dead feeds:    {len(dead_feeds)} ({', '.join(dead_feeds) if dead_feeds else 'none'})")
    print(f"Stale feeds:   {len(stale_feeds)} ({', '.join(stale_feeds) if stale_feeds else 'none'})")
    print(f"No timestamp:  {len(no_timestamp_feeds)} ({', '.join(no_timestamp_feeds) if no_timestamp_feeds else 'none'})")

    with open("feed_health.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "working_count": working_count,
                "total_count": len(FEEDS),
                "dead_feeds": dead_feeds,
                "stale_feeds": stale_feeds,
                "checked_at": isoformat_utc(utc_now()),
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
