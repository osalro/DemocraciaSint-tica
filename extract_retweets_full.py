"""
Reproducibility script — Fetches all retweets (of image-posts) made by 13
candidate/party accounts for the full campaign window (Jul 1 2025 – Feb 1 2026)
in a single run.

Prerequisites:
    bearer_token=<token> in .env  (or BEARER_TOKEN as env var)
    pip install -r requirements.txt

Estimated cost:  ~13,000 reads / ~$25 (retweet expansions bill ~3x per timeline tweet)
Estimated time:  ~2 minutos (15 s/page × 13 accounts)

WARNING: The output file will be OVERWRITTEN on each run.
"""

import tweepy
import csv
import datetime
import unicodedata
import time
import json
import os
import sys
import argparse

# ─── Path setup ──────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from budget import (
    load_tracker, check_budget, record_reads, record_user_lookup,
    print_budget_summary, get_remaining_budget
)

# ─── Configuration ───────────────────────────────────────────────────────────

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
bearer_token = os.getenv("bearer_token") or os.getenv("BEARER_TOKEN")

START_TIME = datetime.datetime(2025, 7, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
END_TIME   = datetime.datetime(2026, 2, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

USERNAMES = [
    "jchidalgo",
    "laurapresi2026",
    "RoblesBarrantes",
    "ClaudiaDobles",
    "JoseAguilarBerr",
    "FabriAlvarado7",
    "FrenteAmplio",
    "accionciudadana",
    "pusc_cr",
    "Avanza_cr",
    "nuevarepublica7",
    "elifeinzaig",
    "liberalcr",
]

KNOWN_USER_IDS = {
    "jchidalgo": 23224517,
    "laurapresi2026": 1970904485686812673,
    "RoblesBarrantes": 1155523807228157952,
    "ClaudiaDobles": 910585823623606272,
    "JoseAguilarBerr": 1497413647,
    "FabriAlvarado7": 1589859326,
    "FrenteAmplio": 29837045,
    "accionciudadana": 19989379,
    "pusc_cr": 1090371633766903809,
    "Avanza_cr": 1912267332279312384,
    "nuevarepublica7": 1059527279615848449,
    "elifeinzaig": 895419694198534148,
    "liberalcr": 313454531,
}

TWEET_FIELDS = [
    "attachments", "author_id", "created_at", "entities",
    "public_metrics", "lang", "possibly_sensitive", "conversation_id",
    "note_tweet", "context_annotations"
]
MEDIA_FIELDS = ["media_key", "type", "url", "width", "height", "alt_text"]
USER_FIELDS  = ["username", "name", "verified", "profile_image_url", "public_metrics"]
EXPANSIONS   = [
    "attachments.media_keys",
    "author_id",
    "referenced_tweets.id",
    "referenced_tweets.id.author_id",
]

PAGE_SLEEP    = 15  # seconds between pages (polite rate limiting)
ACCOUNT_SLEEP = 5   # seconds between accounts

# Matches d3_retweets_tweet_level.csv exactly (23 columns)
CSV_HEADERS = [
    "count", "candidato_retuiteador", "retweet_id", "retweet_created_at",
    "original_tweet_id", "original_author_username", "original_author_id", "original_text",
    "like_count", "retweet_count", "reply_count", "quote_count",
    "bookmark_count", "impression_count",
    "media_urls", "media_type", "image_count",
    "lang", "possibly_sensitive",
    "image_alt_texts", "image_dimensions",
    "note_tweet", "context_annotations"
]

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "deliverables", "retweets_raw_extraction.csv")


def _get(obj, attr, default=""):
    """Unified attribute access for tweepy objects and plain dicts."""
    if type(obj) is dict:
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ─── Argparse ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Full-range retweet reproducibility extractor")
parser.add_argument("--dry-run", action="store_true", help="Estimate cost without calling API")
args = parser.parse_args()

# ─── Dry-run mode ────────────────────────────────────────────────────────────

if args.dry_run:
    print("DRY RUN MODE - no API calls will be made.\n")
    estimated_reads_per_account = 500 * 3  # ~3x billing due to expansions
    estimated_total = estimated_reads_per_account * len(USERNAMES)
    print(f"  Accounts      : {len(USERNAMES)}")
    print(f"  Date range    : {START_TIME.date()} to {END_TIME.date()}")
    print(f"  Est. reads    : ~{estimated_total:,} (~{estimated_reads_per_account}/account incl. expansions)")
    print(f"  Est. cost     : ~${estimated_total * 0.005:.2f}")
    print(f"  Est. time     : ~{(estimated_total / 3 / 100 * PAGE_SLEEP) / 3600:.1f} hours")
    print(f"  Output file   : {OUTPUT_FILE}")
    tracker = load_tracker()
    _, _, msg = check_budget(tracker, estimated_total)
    print(f"\n  {msg}")
    print_budget_summary(tracker)
    sys.exit(0)

# ─── Bearer token check ──────────────────────────────────────────────────────

if not bearer_token:
    raise SystemExit("Error: bearer_token not found. Set it in .env or as BEARER_TOKEN env var.")

# ─── Startup warning ─────────────────────────────────────────────────────────

print("=" * 60)
print("REPRODUCIBILITY EXTRACTION — RETWEETS (Jul 2025 – Feb 2026)")
print("=" * 60)
print(f"Output: {OUTPUT_FILE}")
print("WARNING: This file will be OVERWRITTEN if it exists.")
print("Press Ctrl+C within 5 seconds to abort...")
time.sleep(5)
print()

# ─── Budget check ────────────────────────────────────────────────────────────

tracker = load_tracker()
can_proceed, _, msg = check_budget(tracker)
print(msg)
if not can_proceed:
    print_budget_summary(tracker)
    raise SystemExit("Aborting due to budget constraints.")

# ─── API client ──────────────────────────────────────────────────────────────

client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)

# ─── Main extraction ─────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

total_retweets_saved = 0
total_reads_used     = 0
global_count         = 1

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(CSV_HEADERS)

    for username in USERNAMES:
        can_proceed, _, msg = check_budget(tracker, 100)
        if not can_proceed:
            print(f"\n{msg}")
            print("Stopping — budget exhausted before all accounts were processed.")
            break

        # Resolve user ID
        user_id = KNOWN_USER_IDS.get(username)
        if not user_id:
            try:
                user = client.get_user(username=username)
                record_user_lookup(tracker, count=1, source="full_retweets:user_lookup",
                                   details=f"resolved @{username}")
                if user.data is None:
                    print(f"  Warning: @{username} not found. Skipping.")
                    continue
                user_id = user.data.id
            except Exception as e:
                print(f"  Error looking up @{username}: {e}. Skipping.")
                continue

        print(f"\nFetching retweets for @{username} ({START_TIME.date()} to {END_TIME.date()})...")

        pagination_token = None
        account_saved    = 0
        account_reads    = 0

        while True:
            can_proceed, _, msg = check_budget(tracker, 100)
            if not can_proceed:
                print(f"  {msg}")
                break

            try:
                tweets = client.get_users_tweets(
                    id=user_id,
                    max_results=100,
                    tweet_fields=TWEET_FIELDS,
                    expansions=EXPANSIONS,
                    media_fields=MEDIA_FIELDS,
                    user_fields=USER_FIELDS,
                    start_time=START_TIME.isoformat(),
                    end_time=END_TIME.isoformat(),
                    pagination_token=pagination_token,
                )
            except tweepy.TooManyRequests as e:
                if hasattr(e, "response") and e.response is not None:
                    reset_ts = e.response.headers.get("x-rate-limit-reset")
                    if reset_ts:
                        now_ts   = datetime.datetime.now(datetime.timezone.utc).timestamp()
                        wait_sec = max(0, int(reset_ts) - int(now_ts)) + 5
                        print(f"  Rate limited. Sleeping {wait_sec}s...")
                        time.sleep(wait_sec)
                        continue
                print("  Rate limited (no reset header). Sleeping 15 minutes...")
                time.sleep(900)
                continue
            except Exception as e:
                print(f"  Error fetching page: {e}")
                break

            # Bill conservatively: timeline tweets + expanded tweets + expanded users
            timeline_count = tweets.meta.get("result_count", 0) if tweets.meta else 0
            expanded_count = 0
            if tweets.includes:
                expanded_count += len(tweets.includes.get("tweets", []))
                expanded_count += len(tweets.includes.get("users", []))
            read_cost = timeline_count + expanded_count
            account_reads    += read_cost
            total_reads_used += read_cost
            record_reads(tracker, read_cost, f"full_retweets:{username}",
                         f"{timeline_count} timeline + {expanded_count} expanded objects")

            if hasattr(tweets, "response") and tweets.response is not None:
                hdrs = tweets.response.headers
                print(f"  Page: {timeline_count} tweets | "
                      f"rate {hdrs.get('x-rate-limit-remaining','?')}/{hdrs.get('x-rate-limit-limit','?')} | "
                      f"{get_remaining_budget(tracker):,} reads left")

            # Build lookup tables from includes
            media_lookup = {}
            user_lookup  = {}
            tweet_lookup = {}

            if tweets.includes:
                for m in tweets.includes.get("media", []):
                    key = _get(m, "media_key")
                    if key:
                        media_lookup[key] = m
                for u in tweets.includes.get("users", []):
                    uid = _get(u, "id") or getattr(u, "id", None)
                    if uid is not None:
                        user_lookup[uid] = u
                for t in tweets.includes.get("tweets", []):
                    tid = _get(t, "id") or getattr(t, "id", None)
                    if tid is not None:
                        tweet_lookup[tid] = t

            if tweets.data:
                for tweet in tweets.data:
                    # Client-side date guard
                    if tweet.created_at:
                        created = tweet.created_at
                        if created.tzinfo is None:
                            created = created.replace(tzinfo=datetime.timezone.utc)
                        if not (START_TIME <= created < END_TIME):
                            continue

                    # Identify retweets
                    original_tweet_id = None
                    if hasattr(tweet, "referenced_tweets") and tweet.referenced_tweets:
                        for ref in tweet.referenced_tweets:
                            if ref.type == "retweeted":
                                original_tweet_id = ref.id
                                break

                    if not original_tweet_id:
                        continue

                    orig = tweet_lookup.get(original_tweet_id)
                    if not orig:
                        continue

                    # Extract media from the ORIGINAL tweet
                    media_urls  = []
                    media_types = []
                    alt_texts   = []
                    dimensions  = []
                    image_count = 0

                    orig_attachments = _get(orig, "attachments") or getattr(orig, "attachments", None)
                    if orig_attachments:
                        mk_list = (orig_attachments.get("media_keys", [])
                                   if type(orig_attachments) is dict
                                   else orig_attachments.get("media_keys", []))
                        for mk in mk_list:
                            if mk not in media_lookup:
                                continue
                            m     = media_lookup[mk]
                            mtype = str(_get(m, "type") or getattr(m, "type", ""))
                            if mtype != "photo":
                                continue
                            media_types.append("photo")
                            media_urls.append(str(_get(m, "url") or getattr(m, "url", "")))
                            alt = _get(m, "alt_text") or getattr(m, "alt_text", "")
                            alt_texts.append(str(alt) if alt else "")
                            w = _get(m, "width") or getattr(m, "width", "")
                            h = _get(m, "height") or getattr(m, "height", "")
                            dimensions.append(f"{w}x{h}" if w and h else "")
                            image_count += 1

                    if image_count == 0:
                        continue

                    # Resolve original author
                    orig_author_id = _get(orig, "author_id") or getattr(orig, "author_id", "")
                    orig_author    = user_lookup.get(orig_author_id)
                    orig_username  = _get(orig_author, "username") if orig_author else ""

                    # Original tweet text and metrics
                    orig_text = unicodedata.normalize(
                        "NFKC", str(_get(orig, "text") or getattr(orig, "text", ""))
                    ).replace("\n", " ")

                    metrics = _get(orig, "public_metrics") or getattr(orig, "public_metrics", {}) or {}

                    # note_tweet (3-way guard)
                    n_tweet = _get(orig, "note_tweet") or getattr(orig, "note_tweet", None)
                    if isinstance(n_tweet, dict):
                        note_text = n_tweet.get("text", "")
                    elif hasattr(n_tweet, "text"):
                        note_text = n_tweet.text
                    else:
                        note_text = ""

                    ctx = _get(orig, "context_annotations") or getattr(orig, "context_annotations", None)
                    ctx_json = json.dumps(ctx, default=str) if ctx else ""

                    writer.writerow([
                        global_count,
                        username,
                        tweet.id,
                        tweet.created_at,
                        _get(orig, "id") or getattr(orig, "id", ""),
                        orig_username,
                        orig_author_id,
                        orig_text,
                        metrics.get("like_count", 0) if type(metrics) is dict else getattr(metrics, "like_count", 0),
                        metrics.get("retweet_count", 0) if type(metrics) is dict else getattr(metrics, "retweet_count", 0),
                        metrics.get("reply_count", 0) if type(metrics) is dict else getattr(metrics, "reply_count", 0),
                        metrics.get("quote_count", 0) if type(metrics) is dict else getattr(metrics, "quote_count", 0),
                        metrics.get("bookmark_count", 0) if type(metrics) is dict else getattr(metrics, "bookmark_count", 0),
                        metrics.get("impression_count", 0) if type(metrics) is dict else getattr(metrics, "impression_count", 0),
                        ";".join(media_urls),
                        ";".join(media_types),
                        image_count,
                        _get(orig, "lang") or getattr(orig, "lang", ""),
                        _get(orig, "possibly_sensitive") or getattr(orig, "possibly_sensitive", ""),
                        ";".join(alt_texts),
                        ";".join(dimensions),
                        note_text,
                        ctx_json,
                    ])
                    global_count         += 1
                    account_saved        += 1
                    total_retweets_saved += 1

            if tweets.meta and "next_token" in tweets.meta:
                pagination_token = tweets.meta["next_token"]
                time.sleep(PAGE_SLEEP)
            else:
                break

        print(f"  Done @{username}: {account_saved} retweets with images ({account_reads} billed reads)")
        time.sleep(ACCOUNT_SLEEP)

# ─── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Output : {OUTPUT_FILE}")
print(f"Rows   : {total_retweets_saved}")
print(f"Reads  : {total_reads_used:,}")
print_budget_summary(tracker)
