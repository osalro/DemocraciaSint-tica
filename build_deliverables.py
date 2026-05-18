import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Regex to pull (url, label) pairs back out of the `=HYPERLINK("url","label")&...`
# Sheets formula stored in `enlaces_imagenes`, so the markdown report can
# render the same set of image links as inline `[label](url)` links.
HYPERLINK_RE = re.compile(r'HYPERLINK\("([^"]+)","([^"]+)"\)')

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REVIEW_DIR = PROJECT_ROOT / "data" / "hive_and_human_review"
RAW_TWEETS_DIR = PROJECT_ROOT / "data" / "raw_tweets"
RETWEETS_DIR = PROJECT_ROOT / "data" / "retweets"
MASTER_DIR = PROJECT_ROOT / "src" / "data"
OUT_DIR = PROJECT_ROOT / "data" / "deliverables"

IMAGE_MASTER_CSV = REVIEW_DIR / "image_master_scores.csv"
TWEET_MASTER_CSV = MASTER_DIR / "master_tweet_level.csv"
FLAGGED_INCLUSIVE_CSV = REVIEW_DIR / "flagged_si_dudoso_valoralto.csv"
FOLLOWERS_SOURCE = PROJECT_ROOT / "data" / "extraction_state" / "tweet_estimates.json"
EXPECTED_ACCOUNTS = 13

D1_PATH = OUT_DIR / "d1_aide_raw_scores.csv"
D2A_PATH = OUT_DIR / "d2a_flagged_originals.csv"
D2B_PATH = OUT_DIR / "d2b_flagged_retweets.csv"
D2_LEGACY_PATH = OUT_DIR / "d2_flagged_analysis.csv"  # superseded by d2a + d2b
D3A_PATH = OUT_DIR / "d3_tweets_tweet_level.csv"
D3B_PATH = OUT_DIR / "d3_tweets_image_level.csv"
D3C_PATH = OUT_DIR / "d3_retweets_tweet_level.csv"
D3D_PATH = OUT_DIR / "d3_retweets_image_level.csv"
D4A_PATH = OUT_DIR / "d4a_per_user_metric_medians_originals.csv"
D4B_PATH = OUT_DIR / "d4b_per_user_metric_medians_retweets.csv"
D4_REPORT_PATH = OUT_DIR / "d4_per_user_metric_medians.md"
D5_PATH = OUT_DIR / "d5_pooled_ia_vs_no_ia.csv"
D6A_PATH = OUT_DIR / "d6a_l_summary_overall.csv"
D6B_PATH = OUT_DIR / "d6b_l_summary_per_account.csv"
D7_PATH = OUT_DIR / "d7_aide_confusion_matrices.csv"
D8A_PATH = OUT_DIR / "d8a_disclosure_rate.csv"
D8B_PATH = OUT_DIR / "d8b_b4_mentions.csv"
RESULTADOS_HUB_PATH = OUT_DIR / "resultados_hub.md"

# Read-only sources for D7/D8: the side-projects are the canonical
# producers; the D-numbered deliverables are portable copies so chapter
# Resultados can cite them alongside the rest of the d*_*.csv family.
AIDE_CONFUSION_SOURCE = (
    PROJECT_ROOT / "_side_projects" / "aide_vs_hive" / "results"
    / "confusion_matrices.csv"
)
DISCLOSURE_RATE_SOURCE = (
    PROJECT_ROOT / "_side_projects" / "spanish_ai_disclosure_scan" / "results"
    / "disclosure_rate.csv"
)
DISCLOSURE_MATCHES_SOURCE = (
    PROJECT_ROOT / "_side_projects" / "spanish_ai_disclosure_scan" / "results"
    / "matches_all.csv"
)

# Canonical account-type lists for D6b `account_type` column.
CANDIDATES = {
    "laurapresi2026", "RoblesBarrantes", "ClaudiaDobles", "jchidalgo",
    "JoseAguilarBerr", "FabriAlvarado7", "elifeinzaig",
}
PARTIES = {
    "FrenteAmplio", "accionciudadana", "pusc_cr", "Avanza_cr",
    "nuevarepublica7", "liberalcr",
}

DECISIONS_CSV = REVIEW_DIR / "image_master_decisions.csv"
# Oscar's source verdict CSV — the original review spreadsheet exported.
# `image_master_decisions.csv` is a derivative of this file (verified
# 737/737 rows and identical value counts on `oscar_personal`); we read
# verdicts from the source for traceability and pull only `is_retweet`
# from the master because it's a structural flag, not a verdict.
OSCAR_VERDICTS_CSV = REVIEW_DIR / "Detección Hive y Manual - Detección Oscar.csv"

LEVEL_LABELS = {
    "L1": "L1 Sí/Estricto",
    "L2": "L2 Dudoso",
    "L3": "L3 Valor Alto",
}

STRICT_PERSONAL = {"sí", "si", "dudoso"}
INCLUSIVE_HIVE = {"sí", "si", "dudoso", "valor alto"}

ARRAY_DELIM = ";"

# X API public_metrics field names. Reused everywhere engagement is exposed.
ENGAGEMENT_COLS = [
    "like_count", "retweet_count", "reply_count",
    "quote_count", "bookmark_count", "impression_count",
]

# Strip the project-internal `orig_` prefix from retweet columns so they
# match the canonical X API field names. Identity columns (`original_*`)
# keep their prefix because they disambiguate source vs retweet within the
# same row. `retweeting_candidate` is a project-derived field (not an X API
# field) and is exposed in Spanish to match the human-review label
# convention used elsewhere in the deliverables.
RETWEETS_RENAME = {
    "retweeting_candidate": "candidato_retuiteador",
    "orig_like_count": "like_count",
    "orig_retweet_count": "retweet_count",
    "orig_reply_count": "reply_count",
    "orig_quote_count": "quote_count",
    "orig_bookmark_count": "bookmark_count",
    "orig_impression_count": "impression_count",
    "orig_lang": "lang",
    "orig_possibly_sensitive": "possibly_sensitive",
    "orig_note_tweet": "note_tweet",
    "orig_context_annotations": "context_annotations",
}


def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _load_followers() -> dict[str, int]:
    """Load follower counts keyed by username from the latest snapshot.

    Source: `data/extraction_state/tweet_estimates.json`, field `accounts[].followers`.
    Used by the D4 builder to compute engagement-per-follower variants
    (Oscar WhatsApp 5/6; spec in `algorithms_reference.md` §A.3b).
    """
    with FOLLOWERS_SOURCE.open(encoding="utf-8") as f:
        data = json.load(f)
    out = {row["username"]: int(row["followers"]) for row in data["accounts"]}
    assert len(out) == EXPECTED_ACCOUNTS, (
        f"expected {EXPECTED_ACCOUNTS} accounts in followers snapshot, got {len(out)}"
    )
    return out


def build_d1(image_master: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "username", "image_id", "url_imagen",
        "score_aide_GenImage", "score_aide_ProGAN", "score_aide_SD14",
    ]
    out = image_master[cols].copy()
    out = out.rename(columns={"url_imagen": "media_url"})
    return out


def _flagged_image_master(image_master: pd.DataFrame) -> pd.DataFrame:
    """Apply the inclusive D2 flag rule to the image master and return the
    flagged subset. Selection: Oscar personal ∈ {Sí, Dudoso} OR Oscar Hive
    ∈ {Sí, Dudoso, Valor alto}, normalized via lower+strip."""
    mask = (
        norm(image_master["oscar_personal"]).isin(STRICT_PERSONAL)
        | norm(image_master["oscar_hive"]).isin(INCLUSIVE_HIVE)
    )
    return image_master.loc[mask].copy()


def build_d2a_originals(image_master: pd.DataFrame, tweet_master: pd.DataFrame) -> pd.DataFrame:
    """D2a: flagged originals only. Engagement from master_tweet_level
    (joined on tweet_id derived from image_id), which carries the tweet's
    own metrics for non-retweet rows."""
    df = _flagged_image_master(image_master)
    df = df[df["is_retweet"] == False].copy()
    df["tweet_id"] = df["image_id"].astype(str).str.rsplit("_", n=1).str[0]

    tweet_eng = tweet_master[["tweet_id", "created_at", "text", "note_tweet"] + ENGAGEMENT_COLS].copy()
    tweet_eng["tweet_id"] = tweet_eng["tweet_id"].astype(str)
    df = df.merge(tweet_eng, on="tweet_id", how="left")

    df = df.rename(columns={
        "url_imagen": "media_url",
        "oscar_personal": "humano_personal",
        "oscar_hive": "humano_hive",
    })

    out_cols = [
        "username", "image_id", "media_url", "tweet_id", "created_at",
        "text", "note_tweet",
        "humano_personal", "humano_hive",
        "oscar_comentarios", "paulo_comentarios",
        *ENGAGEMENT_COLS,
    ]
    return df[out_cols].copy()


def build_d2b_retweets(image_master: pd.DataFrame, d3d: pd.DataFrame) -> pd.DataFrame:
    """D2b: flagged retweets only.

    The retweet image_id in `image_master_decisions.csv` is keyed on the
    *retweeter's* tweet id (`{retweeter_tweet_id}_{slot}`), while D3d uses
    `{original_tweet_id}_{slot}`. The two image_id values do NOT match,
    so the join key is `media_url` (verified 66/66 in
    `.tooling/memory_bank/active_context.md`). Engagement metrics come
    from D3d's engagement columns — by design those carry the source
    tweet's metrics (the retweet row itself carries zeros via the API).

    Output exposes both the retweeter (`candidato_retuiteador`) and the
    original poster (`original_author_username`) so the table answers both
    "what was AI?" and "who retweeted it from whom?"."""
    df = _flagged_image_master(image_master)
    df = df[df["is_retweet"] == True].copy()

    rt_cols = [
        "media_url",
        "candidato_retuiteador", "retweet_id", "retweet_created_at",
        "original_tweet_id", "original_author_username",
        "original_text", "note_tweet",
        *ENGAGEMENT_COLS,
    ]
    rt = d3d[rt_cols].drop_duplicates(subset="media_url", keep="first")

    df = df.rename(columns={"url_imagen": "media_url"})
    df = df.merge(rt, on="media_url", how="left")

    df = df.rename(columns={
        "username": "retweeter_username_master",
        "image_id": "retweet_image_id",
        "oscar_personal": "humano_personal",
        "oscar_hive": "humano_hive",
    })

    # Sanity: the retweeter handle from the image master should equal the
    # `candidato_retuiteador` from D3d (both name the project account that
    # made the retweet).
    if df["retweeter_username_master"].notna().any():
        bad = df[df["retweeter_username_master"] != df["candidato_retuiteador"]]
        assert bad.empty, (
            f"retweeter mismatch between image_master.username and "
            f"D3d.candidato_retuiteador on {len(bad)} rows:\n{bad[['retweeter_username_master','candidato_retuiteador','media_url']]}"
        )

    out_cols = [
        "candidato_retuiteador", "retweet_id", "retweet_created_at",
        "retweet_image_id", "media_url",
        "original_author_username", "original_tweet_id",
        "humano_personal", "humano_hive",
        "oscar_comentarios", "paulo_comentarios",
        "original_text", "note_tweet",
        *ENGAGEMENT_COLS,
    ]
    return df[out_cols].copy()


def concat_dir(directory: Path) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(directory.glob("*.csv"))]
    return pd.concat(parts, ignore_index=True)


def explode_media(
    df: pd.DataFrame,
    *,
    tweet_id_col: str,
    drop_extra: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Explode media_urls / media_type / image_alt_texts / image_dimensions
    in lockstep on `;`. Emits one row per image, replacing those four
    columns with per-image singulars: `media_url`, `media_type`,
    `image_alt_text`, `image_dimension`, plus `image_id`. Tweets with
    image_count == 0 are dropped.
    """
    df = df.copy()

    has_media = df["image_count"].fillna(0).astype(int) > 0
    df = df.loc[has_media].copy()

    def split_col(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.split(ARRAY_DELIM)

    df["_urls"] = split_col(df["media_urls"])
    df["_types"] = split_col(df["media_type"])
    df["_alts"] = split_col(df["image_alt_texts"])
    df["_dims"] = split_col(df["image_dimensions"])

    rows = []
    for _, row in df.iterrows():
        urls = row["_urls"]
        types = row["_types"]
        alts = row["_alts"]
        dims = row["_dims"]
        n = len(urls)
        for i in range(n):
            new = row.drop(labels=[
                "media_urls", "media_type", "image_alt_texts",
                "image_dimensions", "_urls", "_types", "_alts", "_dims",
                *drop_extra,
            ]).to_dict()
            new["media_url"] = urls[i] if i < len(urls) else ""
            new["media_type"] = types[i] if i < len(types) else ""
            new["image_alt_text"] = alts[i] if i < len(alts) else ""
            new["image_dimension"] = dims[i] if i < len(dims) else ""
            new["image_id"] = f"{row[tweet_id_col]}_{i + 1}"
            rows.append(new)

    return pd.DataFrame(rows)


def build_d3a(raw_tweets_dir: Path) -> pd.DataFrame:
    # Raw tweets CSVs already use the canonical X API column names — no
    # rename is needed on the way out.
    df = concat_dir(raw_tweets_dir)
    df["is_retweet"] = df["text"].astype(str).str.startswith("RT ")
    return df


def build_d3b(d3a: pd.DataFrame) -> pd.DataFrame:
    exploded = explode_media(d3a, tweet_id_col="tweet_id")
    front = ["count", "username", "tweet_id", "image_id", "media_url",
             "media_type", "image_alt_text", "image_dimension", "is_retweet"]
    rest = [c for c in exploded.columns if c not in front]
    return exploded[front + rest]


def build_d3c(retweets_dir: Path) -> pd.DataFrame:
    df = concat_dir(retweets_dir)
    return df.rename(columns=RETWEETS_RENAME)


def build_d3d(d3c: pd.DataFrame) -> pd.DataFrame:
    exploded = explode_media(d3c, tweet_id_col="original_tweet_id")
    front = ["count", "candidato_retuiteador", "retweet_id",
             "original_tweet_id", "image_id", "media_url",
             "media_type", "image_alt_text", "image_dimension"]
    rest = [c for c in exploded.columns if c not in front]
    return exploded[front + rest]


def _add_image_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per-image L1/L2/L3 boolean flags from oscar_personal + oscar_hive."""
    op = norm(df["oscar_personal"])
    oh = norm(df["oscar_hive"])
    si = {"sí", "si"}
    df = df.copy()
    df["L1"] = op.isin(si) | oh.isin(si)
    df["L2"] = df["L1"] | op.eq("dudoso") | oh.eq("dudoso")
    df["L3"] = df["L2"] | oh.eq("valor alto")
    return df


def _load_oscar_decisions() -> pd.DataFrame:
    """Read Oscar's source verdict CSV and rename to the workspace's
    canonical column names. Pulls `is_retweet` from
    `image_master_decisions.csv` since the Oscar source doesn't carry it
    (it's a structural flag derived from where the image lives in the
    extraction, not a verdict).

    Returns one row per image with columns:
        image_id, username, url_imagen, oscar_personal, oscar_hive,
        hive_pct, oscar_comentarios, is_retweet
    """
    oscar = pd.read_csv(OSCAR_VERDICTS_CSV)
    oscar = oscar.rename(columns={
        "Personalmente considera que es IA?": "oscar_personal",
        "Addon de Hive detecta que es IA?": "oscar_hive",
        "Porcentaje de Detección de Hive": "hive_pct",
        "Comentarios": "oscar_comentarios",
    })
    decisions = pd.read_csv(DECISIONS_CSV, usecols=["image_id", "is_retweet"])
    return oscar.merge(decisions, on="image_id", how="left", validate="one_to_one")


def _hyperlink_cell(items: list[tuple[str, str]]) -> str:
    """Build a Google Sheets `=HYPERLINK(url, label)&...` string for a
    list of (url, label) pairs. Empty list → empty string. Single item →
    just `=HYPERLINK("url","label")`. Multiple → joined by `&" | "&` so
    Sheets renders each as a clickable label separated by " | ".

    The leading `=` makes Sheets parse the cell as a formula on import."""
    if not items:
        return ""
    parts = [f'HYPERLINK("{url}","{label}")' for url, label in items]
    if len(parts) == 1:
        return "=" + parts[0]
    return "=" + '&" | "&'.join(parts)


def _build_d4_wide(
    tweets: pd.DataFrame,
    tweet_flags: pd.DataFrame,
    flagged_images: pd.DataFrame,
    *,
    user_col: str,
    key_col: str,
    followers: dict[str, int],
) -> pd.DataFrame:
    """One row per (usuario × level), with all six metrics as columns.

    `tweets` is one row per tweet with `user_col`, `key_col`, and every
    column in `ENGAGEMENT_COLS`.

    `tweet_flags` is one row per `key_col` with bool `L1`/`L2`/`L3`
    columns. Tweets in `tweets` but not in `tweet_flags` are treated as
    not flagged at any level.

    `flagged_images` is one row per *image* with `key_col`, `image_id`,
    `url_imagen`, and bool `L1`/`L2`/`L3` — used to assemble the
    `enlaces_imagenes` hyperlink cell per (usuario, level).

    Baseline (`sin_ia`, `mediana_sin_ia_*`) is constant per user: tweets
    with no image flagged at L3 (the most inclusive level).

    `proporcion_de_exito_<metric>` = `mediana_ia_<metric>` − `mediana_sin_ia_<metric>`.
    Positive ⇒ flagged (con IA) outperforms organic on that metric.
    """
    df = tweets.merge(tweet_flags, on=key_col, how="left")
    for lvl in ("L1", "L2", "L3"):
        df[lvl] = df[lvl].fillna(False).astype(bool)

    # Pre-index flagged images by (user, key) → list of (url, label) for
    # quick lookup when assembling the enlaces_imagenes cell.
    img_lookup: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for _, r in flagged_images.iterrows():
        img_lookup.setdefault(
            (r[user_col], r[key_col]),
            [],
        ).append((r["url_imagen"], r["image_id"]))

    rows = []
    for user, g in df.groupby(user_col, sort=True):
        organic_mask = ~g["L3"]
        n_org = int(organic_mask.sum())
        organic_medians = {
            m: float(g.loc[organic_mask, m].median()) if n_org else float("nan")
            for m in ENGAGEMENT_COLS
        }
        # Per-follower normalizer (Oscar WhatsApp 5/6; spec in
        # algorithms_reference.md §A.3b). Constant per user; falls back
        # to NaN if the user is missing from the followers snapshot or
        # F == 0.
        F = followers.get(user)
        has_F = bool(F) and F > 0
        for lvl_key, lvl_label in LEVEL_LABELS.items():
            flag_mask = g[lvl_key]
            n_flag = int(flag_mask.sum())
            row = {
                "usuario": user,
                "level": lvl_label,
                "con_ia": n_flag,
                "sin_ia": n_org,
                "porcentaje_ia": round((n_flag / n_org) * 100, 1) if n_org else float("nan"),
            }
            for m in ENGAGEMENT_COLS:
                med_ia = float(g.loc[flag_mask, m].median()) if n_flag else float("nan")
                med_sin = organic_medians[m]
                row[f"mediana_ia_{m}"] = round(med_ia, 1) if n_flag else float("nan")
                row[f"mediana_sin_ia_{m}"] = round(med_sin, 1) if n_org else float("nan")
                row[f"proporcion_de_exito_{m}"] = (
                    round(med_ia - med_sin, 1) if (n_flag and n_org) else float("nan")
                )
                # Per-follower variants. The "rate first, then median"
                # form (rather than median(metric) / F) matches the
                # documented spec and would generalize to a pooled-across-
                # users computation; for a single user with constant F the
                # two forms are mathematically equivalent.
                if has_F:
                    med_ia_ps = float((g.loc[flag_mask, m] / F).median()) if n_flag else float("nan")
                    med_sin_ps = float((g.loc[organic_mask, m] / F).median()) if n_org else float("nan")
                    row[f"mediana_ia_{m}_por_seguidor"] = round(med_ia_ps, 6) if n_flag else float("nan")
                    row[f"mediana_sin_ia_{m}_por_seguidor"] = round(med_sin_ps, 6) if n_org else float("nan")
                    row[f"proporcion_de_exito_{m}_por_seguidor"] = (
                        round(med_ia_ps - med_sin_ps, 6) if (n_flag and n_org) else float("nan")
                    )
                else:
                    row[f"mediana_ia_{m}_por_seguidor"] = float("nan")
                    row[f"mediana_sin_ia_{m}_por_seguidor"] = float("nan")
                    row[f"proporcion_de_exito_{m}_por_seguidor"] = float("nan")
            # Collect images flagged at THIS level for this user.
            flagged_keys = g.loc[flag_mask, key_col].tolist()
            level_images: list[tuple[str, str]] = []
            for k in flagged_keys:
                for url, label in img_lookup.get((user, k), []):
                    # An image contributes to this level only if its own
                    # per-image flag at this level is set; the tweet is
                    # flagged via .any() across its images.
                    pass  # filtering done below
            # More precise: filter flagged_images to this user at this level
            sub = flagged_images[
                (flagged_images[user_col] == user) & flagged_images[lvl_key]
            ]
            level_images = list(zip(sub["url_imagen"].tolist(), sub["image_id"].tolist()))
            row["enlaces_imagenes"] = _hyperlink_cell(level_images)
            rows.append(row)
    return pd.DataFrame(rows)


def build_d4_originals(
    oscar_decisions: pd.DataFrame, d3a: pd.DataFrame, followers: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D4a: per-user × level wide-format median comparison on originals.

    Returns `(wide_df, flagged_images_df)`. `flagged_images_df` is the
    per-image flag table used to build the gallery in the markdown report
    (one row per flagged image with `usuario, image_id, url_imagen, L1,
    L2, L3`). It is *not* written to CSV — only the wide df is.
    """
    dec = oscar_decisions[oscar_decisions["is_retweet"] == False].copy()
    dec = _add_image_flags(dec)
    dec["tweet_id"] = dec["image_id"].astype(str).str.rsplit("_", n=1).str[0]
    tweet_flags = dec.groupby("tweet_id")[["L1", "L2", "L3"]].any().reset_index()

    tweets = d3a[d3a["is_retweet"] == False].copy()
    tweets["tweet_id"] = tweets["tweet_id"].astype(str)

    flagged_images = dec[
        ["username", "tweet_id", "image_id", "url_imagen", "L1", "L2", "L3"]
    ].copy().rename(columns={"username": "usuario"})

    wide = _build_d4_wide(
        tweets, tweet_flags,
        dec[["username", "tweet_id", "image_id", "url_imagen", "L1", "L2", "L3"]],
        user_col="username", key_col="tweet_id",
        followers=followers,
    )
    return wide, flagged_images


def build_d4_retweets(
    oscar_decisions: pd.DataFrame, d3c: pd.DataFrame, followers: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D4b: per-user × level wide-format median comparison on retweets.

    The `username` field in the Oscar source CSV for retweet rows holds
    the *retweeting candidate* (i.e., the project account that retweeted
    the source image). We carry that through as `candidato_retuiteador`
    so the join with D3c works.

    `followers` is keyed by `candidato_retuiteador` (the retweeter, one
    of the 13 project accounts); we normalize engagement by the
    retweeter's follower count, since that is the audience whose
    timeline the retweet promoted into.

    Returns `(wide_df, flagged_images_df)` analogously to D4a.
    """
    dec = oscar_decisions[oscar_decisions["is_retweet"] == True].copy()
    dec = _add_image_flags(dec)
    dec["retweet_id"] = dec["image_id"].astype(str).str.rsplit("_", n=1).str[0]
    dec = dec.rename(columns={"username": "candidato_retuiteador"})
    tweet_flags = dec.groupby("retweet_id")[["L1", "L2", "L3"]].any().reset_index()

    rt = d3c.copy()
    rt["retweet_id"] = rt["retweet_id"].astype(str)

    flagged_images = dec[
        ["candidato_retuiteador", "retweet_id", "image_id", "url_imagen", "L1", "L2", "L3"]
    ].copy().rename(columns={"candidato_retuiteador": "usuario"})

    wide = _build_d4_wide(
        rt, tweet_flags,
        dec[["candidato_retuiteador", "retweet_id", "image_id", "url_imagen", "L1", "L2", "L3"]],
        user_col="candidato_retuiteador", key_col="retweet_id",
        followers=followers,
    )
    return wide, flagged_images


def build_d5_pooled_comparison(
    oscar_decisions: pd.DataFrame, d3a: pd.DataFrame,
) -> pd.DataFrame:
    """D5: pooled IA vs no-IA median comparison across all 13 accounts.

    For each strictness variant (Pooled-any-level, L1, L2, L3) and each
    of the 6 ENGAGEMENT_COLS metrics, compute the median of the flagged
    group (any image flagged at that level) and the unflagged group,
    plus the delta. Originals only — engagement on retweet rows is
    zeroed by the X API.

    Long-format output: 4 levels × 6 metrics = 24 rows × 7 columns
    (`level`, `metric`, `n_ia`, `n_no_ia`, `mediana_ia`,
    `mediana_no_ia`, `delta`).

    Origin: Oscar WhatsApp 5/12/2026; spec in
    `methodology_section_3/algorithms_reference.md` §A.4.
    """
    dec = oscar_decisions[oscar_decisions["is_retweet"] == False].copy()
    dec = _add_image_flags(dec)
    dec["tweet_id"] = dec["image_id"].astype(str).str.rsplit("_", n=1).str[0]
    tweet_flags = dec.groupby("tweet_id")[["L1", "L2", "L3"]].any().reset_index()

    tweets = d3a[d3a["is_retweet"] == False].copy()
    tweets["tweet_id"] = tweets["tweet_id"].astype(str)
    tweets = tweets.merge(tweet_flags, on="tweet_id", how="left")
    for lvl in ("L1", "L2", "L3"):
        tweets[lvl] = tweets[lvl].fillna(False).astype(bool)
    tweets["L_any"] = tweets["L1"] | tweets["L2"] | tweets["L3"]

    rows = []
    for level_label, mask_col in [
        ("Pooled (cualquier L1/L2/L3)", "L_any"),
        ("L1 Sí/Estricto", "L1"),
        ("L2 Dudoso", "L2"),
        ("L3 Valor Alto", "L3"),
    ]:
        ia_mask = tweets[mask_col]
        n_ia = int(ia_mask.sum())
        n_no_ia = int((~ia_mask).sum())
        for m in ENGAGEMENT_COLS:
            med_ia = float(tweets.loc[ia_mask, m].median()) if n_ia else float("nan")
            med_no = float(tweets.loc[~ia_mask, m].median()) if n_no_ia else float("nan")
            rows.append({
                "level": level_label,
                "metric": m,
                "n_ia": n_ia,
                "n_no_ia": n_no_ia,
                "mediana_ia": round(med_ia, 1) if n_ia else float("nan"),
                "mediana_no_ia": round(med_no, 1) if n_no_ia else float("nan"),
                "delta": round(med_ia - med_no, 1) if (n_ia and n_no_ia) else float("nan"),
            })
    return pd.DataFrame(rows)


def build_d6a_l_summary_overall(oscar_decisions: pd.DataFrame) -> pd.DataFrame:
    """D6a: image-level L1/L2/L3 summary across the whole corpus.

    Three rows: `originales`, `retuits`, `total`. Levels are cumulative
    (L3 ⊇ L2 ⊇ L1) so `n_L1 ≤ n_L2 ≤ n_L3` by construction. `n_sin_L`
    is the complement of `n_L3` within the population. Source:
    `_load_oscar_decisions()` (the same verdict source used by D4/D5).

    Justifies the chapter Resultados sub-block at `draft.txt L127`
    (*"Tabla de resumen de resultados de IA en las imágenes (en nivel
    general, cuántas imágenes fueron L1, L2, L3 vs no Ls)"*).
    """
    df = _add_image_flags(oscar_decisions)

    rows = []
    populations = [
        ("originales", df[df["is_retweet"] == False]),
        ("retuits", df[df["is_retweet"] == True]),
        ("total", df),
    ]
    for pop_name, sub in populations:
        n_total = len(sub)
        n_L1 = int(sub["L1"].sum())
        n_L2 = int(sub["L2"].sum())
        n_L3 = int(sub["L3"].sum())
        n_sin = n_total - n_L3
        pct_L3 = round((n_L3 / n_total) * 100, 1) if n_total else 0.0
        rows.append({
            "population": pop_name,
            "n_total": n_total,
            "n_L1": n_L1,
            "n_L2": n_L2,
            "n_L3": n_L3,
            "n_sin_L": n_sin,
            "pct_L3": pct_L3,
        })
    return pd.DataFrame(rows)


def build_d6b_l_summary_per_account(oscar_decisions: pd.DataFrame) -> pd.DataFrame:
    """D6b: per-account L1/L2/L3 summary across all images of the account.

    One row per `username` that contributed at least one image to the
    corpus (originals + retweets together). Levels cumulative
    (L3 ⊇ L2 ⊇ L1); `n_sin_L = n_total − n_L3`. `account_type`
    derived from the canonical CANDIDATES/PARTIES sets.

    Accounts that contributed zero images (e.g. `JoseAguilarBerr` —
    139 posts, 0 with images) are omitted from the table.

    Justifies the chapter Resultados sub-block at `draft.txt L127`
    (*"Tabla de resultados con las cuentas analizadas y su cantidad
    de L1, L2, L3 vs sin Ls"*).
    """
    df = _add_image_flags(oscar_decisions)

    rows = []
    for user, g in df.groupby("username", sort=True):
        n_total = len(g)
        if n_total == 0:
            continue
        n_L1 = int(g["L1"].sum())
        n_L2 = int(g["L2"].sum())
        n_L3 = int(g["L3"].sum())
        n_sin = n_total - n_L3
        pct_L3 = round((n_L3 / n_total) * 100, 1)
        if user in CANDIDATES:
            atype = "candidato"
        elif user in PARTIES:
            atype = "partido"
        else:
            atype = "desconocido"
        rows.append({
            "username": user,
            "account_type": atype,
            "n_total": n_total,
            "n_L1": n_L1,
            "n_L2": n_L2,
            "n_L3": n_L3,
            "n_sin_L": n_sin,
            "pct_L3": pct_L3,
        })
    return pd.DataFrame(rows)


def copy_d7_aide_confusion() -> pd.DataFrame:
    """D7: AIDE-vs-Hive confusion matrices.

    Portable copy of `_side_projects/aide_vs_hive/results/confusion_matrices.csv`.
    The side-project is the canonical producer (3 GT levels × 3 AIDE
    checkpoints = 9 rows with `tp/fp/fn/tn/precision/recall/accuracy`);
    D7 exposes the same table inside `data/deliverables/` so the chapter
    Resultados can cite it alongside the rest of the D-numbered family.

    Justifies the chapter Resultados sub-block at `draft.txt L125`
    (*"Tabla de confusiones para desechar los AIDE y dejar solo HIVE"*).
    Qualitative interpretation stays at
    `_side_projects/aide_vs_hive/reports/aide_vs_hive_interpretation.md`.
    """
    if not AIDE_CONFUSION_SOURCE.exists():
        raise FileNotFoundError(
            f"AIDE confusion source not found: {AIDE_CONFUSION_SOURCE}. "
            "Rerun the aide_vs_hive side-project to regenerate."
        )
    return pd.read_csv(AIDE_CONFUSION_SOURCE)


def copy_d8a_disclosure_rate() -> pd.DataFrame:
    """D8a: Spanish AI-disclosure rate table ("tabla de ceros").

    Portable copy of `_side_projects/spanish_ai_disclosure_scan/results/disclosure_rate.csv`.
    24 rows = 6 populations (`originals_tweets`, `retweets_tweets`,
    `all_tweets`, `L3_originals`, `L3_retweets`, `L3_all`) × 4
    bucket-cumulative groups (`B1_only`, `B1_B2`, `B1_B2_B3`,
    `any_bucket`) with Wilson 95 % CI per row.

    Name acuñado por Oscar (transcripción L719: *"'Resultados sobre
    especificación de IA,' que es cero — la tabla de ceros"*). Justifies
    the chapter Resultados sub-block at `draft.txt L133`.
    """
    if not DISCLOSURE_RATE_SOURCE.exists():
        raise FileNotFoundError(
            f"Disclosure rate source not found: {DISCLOSURE_RATE_SOURCE}. "
            "Rerun the spanish_ai_disclosure_scan side-project to regenerate."
        )
    return pd.read_csv(DISCLOSURE_RATE_SOURCE)


def build_d8b_b4_mentions() -> pd.DataFrame:
    """D8b: tweets that mention IA generically without disclosing it
    ("tabla de B4s").

    Filters `_side_projects/spanish_ai_disclosure_scan/results/matches_all.csv`
    to `bucket == "B4_generic"` and dedupes by `primary_id` so each
    tweet appears once. Channel priority for dedup:
    `text > note_tweet > image_alt` (keep the most prominent match).

    Name acuñado por Oscar (transcripción L723: *"'tabla de tweets
    que solamente se referían a IA,' que sería la de B4"*). These
    tweets MENTION IA generically but do NOT disclose image generation;
    the interpretive verdict (policy positioning vs accusation vs
    promo) is reported in chapter Resultados, not here.
    """
    if not DISCLOSURE_MATCHES_SOURCE.exists():
        raise FileNotFoundError(
            f"Disclosure matches source not found: {DISCLOSURE_MATCHES_SOURCE}. "
            "Rerun the spanish_ai_disclosure_scan side-project to regenerate."
        )
    matches = pd.read_csv(DISCLOSURE_MATCHES_SOURCE)
    b4 = matches[matches["bucket"] == "B4_generic"].copy()
    channel_priority = {"text": 0, "note_tweet": 1, "image_alt": 2}
    b4["_priority"] = b4["channel"].map(channel_priority).fillna(3)
    b4 = b4.sort_values(["primary_id", "_priority"], kind="stable")
    b4 = b4.drop_duplicates(subset="primary_id", keep="first")
    b4 = b4.drop(columns=["_priority"])
    return b4.reset_index(drop=True)


def _hyperlink_formula_to_md(formula: str) -> str:
    """Convert a `=HYPERLINK("url","label")&" | "&HYPERLINK(...)` Sheets
    formula into a markdown-cell-friendly list of `[label](url)` links.

    Image_ids contain underscores, which markdown would otherwise
    interpret as italic — wrap labels in backticks. Separate links with
    middle-dot (`·`) since markdown table cells can't contain raw `|`.
    """
    if not formula:
        return ""
    pairs = HYPERLINK_RE.findall(formula)
    return " · ".join(f"[`{label}`]({url})" for url, label in pairs)


def _format_d4_summary_table_unified(df: pd.DataFrame, metric: str) -> str:
    """Render a per-metric summary as a single unified table.

    One row per usuario at **L3 only** (L3 ⊇ L2 ⊇ L1, so L3 covers every
    marked tweet for that user). Columns use the chapter's own vocabulary
    from draft.txt L87 / L91, rescaled for readability:

        cálculo de éxito          = mediana_ia − mediana_sin_ia
        eficiencia de interacción = (cálculo de éxito / followers) × 1000

    The × 1000 rescaling makes the coefficient readable (e.g., +9.54 likes
    de ventaja por cada 1000 seguidores) instead of the bare ratio +0.009542.
    It's the standard "engagement por 1000 seguidores" unit in social media
    analytics; the comparison-across-cuentas property is unchanged
    (draft.txt L89: *"nuestro coeficiente normalizador"*).
    """
    sub = df[(df["level"] == "L3 Valor Alto") & (df["con_ia"] > 0)].copy()
    sub = sub.sort_values("usuario", kind="stable")

    if sub.empty:
        return "_Ninguna cuenta tiene tuits marcados a L3 para esta población._"

    def fmt_num(x: float) -> str:
        if pd.isna(x):
            return "—"
        return f"{x:.1f}"

    def fmt_eficiencia(x: float) -> str:
        """Coefficient × 1000, signed, 2 decimals. Same magnitude reads
        as 'interacciones de ventaja por cada 1000 seguidores'."""
        if pd.isna(x):
            return "—"
        return f"{x * 1000:+.2f}"

    lines = [
        "| usuario | con_ia | sin_ia | mediana_ia | mediana_sin_ia | cálculo de éxito | eficiencia de interacción (por 1000 seg.) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['usuario']} "
            f"| {int(r['con_ia'])} "
            f"| {int(r['sin_ia'])} "
            f"| {fmt_num(r[f'mediana_ia_{metric}'])} "
            f"| {fmt_num(r[f'mediana_sin_ia_{metric}'])} "
            f"| {fmt_num(r[f'proporcion_de_exito_{metric}'])} "
            f"| {fmt_eficiencia(r[f'proporcion_de_exito_{metric}_por_seguidor'])} |"
        )
    return "\n".join(lines)


def _format_d4_image_gallery(flagged_images: pd.DataFrame) -> str:
    """Per-population image gallery: each flagged image listed once,
    grouped by usuario, with a tag showing the strictest level it
    qualifies at (`L1` / `L2` / `L3`).

    Levels are cumulative (L3 ⊇ L2 ⊇ L1), so an image's *strictest*
    level is the most informative tag — an image at L1 inherits L2 and
    L3, but an image only at L3 (e.g., `oscar_hive == "valor alto"` with
    `oscar_personal == "no"`) is qualitatively weaker evidence.
    """
    if flagged_images.empty:
        return "_Ninguna imagen marcada en esta población._"

    df = flagged_images.copy()
    # Strictest qualifying level: L1 wins over L2 wins over L3.
    def strict(row):
        if row["L1"]:
            return "L1"
        if row["L2"]:
            return "L2"
        if row["L3"]:
            return "L3"
        return ""
    df["strict_level"] = df.apply(strict, axis=1)
    df = df[df["strict_level"] != ""].copy()
    if df.empty:
        return "_Ninguna imagen marcada en esta población._"

    parts = []
    for usuario, g in df.groupby("usuario", sort=True):
        # Order within user: L1 first (strongest), then L2, then L3.
        order = {"L1": 0, "L2": 1, "L3": 2}
        g = g.sort_values(
            by=["strict_level", "image_id"],
            key=lambda s: s.map(order) if s.name == "strict_level" else s,
            kind="stable",
        )
        counts = g["strict_level"].value_counts().reindex(["L1", "L2", "L3"], fill_value=0)
        header = (f"**{usuario}** — {len(g)} imagen{'es' if len(g) != 1 else ''} "
                  f"(L1: {counts['L1']}, L2: {counts['L2']}, L3: {counts['L3']})")
        parts.append(header)
        for _, r in g.iterrows():
            parts.append(
                f"- `{r['strict_level']}` · [`{r['image_id']}`]({r['url_imagen']})"
            )
        parts.append("")  # blank line after each user
    return "\n".join(parts).rstrip() + "\n"


def _split_by_l3_flag(
    oscar_decisions: pd.DataFrame,
    tweets: pd.DataFrame,
    *,
    user_col: str,
    key_col: str,
    is_retweet_value: bool,
    filter_originals_in_tweets: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a tweet-level frame into (flagged_at_L3, organic) populations.

    `oscar_decisions` is per-image. We roll image-level L1/L2/L3 flags up
    to the tweet by `.any()` on `key_col` (the prefix of `image_id`).
    """
    dec = oscar_decisions[oscar_decisions["is_retweet"] == is_retweet_value].copy()
    dec = _add_image_flags(dec)
    dec[key_col] = dec["image_id"].astype(str).str.rsplit("_", n=1).str[0]
    tweet_flags = dec.groupby(key_col)[["L1", "L2", "L3"]].any().reset_index()

    t = tweets.copy()
    if filter_originals_in_tweets:
        t = t[t["is_retweet"] == False].copy()
    t[key_col] = t[key_col].astype(str)

    df = t.merge(tweet_flags, on=key_col, how="left")
    for lvl in ("L1", "L2", "L3"):
        df[lvl] = df[lvl].fillna(False).astype(bool)

    return df[df["L3"]].copy(), df[~df["L3"]].copy()


def _strictest_level_str(row: pd.Series) -> str:
    """Strictest qualifying level for a row with bool L1/L2/L3 columns."""
    if row["L1"]:
        return "L1"
    if row["L2"]:
        return "L2"
    if row["L3"]:
        return "L3"
    return ""


def _format_interesting_cases_block(
    flagged_tweets: pd.DataFrame,
    flagged_images: pd.DataFrame,
    organic_tweets: pd.DataFrame,
    *,
    user_col: str,
    key_col: str,
    top_n: int = 10,
) -> str:
    """Top-N flagged tweets ranked by |robust_z(like_count vs user
    organic baseline)|. Robust z = (likes − account_median) /
    (1.4826 × MAD(account organic)). |z| ≥ 2 = notable; ≥ 5 = atypical.
    """
    if flagged_tweets.empty:
        return "_Sin tuits marcados en esta población._"

    def org_stats(s: pd.Series) -> pd.Series:
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        return pd.Series({"org_median": med, "org_mad": mad})

    stats = (
        organic_tweets.groupby(user_col)["like_count"]
        .apply(org_stats).unstack().reset_index()
    )
    f = flagged_tweets.merge(stats, on=user_col, how="left").copy()

    def rz(row):
        if pd.isna(row.get("org_mad")) or row["org_mad"] == 0:
            return float("nan")
        return (row["like_count"] - row["org_median"]) / (1.4826 * row["org_mad"])

    f["robust_z"] = f.apply(rz, axis=1)
    f["abs_z"] = f["robust_z"].abs()
    f["strict_level"] = f.apply(_strictest_level_str, axis=1)

    # Per-tweet image link cell — list every flagged image of that tweet.
    img_per_tweet: dict[tuple[str, str], str] = {}
    for (u, k), g in flagged_images.groupby(["usuario", key_col]):
        img_per_tweet[(u, k)] = " · ".join(
            f"[`{r['image_id']}`]({r['url_imagen']})" for _, r in g.iterrows()
        )

    flagged_user_counts = f.groupby(user_col).size().to_dict()

    top = (
        f.sort_values(["abs_z", "like_count"], ascending=[False, False],
                      na_position="last")
        .head(top_n)
    )

    def fmt_med(x: float) -> str:
        if pd.isna(x):
            return "—"
        if abs(x - round(x)) < 1e-9:
            return f"{int(round(x))}"
        return f"{x:.1f}"

    def case_note(row) -> str:
        notes = []
        z = row["robust_z"]
        if pd.isna(z):
            notes.append("MAD orgánico = 0 (no se puede normalizar)")
        elif abs(z) >= 5:
            notes.append("outlier extremo del baseline")
        elif abs(z) >= 2:
            notes.append("desvío notable")
        if flagged_user_counts.get(row[user_col], 0) == 1:
            notes.append("único marcado de la cuenta")
        if not pd.isna(row["org_median"]) and row["org_median"] <= 1:
            notes.append("baseline ≈ 0; cualquier engagement amplifica el z")
        return "; ".join(notes)

    lines = [
        "| # | usuario | imagen(es) | level | likes | mediana_org | robust_z | nota |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for i, (_, r) in enumerate(top.iterrows(), 1):
        imgs_md = img_per_tweet.get((r[user_col], r[key_col]), "")
        rz_val = "—" if pd.isna(r["robust_z"]) else f"{r['robust_z']:+.2f}"
        lines.append(
            f"| {i} | {r[user_col]} | {imgs_md} | {r['strict_level']} "
            f"| {int(r['like_count'])} | {fmt_med(r['org_median'])} "
            f"| {rz_val} | {case_note(r)} |"
        )
    return "\n".join(lines)


def _format_correlation_matrix(df: pd.DataFrame, n_label: int) -> str:
    """Spearman rank correlation matrix (6 × 6) over `ENGAGEMENT_COLS`.
    Returns markdown — empty/insufficient sample handled gracefully."""
    if len(df) < 3:
        return f"_n = {n_label}, insuficiente para correlación (mínimo 3)._"
    corr = df[ENGAGEMENT_COLS].corr(method="spearman")
    head = "| métrica | " + " | ".join(ENGAGEMENT_COLS) + " |"
    sep = "|---|" + "|".join(["---:"] * len(ENGAGEMENT_COLS)) + "|"
    rows = [head, sep]
    for m in ENGAGEMENT_COLS:
        cells = [f"{corr.loc[m, n]:+.2f}" for n in ENGAGEMENT_COLS]
        rows.append(f"| **{m}** | " + " | ".join(cells) + " |")
    return f"_(n = {n_label})_\n\n" + "\n".join(rows)


def _summarize_corr_diff(
    flag_corr: pd.DataFrame, org_corr: pd.DataFrame, top_n: int = 3,
) -> str:
    """Pick the metric pairs whose Spearman ρ moves the most between
    flagged and organic — surface them as a small narrative bullet list.
    Skips the diagonal and only counts each unordered pair once."""
    rows = []
    cols = ENGAGEMENT_COLS
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            f = flag_corr.loc[a, b]
            o = org_corr.loc[a, b]
            if pd.isna(f) or pd.isna(o):
                continue
            rows.append((a, b, float(f), float(o), float(f - o)))
    if not rows:
        return ""
    rows.sort(key=lambda r: abs(r[4]), reverse=True)
    bullets = []
    for a, b, f, o, d in rows[:top_n]:
        direction = "más alta" if d > 0 else "más baja"
        bullets.append(
            f"- `{a}` ↔ `{b}` — ρ = {f:+.2f} en marcados vs {o:+.2f} en orgánicos "
            f"({direction} en marcados, Δ = {d:+.2f})."
        )
    return "\n".join(bullets)


def build_d4_report(
    d4a: pd.DataFrame, d4b: pd.DataFrame,
    flagged_images_a: pd.DataFrame, flagged_images_b: pd.DataFrame,
    oscar_decisions: pd.DataFrame, d3a: pd.DataFrame, d3c: pd.DataFrame,
) -> str:
    """Markdown report — narrative + 12 per-metric digest tables (2
    populations × 6 metrics). The full wide CSVs (`d4a_*.csv`,
    `d4b_*.csv`) carry every metric per row; the report's per-metric
    tables are just a slice for fast scanning."""
    parts = []
    parts.append("# D4 — Medianas por candidato, nivel y métrica\n")
    parts.append(
        "Comparación, para cada cuenta del proyecto, entre la mediana de cada "
        "métrica de engagement de tuits **marcados** por revisión humana o "
        "Hive y la mediana de los tuits **sin ningún indicio de IA** "
        "(baseline orgánico). Todo a nivel de tuit (un tuit con varias "
        "imágenes cuenta una vez).\n"
    )
    parts.append(
        "**Fuente de los veredictos:** `data/hive_and_human_review/Detección Hive y Manual - Detección Oscar.csv` "
        "(la hoja de revisión exportada). El campo `is_retweet` se toma de "
        "`image_master_decisions.csv` porque es una bandera estructural "
        "(según dónde vive la imagen en la extracción), no un veredicto.\n"
    )
    parts.append(
        "**Nivel reportado en este reporte** — sólo **L3 Valor Alto** (la "
        "definición más inclusiva, `L3 ⊇ L2 ⊇ L1`). Cualquier tuit marcado "
        "en L1 o L2 también está marcado en L3, así que L3 captura el "
        "universo completo de tuits con indicio de IA para cada cuenta. "
        "Las CSV completas (`d4a/d4b_*.csv`) siguen llevando los 3 niveles "
        "para análisis ulteriores, pero las tablas aquí abajo se condensan "
        "a una fila por cuenta para facilitar la lectura.\n\n"
        "- `L1 Sí/Estricto` — al menos una imagen del tuit con `oscar_personal == Sí` o `oscar_hive == Sí`.\n"
        "- `L2 Dudoso` — L1 o al menos una imagen con `oscar_personal == Dudoso` o `oscar_hive == Dudoso`.\n"
        "- `L3 Valor Alto` — L2 o al menos una imagen con `oscar_hive == Valor alto`.\n"
    )
    parts.append(
        "**Baseline orgánico** (`sin_ia`, `mediana_sin_ia`) — tuits del "
        "usuario donde **ninguna** imagen aparece marcada a nivel L3.\n"
    )
    parts.append(
        "**Equaciones del capítulo** (draft.txt L87, L91):\n\n"
        "- **cálculo de éxito** = `mediana_ia − mediana_sin_ia`. "
        "Positivo significa que los tuits con IA superan al baseline orgánico; "
        "negativo significa que el orgánico supera. Redondeado a **1 decimal**.\n"
        "- **eficiencia de interacción** = `(cálculo de éxito / followers) × 1000`. "
        "Lectura: *interacciones de ventaja (o desventaja) por cada 1000 seguidores*. "
        "Es un **coeficiente normalizador** (draft.txt L89: *\"nuestro "
        "coeficiente normalizador\"*) que permite comparar cuentas con "
        "audiencias muy distintas; el factor × 1000 es sólo una rescalada "
        "de lectura para evitar mostrar números como `+0.009542`. Signo + "
        "= IA gana; signo − = orgánico gana. Mostrado con 2 decimales.\n"
    )
    parts.append(
        "**Fuente de los conteos de followers** — `data/extraction_state/tweet_estimates.json`, "
        "snapshot manual del 2026-05-12. Limitación: es un único punto temporal "
        "post-elección. El conteo de followers crece a lo largo de la campaña, "
        "así que usar el snapshot final como divisor **sub-estima** la "
        "Eficiencia de interacción de los tuits tempranos (cuando la audiencia "
        "real era menor). El sesgo es direccional consistente y **no invalida** "
        "la comparación IA vs. orgánico dentro de la misma cuenta porque el "
        "divisor es idéntico para ambos grupos del mismo usuario.\n"
    )
    parts.append(
        "**Mapeo CSV ↔ columnas mostradas** — las CSV `d4a/d4b_*.csv` "
        "conservan el esquema histórico con nombres de implementación: "
        "`proporcion_de_exito_<m>` (= **Cálculo de éxito** para esa métrica) y "
        "`proporcion_de_exito_<m>_por_seguidor` (= **Eficiencia de interacción**). "
        "Las tablas aquí abajo renombran a la vocabulario del capítulo "
        "(draft.txt L87, L91) y se condensan a una fila por usuario (L3). "
        "La columna `enlaces_imagenes` del CSV contiene fórmulas "
        "`=HYPERLINK(...)` para abrir las imágenes en Google Sheets "
        "(*File → Import → Replace data*).\n"
    )
    parts.append(
        "**Filas mostradas en este reporte** — solo cuentas con `con_ia > 0` "
        "a nivel L3. Cuentas sin nada marcado se omiten para que la tabla se "
        "lea de un vistazo; siguen presentes en las CSV completas.\n"
    )
    parts.append(
        "**Enlaces a imágenes** — para evitar repetir los mismos `image_id` en "
        "cada métrica × nivel, los enlaces clicables a las imágenes marcadas "
        "viven en una **galería al final de cada población** (`Imágenes marcadas — "
        "Originales` y `Imágenes marcadas — Retuits`). Cada imagen aparece una sola "
        "vez con un tag indicando el nivel **más estricto** al que clasifica "
        "(`L1` ⊂ `L2` ⊂ `L3`). Las CSV `d4a/d4b_*.csv` siguen llevando la columna "
        "`enlaces_imagenes` con la fórmula `=HYPERLINK(...)` para Google Sheets.\n"
    )
    parts.append(
        "**Poblaciones separadas** — *Originales* (filas en `d3_tweets_tweet_level.csv` con `is_retweet == False`) y *Retuits* (filas en `d3_retweets_tweet_level.csv`) se reportan por separado. Las métricas en la tabla de retuits son las del **tuit fuente** (la API zerea las métricas de la fila del retuit, así que esas no se usan).\n"
    )

    for pop_label, df, imgs in [
        ("Originales", d4a, flagged_images_a),
        ("Retuits", d4b, flagged_images_b),
    ]:
        parts.append(f"\n## {pop_label}\n")
        for metric in ENGAGEMENT_COLS:
            parts.append(f"\n### {metric}\n")
            parts.append(_format_d4_summary_table_unified(df, metric))
            parts.append("")
        parts.append(f"\n## Imágenes marcadas — {pop_label}\n")
        parts.append(_format_d4_image_gallery(imgs))

    # Tweet-level flagged/organic splits used by the cases + correlation
    # sections. Originals filter on `is_retweet == False` inside D3a;
    # retweets in D3c are all retweets by construction.
    flag_orig, org_orig = _split_by_l3_flag(
        oscar_decisions, d3a,
        user_col="username", key_col="tweet_id",
        is_retweet_value=False, filter_originals_in_tweets=True,
    )
    flag_rt, org_rt = _split_by_l3_flag(
        oscar_decisions, d3c,
        user_col="candidato_retuiteador", key_col="retweet_id",
        is_retweet_value=True, filter_originals_in_tweets=False,
    )

    parts.append("\n## Casos interesantes para análisis cualitativo\n")
    parts.append(
        "Tuits marcados con la mayor desviación (en likes) frente al "
        "**baseline orgánico de su propia cuenta**. La métrica es el "
        "**robust z-score**: `z = (likes − mediana_orgánica) / "
        "(1.4826 × MAD_orgánico)`. La MAD (mediana de las desviaciones "
        "absolutas) hace al z resistente a colas pesadas; la constante "
        "1.4826 lo escala para que coincida con un σ gaussiano. "
        "Lectura: `|z| ≥ 2` es un desvío notable; `|z| ≥ 5` es atípico "
        "para esa cuenta. Los casos con `MAD = 0` (cuentas con baseline "
        "casi constante en cero, p. ej. `liberalcr`) se marcan en la "
        "columna *nota*: el z no normaliza ahí, pero el caso sigue siendo "
        "interesante porque la imagen marcada es el único pico de la cuenta.\n"
    )
    parts.append(
        "Estos no son hallazgos estadísticos — son **leads** para la "
        "discusión cualitativa del capítulo: para cada fila, abrir la "
        "imagen, leer el texto del tuit, contextualizar la fecha y el "
        "tema, y decidir si el patrón engagement-vs-IA tiene una lectura "
        "narrativa.\n"
    )
    parts.append("\n### Originales (top 10)\n")
    parts.append(_format_interesting_cases_block(
        flag_orig, flagged_images_a, org_orig,
        user_col="username", key_col="tweet_id", top_n=10,
    ))
    parts.append("\n\n### Retuits (todos los marcados)\n")
    parts.append(_format_interesting_cases_block(
        flag_rt, flagged_images_b, org_rt,
        user_col="candidato_retuiteador", key_col="retweet_id", top_n=10,
    ))

    parts.append("\n\n## Correlación entre métricas en contenido marcado\n")
    parts.append(
        "**Pregunta:** ¿el éxito en una métrica de engagement refleja éxito "
        "en otra, dentro del contenido marcado? Si los tuits con IA que "
        "tienen más likes también tienen más retweets, replies, etc., las "
        "métricas se mueven juntas — el contenido es \"viralmente coherente\" "
        "en todas las dimensiones. Si una métrica se desacopla (p. ej. "
        "muchos likes pero pocos retweets), el contenido genera una "
        "respuesta selectiva.\n"
    )
    parts.append(
        "La medida es **Spearman ρ** sobre las 6 métricas de "
        "`public_metrics`. Spearman captura monotonicidad (sin asumir "
        "linealidad ni distribución gaussiana), lo cual es importante "
        "porque las métricas de engagement son fuertemente colas-pesadas. "
        "ρ = +1 indica orden idéntico; ρ = 0 sin relación monótona; ρ = −1 "
        "orden invertido.\n"
    )
    parts.append(
        "Se compara la matriz **flagged** (todos los tuits marcados a L3, "
        "originales + retuits) contra la matriz **orgánica** (tuits sin "
        "ninguna imagen marcada a nivel L3). Si la diferencia es pequeña, "
        "el patrón inter-métrico del contenido marcado es indistinguible "
        "del baseline.\n"
    )

    flag_combined = pd.concat([flag_orig, flag_rt], axis=0, ignore_index=True)
    org_originals_only = org_orig  # retweets are too sparse to matter
    parts.append(
        f"\n### Tuits marcados a L3 (n = {len(flag_combined)} = "
        f"{len(flag_orig)} originales + {len(flag_rt)} retuits)\n"
    )
    parts.append(_format_correlation_matrix(flag_combined, len(flag_combined)))

    parts.append(
        f"\n\n### Baseline orgánico — tuits sin marcar (originales, "
        f"n = {len(org_originals_only)})\n"
    )
    parts.append(_format_correlation_matrix(org_originals_only, len(org_originals_only)))

    if len(flag_combined) >= 3 and len(org_originals_only) >= 3:
        flag_corr = flag_combined[ENGAGEMENT_COLS].corr(method="spearman")
        org_corr = org_originals_only[ENGAGEMENT_COLS].corr(method="spearman")
        diffs_md = _summarize_corr_diff(flag_corr, org_corr, top_n=3)
        if diffs_md:
            parts.append("\n\n### Pares con mayor cambio entre marcados y orgánicos\n")
            parts.append(diffs_md)
            parts.append(
                "\nUn cambio de magnitud `|Δρ| < 0.10` se puede atribuir a "
                "ruido de muestreo dado el tamaño del subconjunto marcado; "
                "diferencias mayores merecen lectura cualitativa para "
                "entender por qué el acople inter-métrico difiere."
            )

    # Plain-language explanation of what "coupling" means and how to read
    # the matrices. Static text — kept in-script so it regenerates each
    # build instead of being lost on the next overwrite of the .md file.
    parts.append("\n\n### Qué significa esto, en lenguaje sencillo\n")
    parts.append(
        "**\"Acople\" entre métricas** — cuando dos métricas tienen ρ alto, "
        "los tuits que puntúan alto en una también puntúan alto en la otra. "
        "Si ordenamos los tuits por likes de más a menos, ¿quedan en un "
        "orden parecido al ordenarlos por retweets? Si sí, las dos métricas "
        "se mueven juntas. ρ = +1 es orden idéntico, ρ = 0 es sin relación, "
        "ρ = −1 es orden invertido.\n"
    )
    parts.append(
        "Cuando el reporte dice *\"`quote_count` y `bookmark_count` están "
        "más acoplados en marcados\"*, quiere decir: **en tuits con IA, si "
        "un tuit recibe muchos quote-tweets, casi siempre también recibe "
        "muchos bookmarks; en tuits orgánicos esa relación es más floja** "
        "— hay tuits orgánicos que se comentan mucho pero no se guardan, "
        "o viceversa.\n"
    )
    parts.append("\n#### Las 6 métricas en términos de comportamiento\n")
    parts.append(
        "| métrica | qué está haciendo el lector | esfuerzo |\n"
        "|---|---|---|\n"
        "| `impression_count` | la vio pasar en su feed | ninguno |\n"
        "| `like_count` | tocó el corazón | mínimo |\n"
        "| `retweet_count` | la reenvió a sus seguidores con un toque | mínimo |\n"
        "| `reply_count` | escribió un comentario | alto |\n"
        "| `quote_count` | comentó **y** reenvió | alto |\n"
        "| `bookmark_count` | la guardó para volver | alto |\n"
    )
    parts.append(
        "\n`reply`, `quote` y `bookmark` son las **acciones deliberadas**: "
        "el lector tuvo que detenerse y decidir hacer algo pensado, no solo "
        "tocar un botón.\n"
    )
    parts.append("\n#### Lo que dicen los números\n")
    parts.append(
        "- **Likes y retweets** permanecen muy acoplados en ambas matrices "
        "(ρ ≈ 0.87–0.92). Las dos son reacciones de un toque, así que quien "
        "\"apenas aprueba\" un tuit hace las dos en automático — eso es "
        "verdad sea contenido IA o no.\n"
        "- **Entre acciones deliberadas**, los tuits orgánicos muestran "
        "acople más suelto (ρ ≈ 0.62–0.78). En contenido orgánico un lector "
        "puede guardar sin quotear, o quotear sin guardar; las decisiones "
        "se desacoplan.\n"
        "- **En contenido IA-marcado, esas acciones deliberadas se aprietan** "
        "(ρ ≈ 0.75–0.90). Cuando se quotea, también se guarda. Cuando se "
        "responde, también se quotea.\n"
    )
    parts.append("\n#### Posibles lecturas (para discusión cualitativa)\n")
    parts.append(
        "1. **Hipótesis de notoriedad.** El contenido IA destaca lo "
        "suficiente como para que los lectores que deciden reaccionar lo "
        "hagan \"completo\": comentan, guardan y quotean a la vez. El "
        "contenido orgánico recibe más reacciones casuales por un solo "
        "canal.\n"
        "2. **Hipótesis de polarización.** El contenido IA divide: la "
        "mayoría lo ignora, pero un segmento más pequeño engancha en todos "
        "los canales deliberados a la vez. El orgánico recibe una "
        "distribución más amplia de reacciones casuales.\n"
        "3. **Es ruido.** Con solo 36 tuits marcados a L3, desplazamientos "
        "de ρ de ±0.20 pueden ocurrir por azar. El umbral `|Δρ| < 0.10` "
        "(arriba) ya marca como ruido la mayoría de los pares; solo "
        "`quote_count ↔ bookmark_count` (Δ = +0.20) cae fuera de esa banda "
        "en este corpus.\n"
    )
    parts.append(
        "\n**Lectura honesta:** hay un patrón pequeño y sugestivo de que el "
        "contenido IA-marcado recibe engagement más uniforme entre canales "
        "deliberados que el contenido orgánico. Es un **lead para lectura "
        "cualitativa** (abrir los casos del tope de la lista, leer qué "
        "tienen esos tuits con muchos quotes-y-bookmarks a la vez), no un "
        "efecto confirmado.\n"
    )

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Resultados Hub — master writing document
# ---------------------------------------------------------------------------
#
# `build_resultados_hub` consolidates every table, every case, and every
# headline number that the chapter Resultados section (draft.txt L121–L133)
# needs into a single auto-generated markdown file. It is a superset of:
#   - the deliverables README "Guía de redacción de Resultados" (§R.0–§R.6)
#   - the `d4_per_user_metric_medians.md` report (embedded verbatim)
#   - the §R.2 / §R.6 prose drop-in fragments
#   - one full "case card" per unique flagged tweet (34 originales + 2
#     retuits = 36 cards), sorted chronologically.
#
# The hub regenerates from scratch each builder run. Prose drop-in lives
# in `_format_prose_drop_in_*` functions below; edits to prose require
# editing this file, not the .md output (which would be overwritten).
# ---------------------------------------------------------------------------


def _format_d6a_table(d6a: pd.DataFrame) -> str:
    """Render D6a (3 rows: originales / retuits / total) as a markdown table."""
    lines = [
        "| población | n_total | n_L1 | n_L2 | n_L3 | n_sin_L | pct_L3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in d6a.iterrows():
        lines.append(
            f"| {r['population']} | {int(r['n_total'])} "
            f"| {int(r['n_L1'])} | {int(r['n_L2'])} | {int(r['n_L3'])} "
            f"| {int(r['n_sin_L'])} | {r['pct_L3']:.1f} % |"
        )
    return "\n".join(lines)


def _format_d6b_table(d6b: pd.DataFrame) -> str:
    """Render D6b (12 rows, one per cuenta) sorted by pct_L3 desc so the
    accounts with most IA presence (Robles 15.1 %, accionciudadana 10.5 %,
    …) surface first."""
    sub = d6b.sort_values(["pct_L3", "n_L3", "n_total"],
                          ascending=[False, False, False], kind="stable")
    lines = [
        "| username | tipo | n_total | n_L1 | n_L2 | n_L3 | n_sin_L | pct_L3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in sub.iterrows():
        lines.append(
            f"| `{r['username']}` | {r['account_type']} | {int(r['n_total'])} "
            f"| {int(r['n_L1'])} | {int(r['n_L2'])} | {int(r['n_L3'])} "
            f"| {int(r['n_sin_L'])} | {r['pct_L3']:.1f} % |"
        )
    return "\n".join(lines)


def _format_d5_panels(d5: pd.DataFrame) -> str:
    """Render D5 as 4 panels (Pooled, L1, L2, L3) × 6 métricas — one
    sub-table per nivel. Mirrors the visual layout of `figures/d5_pooled_*.png`."""
    order = [
        "Pooled (cualquier L1/L2/L3)",
        "L1 Sí/Estricto",
        "L2 Dudoso",
        "L3 Valor Alto",
    ]
    parts = []
    for level in order:
        sub = d5[d5["level"] == level].copy()
        if sub.empty:
            continue
        n_ia = int(sub.iloc[0]["n_ia"])
        n_no = int(sub.iloc[0]["n_no_ia"])
        parts.append(f"\n#### {level}  ·  n_IA = {n_ia}  ·  n_no_IA = {n_no}\n")
        parts.append(
            "| métrica | mediana_ia | mediana_no_ia | Δ (ia − no_ia) |\n"
            "|---|---:|---:|---:|"
        )
        for _, r in sub.iterrows():
            delta = r["delta"]
            delta_str = f"{delta:+.1f}" if not pd.isna(delta) else "—"
            parts.append(
                f"| `{r['metric']}` | {r['mediana_ia']:.1f} "
                f"| {r['mediana_no_ia']:.1f} | {delta_str} |"
            )
    return "\n".join(parts)


def _format_d7_confusion(d7: pd.DataFrame) -> str:
    """Render D7 (9 rows = 3 GT × 3 checkpoints) as a markdown table with
    precision / recall / accuracy rendered as percentages."""
    lines = [
        "| GT level | checkpoint | GT+ | pred+ | TP | FP | FN | TN | precisión | recall | accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in d7.iterrows():
        lines.append(
            f"| {r['gt_level']} | {r['checkpoint']} "
            f"| {int(r['gt_positives'])} | {int(r['pred_positives'])} "
            f"| {int(r['tp'])} | {int(r['fp'])} | {int(r['fn'])} | {int(r['tn'])} "
            f"| **{r['precision'] * 100:.1f} %** | {r['recall'] * 100:.1f} % "
            f"| {r['accuracy'] * 100:.1f} % |"
        )
    return "\n".join(lines)


def _format_d8a_table(d8a: pd.DataFrame) -> str:
    """Render D8a (24 rows = 6 poblaciones × 4 bucket-cumulativos) grouped
    by `population`, with Wilson 95 % CI formatted as a range."""
    pop_order = [
        "all_tweets", "originals_tweets", "retweets_tweets",
        "L3_all", "L3_originals", "L3_retweets",
    ]
    bucket_order = ["B1_only", "B1_B2", "B1_B2_B3", "any_bucket"]
    parts = []
    for pop in pop_order:
        sub = d8a[d8a["population"] == pop]
        if sub.empty:
            continue
        parts.append(f"\n**{pop}**\n")
        parts.append(
            "| bucket | numerador | denominador | tasa | Wilson 95 % CI |\n"
            "|---|---:|---:|---:|---|"
        )
        for bg in bucket_order:
            r = sub[sub["bucket_group"] == bg]
            if r.empty:
                continue
            r = r.iloc[0]
            parts.append(
                f"| `{r['bucket_group']}` | {int(r['numerator'])} "
                f"| {int(r['denominator'])} | {r['rate'] * 100:.3f} % "
                f"| [{r['wilson_lo'] * 100:.2f} %, {r['wilson_hi'] * 100:.2f} %] |"
            )
    return "\n".join(parts)


def _format_d8b_table(d8b: pd.DataFrame) -> str:
    """Render D8b (5 rows) with snippets truncated to ~120 chars."""
    lines = [
        "| scope | primary_id | cuenta | canal | términos | L3 | snippet |",
        "|---|---|---|---|---|:-:|---|",
    ]
    for _, r in d8b.iterrows():
        snippet = str(r["snippet"]).replace("|", "\\|").replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        l3_mark = "✓" if bool(r["is_l3"]) else "·"
        lines.append(
            f"| {r['scope']} | `{r['primary_id']}` | `{r['account']}` "
            f"| `{r['channel']}` | {r['matched_terms']} | {l3_mark} | {snippet} |"
        )
    return "\n".join(lines)


def _format_prose_drop_in_R2() -> str:
    """Prose drop-in for §R.2 (Confusión AIDE). Already vetted in
    `data/deliverables/README.md` §R.2 — kept here as a single source so
    the hub regenerates it verbatim."""
    return (
        "> *\"Al contrastar las predicciones de AIDE (t = 0.50) contra los "
        "veredictos de Hive como verdad de referencia, encontramos que el "
        "checkpoint con mejor desempeño (GenImage) acertó sólo 4 de sus 37 "
        "alarmas (precisión 10.8 %), mientras que ProGAN y SD14 produjeron "
        "precisiones por debajo del 2 %. Esta tasa de falsos positivos hace "
        "inviable a AIDE como detector individual y motivó su descarte; los "
        "niveles L1/L2/L3 se construyeron usando únicamente la revisión "
        "humana y Hive Moderation.\"*"
    )


def _format_prose_drop_in_R6() -> str:
    """Prose drop-in for §R.6 (Especificación de IA). Vetted in README §R.6."""
    return (
        "> *\"Aún bajo un vocabulario maximalmente inclusivo (~90 patrones "
        "en 4 buckets B1/B2/B3/B4), no detectamos **ninguna** declaración "
        "directa de generación con IA en los 475 tuits del corpus. Cinco "
        "tuits mencionan IA de forma genérica (bucket B4), pero ninguno "
        "atribuye la generación de la imagen adjunta a IA; cuatro son "
        "positioning de política y uno es una acusación a oponentes.\"*"
    )


def _baseline_medians_for_user(
    tweets: pd.DataFrame, oscar_decisions: pd.DataFrame,
    *, user_col: str, key_col: str, is_retweet_value: bool,
    filter_originals_in_tweets: bool,
) -> dict[str, dict[str, float]]:
    """Per-user organic-baseline medians for the 6 engagement metrics.

    The case-card needs the **same** baseline as D4 / d4_report — tuits
    where ninguna imagen está marcada a L3. We return a dict
    `{username: {metric: median}}` so each card can look up its delta.
    """
    dec = oscar_decisions[oscar_decisions["is_retweet"] == is_retweet_value].copy()
    dec = _add_image_flags(dec)
    dec[key_col] = dec["image_id"].astype(str).str.rsplit("_", n=1).str[0]
    tweet_flags = dec.groupby(key_col)[["L3"]].any().reset_index()

    t = tweets.copy()
    if filter_originals_in_tweets:
        t = t[t["is_retweet"] == False].copy()
    t[key_col] = t[key_col].astype(str)
    t = t.merge(tweet_flags, on=key_col, how="left")
    t["L3"] = t["L3"].fillna(False).astype(bool)

    organic = t[~t["L3"]]
    out: dict[str, dict[str, float]] = {}
    for user, g in organic.groupby(user_col, sort=True):
        out[user] = {m: float(g[m].median()) if len(g) else float("nan")
                     for m in ENGAGEMENT_COLS}
    return out


def _format_case_card(
    *,
    card_no: int,
    scope: str,  # "original" or "retweet"
    primary_id: str,
    username: str,
    created_at: str,
    strictest_level: str,
    image_rows: list[dict],
    text: str,
    note_tweet: str,
    metrics: dict[str, int],
    baseline: dict[str, float],
    followers: int | None,
    robust_z: float | None,
    extra_header: str = "",
) -> str:
    """Render one case card as markdown. `image_rows` is a list of dicts
    with keys `image_id`, `media_url`, `level_tag` ("L1"/"L2"/"L3"),
    `humano_personal`, `humano_hive`, `oscar_comentarios`, `paulo_comentarios`."""
    # Header line.
    date_str = str(created_at)[:10] if created_at else "—"
    header_extra = f" · {extra_header}" if extra_header else ""
    parts = [
        f"### Caso {card_no} · @{username} · {date_str} · {strictest_level}{header_extra}",
        "",
        f"- **scope**: `{scope}` · **primary_id**: `{primary_id}` · "
        f"**imágenes en el tuit**: {len(image_rows)}",
    ]

    # Image links with their per-image level tag.
    parts.append("- **imágenes marcadas**:")
    for img in image_rows:
        parts.append(
            f"  - `{img['level_tag']}` · "
            f"[`{img['image_id']}`]({img['media_url']})"
        )

    # Tweet text + note_tweet.
    safe_text = str(text or "").strip().replace("\n", "\n  > ")
    parts.append("")
    parts.append("**Texto del tuit:**")
    parts.append("")
    if safe_text:
        parts.append(f"> {safe_text}")
    else:
        parts.append("> _(sin texto)_")
    if note_tweet and str(note_tweet).strip() and str(note_tweet).strip() != "nan":
        nt = str(note_tweet).strip().replace("\n", "\n  > ")
        parts.append("")
        parts.append("**`note_tweet` (texto extendido):**")
        parts.append("")
        parts.append(f"> {nt}")

    # Per-image verdicts table.
    parts.append("")
    parts.append("**Veredictos por imagen:**")
    parts.append("")
    parts.append(
        "| image_id | humano_personal | humano_hive | oscar_comentarios | paulo_comentarios |"
    )
    parts.append("|---|---|---|---|---|")
    for img in image_rows:
        op = str(img.get("humano_personal") or "").strip() or "—"
        oh = str(img.get("humano_hive") or "").strip() or "—"
        oc = str(img.get("oscar_comentarios") or "").strip()
        pc = str(img.get("paulo_comentarios") or "").strip()
        if oc == "nan":
            oc = ""
        if pc == "nan":
            pc = ""
        oc = oc.replace("|", "\\|").replace("\n", " ") or "—"
        pc = pc.replace("|", "\\|").replace("\n", " ") or "—"
        parts.append(
            f"| `{img['image_id']}` | {op} | {oh} | {oc} | {pc} |"
        )

    # Engagement metrics + delta vs baseline (raw + por_seguidor).
    parts.append("")
    parts.append("**Métricas de engagement vs. baseline orgánico de la cuenta:**")
    parts.append("")
    parts.append(
        "| métrica | valor del tuit | mediana_sin_ia | Δ (raw) | "
        "valor / followers | Δ por seguidor (p.p.) |"
    )
    parts.append("|---|---:|---:|---:|---:|---:|")
    for m in ENGAGEMENT_COLS:
        v = metrics.get(m)
        med = baseline.get(m, float("nan"))
        if v is None or pd.isna(v):
            v_str = "—"
            delta_str = "—"
            ps_str = "—"
            dps_str = "—"
        else:
            v_int = int(v)
            v_str = f"{v_int}"
            if pd.isna(med):
                delta_str = "—"
            else:
                delta_str = f"{v_int - med:+.1f}"
            if followers and followers > 0:
                v_per = v_int / followers
                ps_str = f"{v_per * 100:.2f} %"
                if pd.isna(med):
                    dps_str = "—"
                else:
                    med_per = med / followers
                    dps_str = f"{(v_per - med_per) * 100:+.2f} p.p."
            else:
                ps_str = "—"
                dps_str = "—"
        if pd.isna(med):
            med_str = "—"
        else:
            med_str = f"{med:.1f}"
        parts.append(
            f"| `{m}` | {v_str} | {med_str} | {delta_str} | {ps_str} | {dps_str} |"
        )

    # Robust z annotation when the case is in the top-10 likes ranking.
    if robust_z is not None and not pd.isna(robust_z):
        z_label = (
            "outlier extremo" if abs(robust_z) >= 5
            else "desvío notable" if abs(robust_z) >= 2
            else "dentro del rango habitual"
        )
        parts.append("")
        parts.append(
            f"**Robust z (likes vs. baseline orgánico de la cuenta):** "
            f"`{robust_z:+.2f}` — {z_label}."
        )

    parts.append("")  # trailing blank for spacing between cards
    return "\n".join(parts)


def _build_case_universe(
    d2a: pd.DataFrame, d2b: pd.DataFrame,
    d3a: pd.DataFrame, d3c: pd.DataFrame,
    oscar_decisions: pd.DataFrame,
    followers: dict[str, int],
) -> str:
    """Build the full §R.5 case-cards section: one card per unique
    flagged tweet (34 originales + 2 retuits = 36), sorted chronologically.

    For each card we collect:
      - all the per-image verdicts from D2a/D2b
      - the parent tweet's engagement metrics (from D2a directly; D2b
        already carries source-tweet engagement via the media_url join)
      - the user's organic baseline (medianas sin_ia) for delta
      - robust z on like_count (computed against organic likes for that
        user's account; same MAD-based formula as
        `_format_interesting_cases_block`)
    """
    # Organic baselines, by population.
    baseline_orig = _baseline_medians_for_user(
        d3a, oscar_decisions,
        user_col="username", key_col="tweet_id",
        is_retweet_value=False, filter_originals_in_tweets=True,
    )
    baseline_rt = _baseline_medians_for_user(
        d3c, oscar_decisions,
        user_col="candidato_retuiteador", key_col="retweet_id",
        is_retweet_value=True, filter_originals_in_tweets=False,
    )

    # Organic like-distributions per user → for robust z denominator.
    def _org_likes(tweets: pd.DataFrame, oscar: pd.DataFrame, *,
                   user_col: str, key_col: str, is_retweet_value: bool,
                   filter_originals_in_tweets: bool) -> dict[str, np.ndarray]:
        dec = oscar[oscar["is_retweet"] == is_retweet_value].copy()
        dec = _add_image_flags(dec)
        dec[key_col] = dec["image_id"].astype(str).str.rsplit("_", n=1).str[0]
        tweet_flags = dec.groupby(key_col)[["L3"]].any().reset_index()
        t = tweets.copy()
        if filter_originals_in_tweets:
            t = t[t["is_retweet"] == False].copy()
        t[key_col] = t[key_col].astype(str)
        t = t.merge(tweet_flags, on=key_col, how="left")
        t["L3"] = t["L3"].fillna(False).astype(bool)
        organic = t[~t["L3"]]
        out: dict[str, np.ndarray] = {}
        for u, g in organic.groupby(user_col, sort=True):
            out[u] = g["like_count"].astype(float).to_numpy()
        return out

    org_likes_orig = _org_likes(
        d3a, oscar_decisions,
        user_col="username", key_col="tweet_id",
        is_retweet_value=False, filter_originals_in_tweets=True,
    )
    org_likes_rt = _org_likes(
        d3c, oscar_decisions,
        user_col="candidato_retuiteador", key_col="retweet_id",
        is_retweet_value=True, filter_originals_in_tweets=False,
    )

    def robust_z(user: str, likes: float, pool: dict[str, np.ndarray]) -> float | None:
        arr = pool.get(user)
        if arr is None or len(arr) == 0:
            return None
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        if mad == 0:
            return None
        return (float(likes) - med) / (1.4826 * mad)

    # Build per-tweet image groups for D2a (group by tweet_id).
    cards_data: list[dict] = []

    # D2a: originals. Each tweet may have several flagged image rows.
    if not d2a.empty:
        d2a_sorted = d2a.copy()
        d2a_sorted["tweet_id"] = d2a_sorted["tweet_id"].astype(str)
        for tid, g in d2a_sorted.groupby("tweet_id", sort=False):
            row0 = g.iloc[0]
            image_rows = []
            for _, r in g.iterrows():
                # Per-image strictest level from the verdict.
                op = str(r["humano_personal"]).strip().lower()
                oh = str(r["humano_hive"]).strip().lower()
                if op in {"sí", "si"} or oh in {"sí", "si"}:
                    lvl_tag = "L1"
                elif op == "dudoso" or oh == "dudoso":
                    lvl_tag = "L2"
                elif oh == "valor alto":
                    lvl_tag = "L3"
                else:
                    lvl_tag = "—"
                image_rows.append({
                    "image_id": str(r["image_id"]),
                    "media_url": str(r["media_url"]),
                    "level_tag": lvl_tag,
                    "humano_personal": r["humano_personal"],
                    "humano_hive": r["humano_hive"],
                    "oscar_comentarios": r["oscar_comentarios"],
                    "paulo_comentarios": r["paulo_comentarios"],
                })
            strictest = min(
                (img["level_tag"] for img in image_rows if img["level_tag"] in {"L1", "L2", "L3"}),
                default="—",
            )
            user = row0["username"]
            metrics = {m: row0[m] for m in ENGAGEMENT_COLS}
            cards_data.append({
                "scope": "original",
                "primary_id": tid,
                "username": user,
                "created_at": row0["created_at"],
                "strictest_level": (
                    "L1 Sí/Estricto" if strictest == "L1"
                    else "L2 Dudoso" if strictest == "L2"
                    else "L3 Valor Alto" if strictest == "L3"
                    else "—"
                ),
                "image_rows": image_rows,
                "text": row0["text"],
                "note_tweet": row0.get("note_tweet", ""),
                "metrics": metrics,
                "baseline": baseline_orig.get(user, {}),
                "followers": followers.get(user),
                "robust_z": robust_z(user, row0["like_count"], org_likes_orig),
                "extra_header": "",
            })

    # D2b: retweets. Group by retweet_id (each retweet may have several
    # flagged images of the source tweet).
    if not d2b.empty:
        d2b_sorted = d2b.copy()
        d2b_sorted["retweet_id"] = d2b_sorted["retweet_id"].astype(str)
        for rid, g in d2b_sorted.groupby("retweet_id", sort=False):
            row0 = g.iloc[0]
            image_rows = []
            for _, r in g.iterrows():
                op = str(r["humano_personal"]).strip().lower()
                oh = str(r["humano_hive"]).strip().lower()
                if op in {"sí", "si"} or oh in {"sí", "si"}:
                    lvl_tag = "L1"
                elif op == "dudoso" or oh == "dudoso":
                    lvl_tag = "L2"
                elif oh == "valor alto":
                    lvl_tag = "L3"
                else:
                    lvl_tag = "—"
                image_rows.append({
                    "image_id": str(r["retweet_image_id"]),
                    "media_url": str(r["media_url"]),
                    "level_tag": lvl_tag,
                    "humano_personal": r["humano_personal"],
                    "humano_hive": r["humano_hive"],
                    "oscar_comentarios": r["oscar_comentarios"],
                    "paulo_comentarios": r["paulo_comentarios"],
                })
            strictest = min(
                (img["level_tag"] for img in image_rows if img["level_tag"] in {"L1", "L2", "L3"}),
                default="—",
            )
            user = row0["candidato_retuiteador"]
            metrics = {m: row0[m] for m in ENGAGEMENT_COLS}
            extra = f"retuit de @{row0['original_author_username']}"
            cards_data.append({
                "scope": "retweet",
                "primary_id": rid,
                "username": user,
                "created_at": row0["retweet_created_at"],
                "strictest_level": (
                    "L1 Sí/Estricto" if strictest == "L1"
                    else "L2 Dudoso" if strictest == "L2"
                    else "L3 Valor Alto" if strictest == "L3"
                    else "—"
                ),
                "image_rows": image_rows,
                "text": row0["original_text"],
                "note_tweet": row0.get("note_tweet", ""),
                "metrics": metrics,
                "baseline": baseline_rt.get(user, {}),
                "followers": followers.get(user),
                "robust_z": robust_z(user, row0["like_count"], org_likes_rt),
                "extra_header": extra,
            })

    # Sort chronologically (ascending created_at).
    cards_data.sort(key=lambda d: str(d["created_at"]))

    # Render.
    rendered = []
    for i, c in enumerate(cards_data, 1):
        rendered.append(_format_case_card(
            card_no=i,
            scope=c["scope"],
            primary_id=c["primary_id"],
            username=c["username"],
            created_at=c["created_at"],
            strictest_level=c["strictest_level"],
            image_rows=c["image_rows"],
            text=c["text"],
            note_tweet=c["note_tweet"],
            metrics=c["metrics"],
            baseline=c["baseline"],
            followers=c["followers"],
            robust_z=c["robust_z"],
            extra_header=c["extra_header"],
        ))
    return "\n".join(rendered)


def build_resultados_hub(
    d4_report: str,
    d5: pd.DataFrame,
    d6a: pd.DataFrame,
    d6b: pd.DataFrame,
    d7: pd.DataFrame,
    d8a: pd.DataFrame,
    d8b: pd.DataFrame,
    d2a: pd.DataFrame,
    d2b: pd.DataFrame,
    oscar_decisions: pd.DataFrame,
    d3a: pd.DataFrame,
    d3c: pd.DataFrame,
    followers: dict[str, int],
) -> str:
    """Assemble the master writing document.

    The doc is structured around the **six Resultados sub-blocks** of the
    chapter draft (`data/meeting_overviews/draft.txt` L121–L133). Each
    sub-block is rendered with: a brief "qué cubrir" line, the headline
    drop-in numbers, the full table(s), a prose drop-in when ready, and
    a "frontera con Discusión" callout reminding what NOT to put in
    Resultados.

    §R.4 inlines `d4_report` verbatim (stripping its top-level `# D4 …`
    header) so the chapter writer never leaves the hub. The standalone
    `d4_per_user_metric_medians.md` stays available unchanged — both come
    from the same `build_d4_report(...)` call.
    """
    # Counts used in §R.0 / §R.1. Canonical convention used in draft.txt
    # (L9, L59) and the deliverables README: 424 tuits (D3a total, including
    # RT-prefixed timeline rows) + 51 retuits (D3c total) = 475 tuits únicos.
    # `n_orig_no_rt` (373) is the D5-engagement working set (D3a filtered to
    # is_retweet == False); it's *not* the corpus tuit count.
    n_d3a_rows = len(d3a)  # 424
    n_d3c_rows = len(d3c)  # 51
    n_d3b_images = int(d3a["image_count"].fillna(0).astype(int).sum())  # 738
    n_d3d_images = int(d3c["image_count"].fillna(0).astype(int).sum())  # 66
    n_unique_tweets = n_d3a_rows + n_d3c_rows  # 475
    n_orig_no_rt = int((d3a["is_retweet"] == False).sum())  # 373 — for §R.4 D5 context only
    n_images_total = int(d6a[d6a["population"] == "total"]["n_total"].iloc[0])
    n_images_orig = int(d6a[d6a["population"] == "originales"]["n_total"].iloc[0])
    n_images_rt = int(d6a[d6a["population"] == "retuits"]["n_total"].iloc[0])
    n_L3_total = int(d6a[d6a["population"] == "total"]["n_L3"].iloc[0])
    pct_L3_total = float(d6a[d6a["population"] == "total"]["pct_L3"].iloc[0])

    # Strip D4 report's top H1 so this hub's H2 wraps cleanly.
    d4_body = d4_report
    if d4_body.startswith("# D4 — Medianas por candidato, nivel y métrica\n"):
        d4_body = d4_body.split("\n", 1)[1].lstrip("\n")

    parts: list[str] = []

    # -----------------------------------------------------------------
    # Header + TOC + usage notes.
    # -----------------------------------------------------------------
    parts.append("# Resultados — Hub de redacción\n")
    parts.append(
        "**Qué es este archivo.** Documento único auto-generado para "
        "redactar la sección **Resultados** del capítulo *\"Democracia "
        "sintética…\"* (`data/meeting_overviews/draft.txt` L121–L133). "
        "Embebido: cada tabla, cada cifra y los 36 casos marcados (uno "
        "por tuit único) — todo en un solo lugar para que no haga falta "
        "abrir el CSV ni saltar entre documentos mientras se escribe.\n"
    )
    parts.append(
        "**Cómo se mantiene en sincronía.** Generado por "
        "`src/utils/build_deliverables.py` desde los CSV de "
        "`data/deliverables/`. Cada rerun reemplaza el archivo entero. "
        "Las prosa drop-in viven como funciones de Python "
        "(`_format_prose_drop_in_R*`) en el builder — para editarla, "
        "editá el builder, no este `.md` (se sobreescribe al regenerar).\n"
    )
    parts.append(
        "**Convención de fronteras.** Cada sub-bloque cierra con una "
        "*frontera con Discusión* que recuerda qué NO debe entrar en "
        "Resultados. Regla operativa (heredada de Oscar, Reunión 2 "
        "L715): si una oración contiene **verdicto interpretativo + "
        "nombre de cuenta**, esa oración pertenece a Discusión.\n"
    )
    parts.append("**Índice:**\n")
    parts.append(
        "- [§R.0 — Apertura](#r-0--apertura)\n"
        "- [§R.1 (draft.txt L123) — Aportes de BD](#r-1-drafttxt-l123--aportes-de-bd)\n"
        "- [§R.2 (draft.txt L125) — Confusión AIDE](#r-2-drafttxt-l125--confusión-aide)\n"
        "- [§R.3 (draft.txt L127) — Resumen L1/L2/L3](#r-3-drafttxt-l127--resumen-l1l2l3)\n"
        "- [§R.4 (draft.txt L129) — Engagement (general + por cuenta)](#r-4-drafttxt-l129--engagement-general--por-cuenta)\n"
        "- [§R.5 (draft.txt L131) — Cálculo de éxito + Casos específicos](#r-5-drafttxt-l131--cálculo-de-éxito--casos-específicos)\n"
        "- [§R.6 (draft.txt L133) — Especificación de IA](#r-6-drafttxt-l133--especificación-de-ia)\n"
        "- [Apéndice — Cruces y referencias](#apéndice--cruces-y-referencias)\n"
    )
    parts.append("---\n")

    # -----------------------------------------------------------------
    # §R.0 — Apertura
    # -----------------------------------------------------------------
    parts.append("## §R.0 — Apertura\n")
    parts.append(
        "**Qué cubrir.** Una oración que recuerde el alcance del corpus "
        "(período, cuentas, totales) antes de entrar a los sub-bloques.\n"
    )
    parts.append("**Headline numbers (canónicos del workspace):**\n")
    parts.append(
        f"- Corpus: **{n_images_total} imágenes** ({n_images_orig} originales + "
        f"{n_images_rt} retuiteadas) distribuidas en **{n_d3a_rows} tuits + "
        f"{n_d3c_rows} retuits = {n_unique_tweets} tuits únicos** "
        "(esta es la convención del proyecto: D3a + D3c, sin deduplicar las "
        f"filas RT-prefijadas del timeline; el subconjunto {n_orig_no_rt} = "
        "`is_retweet == False` aparece sólo como working-set de engagement "
        "en §R.4 / D5).\n"
        "- Período: 2025-07-01 a 2026-01-31 (campaña electoral CR 2026).\n"
        "- Cuentas: 12 de las 13 objetivo aportaron imágenes "
        "(`JoseAguilarBerr` quedó fuera: 139 posts, 0 con imagen).\n"
        f"- Marcado a algún nivel L1/L2/L3: **{n_L3_total} imágenes "
        f"({pct_L3_total:.1f} %)** del corpus completo.\n"
    )
    parts.append(
        "> ⚠ **Discrepancia con draft.txt L9** (a resolver con Oscar antes "
        "de publicar): el borrador del capítulo dice *\"804 imágenes en "
        f"475 tweets\"*; el canónico del workspace es **{n_images_total} "
        f"imágenes / {n_unique_tweets} tuits** — la cuenta de tuits sí "
        f"coincide, pero las imágenes difieren (804 vs {n_images_total}). "
        "La aritmética también aparece en draft.txt L93 "
        "(*\"41 IGA comparada con 650 orgánicas\"*) y L137.\n"
    )
    parts.append(
        "**Frontera con Discusión:** §R.0 sólo presenta totales y "
        "composición. No anticipar prevalencia ni representatividad "
        "(eso vive en §R.3 y en Discusión).\n"
    )
    parts.append("---\n")

    # -----------------------------------------------------------------
    # §R.1 — Aportes de BD (D3)
    # -----------------------------------------------------------------
    parts.append("## §R.1 (draft.txt L123) — Aportes de BD\n")
    parts.append(
        "**Qué cubrir.** Referencia transversal a la base de datos pública "
        "compartible que sustenta todo el análisis. Este sub-bloque "
        "explica que la BD existe y cómo está estructurada.\n"
    )
    parts.append("**Conteos de filas (vista panorámica):**\n")
    parts.append(
        "| Deliverable | Granularidad | Filas |\n"
        "|---|---|---:|\n"
        f"| `d3_tweets_tweet_level.csv` (D3a) | tuit | {n_d3a_rows} |\n"
        f"| `d3_tweets_image_level.csv` (D3b) | imagen | {n_d3b_images} (D3a exploded; incluye 1 imagen con descarga fallida) |\n"
        f"| `d3_retweets_tweet_level.csv` (D3c) | retuit | {n_d3c_rows} |\n"
        f"| `d3_retweets_image_level.csv` (D3d) | imagen | {n_d3d_images} (D3c exploded) |\n"
    )
    parts.append(
        "**Esquemas de columnas:** ver `data/deliverables/README.md` "
        "§D3 (cada uno de D3a/D3b/D3c/D3d tiene su tabla `| columna | "
        "descripción |` con todos los campos de la API X/Twitter).\n"
    )
    parts.append("**Headline drop-in:**\n")
    parts.append(
        f"> *\"La extracción produjo {n_d3a_rows} tuits con imagen + "
        f"{n_d3c_rows} retuits, totalizando **{n_unique_tweets} tuits únicos** "
        f"y **{n_images_total} imágenes** ({n_images_orig} originales + "
        f"{n_images_rt} retuiteadas; 1 descarga fallida documentada).\"*\n"
    )
    parts.append(
        "**Frontera con Discusión:** §R.1 introduce la BD como aporte. "
        "No interpreta composición temática ni sesgo de cobertura — eso "
        "es Discusión.\n"
    )
    parts.append("---\n")

    # -----------------------------------------------------------------
    # §R.2 — Confusión AIDE (D7)
    # -----------------------------------------------------------------
    parts.append("## §R.2 (draft.txt L125) — Confusión AIDE\n")
    parts.append(
        "**Qué cubrir.** El resultado empírico que justifica descartar "
        "AIDE como detector standalone y construir L1/L2/L3 sólo desde "
        "Hive + revisión humana.\n"
    )
    parts.append("**Tabla D7 — Matriz de confusiones AIDE-vs-Hive (9 filas):**\n")
    parts.append(_format_d7_confusion(d7))
    parts.append("")
    parts.append(
        "**Headline numbers** (al nivel de ground truth estricto, "
        f"Hive `Sí`, tasa base {int(d7[d7['gt_level']=='strict'].iloc[0]['gt_positives'])}/737 ≈ 1.9 %):\n"
    )
    parts.append(
        "- **GenImage** (el mejor checkpoint): 4 TP de 37 alarmas → **precisión 10.8 %**.\n"
        "- **ProGAN**: 2 TP de 113 alarmas → **precisión 1.8 %**.\n"
        "- **SD14**: 1 TP de 145 alarmas → **precisión 0.7 %**.\n"
        "- Accuracy parece alta (~94–99 %) sólo porque la mayoría son verdaderos negativos triviales; reportar accuracy aquí sería engañoso.\n"
    )
    parts.append("**Prosa drop-in (lista para pegar en el capítulo):**\n")
    parts.append(_format_prose_drop_in_R2())
    parts.append("")
    parts.append(
        "**Frontera con Discusión:** los números van aquí. La conexión "
        "con literatura sobre detectores académicos vs. comerciales "
        "(Ferrara 2026, Gao et al. 2026) ya está parcialmente en "
        "draft.txt L155 — eso es Discusión.\n"
    )
    parts.append(
        "**Interpretación cualitativa profunda** (qué significa que SD14 "
        "y ProGAN sean *\"esencialmente ruido\"*): "
        "`_side_projects/aide_vs_hive/reports/aide_vs_hive_interpretation.md`.\n"
    )
    parts.append("---\n")

    # -----------------------------------------------------------------
    # §R.3 — Resumen L1/L2/L3 (D6a + D6b)
    # -----------------------------------------------------------------
    parts.append("## §R.3 (draft.txt L127) — Resumen L1/L2/L3\n")
    parts.append(
        "**Qué cubrir.** Cuántas imágenes terminaron en cada nivel — "
        "primero general (3 filas: originales / retuits / total), "
        "después por cuenta (una fila por cada cuenta con ≥1 imagen).\n"
    )
    parts.append(
        "**Recordatorio metodológico (no redefinir en prosa, ya está en §2):** "
        "los niveles son acumulativos por construcción — `L3 ⊇ L2 ⊇ L1`. "
        "Una imagen marcada a L1 también cuenta en L2 y L3. `n_sin_L` "
        "es el complemento del L más inclusivo: `n_sin_L = n_total − n_L3`.\n"
    )
    parts.append("### D6a — Resumen general (3 filas)\n")
    parts.append(_format_d6a_table(d6a))
    parts.append("")
    parts.append("### D6b — Resumen por cuenta (ordenada por `pct_L3` desc)\n")
    parts.append(_format_d6b_table(d6b))
    parts.append("")
    parts.append("**Headline numbers:**\n")
    # Compute the top-3 dynamically so it reflects the data.
    top3 = (d6b.sort_values("pct_L3", ascending=False, kind="stable")
            .head(3)[["username", "pct_L3", "n_L3", "n_total"]])
    cero_l3 = d6b[d6b["n_L3"] == 0]["username"].tolist()
    parts.append(
        f"- Total imágenes marcadas (cualquier L): **{n_L3_total} de "
        f"{n_images_total}** ({pct_L3_total:.1f} %).\n"
        f"- **Top L3 share por cuenta:** "
        + ", ".join(
            f"`{r['username']}` {r['pct_L3']:.1f} % "
            f"({int(r['n_L3'])}/{int(r['n_total'])})"
            for _, r in top3.iterrows()
        )
        + ".\n"
        + (f"- **Cuentas sin nada marcado:** "
           + ", ".join(f"`{u}`" for u in cero_l3) + ".\n"
           if cero_l3 else "")
    )
    parts.append(
        "**Frontera con Discusión:** presentar conteos y porcentajes. "
        "**No especular** sobre **por qué** `RoblesBarrantes` encabeza ni "
        "**por qué** `FrenteAmplio` quedó en cero — eso es Discusión "
        "(la conexión con uso ideológico por partido vive en draft.txt "
        "L145–L149 / L519–L521).\n"
    )
    parts.append("---\n")

    # -----------------------------------------------------------------
    # §R.4 — Engagement (D5 panels + figures + full D4 report inlined)
    # -----------------------------------------------------------------
    parts.append("## §R.4 (draft.txt L129) — Engagement (general + por cuenta)\n")
    parts.append(
        "**Qué cubrir.** Comparación de engagement entre tuits IA y "
        "tuits no-IA, primero **poolada** (todos los candidatos juntos), "
        "después **por cuenta** (cada candidato × nivel × métrica).\n"
    )
    parts.append("### D5 — Comparación poolada IA vs no-IA (4 paneles)\n")
    parts.append(
        f"**Solo originales** (n_total = {n_orig_no_rt} = {n_d3a_rows} − "
        "filas RT-prefijadas del timeline; D5 filtra D3a a `is_retweet == False`). "
        "Las métricas de las filas RT-prefijadas corresponden al tuit fuente, "
        "no al retuiteador — incluirlas sesgaría la comparación. "
        "*Pooled* y *L3* son idénticos por construcción matemática "
        "(L3 es el nivel más inclusivo y *Pooled* agrupa cualquier "
        "L1/L2/L3).\n"
    )
    parts.append(_format_d5_panels(d5))
    parts.append("")
    parts.append("**Headline numbers (drop-in para prosa):**\n")
    # Compute D5 deltas dynamically so the hub reflects the CSV exactly.
    def _d5_delta(level: str, metric: str) -> float:
        sub = d5[(d5["level"] == level) & (d5["metric"] == metric)]
        return float(sub["delta"].iloc[0]) if not sub.empty else float("nan")
    d_pool_likes = _d5_delta("Pooled (cualquier L1/L2/L3)", "like_count")
    d_pool_imps = _d5_delta("Pooled (cualquier L1/L2/L3)", "impression_count")
    d_l1_likes = _d5_delta("L1 Sí/Estricto", "like_count")
    d_l1_imps = _d5_delta("L1 Sí/Estricto", "impression_count")
    d_l2_likes = _d5_delta("L2 Dudoso", "like_count")
    d_l2_imps = _d5_delta("L2 Dudoso", "impression_count")
    parts.append(
        f"| Nivel | n_IA | Δ likes | Δ impressions |\n"
        f"|---|---:|---:|---:|\n"
        f"| Pooled / L3 | 34 | {d_pool_likes:+.1f} | {d_pool_imps:+.1f} |\n"
        f"| L1 estricto | 10 | {d_l1_likes:+.1f} | {d_l1_imps:+.1f} |\n"
        f"| **L2 dudoso** | **15** | **{d_l2_likes:+.1f}** | **{d_l2_imps:+.1f}** |\n"
    )
    parts.append(
        "- **L2 es el único nivel** donde la mediana de likes IA supera "
        "claramente al orgánico. El resto muestra deltas marginales o "
        "negativos en likes; las impresiones suben en todos los niveles, "
        "pero la magnitud relativa es modesta.\n"
    )
    parts.append(
        "**Coherencia con metodología inferencial.** El delta descriptivo "
        f"{d_l2_likes:+.1f} a L2 **no contradice** el verdicto nulo de "
        "Mann–Whitney pooled (p = 0.343). Describen objetos distintos a "
        "niveles de agregación distintos. El diseño emparejado por "
        "fechas dentro de cuenta sí rechaza (Wilcoxon p = 0.023). "
        "Detalle: `data/meeting_overviews/methodology_section_3/coherence_audit.md` §3.\n"
    )
    parts.append(
        "**Figuras D5** (cuatro PNG generados por `src/utils/build_pooled_bar_charts.py`):\n\n"
        "- `figures/d5_pooled_any_level.png` — Pooled (cualquier L1/L2/L3).\n"
        "- `figures/d5_pooled_l1.png` — L1 estricto.\n"
        "- `figures/d5_pooled_l2.png` — L2 dudoso.\n"
        "- `figures/d5_pooled_l3.png` — L3 valor alto.\n\n"
        "Cada figura es un panel 2×3 (6 paneles, uno por métrica) con "
        "escala-y independiente. Barras coral `#E07A5F` para IA, acero "
        "`#4A90A4` para no-IA.\n"
    )
    parts.append(
        "**Frontera con Discusión:** reportar los deltas y los p-valores. "
        "La lectura *\"la IA por sí sola no genera más engagement, pero "
        "el contexto importa\"* es Discusión (draft.txt L145–L149 ya la "
        "enmarca).\n"
    )
    parts.append("")
    parts.append("### D4 — Medianas por candidato, nivel y métrica\n")
    parts.append(
        "**Vista por cuenta.** A continuación se embebe verbatim el "
        "reporte `d4_per_user_metric_medians.md` (también disponible "
        "como archivo independiente). Cubre las 6 métricas × 2 vistas "
        "(*por post* + *por seguidor*) × 2 poblaciones (Originales + "
        "Retuits), galería de imágenes marcadas, top robust z, y "
        "matrices de correlación Spearman flagged vs. orgánico.\n"
    )
    parts.append("---\n")
    parts.append(d4_body.rstrip() + "\n")
    parts.append("---\n")

    # -----------------------------------------------------------------
    # §R.5 — Cálculo de éxito + Casos específicos (D2 + cards)
    # -----------------------------------------------------------------
    parts.append("## §R.5 (draft.txt L131) — Cálculo de éxito + Casos específicos\n")
    parts.append(
        "**Qué cubrir.** Dos cosas en un mismo sub-bloque del borrador:\n\n"
        "1. **Tablas de cálculo de éxito** — son las columnas "
        "`proporcion_de_exito_<metric>` (raw + `_por_seguidor`) de D4a/b, "
        "ya tabuladas en §R.4 arriba. **No se duplican** acá; basta con "
        "referenciar.\n"
        "2. **Casos específicos de éxito y fallo** — 2–6 casos canónicos "
        "discutidos en prosa cualitativa, alimentados por las tarjetas "
        "que siguen.\n"
    )
    parts.append("**Casos canónicos comprometidos en draft.txt L141 / L147 + transcripción Reunión 2:**\n")
    parts.append(
        "| Cuenta | Tipo | Hallazgo descriptivo | Origen narrativo |\n"
        "|---|---|---|---|\n"
        "| `FabriAlvarado7` | Éxito | Imagen del arresto de Maduro (tweet `1966286626788638780`), `+264.5` likes vs baseline, robust z ≈ +11.5 | draft.txt L141 |\n"
        "| `elifeinzaig` | Éxito | *\"Cara a cara sin excusas\"* (tweet `2009004380276859020`), debate con Laura, robust z ≈ +21.4 | draft.txt L141 |\n"
        "| `RoblesBarrantes` | Contrapunto (fallo) | Memes 4o (tweet `1955848205087739936`) — engagement alto pero deltas negativos vs baseline alto de la cuenta | Transcripción Reunión 2 L113 |\n"
        "| `liberalcr` | Atípico | Tweet `1981801143987503127` único marcado de la cuenta, baseline ≈ 0, robust z ≈ +38.5 (caveat: amplifica por baseline) | top robust z (`d4_per_user_metric_medians.md`) |\n"
    )
    parts.append(
        "**Universo total:** **{n_total_cards} tuits únicos** marcados a "
        "algún nivel (L1 / L2 / L3) — {n_orig_cards} originales (D2a) + "
        "{n_rt_cards} retuits (D2b). Cada uno aparece abajo como una "
        "**tarjeta de caso** con: enlaces a las imágenes, texto del "
        "tuit, veredictos humano + Hive por imagen, comentarios "
        "Oscar/Paulo, las 6 métricas crudas, Δ vs baseline (raw + por "
        "seguidor), y robust z cuando aplica.\n".format(
            n_total_cards=int(d2a["tweet_id"].nunique() + (d2b["retweet_id"].nunique() if not d2b.empty else 0)),
            n_orig_cards=int(d2a["tweet_id"].nunique()),
            n_rt_cards=int(d2b["retweet_id"].nunique()) if not d2b.empty else 0,
        )
    )
    parts.append(
        "**Orden:** cronológico ascendente (la campaña avanza al leer "
        "la tarjeta N+1). Para buscar por cuenta: `Ctrl-F` por "
        "`@username`.\n"
    )
    parts.append(
        "**Frontera con Discusión:** los verdictos por caso "
        "(*\"Fabricio funcionó porque la imagen apela a Nueva "
        "República\"*) **van aquí, en Resultados**. La **teoría "
        "general** (*\"el contenido pertinente apela cultural e "
        "ideológicamente\"*) **va en Discusión** (draft.txt L141, "
        "L519–L523). Ver "
        "`methodology_section_4/section_4_outline_and_results_map.md` "
        "§4.5 + §5.5 para la frontera operativa.\n"
    )
    parts.append("")
    parts.append("### Tarjetas de caso (todas las imágenes marcadas, orden cronológico)\n")
    parts.append(
        _build_case_universe(
            d2a, d2b, d3a, d3c, oscar_decisions, followers,
        )
    )
    parts.append("---\n")

    # -----------------------------------------------------------------
    # §R.6 — Especificación de IA (D8a + D8b)
    # -----------------------------------------------------------------
    parts.append("## §R.6 (draft.txt L133) — Especificación de IA\n")
    parts.append(
        "**Qué cubrir.** Dos tablas, cero ambigüedad narrativa: "
        "la *\"tabla de ceros\"* (D8a) que muestra que ninguna cuenta "
        "declara explícitamente uso de IA, y la *\"tabla de B4s\"* "
        "(D8b) que lista los 5 tuits que mencionan IA genéricamente "
        "(sin atribuir generación de imagen).\n"
    )
    parts.append("### D8a — Tabla de ceros (24 filas)\n")
    parts.append(
        "Tasa de disclosure en 6 poblaciones × 4 bucket-cumulativos. "
        "Wilson 95 % CI por fila — interpretar como *\"el upper bound "
        "es la tasa máxima compatible con cero observados a 95 % de "
        "confianza\"*.\n"
    )
    parts.append(_format_d8a_table(d8a))
    parts.append("")
    parts.append("### D8b — Tabla de B4s (5 tuits)\n")
    parts.append(
        "Tuits con mención **genérica** de IA (`inteligencia "
        "artificial`, `IA`, `chatbot`, etc.) que NO disclosan "
        "generación de imagen. Lectura interpretativa (policy "
        "positioning / promo / acusación) se hace en prosa.\n"
    )
    parts.append(_format_d8b_table(d8b))
    parts.append("")
    parts.append("**Headline numbers (drop-in para prosa):**\n")
    parts.append(
        "- **0 / 475 disclosures en B1** (mención directa de generación con IA). "
        "Wilson 95 % CI **[0, 0.8 %]** sobre el corpus completo.\n"
        "- 0 / 475 en B1+B2 (incluyendo nombres de herramienta: Midjourney, DALL·E, etc.).\n"
        "- 0 / 475 en B1+B2+B3 (incluyendo cues suaves: `prompt`, `render`, `deepfake`).\n"
        "- **5 tuits únicos** en B4 (mención genérica de IA sin atribución a imagen): "
        "2× `Avanza_cr` (policy positioning), 1× `jchidalgo` (promo de chatbot del "
        "Plan de Gobierno), 2× `laurapresi2026` (uno policy positioning sobre IA "
        "de seguridad; uno **anti-disclosure**: acusa a oponentes de usar IA para "
        "falsificar imágenes suyas).\n"
    )
    parts.append("**Prosa drop-in (lista para pegar):**\n")
    parts.append(_format_prose_drop_in_R6())
    parts.append("")
    parts.append(
        "**Cobertura del *\"platform side\"*** (draft.txt L161 — *\"ni "
        "por parte de la plataforma que las distribuyó\"*): mencionar "
        "que el procedimiento de revisión manual de §2 abrió las 737 "
        "imágenes en navegador con el plugin de Hive activo; cualquier "
        "label visible de X(Twitter) habría sido detectado y ninguno "
        "aparece en los datos. No requirió un escaneo NLP separado.\n"
    )
    parts.append(
        "**Frontera con Discusión:** reportar hallazgos numéricos + "
        "ejemplos. **Recomendaciones regulatorias, propuestas de "
        "blockchain y marcaje obligatorio van a Discusión** "
        "(draft.txt L159–L177 ya tiene gran parte escrita).\n"
    )
    parts.append("---\n")

    # -----------------------------------------------------------------
    # Apéndice — Cruces y referencias
    # -----------------------------------------------------------------
    parts.append("## Apéndice — Cruces y referencias\n")
    parts.append("**Cruces entre archivos** (útil cuando algún número no cuadra y se quiere rastrear su origen):\n")
    parts.append(
        "- **D2a ↔ D3a** por `tweet_id` (registro API del tuit padre).\n"
        "- **D2a ↔ D3b** por `image_id` (registro API por imagen).\n"
        "- **D2b ↔ D3c/D3d**: `retweet_id` para el registro API; cruce por imagen vía `media_url` con D3d.\n"
        "- **D1 ↔ D3b** por `image_id` (puntajes AIDE crudos).\n"
        "- **D4a ↔ D3a** por `usuario` ↔ `username` (filtrado a `is_retweet == False`).\n"
        "- **D4b ↔ D3c** por `usuario` ↔ `candidato_retuiteador`.\n"
        "- **D6a ↔ D2a/D2b**: `d6a` originales.`n_L3` ≡ `len(d2a)` (41); `d6a` retuits.`n_L3` ≡ `len(d2b)` (2). Convergencia entre dos rutas de código independientes.\n"
        "- **D6b → D6a**: `sum(d6b.n_total)` ≡ `d6a.total.n_total` (737); `sum(d6b.n_L3)` ≡ `d6a.total.n_L3` (43).\n"
        "- **D7 ← side-project**: copia byte-equivalente de `_side_projects/aide_vs_hive/results/confusion_matrices.csv`.\n"
        "- **D8a/D8b ← side-project**: copias de `_side_projects/spanish_ai_disclosure_scan/results/`.\n"
    )
    parts.append("**Bridge docs metodológicos** (para entender el *cómo* detrás del *qué* en cada sub-bloque):\n")
    parts.append(
        "- `data/meeting_overviews/methodology_section_3/section_3_outline_and_results_map.md` — métodos cuantitativos (D4, D5, MWU, Wilcoxon, robust z, Spearman).\n"
        "- `data/meeting_overviews/methodology_section_3/coherence_audit.md` §3 — reconciliación D5 +39.5 likes vs MWU p = 0.343.\n"
        "- `data/meeting_overviews/methodology_section_4/section_4_outline_and_results_map.md` — método cualitativo de casos (lente *contenido pertinente*, dimensiones qué/dónde/cuándo/público, universo de 43 imágenes).\n"
        "- `data/meeting_overviews/methodology_section_5/section_5_outline_and_results_map.md` — método NLP / regex (4 buckets B1–B4, vocabulario maximal).\n"
    )
    parts.append("**Side-projects canónicos** (productores de D7 y D8):\n")
    parts.append(
        "- `_side_projects/aide_vs_hive/` — AIDE-vs-Hive characterization. Reportes interpretativos vivos.\n"
        "- `_side_projects/spanish_ai_disclosure_scan/` — escaneo regex sobre los 475 tuits. Reporte interpretativo en `reports/spanish_ai_disclosure_scan.md`.\n"
    )
    parts.append("**README de `data/deliverables/`** — catálogo por número de deliverable (`Ctrl-F` por D1, D2a, …). Este hub es el documento de **redacción**; el README es el documento de **catalogación**.\n")

    return "\n".join(parts) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_master = pd.read_csv(IMAGE_MASTER_CSV)
    tweet_master = pd.read_csv(TWEET_MASTER_CSV)
    tweet_master["tweet_id"] = tweet_master["tweet_id"].astype(str)
    oscar_decisions = _load_oscar_decisions()

    flagged_inclusive = pd.read_csv(FLAGGED_INCLUSIVE_CSV)

    d1 = build_d1(image_master)
    d3a = build_d3a(RAW_TWEETS_DIR)
    d3b = build_d3b(d3a)
    d3c = build_d3c(RETWEETS_DIR)
    d3d = build_d3d(d3c)
    d2a = build_d2a_originals(image_master, tweet_master)
    d2b = build_d2b_retweets(image_master, d3d)
    followers = _load_followers()
    d4a, flagged_images_a = build_d4_originals(oscar_decisions, d3a, followers)
    d4b, flagged_images_b = build_d4_retweets(oscar_decisions, d3c, followers)
    d4_report = build_d4_report(
        d4a, d4b, flagged_images_a, flagged_images_b,
        oscar_decisions, d3a, d3c,
    )
    d5 = build_d5_pooled_comparison(oscar_decisions, d3a)
    d6a = build_d6a_l_summary_overall(oscar_decisions)
    d6b = build_d6b_l_summary_per_account(oscar_decisions)
    d7 = copy_d7_aide_confusion()
    d8a = copy_d8a_disclosure_rate()
    d8b = build_d8b_b4_mentions()
    resultados_hub = build_resultados_hub(
        d4_report, d5, d6a, d6b, d7, d8a, d8b,
        d2a, d2b, oscar_decisions, d3a, d3c, followers,
    )

    d1.to_csv(D1_PATH, index=False, encoding="utf-8-sig")
    d2a.to_csv(D2A_PATH, index=False, encoding="utf-8-sig")
    d2b.to_csv(D2B_PATH, index=False, encoding="utf-8-sig")
    d3a.to_csv(D3A_PATH, index=False, encoding="utf-8-sig")
    d3b.to_csv(D3B_PATH, index=False, encoding="utf-8-sig")
    d3c.to_csv(D3C_PATH, index=False, encoding="utf-8-sig")
    d3d.to_csv(D3D_PATH, index=False, encoding="utf-8-sig")
    d4a.to_csv(D4A_PATH, index=False, encoding="utf-8-sig")
    d4b.to_csv(D4B_PATH, index=False, encoding="utf-8-sig")
    D4_REPORT_PATH.write_text(d4_report, encoding="utf-8")
    d5.to_csv(D5_PATH, index=False, encoding="utf-8-sig")
    d6a.to_csv(D6A_PATH, index=False, encoding="utf-8-sig")
    d6b.to_csv(D6B_PATH, index=False, encoding="utf-8-sig")
    d7.to_csv(D7_PATH, index=False, encoding="utf-8-sig")
    d8a.to_csv(D8A_PATH, index=False, encoding="utf-8-sig")
    d8b.to_csv(D8B_PATH, index=False, encoding="utf-8-sig")
    RESULTADOS_HUB_PATH.write_text(resultados_hub, encoding="utf-8")

    # The single d2_flagged_analysis.csv is superseded by d2a + d2b.
    if D2_LEGACY_PATH.exists():
        D2_LEGACY_PATH.unlink()

    print()
    print("Wrote:")
    for path, df in [
        (D1_PATH, d1), (D2A_PATH, d2a), (D2B_PATH, d2b),
        (D3A_PATH, d3a), (D3B_PATH, d3b),
        (D3C_PATH, d3c), (D3D_PATH, d3d),
        (D4A_PATH, d4a), (D4B_PATH, d4b),
        (D5_PATH, d5),
        (D6A_PATH, d6a), (D6B_PATH, d6b),
        (D7_PATH, d7),
        (D8A_PATH, d8a), (D8B_PATH, d8b),
    ]:
        print(f"  {path.relative_to(PROJECT_ROOT)}  ({len(df):>5} rows, {len(df.columns):>3} cols)")
    print(f"  {D4_REPORT_PATH.relative_to(PROJECT_ROOT)}  (markdown report, "
          f"{len(d4_report.splitlines())} lines)")
    print(f"  {RESULTADOS_HUB_PATH.relative_to(PROJECT_ROOT)}  (markdown hub, "
          f"{len(resultados_hub.splitlines())} lines)")

    print()
    print("Sanity checks:")

    assert len(d1) == 737, f"D1 expected 737 rows, got {len(d1)}"
    score_cols = ["score_aide_GenImage", "score_aide_ProGAN", "score_aide_SD14"]
    assert d1[score_cols].notna().all().all(), "D1 has NaN in AIDE score columns"
    print(f"  [OK] D1: 737 rows, no NaN in score columns")

    assert len(d2a) + len(d2b) == len(flagged_inclusive), (
        f"D2a+D2b row count {len(d2a)}+{len(d2b)}={len(d2a)+len(d2b)} != "
        f"flagged_si_dudoso_valoralto {len(flagged_inclusive)}"
    )
    print(f"  [OK] D2a originals ({len(d2a)}) + D2b retweets ({len(d2b)}) "
          f"= {len(d2a) + len(d2b)} == flagged_si_dudoso_valoralto.csv")
    # Retweet metrics must come from D3d (source-tweet engagement) — never
    # zero unless the source tweet genuinely has zero engagement.
    if not d2b.empty:
        assert d2b["impression_count"].notna().all(), \
            "D2b has NaN impression_count — media_url join failed"
        assert (d2b["impression_count"] > 0).all(), \
            "D2b has zero impression_count — likely the engagement was sourced from the retweet row, not from D3d"
        print(f"  [OK] D2b: every row has nonzero impression_count (sourced from D3d, not zeroed retweet rows)")

    d3a_imgsum = int(d3a["image_count"].fillna(0).sum())
    assert len(d3b) == d3a_imgsum, f"D3b rows {len(d3b)} != sum(d3a.image_count) {d3a_imgsum}"
    print(f"  [OK] D3b: {len(d3b)} rows == sum of d3a.image_count")

    d3c_imgsum = int(d3c["image_count"].fillna(0).sum())
    assert len(d3d) == d3c_imgsum, f"D3d rows {len(d3d)} != sum(d3c.image_count) {d3c_imgsum}"
    print(f"  [OK] D3d: {len(d3d)} rows == sum of d3c.image_count")

    # Every master_image_level.image_id should appear in D3b. D3b may have
    # *more* image rows than master because master excludes failed downloads
    # (1 known failure: ClaudiaDobles_1984104258829218217_2).
    master_image = pd.read_csv(MASTER_DIR / "master_image_level.csv", usecols=["image_id"])
    master_ids = set(master_image["image_id"].astype(str))
    d3b_ids = set(d3b["image_id"].astype(str))
    missing_from_d3b = master_ids - d3b_ids
    extra_in_d3b = d3b_ids - master_ids
    assert not missing_from_d3b, f"master image_ids missing from D3b: {missing_from_d3b}"
    print(f"  [OK] D3b ⊇ master_image_level (737 master ids all present; {len(extra_in_d3b)} extra in D3b: {extra_in_d3b or 'none'})")

    # D4 sanity checks.
    expected_levels = set(LEVEL_LABELS.values())
    for name, d4 in [("D4a", d4a), ("D4b", d4b)]:
        assert set(d4["level"]) == expected_levels, \
            f"{name} level set mismatch: {set(d4['level'])} != {expected_levels}"
        # Each user has exactly 3 rows (one per level) in the wide schema.
        per_user = d4.groupby("usuario").size()
        assert (per_user == 3).all(), \
            f"{name} not 3 rows per user — got: {per_user.to_dict()}"
        # sin_ia constant per user (baseline doesn't move with level).
        const_check = d4.groupby("usuario")["sin_ia"].nunique()
        assert (const_check == 1).all(), \
            f"{name} sin_ia varies within a user: {const_check[const_check > 1].to_dict()}"
        # All six metric columns must exist — raw + por_seguidor.
        for m in ENGAGEMENT_COLS:
            for prefix in ("mediana_ia_", "mediana_sin_ia_", "proporcion_de_exito_"):
                col = f"{prefix}{m}"
                assert col in d4.columns, f"{name} missing column {col}"
                ps_col = f"{prefix}{m}_por_seguidor"
                assert ps_col in d4.columns, f"{name} missing column {ps_col}"
        # proporcion_de_exito_<m> = mediana_ia_<m> − mediana_sin_ia_<m>
        # Raw rounded to 1 decimal → tolerance 0.05.
        # Por seguidor rounded to 6 decimals → tolerance 1e-5.
        for m in ENGAGEMENT_COLS:
            mask = d4[[f"mediana_ia_{m}", f"mediana_sin_ia_{m}"]].notna().all(axis=1)
            sub = d4.loc[mask]
            diffs = (sub[f"mediana_ia_{m}"] - sub[f"mediana_sin_ia_{m}"]) - sub[f"proporcion_de_exito_{m}"]
            assert (diffs.abs() < 0.05).all(), \
                f"{name} proporcion_de_exito_{m} disagrees with ia − sin_ia"
            mask_ps = d4[[f"mediana_ia_{m}_por_seguidor", f"mediana_sin_ia_{m}_por_seguidor"]].notna().all(axis=1)
            sub_ps = d4.loc[mask_ps]
            if len(sub_ps):
                diffs_ps = (
                    sub_ps[f"mediana_ia_{m}_por_seguidor"]
                    - sub_ps[f"mediana_sin_ia_{m}_por_seguidor"]
                ) - sub_ps[f"proporcion_de_exito_{m}_por_seguidor"]
                assert (diffs_ps.abs() < 1e-5).all(), (
                    f"{name} proporcion_de_exito_{m}_por_seguidor disagrees with ia − sin_ia "
                    f"(max diff {diffs_ps.abs().max():.2e})"
                )
        # Cross-form consistency: mediana_ia_<m>_por_seguidor × F ≈ mediana_ia_<m>.
        # Validates that "rate first, then median" agrees with "median, then divide"
        # for constant-F (single user). Tolerance 0.5 because mediana_ia (raw) is
        # rounded to 1 decimal so reconstruction can drift by up to ±0.5. The
        # por_seguidor side is rounded to 6 decimals so its own rounding
        # contribution is ~10⁻⁶ × F (≤ 0.06 at the largest follower count,
        # F = 52,800 for ClaudiaDobles — well within the 0.5 budget).
        for m in ENGAGEMENT_COLS:
            mask = d4[f"mediana_ia_{m}_por_seguidor"].notna() & d4[f"mediana_ia_{m}"].notna()
            sub = d4.loc[mask]
            if len(sub):
                F_series = sub["usuario"].map(followers)
                assert F_series.notna().all(), \
                    f"{name} unknown follower count for users: {sub.loc[F_series.isna(), 'usuario'].unique()}"
                reconstructed = sub[f"mediana_ia_{m}_por_seguidor"] * F_series
                diffs = (reconstructed - sub[f"mediana_ia_{m}"]).abs()
                assert (diffs < 0.5).all(), (
                    f"{name} cross-form check failed for {m}: "
                    f"mediana_ia_por_seguidor × F differs from mediana_ia by up to {diffs.max():.3f}"
                )
        print(f"  [OK] {name}: {len(d4)} rows, {d4['usuario'].nunique()} users (3 levels each), "
              f"baseline constant per user, proporcion_de_exito_<m> = ia − sin_ia "
              f"for all 6 metrics (raw + por_seguidor), cross-form consistency holds")

    # D4b should reflect the known retweet flag scarcity: only 2 retweets are
    # flagged at L3 across the entire corpus (one per account).
    rt_l3_flagged = d4b[(d4b["level"] == "L3 Valor Alto")
                       & (d4b["con_ia"] > 0)]
    assert len(rt_l3_flagged) == 2, \
        f"D4b expected exactly 2 users with L3 flagged retweet, got {len(rt_l3_flagged)}"
    print(f"  [OK] D4b: {len(rt_l3_flagged)} accounts with at least one L3-flagged retweet "
          f"({sorted(rt_l3_flagged['usuario'].tolist())})")

    # `enlaces_imagenes` should be populated wherever con_ia > 0.
    for name, d4 in [("D4a", d4a), ("D4b", d4b)]:
        flagged_rows = d4[d4["con_ia"] > 0]
        empty_links = flagged_rows[flagged_rows["enlaces_imagenes"].fillna("") == ""]
        assert empty_links.empty, \
            f"{name} has {len(empty_links)} flagged rows with empty enlaces_imagenes"
        print(f"  [OK] {name}: every flagged row carries a HYPERLINK formula in enlaces_imagenes "
              f"({len(flagged_rows)} flagged rows)")

    # D5 sanity: 24 rows = 4 levels × 6 metrics; expected levels present;
    # n_ia + n_no_ia constant within level (every original tweet falls in
    # exactly one group); delta arithmetic holds.
    expected_d5_levels = {
        "Pooled (cualquier L1/L2/L3)", "L1 Sí/Estricto",
        "L2 Dudoso", "L3 Valor Alto",
    }
    assert len(d5) == 24, f"D5 expected 24 rows (4 levels × 6 metrics), got {len(d5)}"
    assert set(d5["level"]) == expected_d5_levels, \
        f"D5 level set mismatch: {set(d5['level'])} != {expected_d5_levels}"
    originals_count = int((d3a["is_retweet"] == False).sum())
    for lvl in expected_d5_levels:
        sub = d5[d5["level"] == lvl]
        totals = (sub["n_ia"] + sub["n_no_ia"]).unique()
        assert len(totals) == 1, f"D5 {lvl}: n_ia + n_no_ia varies across metrics: {totals}"
        assert int(totals[0]) == originals_count, (
            f"D5 {lvl}: n_ia + n_no_ia = {int(totals[0])} != originals {originals_count}"
        )
    mask = d5[["mediana_ia", "mediana_no_ia"]].notna().all(axis=1)
    sub = d5.loc[mask]
    diffs = (sub["mediana_ia"] - sub["mediana_no_ia"]) - sub["delta"]
    assert (diffs.abs() < 0.05).all(), (
        f"D5 delta arithmetic disagrees with ia − no_ia (max diff {diffs.abs().max():.3f})"
    )
    print(f"  [OK] D5: {len(d5)} rows (4 levels × 6 metrics), "
          f"n_ia + n_no_ia = {originals_count} per level, delta arithmetic holds")

    # D5 independent reproduction: reproduce L2 originals like_count median
    # using a different code path than `build_d5_pooled_comparison`. If the
    # builder introduces a latent bug (e.g., merge dtype mismatch, rollup
    # order, NaN handling), the two paths diverge and the assert fires.
    # See coherence_audit.md §6.1 #1 for context.
    dec_repro = oscar_decisions[oscar_decisions["is_retweet"] == False].copy()
    dec_repro = _add_image_flags(dec_repro)
    dec_repro["tweet_id"] = dec_repro["image_id"].astype(str).str.rsplit("_", n=1).str[0]
    l2_flagged_tweet_ids = set(
        dec_repro.groupby("tweet_id")["L2"].any().pipe(lambda s: s[s].index)
    )
    originals_repro = d3a[d3a["is_retweet"] == False].copy()
    originals_repro["is_l2_repro"] = originals_repro["tweet_id"].astype(str).isin(l2_flagged_tweet_ids)
    repro_ia = originals_repro.loc[originals_repro["is_l2_repro"], "like_count"].median()
    repro_no = originals_repro.loc[~originals_repro["is_l2_repro"], "like_count"].median()
    d5_l2_likes = d5[(d5["level"] == "L2 Dudoso") & (d5["metric"] == "like_count")].iloc[0]
    assert abs(round(float(repro_ia), 1) - float(d5_l2_likes["mediana_ia"])) < 0.05, (
        f"D5 reproduction mismatch: independent calc mediana_ia = {repro_ia} "
        f"vs D5 {d5_l2_likes['mediana_ia']}"
    )
    assert abs(round(float(repro_no), 1) - float(d5_l2_likes["mediana_no_ia"])) < 0.05, (
        f"D5 reproduction mismatch: independent calc mediana_no_ia = {repro_no} "
        f"vs D5 {d5_l2_likes['mediana_no_ia']}"
    )
    print(f"  [OK] D5: L2 originals like_count reproduces independently "
          f"(mediana_ia = {d5_l2_likes['mediana_ia']}, "
          f"mediana_no_ia = {d5_l2_likes['mediana_no_ia']})")

    # D4 markdown ↔ CSV consistency spot-check. After the report is written,
    # read it back and regex-extract FabriAlvarado7's L3 like_count row from
    # the unified table; assert mediana_ia + Eficiencia de interacción match
    # the CSV values. Catches any drift introduced by
    # `_format_d4_summary_table_unified` (format-string change, column
    # reorder, sign flip). See coherence_audit.md §6.1 #2 for context.
    report_text = D4_REPORT_PATH.read_text(encoding="utf-8")
    # Schema: | usuario | con_ia | sin_ia | mediana_ia | mediana_sin_ia |
    #         | Cálculo de éxito | Eficiencia de interacción |
    md_row_pat = re.compile(
        r"FabriAlvarado7\s*\|\s*2\s*\|\s*16\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        r"\s*([+\-\d.]+)\s*\|\s*([+\-\d.]+)\s*\|"
    )
    md_match = md_row_pat.search(report_text)
    assert md_match, "FabriAlvarado7 L3 like_count row not found in d4 markdown report"
    md_mediana_ia = float(md_match.group(1))
    md_eficiencia_x1000 = float(md_match.group(4))
    csv_row = d4a[
        (d4a["usuario"] == "FabriAlvarado7") & (d4a["level"] == "L3 Valor Alto")
    ].iloc[0]
    assert abs(md_mediana_ia - float(csv_row["mediana_ia_like_count"])) < 0.05, (
        f"D4 report ↔ CSV mismatch: report mediana_ia = {md_mediana_ia}, "
        f"CSV says {csv_row['mediana_ia_like_count']}"
    )
    # Displayed eficiencia = csv coefficient × 1000 (per 1000 followers rescaling).
    csv_eficiencia_x1000 = float(csv_row["proporcion_de_exito_like_count_por_seguidor"]) * 1000
    assert abs(md_eficiencia_x1000 - csv_eficiencia_x1000) < 0.01, (
        f"D4 report ↔ CSV mismatch: report eficiencia (×1000) = {md_eficiencia_x1000}, "
        f"CSV proporcion_de_exito_*_por_seguidor × 1000 = {csv_eficiencia_x1000}"
    )
    print(f"  [OK] D4 report ↔ CSV: FabriAlvarado7 L3 like_count "
          f"mediana_ia = {md_mediana_ia}, eficiencia de interacción = "
          f"{md_eficiencia_x1000:+.2f} por 1000 seg. (matches CSV × 1000)")

    # D6a sanity: 3 rows (originales / retuits / total), L3 ⊇ L2 ⊇ L1
    # within each row, and the totals match the canonical image counts.
    assert list(d6a["population"]) == ["originales", "retuits", "total"], (
        f"D6a row order mismatch: {list(d6a['population'])}"
    )
    row_orig = d6a.iloc[0]
    row_rt = d6a.iloc[1]
    row_tot = d6a.iloc[2]
    assert int(row_orig["n_total"]) == 671, (
        f"D6a originales n_total {int(row_orig['n_total'])} != 671 (canonical)"
    )
    assert int(row_rt["n_total"]) == 66, (
        f"D6a retuits n_total {int(row_rt['n_total'])} != 66 (canonical)"
    )
    assert int(row_tot["n_total"]) == 737, (
        f"D6a total n_total {int(row_tot['n_total'])} != 737 (canonical)"
    )
    for _, r in d6a.iterrows():
        assert int(r["n_L1"]) <= int(r["n_L2"]) <= int(r["n_L3"]), (
            f"D6a {r['population']}: L1/L2/L3 not cumulative: "
            f"L1={r['n_L1']} L2={r['n_L2']} L3={r['n_L3']}"
        )
        assert int(r["n_sin_L"]) == int(r["n_total"]) - int(r["n_L3"]), (
            f"D6a {r['population']}: n_sin_L != n_total − n_L3"
        )
    # Cross-check D6a originales n_L3 against D2a (independent code path:
    # D2a filters image_master_scores by the equivalent _flagged_image_master
    # rule + is_retweet == False; D6a runs _add_image_flags on
    # oscar_decisions then filters by is_retweet == False. Both should
    # yield the same L3-originals count).
    assert int(row_orig["n_L3"]) == len(d2a), (
        f"D6a originales n_L3 {int(row_orig['n_L3'])} != D2a row count {len(d2a)}"
    )
    # Same for retweets: D2b row count == D6a retuits n_L3.
    assert int(row_rt["n_L3"]) == len(d2b), (
        f"D6a retuits n_L3 {int(row_rt['n_L3'])} != D2b row count {len(d2b)}"
    )
    print(f"  [OK] D6a: 3 rows; totals 671/66/737; L1⊆L2⊆L3 per row; "
          f"originales n_L3 = {int(row_orig['n_L3'])} matches D2a; "
          f"retuits n_L3 = {int(row_rt['n_L3'])} matches D2b")

    # D6b sanity: per-account totals sum to the overall total; per-account
    # L3 sum equals the overall L3; cumulative within each account.
    assert int(d6b["n_total"].sum()) == int(row_tot["n_total"]), (
        f"D6b sum(n_total) {int(d6b['n_total'].sum())} != "
        f"D6a total n_total {int(row_tot['n_total'])}"
    )
    assert int(d6b["n_L3"].sum()) == int(row_tot["n_L3"]), (
        f"D6b sum(n_L3) {int(d6b['n_L3'].sum())} != "
        f"D6a total n_L3 {int(row_tot['n_L3'])}"
    )
    for _, r in d6b.iterrows():
        assert int(r["n_L1"]) <= int(r["n_L2"]) <= int(r["n_L3"]), (
            f"D6b {r['username']}: L1/L2/L3 not cumulative"
        )
        assert r["account_type"] in {"candidato", "partido", "desconocido"}, (
            f"D6b {r['username']}: unexpected account_type {r['account_type']}"
        )
    print(f"  [OK] D6b: {len(d6b)} accounts; sum(n_total) = {int(d6b['n_total'].sum())} "
          f"matches D6a; sum(n_L3) = {int(d6b['n_L3'].sum())} matches D6a")

    # D7 byte equivalence: identical row count + identical column set to
    # the side-project source. Detects accidental schema drift during the
    # copy step.
    d7_source = pd.read_csv(AIDE_CONFUSION_SOURCE)
    assert len(d7) == len(d7_source), (
        f"D7 row count {len(d7)} != source {len(d7_source)}"
    )
    assert list(d7.columns) == list(d7_source.columns), (
        f"D7 column drift: {list(d7.columns)} vs source {list(d7_source.columns)}"
    )
    assert len(d7) == 9, f"D7 expected 9 rows (3 GT × 3 checkpoints), got {len(d7)}"
    print(f"  [OK] D7: {len(d7)} rows × {len(d7.columns)} cols; byte-equivalent to "
          f"_side_projects/aide_vs_hive/results/confusion_matrices.csv")

    # D8a row count + column shape check against side-project source.
    d8a_source = pd.read_csv(DISCLOSURE_RATE_SOURCE)
    assert len(d8a) == len(d8a_source), (
        f"D8a row count {len(d8a)} != source {len(d8a_source)}"
    )
    assert len(d8a) == 24, (
        f"D8a expected 24 rows (6 populations × 4 bucket-cumulative), got {len(d8a)}"
    )
    # B1_only across all populations should be 0 (the "tabla de ceros"
    # invariant — if this fires, either the scan or this deliverable
    # builder regressed).
    b1_only_hits = int(d8a[d8a["bucket_group"] == "B1_only"]["numerator"].sum())
    assert b1_only_hits == 0, (
        f"D8a invariant violated: B1_only hits = {b1_only_hits} (expected 0). "
        "Re-check disclosure_patterns.py and re-run the side-project."
    )
    print(f"  [OK] D8a: {len(d8a)} rows (6 populations × 4 bucket-cumulative); "
          f"B1_only numerator = 0 across all populations (the 'tabla de ceros')")

    # D8b: filtered B4 mentions, dedup by primary_id. Expected ~5 tweets.
    # Each row must have bucket == "B4_generic" and a non-empty snippet.
    assert (d8b["bucket"] == "B4_generic").all(), (
        f"D8b has rows with bucket != B4_generic: {d8b['bucket'].unique()}"
    )
    assert d8b["primary_id"].is_unique, (
        f"D8b primary_id not unique after dedup: {d8b['primary_id'].duplicated().sum()} duplicates"
    )
    assert d8b["snippet"].notna().all() and (d8b["snippet"].str.len() > 0).all(), (
        "D8b has rows with empty snippet"
    )
    print(f"  [OK] D8b: {len(d8b)} unique B4 tweets; all rows have bucket=B4_generic "
          f"+ non-empty snippet")

    # Resultados hub: structural markers + case-card count.
    hub_text = RESULTADOS_HUB_PATH.read_text(encoding="utf-8")
    for marker in ("§R.0", "§R.1", "§R.2", "§R.3", "§R.4", "§R.5", "§R.6"):
        assert marker in hub_text, f"Hub missing section marker {marker}"
    print(f"  [OK] resultados_hub.md: contains all 6 sub-block markers (§R.0–§R.6)")

    # One case card per unique flagged tweet: 34 originales + 2 retuits = 36.
    n_unique_d2a = int(d2a["tweet_id"].astype(str).nunique())
    n_unique_d2b = int(d2b["retweet_id"].astype(str).nunique()) if not d2b.empty else 0
    expected_cards = n_unique_d2a + n_unique_d2b
    n_case_headers = hub_text.count("### Caso ")
    assert n_case_headers == expected_cards, (
        f"Hub case-card count {n_case_headers} != expected {expected_cards} "
        f"(D2a unique tweets {n_unique_d2a} + D2b unique retweets {n_unique_d2b})"
    )
    print(f"  [OK] resultados_hub.md: {n_case_headers} case cards "
          f"({n_unique_d2a} originales + {n_unique_d2b} retuits)")


if __name__ == "__main__":
    main()
