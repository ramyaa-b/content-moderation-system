#!/usr/bin/env python3
"""
scripts/hasoc_dataprep.py

Purpose:
- Combine raw HASOC files (any CSVs you provide) and produce a clean,
  minimal 4-class training CSV plus a meta CSV for debugging.
- Designed for HASOC datasets where labels may be in different columns
  (task1/task2, subtask_a/subtask_b, label, hate/offn/prfn etc.)

Usage:
    python scripts/hasoc_dataprep.py --raw-dir datasets/raw
    python scripts/hasoc_dataprep.py --input-files datasets/raw/hasoc_2020_en_train.csv datasets/raw/train.csv --no-clean --oversample

Output:
- datasets/processed_4class/train.csv  (columns: text,label,language)
- datasets/processed_4class/train_meta.csv (adds orig_task* columns + source)
"""

import argparse
from pathlib import Path
import pandas as pd
import logging
from collections import Counter
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hasoc_dataprep")

# ---------- Helpers to detect columns ----------
COMMON_TEXT_COLS = ["text", "tweet", "content", "tweet_text"]
COMMON_TASK1_COLS = ["task1", "subtask_a", "subtaskA", "A", "task_a"]
COMMON_TASK2_COLS = ["task2", "subtask_b", "subtaskB", "B", "task_b"]
COMMON_LABEL_COLS = ["label", "labels", "class", "hate_label"]

def detect_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    low = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

# ---------- reading input ----------
def read_inputs(paths):
    """
    Accepts a list of file paths. Reads each CSV, attaches a `source` column
    with filename, and returns concatenated DataFrame.
    """
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            logger.warning(f"Input file not found: {p}")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logger.error(f"Failed to read {p}: {e}")
            continue
        df["__source_file"] = p.name
        frames.append(df)
        logger.info(f"Read {p} -> {len(df)} rows")
    if not frames:
        raise FileNotFoundError("No valid input CSVs found.")
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined dataset size: {len(combined)}")
    return combined

# ---------- robust mapping to 4 classes ----------
def map_hasoc_to_4(df):
    """
    Create/standardize the following columns:
      - text  (ensure exists)
      - orig_task1, orig_task2  (if present)
      - label (one of HATE, PRFN, OFFN, NONE, UNKNOWN)
      - language (if present)
      - source (filename)
    """
    df = df.copy()

    # detect text column
    text_col = detect_column(df, COMMON_TEXT_COLS)
    if text_col is None:
        # try to find any long text-like column
        cand = None
        for c in df.columns:
            if df[c].dtype == object and df[c].astype(str).str.len().median() > 10:
                cand = c
                break
        if cand is None:
            raise ValueError("Could not find a text column. Please ensure one exists.")
        text_col = cand
        logger.info(f"Inferred text column as '{text_col}'")

    # standardize text column name
    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].astype(str).str.strip()

    # detect label columns
    t1 = detect_column(df, COMMON_TASK1_COLS)
    t2 = detect_column(df, COMMON_TASK2_COLS)
    lab = detect_column(df, COMMON_LABEL_COLS)

    # fill orig task columns
    if t1:
        df["orig_task1"] = df[t1].astype(str)
    else:
        df["orig_task1"] = None
    if t2:
        df["orig_task2"] = df[t2].astype(str)
    else:
        df["orig_task2"] = None
    if lab:
        df["orig_label"] = df[lab].astype(str)
    else:
        df["orig_label"] = None

    # try to map using the most granular column available (orig_label > orig_task2 > orig_task1)
    def map_row(r):
        # prefer orig_label if it is one of expected tokens
        for src in ["orig_label", "orig_task2", "orig_task1"]:
            val = r.get(src)
            if pd.isna(val) or val is None:
                continue
            s = str(val).strip().upper()
            # handle common synonyms
            if s in {"HATE", "HATE_SPEECH", "HATESPEECH", "H"}:
                return "HATE"
            if s in {"PRFN", "PROFANITY", "PROFANICAL", "P", "PRF"}:
                return "PRFN"
            if s in {"OFFN", "OFFENSE", "OFFENCE", "OFFENSIVE", "OFF"}:
                return "OFFN"
            if s in {"NONE", "NOT", "NOT_ABUSIVE", "NO", "N"}:
                return "NONE"
            # sometimes multi-label or numeric codes appear: check numeric codes
            if s.isdigit():
                # common HASOC encodings vary; don't attempt to auto-interpret digits here
                # return UNKNOWN for digits so user can inspect and expand mapping
                return "UNKNOWN"
            # sometimes labels come like "HOF" or "HOF?" meaning hateful/offensive
            if s in {"HOF", "HOF?","H/O","H/OF"}:
                return "HATE"
            # small heuristics: offensive markers
            if "HATE" in s:
                return "HATE"
            if "PROF" in s or "PRFN" in s:
                return "PRFN"
            if "OFF" in s:
                return "OFFN"
            if "NOT" in s or "NONE" in s:
                return "NONE"
        # if nothing matched
        return "UNKNOWN"

    df["label"] = df.apply(map_row, axis=1)

    # bring language into standardized column if present
    lang_col = None
    for c in ["language", "lang", "tweet_lang"]:
        if c in df.columns:
            lang_col = c
            break
    if lang_col:
        df["language"] = df[lang_col].astype(str)
    else:
        df["language"] = None

    # add a simpler source column for debugging
    df["source"] = df.get("__source_file", None)

    # keep only relevant cols plus audit cols
    keep = ["text", "label", "language", "orig_task1", "orig_task2", "orig_label", "source"]
    for k in keep:
        if k not in df.columns:
            df[k] = None

    return df[keep]

# ---------- light, label-aware cleaning ----------
def light_clean(df, min_words=3, max_words=200, dedupe_within_label=True):
    df = df.copy()
    logger.info("Performing light cleaning (exact dedupe + short/long filters)")
    before = Counter(df["label"].fillna("MISSING").tolist())
    logger.info(f"Counts before clean: {dict(before)}")

    df["word_count"] = df["text"].astype(str).str.split().str.len()

    # exact duplicates removal
    if dedupe_within_label:
        rows_before = len(df)
        df = df[~df.duplicated(subset=["text", "label"], keep="first")]
        logger.info(f"Removed {rows_before - len(df)} exact duplicates (label-aware)")
    else:
        rows_before = len(df)
        df = df[~df.duplicated(subset=["text"], keep="first")]
        logger.info(f"Removed {rows_before - len(df)} exact duplicates (global)")

    # short/long filters
    short_mask = df["word_count"] < min_words
    long_mask = df["word_count"] > max_words
    if short_mask.any():
        logger.info(f"Short removal counts: {Counter(df.loc[short_mask, 'label'])}")
    if long_mask.any():
        logger.info(f"Long removal counts: {Counter(df.loc[long_mask, 'label'])}")

    df = df[~short_mask & ~long_mask]
    after = Counter(df["label"].fillna("MISSING").tolist())
    logger.info(f"Counts after clean: {dict(after)}")
    df = df.drop(columns=["word_count"])
    return df

# ---------- oversample minorities ----------
def oversample(df, ignore_labels=None, target_count=None, seed=42):
    if ignore_labels is None:
        ignore_labels = ["UNKNOWN"]
    counts = Counter(df["label"].tolist())
    logger.info(f"Counts before oversample: {dict(counts)}")
    classes = [c for c in counts.keys() if c not in ignore_labels]
    if not classes:
        logger.warning("No classes available for oversampling (after ignoring). Returning original df.")
        return df
    if target_count is None:
        # aim to bring all classes to the max class count
        target_count = max(counts[c] for c in classes)
    parts = []
    for cls in classes:
        subset = df[df["label"] == cls]
        n = len(subset)
        if n >= target_count:
            parts.append(subset)
        else:
            extra = subset.sample(n=target_count - n, replace=True, random_state=seed)
            parts.append(pd.concat([subset, extra], ignore_index=True))
    # keep ignored labels, e.g., UNKNOWN
    for ig in ignore_labels:
        if ig in counts:
            parts.append(df[df["label"] == ig])
    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(f"Counts after oversample: {dict(Counter(result['label'].tolist()))}")
    return result

# ---------- save minimal + meta ----------
def save_outputs(df, out_dir=Path("datasets/processed_4class")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    minimal_cols = ["text", "label"]
    if "language" in df.columns:
        minimal_cols.append("language")
    df[minimal_cols].to_csv(out_dir / "train.csv", index=False)
    meta_cols = minimal_cols + [c for c in ["orig_task1", "orig_task2", "orig_label", "source"] if c in df.columns]
    df[meta_cols].to_csv(out_dir / "train_meta.csv", index=False)
    logger.info(f"Saved minimal train.csv and train_meta.csv in {out_dir}")

# ---------- main ----------
def main(args):
    # build input file list
    if args.input_files:
        files = args.input_files
    else:
        raw_dir = Path(args.raw_dir)
        files = sorted([str(p) for p in raw_dir.glob("*.csv")])
        # optionally filter to common HASOC filenames
        if args.filter_hasoc:
            files = [f for f in files if ("hasoc" in Path(f).name.lower()) or ("train" in Path(f).name.lower())]
    if not files:
        raise FileNotFoundError("No input files found. Provide --input-files or set --raw-dir with CSVs.")

    combined = read_inputs(files)
    mapped = map_hasoc_to_4(combined)

    # optionally run light cleaning
    if args.clean:
        cleaned = light_clean(mapped, min_words=args.min_words, max_words=args.max_words,
                              dedupe_within_label=not args.global_dedupe)
    else:
        cleaned = mapped
        logger.info("Cleaning disabled (--no-clean)")

    # optionally oversample
    if args.oversample:
        final = oversample(cleaned, ignore_labels=args.ignore_labels.split(","))
    else:
        final = cleaned

    save_outputs(final, out_dir=Path(args.out_dir))
    # print counts for user quick-check
    print("Final label counts:\n", final["label"].value_counts())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=str, default="datasets/raw",
                   help="Directory containing raw CSVs (default: datasets/raw)")
    p.add_argument("--input-files", nargs="*", help="List input CSV files explicitly (overrides --raw-dir)")
    p.add_argument("--out-dir", type=str, default="datasets/processed_4class", dest="out_dir")
    p.add_argument("--no-clean", action="store_false", dest="clean", help="Disable light cleaning")
    p.add_argument("--clean", action="store_true", dest="clean", help="Enable light cleaning (default)")
    p.add_argument("--oversample", action="store_true", help="Oversample minority classes by duplication")
    p.add_argument("--min-words", type=int, default=3, help="Minimum token count to keep text")
    p.add_argument("--max-words", type=int, default=200, help="Maximum token count to keep text")
    p.add_argument("--global-dedupe", action="store_true", help="Use global dedupe instead of label-aware dedupe")
    p.add_argument("--filter-hasoc", action="store_true", help="Filter CSVs in raw-dir to those with 'hasoc' or 'train' in filename")
    p.add_argument("--ignore-labels", type=str, default="UNKNOWN", help="Comma-separated labels to ignore when oversampling")
    p.set_defaults(clean=True)
    args = p.parse_args()
    main(args)
