#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream and normalize raw iPinYou contest logs directly from assets/ipinyou.zip
following wnzhang/make-ipinyou-data conventions, train a calibrated CTR model
(pctr), write per-advertiser train.pickle / test.pickle files, and build the
FpBidder Linear Programming checkpoints required by RTBAgent.
"""
import argparse
import bz2
import datetime as dt
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.fp_bidder import generate_checkpoint

# Canonical campaign-to-season mapping (matching make-ipinyou-data & RTBAgent)
SEASON2_ADVERTISERS = {"1458", "3358", "3386", "3427", "3476"}
SEASON3_ADVERTISERS = {"2259", "2261", "2821", "2997"}
ALL_STANDARD_ADVERTISERS = sorted(SEASON2_ADVERTISERS | SEASON3_ADVERTISERS, key=int)

OS_LIST = ("windows", "ios", "mac", "android", "linux")
BROWSER_LIST = ("chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie")


def parse_useragent(ua: str) -> str:
    """Normalize raw User-Agent string into OS_Browser token (make-ipinyou-data style)."""
    ua_low = (ua or "").lower()
    detected_os = "other"
    for os_name in OS_LIST:
        if os_name in ua_low:
            detected_os = os_name
            break
    detected_browser = "other"
    for br_name in BROWSER_LIST:
        if br_name in ua_low or (br_name == "ie" and "msie" in ua_low):
            detected_browser = br_name
            break
    return f"{detected_os}_{detected_browser}"


def bucket_slotprice(sp: float) -> str:
    """Bucket floor price following make-ipinyou-data mkyzx.py."""
    if sp <= 0:
        return "0"
    if sp <= 10:
        return "1-10"
    if sp <= 50:
        return "11-50"
    if sp <= 100:
        return "51-100"
    return "101+"


def clean_cat(val: str) -> str:
    v = (val or "").strip()
    if not v or v.lower() in {"null", "na", "nan", "none"}:
        return "null"
    return v


def normalize_fields(cols: list[str], season: int, is_test: bool) -> dict | None:
    """
    Normalize inconsistent Season 1 (22/24 cols) and Season 2/3 (24/26 cols)
    raw tab-separated rows into a unified schema dictionary.
    Returns None if row is corrupted/truncated.
    """
    expected = (24 if is_test else 22) if season == 1 else (26 if is_test else 24)
    if len(cols) < expected:
        return None

    bidid = cols[0].strip()
    ts = cols[1].strip()
    if not bidid or len(ts) < 10 or not ts[:10].isdigit():
        return None

    try:
        weekday = dt.datetime.strptime(ts[:8], "%Y%m%d").weekday()
        hour = int(ts[8:10])
        if not (0 <= hour <= 23):
            return None
        slotprice = float(cols[17]) if cols[17].strip() else 0.0
        bidprice = float(cols[19]) if cols[19].strip() else 0.0
        payprice = float(cols[20]) if cols[20].strip() else 0.0
    except ValueError:
        return None

    if payprice < 0 or bidprice < 0 or slotprice < 0:
        return None

    if season == 1:
        advertiser = "s1"
        usertag = "null"
        click = (1 if int(float(cols[22])) > 0 else 0) if is_test else 0
    else:
        advertiser = cols[22].strip()
        usertag = clean_cat(cols[23])
        click = (1 if int(float(cols[24])) > 0 else 0) if is_test else 0

    ip_parts = cols[5].strip().split(".")
    ip_sub = f"{ip_parts[0]}.{ip_parts[1]}" if len(ip_parts) >= 2 else "null"

    return {
        "click": int(click),
        "weekday": int(weekday),
        "hour": int(hour),
        "bidid": bidid,
        "timestamp": ts,
        "ua": parse_useragent(cols[4]),
        "ip_sub": ip_sub,
        "region": clean_cat(cols[6]),
        "city": clean_cat(cols[7]),
        "adexchange": clean_cat(cols[8]),
        "domain": clean_cat(cols[9]),
        "slotid": clean_cat(cols[12]),
        "slotwidth": clean_cat(cols[13]),
        "slotheight": clean_cat(cols[14]),
        "slotvisibility": clean_cat(cols[15]),
        "slotformat": clean_cat(cols[16]),
        "slotprice": float(slotprice),
        "creative": clean_cat(cols[18]),
        "bidprice": float(bidprice),
        "payprice": float(payprice),
        "keypage": clean_cat(cols[21]),
        "advertiser": advertiser,
        "usertag": usertag,
    }


def extract_season_for_advertisers(
    zf: zipfile.ZipFile,
    season: int,
    target_advs: set[str],
    max_train_rows_per_adv: int | None = None,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict]:
    """Stream clk, imp, and leaderboard.test.data for a given season directly from zip."""
    season_tag = {1: "1st", 2: "2nd", 3: "3rd"}[season]
    train_prefix = f"ipinyou.contest.dataset/training{season_tag}/"
    test_prefix = f"ipinyou.contest.dataset/testing{season_tag}/"

    names = sorted(zf.namelist())
    clk_files = [n for n in names if n.startswith(train_prefix) and "/clk." in n and n.endswith(".txt.bz2")]
    imp_files = [n for n in names if n.startswith(train_prefix) and "/imp." in n and n.endswith(".txt.bz2")]
    test_files = [n for n in names if n.startswith(test_prefix) and n.endswith(".txt.bz2")]

    stats = {
        "season": season,
        "clk_files": len(clk_files),
        "imp_files": len(imp_files),
        "test_files": len(test_files),
        "clicked_bidids": 0,
        "duplicate_clicks_deduped": 0,
        "malformed_rows_skipped": 0,
    }

    # 1. Collect clicked bidids per advertiser from clk.*.txt.bz2
    clicked_bidids: set[str] = set()
    for clk_name in clk_files:
        print(f"  [S{season} CLK] Reading {clk_name} ...", flush=True)
        with zf.open(clk_name) as raw, bz2.open(raw, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                cols = line.rstrip("\r\n").split("\t")
                if season == 1:
                    if len(cols) < 22:
                        stats["malformed_rows_skipped"] += 1
                        continue
                    adv = "s1"
                else:
                    if len(cols) < 24:
                        stats["malformed_rows_skipped"] += 1
                        continue
                    adv = cols[22].strip()
                if adv not in target_advs:
                    continue
                bidid = cols[0].strip()
                if bidid in clicked_bidids:
                    stats["duplicate_clicks_deduped"] += 1
                else:
                    clicked_bidids.add(bidid)
    stats["clicked_bidids"] = len(clicked_bidids)

    # 2. Stream training impressions and join click labels by bidid
    train_rows: dict[str, list[dict]] = {a: [] for a in target_advs}
    for imp_name in imp_files:
        print(f"  [S{season} IMP] Streaming {imp_name} ...", flush=True)
        with zf.open(imp_name) as raw, bz2.open(raw, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                cols = line.rstrip("\r\n").split("\t")
                adv_fast = "s1" if season == 1 else (cols[22].strip() if len(cols) > 22 else "")
                if adv_fast not in target_advs:
                    continue
                if max_train_rows_per_adv and len(train_rows[adv_fast]) >= max_train_rows_per_adv:
                    continue
                rec = normalize_fields(cols, season=season, is_test=False)
                if rec is None:
                    stats["malformed_rows_skipped"] += 1
                    continue
                if rec["bidid"] in clicked_bidids:
                    rec["click"] = 1
                train_rows[adv_fast].append(rec)
        if max_train_rows_per_adv and all(len(train_rows[a]) >= max_train_rows_per_adv for a in target_advs):
            break

    # 3. Stream leaderboard test data
    test_rows: dict[str, list[dict]] = {a: [] for a in target_advs}
    for test_name in test_files:
        print(f"  [S{season} TEST] Streaming {test_name} ...", flush=True)
        with zf.open(test_name) as raw, bz2.open(raw, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                cols = line.rstrip("\r\n").split("\t")
                adv_fast = "s1" if season == 1 else (cols[22].strip() if len(cols) > 22 else "")
                if adv_fast not in target_advs:
                    continue
                rec = normalize_fields(cols, season=season, is_test=True)
                if rec is None:
                    stats["malformed_rows_skipped"] += 1
                    continue
                test_rows[adv_fast].append(rec)

    return train_rows, test_rows, stats


def build_feature_tokens(df: pd.DataFrame):
    """Yield sparse string tokens per row matching make-ipinyou-data feature engineering."""
    cat_cols = [
        "weekday",
        "hour",
        "ua",
        "ip_sub",
        "region",
        "city",
        "adexchange",
        "domain",
        "slotid",
        "slotwidth",
        "slotheight",
        "slotvisibility",
        "slotformat",
        "creative",
        "keypage",
    ]
    col_arrays = {c: df[c].astype(str).tolist() for c in cat_cols}
    sp_list = df["slotprice"].tolist()
    ut_list = df["usertag"].astype(str).tolist()
    n = len(df)
    for i in range(n):
        tokens = [f"{c}={col_arrays[c][i]}" for c in cat_cols]
        tokens.append(f"sp_bkt={bucket_slotprice(sp_list[i])}")
        tokens.append(f"wh={col_arrays['slotwidth'][i]}x{col_arrays['slotheight'][i]}")
        tokens.append(f"hr_adx={col_arrays['hour'][i]}_{col_arrays['adexchange'][i]}")
        ut = ut_list[i]
        if ut and ut != "null":
            for tag in ut.split(","):
                t = tag.strip()
                if t:
                    tokens.append(f"ut={t}")
        yield tokens


def train_and_predict_pctr(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train L2-regularized LogisticRegression CTR predictor with 5-fold OOF predictions on train."""
    hasher = FeatureHasher(n_features=2**18, input_type="string", alternate_sign=False)
    print(f"    Hashing {len(train_df):,} train and {len(test_df):,} test impressions ...", flush=True)
    X_train = hasher.transform(build_feature_tokens(train_df))
    X_test = hasher.transform(build_feature_tokens(test_df))
    y_train = train_df["click"].values.astype(int)
    y_test = test_df["click"].values.astype(int)

    oof_pctr = np.zeros(len(train_df), dtype=np.float64)
    n_pos = int(y_train.sum())
    if n_pos >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
            clf = LogisticRegression(solver="lbfgs", C=0.1, max_iter=200)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            oof_pctr[val_idx] = clf.predict_proba(X_train[val_idx])[:, 1]
    else:
        base_rate = max(float(y_train.mean()), 1e-6)
        oof_pctr[:] = base_rate

    final_clf = LogisticRegression(solver="lbfgs", C=0.1, max_iter=200)
    final_clf.fit(X_train, y_train)
    test_pctr = final_clf.predict_proba(X_test)[:, 1]

    # Clip extreme zeros to small positive floor
    oof_pctr = np.clip(oof_pctr, 1e-7, 1.0 - 1e-7)
    test_pctr = np.clip(test_pctr, 1e-7, 1.0 - 1e-7)

    metrics = {
        "train_rows": int(len(train_df)),
        "train_clicks": int(y_train.sum()),
        "train_ctr": float(y_train.mean()),
        "train_mean_pctr": float(oof_pctr.mean()),
        "train_auc": float(roc_auc_score(y_train, oof_pctr)) if len(np.unique(y_train)) > 1 else None,
        "train_logloss": float(log_loss(y_train, oof_pctr)) if len(np.unique(y_train)) > 1 else None,
        "test_rows": int(len(test_df)),
        "test_clicks": int(y_test.sum()),
        "test_ctr": float(y_test.mean()),
        "test_mean_pctr": float(test_pctr.mean()),
        "test_auc": float(roc_auc_score(y_test, test_pctr)) if len(np.unique(y_test)) > 1 else None,
        "test_logloss": float(log_loss(y_test, test_pctr)) if len(np.unique(y_test)) > 1 else None,
    }
    return oof_pctr, test_pctr, metrics


def main():
    parser = argparse.ArgumentParser(description="Normalize ipinyou.zip and prepare RTBAgent dataset + LP checkpoints.")
    parser.add_argument("--zip-path", default=str(PROJECT_ROOT / "assets" / "ipinyou.zip"))
    parser.add_argument(
        "--advertisers",
        nargs="+",
        default=["1458"],
        help="Advertiser IDs to process (e.g. 1458 3358, or 'all' for all 9 S2/S3 campaigns, or 's1' for Season 1).",
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "ipinyouData" / "new_all"))
    parser.add_argument("--checkpoint-root", default=str(PROJECT_ROOT / "checkpoint"))
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional cap on training rows per advertiser.")
    args = parser.parse_args()

    req_advs = set()
    for item in args.advertisers:
        if item.lower() == "all":
            req_advs.update(ALL_STANDARD_ADVERTISERS)
        else:
            req_advs.add(str(item))

    season_groups: dict[int, set[str]] = {1: set(), 2: set(), 3: set()}
    for adv in req_advs:
        if adv == "s1":
            season_groups[1].add(adv)
        elif adv in SEASON2_ADVERTISERS:
            season_groups[2].add(adv)
        elif adv in SEASON3_ADVERTISERS:
            season_groups[3].add(adv)
        else:
            raise ValueError(
                f"Unknown advertiser '{adv}'. Valid IDs: {ALL_STANDARD_ADVERTISERS} or 's1' or 'all'."
            )

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_report = {"seasons": [], "advertisers": {}}

    keep_cols = [
        "click",
        "weekday",
        "hour",
        "bidid",
        "timestamp",
        "region",
        "city",
        "adexchange",
        "slotwidth",
        "slotheight",
        "slotvisibility",
        "slotformat",
        "slotprice",
        "creative",
        "bidprice",
        "payprice",
        "advertiser",
        "pctr",
    ]

    processed_advs = []
    with zipfile.ZipFile(args.zip_path, "r") as zf:
        for season, adv_set in season_groups.items():
            if not adv_set:
                continue
            print(f"\n=== Processing Season {season} for advertisers: {sorted(adv_set)} ===", flush=True)
            train_map, test_map, season_stats = extract_season_for_advertisers(
                zf=zf,
                season=season,
                target_advs=adv_set,
                max_train_rows_per_adv=args.max_train_rows,
            )
            summary_report["seasons"].append(season_stats)

            for adv in sorted(adv_set):
                tr_list = train_map[adv]
                te_list = test_map[adv]
                if not tr_list or not te_list:
                    print(f"  [WARN] Advertiser {adv} missing train ({len(tr_list)}) or test ({len(te_list)}) rows!")
                    continue

                train_df = pd.DataFrame(tr_list).sort_values("timestamp").reset_index(drop=True)
                test_df = pd.DataFrame(te_list).sort_values("timestamp").reset_index(drop=True)
                del tr_list, te_list

                print(f"  [CTR] Training CTR model for advertiser {adv} ...", flush=True)
                oof_pctr, test_pctr, ctr_metrics = train_and_predict_pctr(train_df, test_df)
                train_df["pctr"] = oof_pctr
                test_df["pctr"] = test_pctr

                adv_dir = out_root / adv
                adv_dir.mkdir(parents=True, exist_ok=True)
                train_df[keep_cols].to_pickle(adv_dir / "train.pickle")
                test_df[keep_cols].to_pickle(adv_dir / "test.pickle")

                summary_report["advertisers"][adv] = ctr_metrics
                processed_advs.append(adv)
                print(
                    f"  [DONE] Advertiser {adv}: train={ctr_metrics['train_rows']:,} "
                    f"(CTR={ctr_metrics['train_ctr']:.5f}, AUC={ctr_metrics['train_auc']:.4f}), "
                    f"test={ctr_metrics['test_rows']:,} "
                    f"(CTR={ctr_metrics['test_ctr']:.5f}, AUC={ctr_metrics['test_auc']:.4f})",
                    flush=True,
                )

    if processed_advs:
        print(f"\n=== Generating FpBidder LP Checkpoints for {processed_advs} ===", flush=True)
        generate_checkpoint(
            data_root=str(out_root),
            checkpoint_root=args.checkpoint_root,
            advertiser_filter=processed_advs,
        )

    report_path = out_root / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved preprocessing report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
