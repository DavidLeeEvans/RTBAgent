#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end Real-Time Bidding (RTB) simulation & evaluation runner for RTBAgent.
Compares the static Linear Programming baseline (BasicAgent / FpBidder) against
the two-step LLM reasoning agent (RTBAgent / llm_fp) on normalized iPinYou test data.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basics.basic_agent import BasicAgent
from llm_based.prompts.role_prompts import role_prompts
from llm_based.roles.rtb_agent import RTBAgent


def simulate_sequential_hour(
    pctrs: np.ndarray,
    payprices: np.ndarray,
    clicks: np.ndarray,
    bidding_factor: float,
    remaining_budget: float,
    min_remaining_budget: float = 0.001,
) -> dict:
    """
    Simulate sequential second-price RTB auctions over one hour of impressions
    subject to a hard remaining budget constraint.
    """
    n = len(pctrs)
    if n == 0:
        return {
            "traffic_num": 0,
            "num_won": 0,
            "clicks": 0,
            "cost": 0.0,
            "remaining_budget": float(remaining_budget),
            "bid_price_avg": 0.0,
            "market_price_avg": 0.0,
            "win_rate": 0.0,
            "actual_ctr_all": 0.0,
            "actual_ctr_won": 0.0,
            "predict_ctr_avg": 0.0,
        }

    raw_bids = pctrs * float(bidding_factor)
    effective_bids = np.zeros(n, dtype=np.float64)
    won_mask = np.zeros(n, dtype=bool)
    rem = float(remaining_budget)
    cost_sum = 0.0
    clicks_won = 0
    num_won = 0

    for i in range(n):
        if rem < min_remaining_budget:
            break
        bid = raw_bids[i]
        effective_bids[i] = bid
        mp = payprices[i]
        if bid >= mp and rem >= mp:
            won_mask[i] = True
            rem -= mp
            cost_sum += mp
            num_won += 1
            clicks_won += int(clicks[i])

    return {
        "traffic_num": int(n),
        "num_won": int(num_won),
        "clicks": int(clicks_won),
        "cost": float(cost_sum),
        "remaining_budget": float(rem),
        "bid_price_avg": float(effective_bids.mean()),
        "market_price_avg": float(payprices.mean()),
        "win_rate": float(num_won / n),
        "actual_ctr_all": float(clicks_won / n),
        "actual_ctr_won": float(clicks_won / num_won) if num_won > 0 else 0.0,
        "predict_ctr_avg": float(pctrs.mean()),
    }


def run_baseline_day(day_df: pd.DataFrame, advertiser_id: str, budget_ratio: str, max_hours: int | None = None) -> dict:
    """Run static FpBidder (BasicAgent) across a single test day."""
    given_budget = float(day_df["payprice"].sum()) * float(budget_ratio)
    basic = BasicAgent(
        remaining_budget=given_budget,
        budget_ratio=str(budget_ratio),
        advertiser_id=str(advertiser_id),
        bidder_name="fp",
        is_train=False,
    )
    w_star = float(basic.tools["bidder"].bidding_factor)

    hours = sorted(day_df["hour"].unique().tolist())
    if max_hours is not None:
        hours = hours[:max_hours]
        sub_df = day_df[day_df["hour"].isin(hours)]
        given_budget = float(sub_df["payprice"].sum()) * float(budget_ratio)

    rem = given_budget
    hourly_records = []
    for hr in hours:
        hr_df = day_df[day_df["hour"] == hr]
        res = simulate_sequential_hour(
            pctrs=hr_df["pctr"].to_numpy(dtype=np.float64),
            payprices=hr_df["payprice"].to_numpy(dtype=np.float64),
            clicks=hr_df["click"].to_numpy(dtype=np.int64),
            bidding_factor=w_star,
            remaining_budget=rem,
            min_remaining_budget=basic.min_remaining_budget,
        )
        rem = res["remaining_budget"]
        res["hour"] = int(hr)
        res["bidding_factor"] = w_star
        hourly_records.append(res)

    total_traffic = sum(r["traffic_num"] for r in hourly_records)
    total_won = sum(r["num_won"] for r in hourly_records)
    total_clicks = sum(r["clicks"] for r in hourly_records)
    total_cost = sum(r["cost"] for r in hourly_records)

    return {
        "agent": "BasicAgent (fp)",
        "given_budget": given_budget,
        "spent_budget": total_cost,
        "remaining_budget": rem,
        "budget_utilization": (total_cost / given_budget) if given_budget > 0 else 0.0,
        "base_bidding_factor": w_star,
        "traffic_num": total_traffic,
        "impressions_won": total_won,
        "win_rate": (total_won / total_traffic) if total_traffic > 0 else 0.0,
        "clicks": total_clicks,
        "ctr_won": (total_clicks / total_won) if total_won > 0 else 0.0,
        "cpc": (total_cost / total_clicks) if total_clicks > 0 else None,
        "hourly": hourly_records,
    }


async def run_llm_agent_day(
    day_df: pd.DataFrame,
    date_str: str,
    advertiser_id: str,
    budget_ratio: str,
    workspace_root: Path,
    max_hours: int | None = None,
) -> dict:
    """Run two-step LLM RTBAgent hour-by-hour across a single test day."""
    hours = sorted(day_df["hour"].unique().tolist())
    if max_hours is not None:
        hours = hours[:max_hours]
        sub_df = day_df[day_df["hour"].isin(hours)]
        given_budget = float(sub_df["payprice"].sum()) * float(budget_ratio)
    else:
        given_budget = float(day_df["payprice"].sum()) * float(budget_ratio)

    agent = RTBAgent(
        name=f"RTBAgent_{advertiser_id}",
        profile="Senior RTB Bidding Strategist",
        goal="Maximize ad clicks within the daily budget",
        advertiser_id=str(advertiser_id),
        role_prompt=role_prompts["role_prompt"],
        is_train=False,
        remaining_budget=given_budget,
        budget_ratio=str(budget_ratio),
        given_budget=given_budget,
        bidder_name="llm_fp",
        workspace_root=str(workspace_root),
    )

    total_episodes = len(hours)
    hourly_records = []
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    for idx, hr in enumerate(hours):
        remaining_episodes = total_episodes - idx
        formatted_time_str = f"{formatted_date}-{hr:02d}"
        t0 = time.time()

        if agent.remaining_budget >= agent.min_remaining_budget:
            strategy = await agent.get_bidding_strategy(
                formatted_time_str=formatted_time_str,
                total_bid_episode_num=total_episodes,
                remaining_bid_episode_num=remaining_episodes,
            )
            adjustment = float(strategy["adjustment"])
            reason = str(strategy["reason"])
            base_factor = float(strategy["bidding_factor"])
            final_factor = float(strategy["final_bidding_factor"])
        else:
            adjustment = 0.0
            reason = "Budget exhausted; skipping LLM call."
            base_factor = float(agent.bidder.get_bidding_factor())
            final_factor = 0.0

        hr_df = day_df[day_df["hour"] == hr]
        sim = simulate_sequential_hour(
            pctrs=hr_df["pctr"].to_numpy(dtype=np.float64),
            payprices=hr_df["payprice"].to_numpy(dtype=np.float64),
            clicks=hr_df["click"].to_numpy(dtype=np.int64),
            bidding_factor=final_factor,
            remaining_budget=agent.remaining_budget,
            min_remaining_budget=agent.min_remaining_budget,
        )
        agent.remaining_budget = sim["remaining_budget"]
        agent.click_num += sim["clicks"]
        elapsed = time.time() - t0

        # Persist hourly environment memory in exact format read by EnvironmentalStatusProvider & HistorySummaryProvider
        env_hour_dir = workspace_root / "env" / formatted_time_str
        env_hour_dir.mkdir(parents=True, exist_ok=True)
        env_record = {
            "datetime": formatted_time_str,
            "traffic_num": sim["traffic_num"],
            "num_won": sim["num_won"],
            "click": sim["clicks"],
            "actual_click_rate": sim["actual_ctr_all"],
            "predict_click_rate": sim["predict_ctr_avg"],
            "bid_price_avg": sim["bid_price_avg"],
            "market_price_avg": sim["market_price_avg"],
            "win_rate": sim["win_rate"],
            "market_cost": sim["cost"],
            "given_budget": given_budget,
            "remaining_budget": agent.remaining_budget,
            "remaining_bid_episode_num": remaining_episodes - 1,
            "adjustment": adjustment,
            "final_bidding_factor": final_factor,
            "reason": reason,
        }
        with open(env_hour_dir / f"{advertiser_id}.json", "w", encoding="utf-8") as f:
            json.dump(env_record, f, ensure_ascii=False, indent=2)

        sim_row = {
            **sim,
            "hour": int(hr),
            "datetime": formatted_time_str,
            "base_bidding_factor": base_factor,
            "adjustment": adjustment,
            "final_bidding_factor": final_factor,
            "reason": reason,
            "latency_sec": round(elapsed, 2),
        }
        hourly_records.append(sim_row)
        print(
            f"  [{formatted_time_str}] traffic={sim['traffic_num']:5d} | adj={adjustment:+.4f} "
            f"(w={final_factor:7.2f}) | won={sim['num_won']:5d} ({sim['win_rate']*100:5.1f}%) | "
            f"clicks={sim['clicks']:2d} | cost={sim['cost']:9.1f} | rem_B={agent.remaining_budget:10.1f} "
            f"| {elapsed:.1f}s",
            flush=True,
        )

    # End-of-day reflection and env_summary persistence
    end_time_str = f"{formatted_date}-{hours[-1]:02d}"
    reflection = await agent.reflect(formatted_time_str=end_time_str)

    total_traffic = sum(r["traffic_num"] for r in hourly_records)
    total_won = sum(r["num_won"] for r in hourly_records)
    total_clicks = sum(r["clicks"] for r in hourly_records)
    total_cost = sum(r["cost"] for r in hourly_records)

    env_sum_dir = workspace_root / "env_summary" / end_time_str
    env_sum_dir.mkdir(parents=True, exist_ok=True)
    day_summary_record = {
        "datetime": end_time_str,
        "traffic_num_sum": total_traffic,
        "num_won_sum": total_won,
        "market_cost_sum": round(total_cost, 2),
        "given_budget": round(given_budget, 2),
        "remaining_budget": round(agent.remaining_budget, 2),
        "budget_utilization": round((total_cost / given_budget) if given_budget > 0 else 0.0, 4),
        "click_sum": total_clicks,
        "day_ctr": round((total_clicks / total_won) if total_won > 0 else 0.0, 6),
        "day_cpc": round((total_cost / total_clicks) if total_clicks > 0 else 0.0, 2),
    }
    with open(env_sum_dir / f"{advertiser_id}.json", "w", encoding="utf-8") as f:
        json.dump(day_summary_record, f, ensure_ascii=False, indent=2)

    return {
        "agent": f"RTBAgent (llm_fp / {os.getenv('MY_API_PARAM', 'default')})",
        "given_budget": given_budget,
        "spent_budget": total_cost,
        "remaining_budget": agent.remaining_budget,
        "budget_utilization": (total_cost / given_budget) if given_budget > 0 else 0.0,
        "base_bidding_factor": float(agent.bidder.get_bidding_factor()),
        "traffic_num": total_traffic,
        "impressions_won": total_won,
        "win_rate": (total_won / total_traffic) if total_traffic > 0 else 0.0,
        "clicks": total_clicks,
        "ctr_won": (total_clicks / total_won) if total_won > 0 else 0.0,
        "cpc": (total_cost / total_clicks) if total_clicks > 0 else None,
        "prompt_tokens": int(agent.prompt_tokens),
        "completion_tokens": int(agent.completion_tokens),
        "reflection": reflection,
        "hourly": hourly_records,
    }


async def async_main():
    parser = argparse.ArgumentParser(description="Run RTBAgent vs BasicAgent evaluation on iPinYou test data.")
    parser.add_argument("--advertiser", default="1458", help="Advertiser campaign ID (default: 1458)")
    parser.add_argument(
        "--budget-ratio",
        default="0.125",
        choices=["0.5", "0.25", "0.125", "0.0625", "0.03125"],
        help="Daily budget ratio relative to total market payprice (default: 0.125)",
    )
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "ipinyouData" / "new_all"))
    parser.add_argument("--workspace-root", default=str(PROJECT_ROOT / "workspace" / "eval_run"))
    parser.add_argument("--max-days", type=int, default=1, help="Number of test days to simulate (default: 1)")
    parser.add_argument("--max-hours", type=int, default=None, help="Optional cap on hours per day for quick testing")
    parser.add_argument("--clean-workspace", action="store_true", help="Clear workspace-root before running")
    parser.add_argument("--baseline-only", action="store_true", help="Only run the non-LLM BasicAgent (fp) baseline")
    args = parser.parse_args()

    test_pickle = Path(args.data_root) / str(args.advertiser) / "test.pickle"
    if not test_pickle.exists():
        raise FileNotFoundError(
            f"Missing {test_pickle}. Run `tools/prepare_ipinyou_data.py --advertisers {args.advertiser}` first."
        )

    ws_root = Path(args.workspace_root)
    if args.clean_workspace and ws_root.exists():
        shutil.rmtree(ws_root)
    ws_root.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_pickle(test_pickle).sort_values("timestamp").reset_index(drop=True)
    test_df["date_str"] = test_df["timestamp"].astype(str).str.slice(0, 8)
    dates = sorted(test_df["date_str"].unique().tolist())[: args.max_days]

    print(
        f"Loaded test.pickle for advertiser={args.advertiser}: {len(test_df):,} rows across dates "
        f"{sorted(test_df['date_str'].unique().tolist())}. Simulating dates={dates}, budget_ratio={args.budget_ratio}."
    )

    all_results = {"advertiser": args.advertiser, "budget_ratio": args.budget_ratio, "days": {}}

    for d_str in dates:
        day_df = test_df[test_df["date_str"] == d_str].copy()
        print(f"\n==================== Day {d_str} ({len(day_df):,} impressions) ====================", flush=True)

        bsl_res = run_baseline_day(
            day_df=day_df,
            advertiser_id=args.advertiser,
            budget_ratio=args.budget_ratio,
            max_hours=args.max_hours,
        )
        print(
            f"[Baseline BasicAgent (fp)] w*={bsl_res['base_bidding_factor']:.2f} | "
            f"Won={bsl_res['impressions_won']:,}/{bsl_res['traffic_num']:,} ({bsl_res['win_rate']*100:.2f}%) | "
            f"Clicks={bsl_res['clicks']} | Cost={bsl_res['spent_budget']:,.1f}/{bsl_res['given_budget']:,.1f} "
            f"({bsl_res['budget_utilization']*100:.2f}%) | CPC={bsl_res['cpc']}",
            flush=True,
        )

        day_entry = {"baseline_fp": bsl_res}
        if not args.baseline_only:
            llm_res = await run_llm_agent_day(
                day_df=day_df,
                date_str=d_str,
                advertiser_id=args.advertiser,
                budget_ratio=args.budget_ratio,
                workspace_root=ws_root,
                max_hours=args.max_hours,
            )
            day_entry["rtb_agent_llm_fp"] = llm_res
            print("\n-------------------- Comparison Summary --------------------")
            print(f"{'Metric':<24} | {'BasicAgent (fp)':>18} | {'RTBAgent (llm_fp)':>18}")
            print("-" * 66)
            print(f"{'Total Traffic':<24} | {bsl_res['traffic_num']:>18,d} | {llm_res['traffic_num']:>18,d}")
            print(f"{'Impressions Won':<24} | {bsl_res['impressions_won']:>18,d} | {llm_res['impressions_won']:>18,d}")
            print(f"{'Win Rate (%)':<24} | {bsl_res['win_rate']*100:>17.2f}% | {llm_res['win_rate']*100:>17.2f}%")
            print(f"{'Given Budget':<24} | {bsl_res['given_budget']:>18,.1f} | {llm_res['given_budget']:>18,.1f}")
            print(f"{'Spent Budget':<24} | {bsl_res['spent_budget']:>18,.1f} | {llm_res['spent_budget']:>18,.1f}")
            print(
                f"{'Budget Utilization (%)':<24} | "
                f"{bsl_res['budget_utilization']*100:>17.2f}% | {llm_res['budget_utilization']*100:>17.2f}%"
            )
            print(f"{'Clicks Won':<24} | {bsl_res['clicks']:>18,d} | {llm_res['clicks']:>18,d}")
            bsl_cpc_s = f"{bsl_res['cpc']:,.2f}" if bsl_res["cpc"] is not None else "N/A"
            llm_cpc_s = f"{llm_res['cpc']:,.2f}" if llm_res["cpc"] is not None else "N/A"
            print(f"{'CPC (Cost per Click)':<24} | {bsl_cpc_s:>18} | {llm_cpc_s:>18}")
            print("-" * 66, flush=True)

        all_results["days"][d_str] = day_entry

    summary_path = ws_root / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full evaluation summary to {summary_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(async_main())
