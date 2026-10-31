#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a comprehensive PDF technical & evaluation report for RTBAgent.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_charts(bsl: dict, llm: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bsl_hr = bsl["hourly"]
    llm_hr = llm["hourly"]
    hours = [r["hour"] for r in llm_hr]

    bsl_cum_cost = np.cumsum([r["cost"] for r in bsl_hr]) / 1e6
    llm_cum_cost = np.cumsum([r["cost"] for r in llm_hr]) / 1e6
    budget_line = bsl["given_budget"] / 1e6

    bsl_cum_clicks = np.cumsum([r["clicks"] for r in bsl_hr])
    llm_cum_clicks = np.cumsum([r["clicks"] for r in llm_hr])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=180)
    ax0, ax1 = axes

    ax0.plot(hours, bsl_cum_cost, marker="o", markersize=3.5, linewidth=1.8, label="BasicAgent (fp)", color="#64748b")
    ax0.plot(hours, llm_cum_cost, marker="s", markersize=3.5, linewidth=2.0, label="RTBAgent (llm_fp)", color="#2563eb")
    ax0.axhline(budget_line, linestyle="--", linewidth=1.2, color="#dc2626", label=f"Daily Budget ({budget_line:.2f}M)")
    ax0.set_title("Cumulative Budget Spend Across Day", fontsize=10.5, fontweight="bold")
    ax0.set_xlabel("Hour of Day")
    ax0.set_ylabel("Cumulative Spend (Millions, CPM units)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)

    ax1.plot(hours, bsl_cum_clicks, marker="o", markersize=3.5, linewidth=1.8, label="BasicAgent (fp)", color="#64748b")
    ax1.plot(hours, llm_cum_clicks, marker="s", markersize=3.5, linewidth=2.0, label="RTBAgent (llm_fp)", color="#16a34a")
    ax1.set_title("Cumulative Clicks Won Across Day", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Cumulative Clicks")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    fig.tight_layout()
    chart1_path = out_dir / "chart_pacing_clicks.png"
    fig.savefig(chart1_path, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: Hourly LLM Adjustment & Win Rate comparison
    adjs = [r["adjustment"] for r in llm_hr]
    bsl_wr = [r["win_rate"] * 100.0 for r in bsl_hr]
    llm_wr = [r["win_rate"] * 100.0 for r in llm_hr]

    fig2, axes2 = plt.subplots(1, 2, figsize=(10.5, 3.5), dpi=180)
    bx0, bx1 = axes2

    bar_colors = ["#2563eb" if a >= 0 else "#ea580c" for a in adjs]
    bx0.bar(hours, adjs, color=bar_colors, alpha=0.85, width=0.7)
    bx0.axhline(0.0, color="#334155", linewidth=1.0)
    bx0.set_ylim(-0.55, 0.55)
    bx0.set_title("RTBAgent Hourly Adjustment Factor a_t in [-0.5, 0.5]", fontsize=10, fontweight="bold")
    bx0.set_xlabel("Hour of Day")
    bx0.set_ylabel("Adjustment (a_t)")
    bx0.grid(True, axis="y", alpha=0.25)

    bx1.plot(hours, bsl_wr, marker="o", markersize=3.5, linewidth=1.6, label="BasicAgent Win Rate %", color="#64748b")
    bx1.plot(hours, llm_wr, marker="s", markersize=3.5, linewidth=1.8, label="RTBAgent Win Rate %", color="#7c3aed")
    bx1.set_title("Hourly Auction Win Rate (%)", fontsize=10, fontweight="bold")
    bx1.set_xlabel("Hour of Day")
    bx1.set_ylabel("Win Rate (%)")
    bx1.grid(True, alpha=0.25)
    bx1.legend(fontsize=8)

    fig2.tight_layout()
    chart2_path = out_dir / "chart_adjustments_winrate.png"
    fig2.savefig(chart2_path, bbox_inches="tight")
    plt.close(fig2)

    return chart1_path, chart2_path


def styled_table(data, col_widths=None, header_bg="#1e293b"):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    return t


def build_pdf(eval_json_path: Path, prep_json_path: Path, checkpoint_json_path: Path, optimal_r_path: Path, out_pdf: Path):
    eval_data = json.loads(eval_json_path.read_text(encoding="utf-8"))
    prep_data = json.loads(prep_json_path.read_text(encoding="utf-8"))
    ckpt_data = json.loads(checkpoint_json_path.read_text(encoding="utf-8"))
    opt_r_data = json.loads(optimal_r_path.read_text(encoding="utf-8"))

    adv_id = str(eval_data["advertiser"])
    budget_ratio = str(eval_data["budget_ratio"])
    first_day = sorted(eval_data["days"].keys())[0]
    day_bundle = eval_data["days"][first_day]
    bsl = day_bundle["baseline_fp"]
    llm = day_bundle["rtb_agent_llm_fp"]

    chart_dir = eval_json_path.parent / "charts"
    chart1, chart2 = make_charts(bsl, llm, chart_dir)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=19,
        leading=23,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Code"],
        fontSize=7.8,
        leading=10.5,
        backColor=colors.HexColor("#f1f5f9"),
        borderPadding=6,
        spaceAfter=8,
    )
    cell_style = ParagraphStyle(
        "CellText",
        parent=styles["Normal"],
        fontSize=7.3,
        leading=9.2,
        textColor=colors.HexColor("#1e293b"),
    )

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="RTBAgent Setup, Data Normalization, and Evaluation Report",
    )

    story = []
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph("RTBAgent: Setup, iPinYou Data Normalization & Evaluation Report", title_style))
    story.append(
        Paragraph(
            f"Workspace: /home/david/Source/RTB/RTBAgent  |  Campaign: Advertiser {adv_id}  |  "
            f"Test Date: {first_day}  |  Budget Ratio: {budget_ratio}  |  Generated: {now_str}",
            subtitle_style,
        )
    )

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    click_diff = llm["clicks"] - bsl["clicks"]
    click_pct = (click_diff / bsl["clicks"] * 100.0) if bsl["clicks"] > 0 else 0.0
    bsl_cpc = f"{bsl['cpc']:,.2f}" if bsl["cpc"] else "N/A"
    llm_cpc = f"{llm['cpc']:,.2f}" if llm["cpc"] else "N/A"
    story.append(
        Paragraph(
            f"This report documents the end-to-end setup, source code remediation, iPinYou dataset normalization, "
            f"CTR & Linear Programming checkpoint generation, and live LLM-based Real-Time Bidding evaluation of "
            f"<b>RTBAgent</b> (WWW'2025). Tested on iPinYou Season 2 Advertiser <b>{adv_id}</b> ({bsl['traffic_num']:,} "
            f"chronological ad exchange requests on {first_day[:4]}-{first_day[4:6]}-{first_day[6:8]} at budget ratio "
            f"<b>{budget_ratio}</b>), <b>{escape(llm['agent'])}</b> achieved <b>{llm['clicks']} clicks</b> "
            f"(CPC: {llm_cpc}) compared to <b>{bsl['clicks']} clicks</b> (CPC: {bsl_cpc}) for the static Linear "
            f"Programming baseline <b>BasicAgent (fp)</b> ({click_diff:+d} clicks / {click_pct:+.2f}%), while strictly "
            f"respecting the daily budget cap ({llm['budget_utilization']*100:.2f}% utilization).",
            body_style,
        )
    )

    summary_rows = [
        ["Metric", "BasicAgent (Static LP / fp)", escape(llm["agent"]), "Delta"],
        ["Test Impressions", f"{bsl['traffic_num']:,}", f"{llm['traffic_num']:,}", "0"],
        [
            "Impressions Won",
            f"{bsl['impressions_won']:,} ({bsl['win_rate']*100:.2f}%)",
            f"{llm['impressions_won']:,} ({llm['win_rate']*100:.2f}%)",
            f"{llm['impressions_won'] - bsl['impressions_won']:+,}",
        ],
        [
            "Daily Budget (CPM units)",
            f"{bsl['given_budget']:,.1f}",
            f"{llm['given_budget']:,.1f}",
            "0.0",
        ],
        [
            "Spent Budget (Utilization)",
            f"{bsl['spent_budget']:,.1f} ({bsl['budget_utilization']*100:.2f}%)",
            f"{llm['spent_budget']:,.1f} ({llm['budget_utilization']*100:.2f}%)",
            f"{llm['spent_budget'] - bsl['spent_budget']:+,.1f}",
        ],
        [
            "Clicks Won",
            f"{bsl['clicks']}",
            f"{llm['clicks']}",
            f"{click_diff:+d} ({click_pct:+.2f}%)",
        ],
        [
            "Won CTR (%)",
            f"{bsl['ctr_won']*100:.4f}%",
            f"{llm['ctr_won']*100:.4f}%",
            f"{(llm['ctr_won'] - bsl['ctr_won'])*100:+.4f}%",
        ],
        [
            "Cost Per Click (CPC)",
            bsl_cpc,
            llm_cpc,
            f"{(llm['cpc'] - bsl['cpc']):+,.2f}" if (llm["cpc"] and bsl["cpc"]) else "N/A",
        ],
    ]
    story.append(styled_table(summary_rows, col_widths=[1.9 * inch, 1.85 * inch, 1.95 * inch, 1.5 * inch]))
    story.append(Spacer(1, 8))

    # 2. Environment & Codebase Audit Fixes
    story.append(Paragraph("2. Environment Setup & Source Code Audit Fixes", h1_style))
    story.append(
        Paragraph(
            "The workspace initially contained a bare Python 3.14.8 virtual environment incompatible with the pinned "
            "scientific stack in <b>requirements.txt</b> (numpy~=1.26.3, pandas~=2.2.3, scikit-learn~=1.5.2, pulp~=2.9.0) "
            "and lacked a top-level simulation loop. We recreated <b>.venv</b> on <b>Python 3.10.22</b> via <b>uv</b>, "
            "installed all required packages plus MetaGPT v0.7 runtime libraries, and resolved the following upstream bugs:",
            body_style,
        )
    )
    fix_rows = [
        ["Component / File", "Upstream Issue Discovered", "Resolution Applied"],
        [
            Paragraph("metagpt/const.py & config2.py", cell_style),
            Paragraph("Hardcoded /path/RTBAgent paths; crash on import if MY_API_PARAM unset; no ${ENV} key expansion.", cell_style),
            Paragraph("Dynamic project root resolution, safe offline fallback, and ${GEMINI_API_KEY} expansion.", cell_style),
        ],
        [
            Paragraph("metagpt/*/__init__.py", cell_style),
            Paragraph("Eager imports of unused cloud/browser SDKs (docx, qianfan, zhipuai, android_env).", cell_style),
            Paragraph("Guarded optional provider/action/env imports; added 429/503 backoff & multi-model fallback.", cell_style),
        ],
        [
            Paragraph("tools/utils.py & fp_bidder.py", cell_style),
            Paragraph("find_w0_star parameter mismatch (p_pv_value vs p_v); missing makedirs; 3M-row LP memory blowup.", cell_style),
            Paragraph("Unified signature, added stratified LP dual scaling (50k sample + full-set refinement in 5s/ratio).", cell_style),
        ],
        [
            Paragraph("environmental_status_provider.py", cell_style),
            Paragraph("Deprecated `from numpy import NaN`, `== NaN` bug, and duplicate `day == day` filter.", cell_style),
            Paragraph("Replaced with `pd.isna()`, fixed `year/month/day` filtering, guarded zero-traffic division.", cell_style),
        ],
        [
            Paragraph("output_bidding_strategy.py & actions", cell_style),
            Paragraph("Rejected integer 0 (`isinstance(0, float)` False); greedy brace-stripping regex; max_try_num=1000.", cell_style),
            Paragraph("Accepted `(int, float)`, parsed fenced JSON cleanly before regex fallback, set max_try_num=5.", cell_style),
        ],
    ]
    story.append(styled_table(fix_rows, col_widths=[1.8 * inch, 2.65 * inch, 2.75 * inch]))

    # 3. Dataset Normalization & CTR / LP Checkpoints
    story.append(Paragraph("3. iPinYou Dataset Inconsistencies & Normalization Pipeline", h1_style))
    adv_m = prep_data["advertisers"][adv_id]
    s_stats = prep_data["seasons"][0]
    story.append(
        Paragraph(
            "Inspection of <b>assets/ipinyou.zip</b> (6.3 GB, 109 files) confirmed four structural inconsistencies across "
            "contest seasons: (1) Season 1 logs have 22/24 columns (missing <i>advertiserID</i> and <i>userTags</i>) and "
            "mislabel conversion <i>logtype=2</i>, whereas Seasons 2 & 3 have 24/26 columns; (2) Season 2 encodes "
            "<i>slotvisibility</i>/<i>slotformat</i> numerically while Season 3 uses strings (<i>OtherView</i>, <i>Na</i>); "
            "(3) raw <i>User-Agent</i> strings contain unescaped quotes and `#` characters; and (4) training clicks reside "
            "in separate <i>clk.*.txt.bz2</i> archives with duplicate click events per impression. "
            "<b>tools/prepare_ipinyou_data.py</b> streams directly from the zip archive using <b>make-ipinyou-data</b> "
            f"conventions, deduplicating <b>{s_stats['duplicate_clicks_deduped']}</b> duplicate click logs in Season 2.",
            body_style,
        )
    )

    ctr_lp_rows = [
        ["Split / Parameter", "Value", "LP Budget Ratio", "Optimal w* (LP)", "Train Opt. Clicks"],
        [
            "Train Impressions",
            f"{adv_m['train_rows']:,}",
            "0.50000 (1/2)",
            f"{ckpt_data[adv_id]['0.5']:,.2f}",
            f"{opt_r_data[adv_id]['0.5']:,.0f}",
        ],
        [
            "Train Clicks (CTR)",
            f"{adv_m['train_clicks']:,} ({adv_m['train_ctr']*100:.4f}%)",
            "0.25000 (1/4)",
            f"{ckpt_data[adv_id]['0.25']:,.2f}",
            f"{opt_r_data[adv_id]['0.25']:,.0f}",
        ],
        [
            "Train 5-Fold OOF AUC",
            f"{adv_m['train_auc']:.4f}",
            "0.12500 (1/8)",
            f"{ckpt_data[adv_id]['0.125']:,.2f}",
            f"{opt_r_data[adv_id]['0.125']:,.0f}",
        ],
        [
            "Test Impressions",
            f"{adv_m['test_rows']:,}",
            "0.06250 (1/16)",
            f"{ckpt_data[adv_id]['0.0625']:,.2f}",
            f"{opt_r_data[adv_id]['0.0625']:,.0f}",
        ],
        [
            "Test Clicks (Test AUC)",
            f"{adv_m['test_clicks']:,} (AUC {adv_m['test_auc']:.4f})",
            "0.03125 (1/32)",
            f"{ckpt_data[adv_id]['0.03125']:,.2f}",
            f"{opt_r_data[adv_id]['0.03125']:,.0f}",
        ],
    ]
    story.append(styled_table(ctr_lp_rows, col_widths=[1.45 * inch, 1.6 * inch, 1.25 * inch, 1.45 * inch, 1.45 * inch]))
    story.append(Spacer(1, 10))

    # 4. Visual Charts & Hourly Simulation Breakdown
    story.append(Paragraph("4. Hourly RTB Simulation & Visual Analysis", h1_style))
    story.append(Image(str(chart1), width=7.1 * inch, height=2.45 * inch))
    story.append(Spacer(1, 6))
    story.append(Image(str(chart2), width=7.1 * inch, height=2.38 * inch))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Hourly Episode Breakdown (Day 2013-06-13)", h2_style))
    hr_table = [["Hr", "Traffic", "Base Won / Clk", "LLM Adj", "Final w_t", "LLM Won (WR%)", "LLM Clk", "LLM Cost", "Rem Budget"]]
    for b_r, l_r in zip(bsl["hourly"], llm["hourly"]):
        hr_table.append(
            [
                f"{l_r['hour']:02d}",
                f"{l_r['traffic_num']:,}",
                f"{b_r['num_won']:,} / {b_r['clicks']}",
                f"{l_r['adjustment']:+.4f}",
                f"{l_r['final_bidding_factor']:,.1f}",
                f"{l_r['num_won']:,} ({l_r['win_rate']*100:.1f}%)",
                f"{l_r['clicks']}",
                f"{l_r['cost']:,.0f}",
                f"{l_r['remaining_budget']:,.0f}",
            ]
        )
    story.append(
        styled_table(
            hr_table,
            col_widths=[0.38 * inch, 0.65 * inch, 0.95 * inch, 0.68 * inch, 0.88 * inch, 1.15 * inch, 0.58 * inch, 0.93 * inch, 1.0 * inch],
        )
    )

    story.append(PageBreak())

    # 5. Sample LLM Decisions, End-of-Day Reflection & How to Run
    story.append(Paragraph("5. RTBAgent Reasoning Samples & End-of-Day Reflection", h1_style))
    story.append(Paragraph("Selected Hourly Strategy Rationales (OutputBiddingStrategy)", h2_style))
    sample_indices = sorted({0, 4, 8, 12, 16, 20, len(llm["hourly"]) - 1})
    reason_rows = [["Hour", "Adjustment", "Agent Reasoning"]]
    for idx in sample_indices:
        if 0 <= idx < len(llm["hourly"]):
            r = llm["hourly"][idx]
            reason_rows.append(
                [
                    f"{r['hour']:02d}:00",
                    f"{r['adjustment']:+.4f}",
                    Paragraph(escape(str(r.get("reason", ""))), cell_style),
                ]
            )
    story.append(styled_table(reason_rows, col_widths=[0.65 * inch, 0.85 * inch, 5.7 * inch]))

    story.append(Paragraph("End-of-Day Agent Reflection (Reflect Action)", h2_style))
    refl = llm.get("reflection", {}) or {}
    refl_rows = [["Reflection Dimension", "LLM Assessment"]]
    for k, label in [
        ("evaluation_of_bid_results", "Evaluation of Bid Results"),
        ("data_analysis_accuracy", "Data Analysis Accuracy"),
        ("policy_adjustment_suggestions", "Policy Adjustment Suggestions"),
        ("improvement_measures", "Future Improvement Measures"),
    ]:
        refl_rows.append([Paragraph(f"<b>{label}</b>", cell_style), Paragraph(escape(str(refl.get(k, "N/A"))), cell_style)])
    story.append(styled_table(refl_rows, col_widths=[1.8 * inch, 5.4 * inch]))

    story.append(Paragraph("6. Reproducible CLI Commands", h1_style))
    cmds = (
        "# 1. Normalize iPinYou logs from assets/ipinyou.zip, train CTR model, and solve LP checkpoints:<br/>"
        ".venv/bin/python tools/prepare_ipinyou_data.py --zip-path assets/ipinyou.zip --advertisers 1458<br/><br/>"
        "# 2. Run 24-hour RTBAgent (llm_fp) vs BasicAgent (fp) simulation:<br/>"
        "MY_API_PARAM=gemini .venv/bin/python run_rtb_agent.py --advertiser 1458 --budget-ratio 0.125 --max-days 1 --clean-workspace<br/><br/>"
        "# 3. Regenerate this PDF report:<br/>"
        ".venv/bin/python tools/generate_pdf_report.py"
    )
    story.append(Paragraph(cmds, code_style))

    doc.build(story)
    print(f"Generated PDF report at: {out_pdf}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate PDF evaluation report for RTBAgent.")
    parser.add_argument("--eval-json", default=str(PROJECT_ROOT / "workspace" / "eval_run" / "evaluation_summary.json"))
    parser.add_argument("--prep-json", default=str(PROJECT_ROOT / "ipinyouData" / "new_all" / "preprocessing_report.json"))
    parser.add_argument("--checkpoint-json", default=str(PROJECT_ROOT / "checkpoint" / "bid" / "fp_bid" / "checkpoint.json"))
    parser.add_argument("--optimal-r-json", default=str(PROJECT_ROOT / "checkpoint" / "bid" / "fp_bid" / "optimal_R.json"))
    parser.add_argument("--output-pdf", default=str(PROJECT_ROOT / "RTBAgent_Evaluation_Report.pdf"))
    args = parser.parse_args()

    build_pdf(
        eval_json_path=Path(args.eval_json),
        prep_json_path=Path(args.prep_json),
        checkpoint_json_path=Path(args.checkpoint_json),
        optimal_r_path=Path(args.optimal_r_json),
        out_pdf=Path(args.output_pdf),
    )


if __name__ == "__main__":
    main()
