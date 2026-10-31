import math
import numpy as np
import pandas as pd
import pulp

def simulate_ad_bidding( clicks, bids, market_prices):
    status = bids >= market_prices
    cost = market_prices * status
    value = clicks * status
    return value, cost, status

def find_w0_star(
    p_v=None,
    c=None,
    B=None,
    v=None,
    c_star=0,
    *,
    p_pv_value=None,
    market_price=None,
    budget=None,
    pv_value=None,
    max_lp_samples=50000,
):
    if p_v is None:
        p_v = p_pv_value
    if c is None:
        c = market_price
    if B is None:
        B = budget
    if v is None:
        v = pv_value

    p_v = np.asarray(p_v, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    current_budget = float(B - c_star)
    if current_budget <= 0 or len(c) == 0:
        return 0.0, 0.0

    # If the dataset has hundreds of thousands / millions of rows, solve the LP dual on a representative subsample
    # (scaling budget proportionally) and then evaluate & refine on the full array.
    n = len(c)
    if max_lp_samples and n > max_lp_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_lp_samples, replace=False)
        lp_pv = p_v[idx]
        lp_c = c[idx]
        lp_B = current_budget * (max_lp_samples / float(n))
    else:
        lp_pv = p_v
        lp_c = c
        lp_B = current_budget

    # traditional optimization approach
    prob = pulp.LpProblem("LP_Problem", pulp.LpMinimize)
    alpha = pulp.LpVariable("alpha", lowBound=1e-12)
    r = {i: pulp.LpVariable(f"r_{i}", lowBound=0) for i in range(len(lp_c))}
    prob += lp_B * alpha + pulp.lpSum(r.values())
    for i in range(len(lp_c)):
        prob += float(lp_c[i]) * alpha + r[i] >= float(lp_pv[i])
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    alpha_val = pulp.value(alpha)
    w_star = float(1.0 / max(alpha_val if alpha_val is not None else 1e-6, 1e-12))

    bids = p_v * w_star
    value, cost, status = simulate_ad_bidding(v, bids, c)
    over_cost_ratio = max((np.sum(cost) - current_budget) / (np.sum(cost) + 1e-4), 0)
    rng_drop = np.random.default_rng(42)
    while over_cost_ratio > 0:
        pv_index = np.where(status == 1)[0]
        if pv_index.shape[0] == 0:
            break
        drop_cnt = min(pv_index.shape[0], int(math.ceil(pv_index.shape[0] * over_cost_ratio)))
        dropped_pv_index = rng_drop.choice(pv_index, drop_cnt, replace=False)
        bids[dropped_pv_index] = 0
        value, cost, status = simulate_ad_bidding(v, bids, c)
        over_cost_ratio = max((np.sum(cost) - current_budget) / (np.sum(cost) + 1e-4), 0)

    return float(np.sum(value)), float(w_star)

