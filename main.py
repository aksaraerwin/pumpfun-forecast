#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: a horizon-aware Monte Carlo forecaster.

Reads /input/history.ndjson (10-minute marks of pump.fun coins launched at or
after the opening) and /input/task.json (the seven answer reads, the size-bin
edges), and writes one static /output/forecast.json: per place 1..5 a
probability for each named coin, a top5, and per place a probability for each
size bin.

Model, in one paragraph. Market caps move in log10 SOL. An incumbent coin's
cap at a read is drawn from a two-component mixture around its projected
level: most 10-minute steps barely move (the caps are sticky), a minority
jump, so the base component is narrow and the jump component wide, both
widening with the square root of the horizon. The projected level follows the
coin's own trend, estimated over its observed life and shrunk hard toward a
winsorised class drift (graduated vs still on the curve), with the drift's
effect saturating over ~1.5 hours so a 13-hour horizon cannot extrapolate a
10-minute run-up, and a fade term that takes the typical incumbent down a
fraction of a decade as the horizon grows (most leaders are displaced or
decay; the practice answer's last read had 3 of its top 5 born after the
cut). Coins launched after the cut, or never ranked in the history, are
drawn as ghosts: bootstrap samples from the (age, cap) points the history
itself recorded, restricted to ages the horizon allows, as many draws as
the measured birth rate times the horizon implies reach the recorded rows.
Place sizes blend the Monte Carlo place-cap histogram (shifted by the
market-wide place-cap trend the marks themselves show: a young universe
inflates, a mature one flattens) with a normal around each place's own
trend-projected cap, plus a small uniform mix. A fixed-seed Monte Carlo
over the seven read horizons produces the marginals; the one forecast is
their average over reads, the log-score-optimal static forecast against
per-read marginals. Deterministic: same input, same bytes out.

Python 3.13, standard library only, no network.
"""

import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
SIMS = 6000                 # Monte Carlo draws per answer read
MAX_CANDIDATES = 140        # named coins considered per simulation
GHOST_MIN, GHOST_MAX = 8, 640
TAU_DRIFT_H = 1.5           # drift saturates with this time constant (hours)
FADE, TAU_FADE_H = 0.30, 2.5  # the typical incumbent loses this much altitude
                              # (log10) as the horizon grows: most top coins
                              # are displaced or decay, they do not hold level
SLOPE_CAP = 0.8             # per-hour log10 slope cap, either way
JUMP_W = 0.12               # weight of the jump component
S_BASE_STEP = 0.045         # base component sd per cadence step (log10)
S_JUMP_STEP = 0.33          # jump component sd per cadence step (log10)
S_BASE_CAP, S_JUMP_CAP = 0.30, 1.0
GHOST_JITTER = 0.05
ROW_MIN_P = 0.003           # smallest named-coin probability row emitted
BIN_MIN_P = 0.001           # smallest size-bin row emitted
TREND_W = 0.35              # weight of the place-cap trend component in sizes
SIZE_UNIFORM = 0.015        # uniform mix over bins, insurance against zeros
SLOPE_CAP_MARKET = 0.5      # per-hour cap on place-cap trend slopes


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def logcap(row):
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def load_input():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    marks.sort(key=lambda m: m["at"])
    return task, marks


class Coin:
    __slots__ = ("mint", "obs", "created", "graduated", "last_L")

    def __init__(self, mint):
        self.mint = mint
        self.obs = []          # (t_hours_since_t0, logcap)
        self.created = 0.0
        self.graduated = False
        self.last_L = 0.0


def build_coins(marks, t0):
    coins = {}
    age_points = []            # (age_hours, logcap) of every recorded row
    for m in marks:
        t = (parse_ts(m["at"]) - t0) / 3600.0
        for r in m["rows"]:
            c = coins.get(r["mint"])
            if c is None:
                c = coins[r["mint"]] = Coin(r["mint"])
                c.created = (parse_ts(r["created_at"]) - t0) / 3600.0
            L = logcap(r)
            c.obs.append((t, L))
            if r.get("venue") == "pool" or r.get("complete"):
                c.graduated = True
            age_points.append((t - c.created, L))
    for c in coins.values():
        c.obs.sort()
        c.last_L = c.obs[-1][1]
    age_points.sort()
    return coins, age_points


def calibrate(marks, coins, cadence_h):
    """Class drift (winsorised, shrunk) and the birth/visibility rates."""
    steps = {"pool": [], "curve": []}
    for c in coins.values():
        cls = "pool" if c.graduated else "curve"
        for a, b in zip(c.obs, c.obs[1:]):
            if abs((b[0] - a[0]) - cadence_h) < cadence_h / 2:
                steps[cls].append(max(-0.3, min(0.3, b[1] - a[1])))
    drift = {}
    for cls, v in steps.items():
        if v:
            mean_step = sum(v) / len(v)
            drift[cls] = mean_step / cadence_h * (len(v) / (len(v) + 25.0))
        else:
            drift[cls] = 0.0
    first, last = marks[0], marks[-1]
    span_h = max((parse_ts(last["at"]) - parse_ts(first["at"])) / 3600.0, cadence_h)
    births_per_h = max(0.0, (last["counts"]["eligible"] - first["counts"]["eligible"])) / span_h
    seen_per_h = len(coins) / (span_h + cadence_h)
    row_share = min(1.0, seen_per_h / births_per_h) if births_per_h > 0 else 0.25
    return drift, births_per_h, row_share


def wls_slope(points):
    """Recency-weighted slope (per hour) of a (t_hours, value) series."""
    if len(points) < 3:
        return 0.0
    t_last = points[-1][0]
    span = max(t_last - points[0][0], 1e-9)
    half = max(span / 2.0, 1e-9)
    sw = sx = sy = sxx = sxy = 0.0
    for t, v in points:
        w = 0.5 ** ((t_last - t) / half)
        sw += w
        sx += w * t
        sy += w * v
        sxx += w * t * t
        sxy += w * t * v
    den = sw * sxx - sx * sx
    if den <= 0:
        return 0.0
    return (sw * sxy - sx * sy) / den


def place_cap_trends(marks, t0, places):
    """Per-place (last cap, shrunk slope) from the marks' own top rows, plus
    the cross-place market slope: a young universe inflates, a mature one
    flattens, and the recent marks say which."""
    series = [[] for _ in range(places)]
    for m in marks:
        t = (parse_ts(m["at"]) - t0) / 3600.0
        rows = sorted(m["rows"], key=lambda r: r["rank"])
        for k in range(min(places, len(rows))):
            series[k].append((t, logcap(rows[k])))
    shrink = len(marks) / (len(marks) + 4.0)
    trends = []
    for k in range(places):
        if series[k]:
            slope = max(-SLOPE_CAP_MARKET, min(SLOPE_CAP_MARKET, wls_slope(series[k]) * shrink))
            trends.append((series[k][-1][1], slope))
        else:
            trends.append((None, 0.0))
    slopes = [s for (lv, s) in trends if lv is not None]
    market_slope = sum(slopes) / len(slopes) if slopes else 0.0
    return trends, market_slope


def coin_slope(c, drift):
    cls = "pool" if c.graduated else "curve"
    n = len(c.obs)
    if n >= 2:
        (t0_, l0_), (t1_, l1_) = c.obs[0], c.obs[-1]
        if t1_ > t0_:
            own = max(-SLOPE_CAP, min(SLOPE_CAP, (l1_ - l0_) / (t1_ - t0_)))
            w = n / (n + 6.0)
            return w * own + (1 - w) * drift[cls]
    return drift[cls]


def write_out(task, doc):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(doc) + "\n")


def main():
    task, marks = load_input()
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]
    nbins = len(edges) + 1
    t0 = parse_ts(task["t0"])
    cadence_h = float(task.get("cadence_minutes", 10)) / 60.0

    if not marks:
        write_out(task, {"probabilities": [], "top5": [("1" * 31 + str(i)) for i in range(places)], "sizes": []})
        return
    cut_t = parse_ts(marks[-1]["at"])
    horizons_h = [max((parse_ts(r) - cut_t) / 3600.0, cadence_h / 2) for r in task["reads"]]

    coins, age_points = build_coins(marks, t0)
    drift, births_per_h, row_share = calibrate(marks, coins, cadence_h)
    trends, market_slope = place_cap_trends(marks, t0, places)

    # Candidate shortlist by an optimistic bound at the longest horizon.
    h_max = max(horizons_h)
    steps_max = h_max / cadence_h
    s_jump_max = min(S_JUMP_CAP, S_JUMP_STEP * math.sqrt(steps_max))
    scored = []
    for c in coins.values():
        slope = coin_slope(c, drift)
        bound = c.last_L + max(slope, 0.0) * TAU_DRIFT_H + 3.0 * s_jump_max
        scored.append((bound, c))
    scored.sort(key=lambda x: -x[0])
    cands = [c for _, c in scored[:MAX_CANDIDATES]]
    cand_mus_base = [c.last_L for c in cands]
    cand_slopes = [coin_slope(c, drift) for c in cands]

    rng = random.Random(SEED)
    place_hits = [dict() for _ in range(places)]
    bin_hits = [[0.0] * nbins for _ in range(places)]
    total_draws = 0
    for h in horizons_h:
        steps = h / cadence_h
        drift_h = TAU_DRIFT_H * (1.0 - math.exp(-h / TAU_DRIFT_H))
        fade_h = FADE * (1.0 - math.exp(-h / TAU_FADE_H))
        mus = [cand_mus_base[i] + cand_slopes[i] * drift_h - fade_h for i in range(len(cands))]
        s_base = min(S_BASE_CAP, S_BASE_STEP * math.sqrt(steps))
        s_jump = min(S_JUMP_CAP, S_JUMP_STEP * math.sqrt(steps))
        # Ghost pool: recorded caps at ages this horizon allows a newborn.
        pool = [L for (age, L) in age_points if age <= h + cadence_h] or [L for (_, L) in age_points]
        n_pool = len(pool)
        n_ghost = int(max(GHOST_MIN, min(GHOST_MAX, round(births_per_h * h * row_share))))
        for _ in range(SIMS):
            draws = []
            for i in range(len(cands)):
                s = s_jump if rng.random() < JUMP_W else s_base
                draws.append((rng.gauss(mus[i], s), i))
            ghosts = sorted(
                (pool[rng.randrange(n_pool)] + rng.gauss(0.0, GHOST_JITTER) for _ in range(n_ghost)),
                reverse=True,
            )
            draws += [(g, -1) for g in ghosts[:places + 2]]
            draws.sort(key=lambda x: -x[0])
            total_draws += 1
            shift = market_slope * drift_h  # ranking-invariant; sizes only
            for k in range(min(places, len(draws))):
                val, idx = draws[k]
                if idx >= 0:
                    d = place_hits[k]
                    mint = cands[idx].mint
                    d[mint] = d.get(mint, 0) + 1
                b = 0
                sval = val + shift
                while b < len(edges) and edges[b] <= sval:
                    b += 1
                bin_hits[k][b] += 1
    denom = float(total_draws)
    probabilities = []
    for k in range(places):
        rows_k = []
        for mint, cnt in sorted(place_hits[k].items(), key=lambda kv: -kv[1]):
            p = math.floor(cnt / denom * 1e6) / 1e6
            if p >= ROW_MIN_P:
                rows_k.append({"mint": mint, "place": k + 1, "p": p})
        probabilities += rows_k[:100]  # a place may name at most 100 rows
    # Sizes: Monte Carlo place-cap histogram blended with a trend component
    # (a normal around each place's own projected cap, averaged over reads),
    # plus a small uniform mix so no plausible bin is ever given zero.
    def phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    sizes = []
    for k in range(places):
        level, slope_k = trends[k]
        trend = [0.0] * nbins
        if level is not None:
            for h in horizons_h:
                drift_h = TAU_DRIFT_H * (1.0 - math.exp(-h / TAU_DRIFT_H))
                mu = level + slope_k * drift_h
                sd_t = min(0.6, 0.35 + 0.05 * math.sqrt(h))
                for b in range(nbins):
                    lo = edges[b - 1] if b > 0 else -math.inf
                    hi = edges[b] if b < len(edges) else math.inf
                    trend[b] += (phi((hi - mu) / sd_t) - phi((lo - mu) / sd_t)) / len(horizons_h)
        for b in range(nbins):
            p = (1.0 - TREND_W) * (bin_hits[k][b] / denom) + TREND_W * trend[b]
            p = (1.0 - SIZE_UNIFORM) * p + SIZE_UNIFORM / nbins
            p = math.floor(p * 1e6) / 1e6
            if p >= BIN_MIN_P:
                sizes.append({"place": k + 1, "bin": b, "p": p})
    marginals = [dict() for _ in range(places)]
    for r in probabilities:
        marginals[r["place"] - 1][r["mint"]] = r["p"]
    used, top5 = set(), []
    for k in range(places):
        best = max(((p, m) for m, p in marginals[k].items() if m not in used), default=None)
        if best:
            top5.append(best[1])
            used.add(best[1])
    if len(top5) < places:
        for _, c in scored:
            if c.mint not in used:
                top5.append(c.mint)
                used.add(c.mint)
            if len(top5) == places:
                break
    i = 0
    while len(top5) < places:  # history held fewer distinct coins than places
        pad = "1" * 31 + str(i)
        i += 1
        if pad not in used:
            top5.append(pad)
            used.add(pad)
    write_out(task, {"probabilities": probabilities, "top5": top5[:places], "sizes": sizes})


if __name__ == "__main__":
    main()
