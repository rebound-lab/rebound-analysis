#!/usr/bin/env python3
"""
analyse.py - rebound height of a dropped ball from the tracking CSV made by
track.py, with the parallax between the ball and the tape measure worked out
from the free-fall acceleration g.

    python analyse.py config.json                 # every video in the config
    python analyse.py config.json --only 0mm 2mm  # just these entries
    python analyse.py config.json --out results   # folder for the CSV output

Method (all heights in metres, image rows counted downwards from the top)
-------------------------------------------------------------------------
1.  Tape scale.  A straight line is fitted to the calibration marks
    (pixel row, tape reading).  Its slope gives the tape scale S_t in px/m.

2.  Line of sight.  The horizontal line of sight through the lens meets the
    ball's plane and the tape at the same height L, so on that image row
    (row_L) there is no parallax.  row_L is either the row where the tape
    reads the measured lens height (lens_height_m), or a given row
    (level_row_px, default the middle row of the frame, which is the line of
    sight when the phone is held level).  L is the tape reading on row_L.

3.  Rebounds.  Frames where the whole ball is visible are linked into runs.
    A run counts as a rebound when it starts low in the frame, moves upwards
    and the ball was higher just before (it fell, hit the floor, came back).

4.  Ball-plane scale.  Near the top of each rebound (within fit_window_s of
    the highest point) the centre row is fitted with y = a t^2 + b t + c.
    The ball is in free fall, so 2a (px/s^2) equals g (m/s^2) times the scale
    S_b of the ball's plane: S_b = 2a/g.  The median S_b of all good fits in a
    video is used for that video.  Because the ball is closer to the camera
    than the tape, S_b > S_t.

5.  Rebound height.  Height of the ball's centre at the top of the parabola:
        H = L + (row_L - y_top) / S_b
    The bottom of the ball is H - r and the rebound height above the landing surface is
        h2 = H - r - d.

6.  Uncorrected (tape) reading.  Reading the same apex against the tape gives
        u = L + (row_L - y_top) / S_t - r - d  =  h2 + (H - L) * k,
    with k = S_b/S_t - 1.  The difference u - h2 is the parallax error dp.

7.  Rejected drops (decided before looking at the heights): the top of the
    rebound fell outside the recorded frames by more than max_extrapolation_s,
    the ball moved sideways faster than max_sideways_m_s, the parabola fit was
    poor, or the ball was off the drop line.

8.  Summary per video: the first n_trials accepted drops are used; a drop
    further than anomaly_factor x half-range from the mean of the other drops
    is marked anomalous and replaced by the next accepted drop.  Then
        mean corrected h2, mean parallax correction dp, dh2 = half-range
        (or the reading uncertainty if that is larger),
        ratio = sqrt(h2/h1) and d(ratio)/ratio = (dh2/h2 + dh1/h1)/2.
"""
import argparse
import csv
import json
import math
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

DEFAULTS = {
    # experiment
    "drop_height_m": 2.000,       # height the ball is released from (bottom of ball)
    "drop_height_unc_m": 0.002,   # uncertainty in h1
    "ball_radius_m": 0.0335,      # ball radius (67 mm diameter)
    "thickness_mm": 0,            # thickness of the layer the ball lands on
    "g": 9.81,                    # m/s^2
    # camera geometry
    "calibration": None,          # [[pixel_row, tape_reading_m], ...]  (required)
    "lens_height_m": None,        # measured height of the lens; if set, the line of sight is the
                                  # image row where the tape reads this height
    "level_row_px": None,         # otherwise: image row of the line of sight (default: middle row)
    "centre_x_px": None,          # column of the optical axis (default: middle column)
    "tape_line": None,            # {"x_px":..,"y_px":..,"slope":..} tape column at one row and
                                  # its drift in px per px (camera roll); None = no roll
    # rebound detection
    "group_gap_s": 1.3,           # a pause this long separates one drop from the next
    "link_gap_s": 0.075,          # max time between linked frames in a rebound
    "min_run_frames": 5,
    "start_row_frac": 0.573,      # a rebound must start below this fraction of the frame height
    "fall_rows_frac": 0.15625,     # ... and the ball must have been this much higher just before
    "fall_window_s": 0.6,
    "require_fall": True,
    "scale_min_width_frac": 0.75,  # settings used to find the rebounds that set the ball-plane
    "scale_require_fall": True,    # scale S_b (kept strict even when the two above are relaxed)
    "min_width_frac": 0.75,       # a frame counts as 'whole ball' when width >= this x ball width
    "width_area_px": 2000,        # blobs larger than this set the typical ball width
    "width_from_top_blobs": False,  # also estimate ball width from the largest 40 % of blobs
    "skip_duplicate_frames": False,  # treat identical consecutive detections as missing
                                     # (for videos re-encoded with repeated frames)
    # fitting and rejection
    "fit_window_s": 0.21,
    "max_extrapolation_s": 0.034,
    "max_sideways_m_s": 0.30,
    "max_rms_m": 0.003,          # parabola fit residual (rms) must be below 3 mm
    "off_line_widths": 1.2,
    # summary
    "n_trials": 5,
    "anomaly_factor": 3.0,
    "round_heights_m": 0.001,
    "reading_unc_m": 0.005,       # smallest uncertainty used for the mean rebound height     # per-drop heights are reported to the nearest mm
}


def rnd(x, step):
    """Round half up to a multiple of `step` (e.g. 0.001)."""
    if step is None:
        return float(x)
    q = Decimal(str(step))
    return float((Decimal(repr(float(x))) / q).quantize(Decimal(1), ROUND_HALF_UP) * q)


# --------------------------------------------------------------------------- input
def read_tracking(path):
    meta, rows = {}, []
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    for ln in lines:
        if ln.startswith("#") and "=" in ln:
            k, v = ln[1:].strip().split("=", 1)
            meta[k.strip()] = v.strip()
    body = [ln for ln in lines if not ln.startswith("#")]
    for r in csv.DictReader(body):
        t = float(r["time_s"])
        if r["x_px"] == "":
            rows.append((t, None))
        else:
            rows.append((t, (float(r["x_px"]), int(r["top_px"]), int(r["bottom_px"]),
                             int(r["width_px"]), int(r["height_px"]), int(r["area_px"]))))
    return meta, rows


# --------------------------------------------------------------------------- analysis
def analyse(rows, meta, c):
    width = int(meta.get("width", 1080))
    height = int(meta.get("height", 1920))
    g = c["g"]
    r_ball = c["ball_radius_m"]
    d = c["thickness_mm"] / 1000.0
    row_L = c["level_row_px"] if c["level_row_px"] is not None else height / 2
    cx = c["centre_x_px"] if c["centre_x_px"] is not None else width / 2

    # 1. tape scale
    cal = np.array(c["calibration"], float)
    pt = np.polyfit(cal[:, 0], cal[:, 1], 1)       # tape reading (m) = pt[0]*row + pt[1]
    s_t = -1.0 / pt[0]                              # px per m on the tape

    # camera roll: rows are shifted by slope*(x - cx) before comparing heights
    tl = c["tape_line"] or {"x_px": cx, "y_px": row_L, "slope": 0.0}
    b0 = tl["slope"]

    def tape_at(row):                # tape reading level with image row `row` at the optical axis
        x_tape = tl["x_px"] + b0 * (row - tl["y_px"])
        return float(np.polyval(pt, row - b0 * (x_tape - cx)))

    # 2. line of sight: the image row level with the lens, and its height L
    if c["lens_height_m"] is not None:
        # find the row where the tape reads the measured lens height
        L = float(c["lens_height_m"])
        for _ in range(3):
            row_L = row_L + (L - tape_at(row_L)) / pt[0]
    L = tape_at(row_L)

    # detections
    if c["skip_duplicate_frames"]:
        clean = []
        for j, (t, b) in enumerate(rows):
            if b is not None and j > 0 and rows[j - 1][1] is not None and rows[j - 1][1][:5] == b[:5]:
                clean.append((t, None))
            else:
                clean.append((t, b))
        rows = clean
    det = [(t, b) for t, b in rows if b is not None]
    if not det:
        sys.exit("no detections in tracking file")
    X, TOP, BOT, WID, HEI, AREA = range(6)

    W = np.median([b[WID] for _, b in det if b[AREA] > c["width_area_px"] and 0.8 < b[WID] / b[HEI] < 1.25])
    if c["width_from_top_blobs"]:
        a95 = np.percentile([b[AREA] for _, b in det], 95)
        W = np.nanmax([W, np.median([b[WID] for _, b in det
                                     if b[AREA] > 0.6 * a95 and 0.8 < b[WID] / b[HEI] < 1.25 and b[TOP] > 2])])
    start_row = c["start_row_frac"] * height
    fall_rows = c["fall_rows_frac"] * height

    def yc(b):                       # centre row; if the top is cut off use bottom - W/2
        return b[BOT] - W / 2 if b[TOP] <= 2 else (b[TOP] + b[BOT]) / 2

    # 3. split into drops
    groups, prev = [], -1e9
    for t, b in det:
        if t - prev > c["group_gap_s"]:
            groups.append([])
        groups[-1].append((t, b))
        prev = t

    def find_fits(wlo, require_fall):
        """Find the rebound in each drop and fit a parabola near its top."""

        def whole(b):                # whole ball visible (top may touch the frame edge)
            if b[TOP] <= 2:
                return wlo * W <= b[WID] <= 1.3 * W and b[BOT] >= 0.35 * W
            return wlo * W <= b[WID] <= 1.3 * W and 0.7 * W <= b[HEI] <= 1.45 * W

        def is_rebound(run):
            return (len(run) >= c["min_run_frames"] and yc(run[0][1]) > start_row
                    and run[1][1][TOP] < run[0][1][TOP])

        fits = []
        for G in groups:
            reb, cur = None, []
            for t, b in G:
                if whole(b) and (not cur or (t - cur[-1][0] <= c["link_gap_s"]
                                             and abs(b[X] - cur[-1][1][X]) < 0.45 * W)):
                    cur.append((t, b))
                else:
                    if is_rebound(cur):
                        reb = cur
                        break
                    cur = [(t, b)] if whole(b) else []
            if reb is None and is_rebound(cur):
                reb = cur
            if reb is not None and require_fall:
                t0, y0 = reb[0][0], yc(reb[0][1])
                if not any(t0 - c["fall_window_s"] <= t < t0 and yc(b) < y0 - fall_rows for t, b in G):
                    reb = None
            if reb is None:
                fits.append(dict(t=G[0][0], note="no clean rebound tracked"))
                continue
            tt = np.array([t for t, _ in reb])
            y = np.array([yc(b) for _, b in reb])
            x = np.array([b[X] for _, b in reb])
            w = np.array([b[WID] for _, b in reb])
            i = int(np.argmin(y))
            seen = 2 <= i <= len(y) - 3                  # highest point inside the run
            o = dict(t=G[0][0], tt=tt, y=y, x=x, w=w, seen=seen)
            if seen:                                     # 4. parabola near the top
                sel = np.abs(tt - tt[i]) <= c["fit_window_s"]
                p = np.polyfit(tt[sel], y[sel], 2)
                tv = -p[1] / (2 * p[0])
                o.update(s=2 * p[0] / g, tv=tv, yv=np.polyval(p, tv),
                         rms=np.std(y[sel] - np.polyval(p, tt[sel])))
            fits.append(o)
        return fits

    fits = find_fits(c["min_width_frac"], c["require_fall"])
    # the ball-plane scale always comes from rebounds found with the standard (strict)
    # settings, even when the detection was relaxed to measure more rebound heights
    if (c["min_width_frac"], c["require_fall"]) == (c["scale_min_width_frac"], c["scale_require_fall"]):
        fits_scale = fits
    else:
        fits_scale = find_fits(c["scale_min_width_frac"], c["scale_require_fall"])

    good = [o["s"] for o in fits_scale if o.get("s") and o["rms"] < c["max_rms_m"] * o["s"]]
    S_b = float(np.median(good)) if good else s_t * 1.12
    k = S_b / s_t - 1.0
    w_med = float(np.median([np.median(o["w"]) for o in fits if "w" in o]))
    x_med = float(np.median([np.median(o["x"]) for o in fits if "x" in o]))

    drops = []
    for n, o in enumerate(fits, 1):
        if "note" in o:
            drops.append(dict(drop=n, time_s=o["t"], accepted=False, reason=o["note"]))
            continue
        tt, y, x, w = o["tt"], o["y"], o["x"], o["w"]
        s_i = S_b * np.median(w) / w_med
        if o["seen"] and 0.8 * S_b < o["s"] < 1.2 * S_b and o["rms"] < c["max_rms_m"] * S_b:
            tv, yv, s_used = o["tv"], o["yv"], o["s"]
        else:                                        # top not seen: fix the curvature from g
            a = s_i * g / 2
            if o["seen"]:
                sel = np.abs(tt - tt[int(np.argmin(y))]) <= c["fit_window_s"]
            else:
                sel = np.arange(len(tt)) >= max(0, len(tt) - 7)
            B = np.polyfit(tt[sel], y[sel] - a * tt[sel] ** 2, 1)
            tv = -B[0] / (2 * a)
            yv = a * tv ** 2 + B[0] * tv + B[1]
            s_used = s_i
        beyond = tv - tt.max()
        xv = float(np.median(x))
        # 5. centre height, bottom of ball, height above the landing surface
        H = L + ((row_L - yv) - b0 * (xv - cx)) / s_used
        h2 = H - r_ball - d
        rise = tt <= tv + 0.01
        vx = np.polyfit(tt[rise], x[rise], 1)[0] / s_used if rise.sum() >= 3 else float("nan")
        why = []
        if beyond > c["max_extrapolation_s"]:
            why.append(f"top of rebound {beyond * 1000:.0f} ms after the last frame")
        if abs(vx) > c["max_sideways_m_s"]:
            why.append(f"moving sideways {vx:+.2f} m/s")
        if o["seen"] and o["rms"] >= c["max_rms_m"] * S_b:
            why.append("tracking unreliable")
        if abs(np.median(x) - x_med) > c["off_line_widths"] * w_med:
            why.append("not on the drop line")
        drops.append(dict(drop=n, time_s=float(o["t"]), accepted=not why, reason="; ".join(why),
                          h2_corrected_m=float(h2), vx_m_s=float(vx), scale_px_per_m=float(s_used)))

    # 6. uncorrected (tape) reading and parallax for every drop
    step = c["round_heights_m"]
    for dr in drops:
        if "h2_corrected_m" in dr:
            h = rnd(dr["h2_corrected_m"], step)
            u = rnd(h + (h + d - L + r_ball) * k, step)
            dr.update(h2_corrected_m=h, h2_uncorrected_m=u, parallax_m=rnd(u - h, step))

    geom = dict(tape_scale_px_per_m=s_t, ball_scale_px_per_m=S_b, k=k, line_of_sight_m=L,
                level_row_px=row_L, ball_width_px=w_med, n_scale_fits=len(good))
    return drops, geom


# --------------------------------------------------------------------------- summary
def summarise(drops, geom, c):
    ok = [dr for dr in drops if dr["accepted"]]
    n = c["n_trials"]
    sel, anom, pool = [], [], list(ok)
    while len(sel) < n and pool:
        sel.append(pool.pop(0))
        if len(sel) == n:
            changed = True
            while changed:
                changed = False
                for i, cand in enumerate(sel):
                    oth = [o["h2_corrected_m"] for j, o in enumerate(sel) if j != i]
                    hr = (max(oth) - min(oth)) / 2
                    if abs(cand["h2_corrected_m"] - np.mean(oth)) > c["anomaly_factor"] * hr and pool:
                        anom.append(sel.pop(i))
                        sel.append(pool.pop(0))
                        changed = True
                        break
    for dr in drops:
        dr["used"] = dr in sel
        dr["anomalous"] = dr in anom
    if len(sel) < 2:
        return None
    d = c["thickness_mm"] / 1000.0
    k, L, r = geom["k"], geom["line_of_sight_m"], c["ball_radius_m"]
    step = c["round_heights_m"]
    h = [o["h2_corrected_m"] for o in sel]          # parallax-corrected heights
    u = [o["h2_uncorrected_m"] for o in sel]        # tape readings, for reference
    h1 = rnd(c["drop_height_m"] - d, step)
    m = float(np.mean(h))
    # half-range worked out in whole steps (mm) so that e.g. 38.5 mm rounds to 39 mm
    steps = [round(x / step) for x in h] if step else h
    hr = rnd((max(steps) - min(steps)) / 2, 1) * step if step else (max(h) - min(h)) / 2
    dp = rnd(float(np.mean(u)) - m, step)           # mean parallax correction that was removed
    dh2 = max(hr, c["reading_unc_m"])               # half-range, or the reading uncertainty if larger
    ratio = math.sqrt(m / h1)
    frac = 0.5 * (dh2 / m + c["drop_height_unc_m"] / h1)
    d_ratio = ratio * frac
    return dict(thickness_mm=c["thickness_mm"], n_used=len(sel), n_anomalous=len(anom),
                h1_m=h1, mean_h2_m=m, half_range_m=hr, parallax_m=dp, dh2_m=dh2,
                ratio=ratio, d_ratio=d_ratio, d_ratio_percent=100 * frac,
                k=k, line_of_sight_m=L)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="JSON file with 'defaults' and a list of 'videos'")
    ap.add_argument("--only", nargs="*", help="names of the config entries to run")
    ap.add_argument("--out", default="results", help="folder for the output CSV files")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    base = os.path.dirname(os.path.abspath(a.config))
    os.makedirs(a.out, exist_ok=True)
    summaries = []
    for entry in cfg["videos"]:
        if a.only and entry["name"] not in a.only:
            continue
        c = dict(DEFAULTS)
        c.update(cfg.get("defaults", {}))
        c.update(entry)
        if not c.get("calibration"):
            sys.exit(f"{entry['name']}: 'calibration' is required")
        meta, rows = read_tracking(os.path.join(base, c["tracking"]))
        drops, geom = analyse(rows, meta, c)
        s = summarise(drops, geom, c)

        for dr in drops:
            for key in ("time_s", "vx_m_s", "scale_px_per_m"):
                if key in dr:
                    dr[key] = round(dr[key], 3)
        cols = ["drop", "time_s", "accepted", "used", "anomalous", "h2_corrected_m", "h2_uncorrected_m",
                "parallax_m", "vx_m_s", "scale_px_per_m", "reason"]
        with open(os.path.join(a.out, f"{entry['name']}_drops.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(drops)
        if not a.quiet:
            print(f"== {entry['name']}  d = {c['thickness_mm']} mm   tape {geom['tape_scale_px_per_m']:.1f} px/m, "
                  f"ball plane {geom['ball_scale_px_per_m']:.1f} px/m (k = {geom['k']:.3f}, "
                  f"{geom['n_scale_fits']} fits), line of sight L = {geom['line_of_sight_m']:.3f} m")
            for dr in drops:
                tag = ("used" if dr.get("used") else "ANOMALY" if dr.get("anomalous") else
                       "spare" if dr["accepted"] else "rejected: " + dr["reason"])
                if "h2_corrected_m" in dr:
                    print(f"   #{dr['drop']:2d} {dr['time_s']:7.1f} s  h2 (tape) {dr['h2_uncorrected_m']:.3f} m  "
                          f"corrected {dr['h2_corrected_m']:.3f} m  dp {dr['parallax_m']:.3f} m   {tag}")
                else:
                    print(f"   #{dr['drop']:2d} {dr['time_s']:7.1f} s  {tag}")
            if s:
                print(f"   mean h2 (corrected) {s['mean_h2_m']:.3f} ± {s['dh2_m']:.3f} m  (parallax correction {s['parallax_m']:.3f} m)  sqrt(h2/h1) = {s['ratio']:.3f} ± {s['d_ratio']:.3f} "
                      f"({s['d_ratio_percent']:.1f} %)")
        if s:
            summaries.append(dict(name=entry["name"], **{key: (round(v, 6) if isinstance(v, float) else v)
                                                         for key, v in s.items()}))

    if summaries:
        with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, list(summaries[0].keys()))
            wr.writeheader()
            wr.writerows(summaries)
        print(f"\nwrote {os.path.join(a.out, 'summary.csv')} and one *_drops.csv per video")


if __name__ == "__main__":
    main()
