#!/usr/bin/env python3
"""
report.py - run the rebound analysis with a readable terminal display and
save a plot of every rebound that was used.

    python report.py data_config.json              # every run
    python report.py data_config.json --only run06 # one run
    python report.py data_config.json --no-plots   # tables only

The calculations are the ones in analyse.py; this script only presents them.
Plots are written to results/plots/.
"""
import argparse
import json
import os
import time

import numpy as np
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

import analyse as A

console = Console()
G = 9.81


def status_text(dr):
    if dr.get("used"):
        return Text("used", style="bold green")
    if dr.get("anomalous"):
        return Text("anomalous", style="bold dark_orange")
    if dr["accepted"]:
        return Text("spare", style="yellow")
    return Text("rejected: " + dr["reason"], style="red")


def run_panel(name, c, geom, drops, s):
    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column(justify="right", style="bold")
    info.add_column(style="dim")
    info.add_column(justify="right", style="bold")
    info.add_row("tape scale", f"{geom['tape_scale_px_per_m']:.0f} px/m",
                 "ball scale (from g)", f"{geom['ball_scale_px_per_m']:.0f} px/m")
    info.add_row("scale ratio k", f"{geom['k']:.3f}",
                 "line of sight L", f"{geom['line_of_sight_m']:.3f} m")

    t = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", pad_edge=False)
    t.add_column("drop", justify="right")
    t.add_column("time / s", justify="right")
    t.add_column("tape reading / m", justify="right")
    t.add_column("parallax / m", justify="right")
    t.add_column("corrected h₂ / m", justify="right", style="bold")
    t.add_column("status")
    for dr in drops:
        if "h2_corrected_m" in dr:
            t.add_row(str(dr["drop"]), f"{dr['time_s']:.1f}", f"{dr['h2_uncorrected_m']:.3f}",
                      f"{dr['parallax_m']:.3f}", f"{dr['h2_corrected_m']:.3f}", status_text(dr))
        else:
            t.add_row(str(dr["drop"]), f"{dr['time_s']:.1f}", "–", "–", "–", status_text(dr))

    if s:
        res = Text.assemble(
            ("mean h₂ = ", "dim"), (f"{s['mean_h2_m']:.3f} ± {s['dh2_m']:.3f} m", "bold"),
            ("    e = √(h₂/h₁) = ", "dim"), (f"{s['ratio']:.3f} ± {s['d_ratio']:.3f}", "bold green"),
            (f"  ({s['d_ratio_percent']:.1f} %)", "dim"))
    else:
        res = Text("not enough accepted drops", style="red")
    title = f"[bold]{name}[/]  ·  layer thickness d = {c['thickness_mm']:g} mm  ·  h₁ = {c['drop_height_m'] - c['thickness_mm'] / 1000:.3f} m"
    return Panel(Group(info, t, res), title=title, title_align="left", border_style="cyan")


def plot_drop(path, name, dr, c, geom):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = dr["_traj"]
    t, h = tr["t"], tr["h"]
    h2, tv = dr["h2_corrected_m"], tr["t_top"]
    k, L = geom["k"], geom["line_of_sight_m"]
    d, r = c["thickness_mm"] / 1000, c["ball_radius_m"]
    tape = h + (h + d + r - L) * k                       # same points read against the tape
    tf = np.linspace(t.min(), max(t.max(), tv) + 0.05, 200)
    hf = h2 - 0.5 * G * (tf - tv) ** 2                   # parabola with curvature g

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(t, tape, "o", ms=4, mfc="none", color="#c0392b", label="read against the tape")
    ax.plot(t, h, "o", ms=4, color="#1f4e79", label="corrected for parallax")
    ax.plot(tf, hf, "-", lw=1.2, color="#1f4e79", alpha=0.8, label="fitted parabola (curvature = g)")
    ax.axhline(h2, ls=":", lw=1, color="#1f4e79")
    ax.annotate(f"top: h₂ = {h2:.3f} m\n(tape read {dr['h2_uncorrected_m']:.3f} m)",
                xy=(tv, h2), xytext=(10, -40), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlabel("time since the ball was first tracked on the way up / s")
    ax.set_ylabel("height of the bottom of the ball / m")
    ax.set_title(f"{name}, drop {dr['drop']}  (d = {c['thickness_mm']:g} mm)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    base = os.path.dirname(os.path.abspath(a.config))
    entries = [e for e in cfg["videos"] if not a.only or e["name"] in a.only]
    plot_dir = os.path.join(a.out, "plots")
    if not a.no_plots:
        os.makedirs(plot_dir, exist_ok=True)

    console.print(Panel.fit(
        "[bold]Rebound height analysis[/]\n"
        "[dim]ball found by colour  →  parabola fitted to each rebound  →  "
        "scale at the ball from g  →  parallax correction[/]",
        border_style="bright_blue"))
    console.print(f"[dim]config:[/] {a.config}   [dim]runs:[/] {len(entries)}\n")

    summaries = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), console=console, transient=True) as prog:
        task = prog.add_task("analysing", total=len(entries))
        results = []
        for e in entries:
            prog.update(task, description=f"analysing {e['name']}")
            c = dict(A.DEFAULTS)
            c.update(cfg.get("defaults", {}))
            c.update(e)
            meta, rows = A.read_tracking(os.path.join(base, c["tracking"]))
            drops, geom = A.analyse(rows, meta, c)
            s = A.summarise(drops, geom, c)
            results.append((e["name"], c, geom, drops, s))
            prog.advance(task)
            time.sleep(0.15)

    n_plots = 0
    for name, c, geom, drops, s in results:
        console.print(run_panel(name, c, geom, drops, s))
        if not a.no_plots:
            for dr in drops:
                if dr.get("used"):
                    plot_drop(os.path.join(plot_dir, f"{name}_drop{dr['drop']:02d}.png"), name, dr, c, geom)
                    n_plots += 1
        if s:
            summaries.append((name, c, s))

    if summaries:
        t = Table(title="Summary", box=box.ROUNDED, header_style="bold magenta")
        for col in ("run", "d / mm", "h₁ / m", "mean h₂ / m", "parallax / m", "e", "Δe", "%"):
            t.add_column(col, justify="right")
        for name, c, s in summaries:
            t.add_row(name, f"{c['thickness_mm']:g}", f"{s['h1_m']:.3f}",
                      f"{s['mean_h2_m']:.3f} ± {s['dh2_m']:.3f}", f"{s['parallax_m']:.3f}",
                      f"[bold green]{s['ratio']:.3f}[/]", f"{s['d_ratio']:.3f}", f"{s['d_ratio_percent']:.1f}")
        console.print(t)
    if n_plots:
        console.print(f"[dim]saved {n_plots} rebound plots to[/] {plot_dir}/")


if __name__ == "__main__":
    main()
