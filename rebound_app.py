#!/usr/bin/env python3
"""
rebound_app.py - interactive rebound analysis.

    python rebound_app.py

You type the thickness of the layer the ball lands on and drop the video
file into the window. The app then
  1. finds the ball in every frame (track.py),
  2. calibrates the tape measure (saved settings, or you click two tape marks),
  3. finds each rebound, fits a parabola to its top, works out the scale at the
     ball from g and corrects the parallax (analyse.py),
  4. prints the results and saves a CSV file and a plot of every rebound used.

Saved per-video settings (tape calibration etc.) are read from videos.json
next to this script, keyed by the video's file name. Results go to
results/<video name>/.
"""
import json
import os
import sys

import cv2
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.prompt import Confirm, FloatPrompt, Prompt

import analyse as A
import report as R
import track as T

HERE = os.path.dirname(os.path.abspath(__file__))
console = Console()


def clean_path(p):
    """Paths dragged into Terminal come quoted or with escaped spaces."""
    p = p.strip()
    if len(p) > 1 and p[0] == p[-1] and p[0] in "'\"":
        p = p[1:-1]
    return os.path.expanduser(p.replace("\\ ", " "))


def load_settings():
    path = os.path.join(HERE, "videos.json")
    if os.path.exists(path):
        return json.load(open(path)), path
    return {"defaults": {}, "videos": {}}, path


def click_calibration(video):
    """Show a frame and let the user click tape marks; returns [[row, reading_m], ...]."""
    import matplotlib.pyplot as plt
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, frame = cap.read()
    if not ok:
        sys.exit("could not read a frame for the calibration")
    console.print("[bold]Calibration:[/] click on two or more marks of the tape measure "
                  "(far apart, e.g. 0.50 m and 1.20 m), then press Enter in the picture window.")
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title("Click tape marks, then press Enter")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    if len(pts) < 2:
        sys.exit("at least two tape marks are needed")
    cal = []
    for i, (x, y) in enumerate(sorted(pts, key=lambda p: p[1]), 1):
        v = FloatPrompt.ask(f"  tape reading at mark {i} (pixel row {y:.0f}) in metres")
        cal.append([round(float(y), 1), v])
    return cal


def main():
    console.print(Panel.fit(
        "[bold]Rebound height analysis[/]\n"
        "[dim]1. type the layer thickness   2. drop the video into this window   3. the program does the rest[/]",
        border_style="bright_blue"))
    settings, settings_path = load_settings()

    while True:
        d_mm = FloatPrompt.ask("\n[bold]Layer thickness d[/] in mm")
        video = clean_path(Prompt.ask("[bold]Video file[/] (drag it into this window, then press Enter)"))
        if not os.path.exists(video):
            console.print(f"[red]File not found:[/] {video}")
            continue
        name = os.path.basename(video)
        stem = os.path.splitext(name)[0].replace(" ", "_")
        out_dir = os.path.join(HERE, "results", stem)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "tracking.csv")

        # 1. tracking
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            console.print(f"[red]Cannot open the video:[/] {video}")
            continue
        if os.path.exists(csv_path) and Confirm.ask("This video was tracked before. Use the saved tracking?",
                                                    default=True):
            pass
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            with Progress(TextColumn("finding the ball"), BarColumn(), TextColumn("{task.completed}/{task.total} frames"),
                          TimeRemainingColumn(), console=console) as prog:
                task = prog.add_task("track", total=total)
                n, found = T.track_video(cap, name, csv_path, progress=lambda i: prog.update(task, completed=i))
            console.print(f"[green]✓[/] ball found in {found} of {n} frames")

        # 2. calibration and settings for this video
        entry = dict(settings.get("videos", {}).get(name, {}))
        if "calibration" in entry:
            console.print(f"[green]✓[/] tape calibration loaded from videos.json ({len(entry['calibration'])} marks)")
        else:
            entry["calibration"] = click_calibration(video)
            if Confirm.ask("Save this calibration in videos.json for next time?", default=True):
                settings.setdefault("videos", {})[name] = {"calibration": entry["calibration"]}
                json.dump(settings, open(settings_path, "w"), indent=1)

        c = dict(A.DEFAULTS)
        c.update(settings.get("defaults", {}))
        c.update(entry)
        c["thickness_mm"] = d_mm

        # 3. analysis
        meta, rows = A.read_tracking(csv_path)
        drops, geom = A.analyse(rows, meta, c)
        s = A.summarise(drops, geom, c)

        # 4. results
        console.print(R.run_panel(f"{name}", c, geom, drops, s))
        n_plots = 0
        for dr in drops:
            if dr.get("used"):
                R.plot_drop(os.path.join(out_dir, f"drop{dr['drop']:02d}.png"), name, dr, c, geom)
                n_plots += 1
        cols = ["drop", "time_s", "accepted", "used", "anomalous", "h2_uncorrected_m", "parallax_m",
                "h2_corrected_m", "reason"]
        import csv
        with open(os.path.join(out_dir, "drops.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(drops)
        console.print(f"[dim]saved drops.csv and {n_plots} plots to[/] {out_dir}")

        if not Confirm.ask("\nAnalyse another video?", default=False):
            break


if __name__ == "__main__":
    main()
