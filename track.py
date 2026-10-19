#!/usr/bin/env python3
"""
track.py - find a yellow ball in every frame of a video.

For each frame the ball is found by colour (HSV threshold), small specks are
removed, and the largest roughly round blob is kept. Its bounding box is
written to a CSV file that analyse.py reads.

Usage
-----
    python track.py VIDEO -o tracking.csv
    python track.py VIDEO --save-frame 120 -o frame120.png   # frame with a pixel-row
                                                            # grid, for reading tape marks

Output columns
--------------
    frame       frame number (0 = first frame)
    time_s      presentation time of the frame in seconds (from the video file,
                so variable-frame-rate phone videos are timed correctly)
    x_px        horizontal centre of the ball's bounding box
    top_px      top edge row of the bounding box (row 0 = top of the image)
    bottom_px   bottom edge row of the bounding box
    width_px, height_px, area_px
                size of the blob (empty when no ball was found in that frame)
The first lines of the file are comments (#) holding the frame size and fps.
"""
import argparse
import csv
import sys

import cv2
import numpy as np


def find_ball(frame, hsv_lo, hsv_hi, min_area, aspect_lo, aspect_hi):
    """Return (x_centre, top, bottom, width, height, area) of the ball, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if a > min_area and aspect_lo < w / max(h, 1) < aspect_hi:
            if best is None or a > best[4]:
                best = (x, y, w, h, a)
    if best is None:
        return None
    x, y, w, h, a = best
    return float(x + w / 2), int(y), int(y + h), int(w), int(h), int(a)


def save_grid_frame(cap, index, out_path, step):
    """Save one frame with horizontal lines every `step` pixel rows, labelled,
    so the rows of the tape-measure marks can be read off for calibration."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"could not read frame {index}")
    for y in range(0, frame.shape[0], step):
        cv2.line(frame, (0, y), (frame.shape[1], y), (0, 0, 255), 1)
        cv2.putText(frame, str(y), (5, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(out_path, frame)
    print(f"saved frame {index} ({frame.shape[1]}x{frame.shape[0]}) to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--output", required=True, help="CSV file to write (or PNG with --save-frame)")
    ap.add_argument("--hsv-lo", default="25,80,120", help="lower HSV bound for the ball colour (default 25,80,120)")
    ap.add_argument("--hsv-hi", default="45,255,255", help="upper HSV bound (default 45,255,255)")
    ap.add_argument("--min-area", type=int, default=300, help="smallest blob area in pixels counted as the ball")
    ap.add_argument("--aspect", default="0.4,2.5", help="allowed width/height range of the blob")
    ap.add_argument("--fps", type=float, default=None,
                    help="ignore the file's timestamps and use frame/fps instead")
    ap.add_argument("--save-frame", type=int, default=None, metavar="N",
                    help="only save frame N with a pixel-row grid, then stop")
    ap.add_argument("--grid-step", type=int, default=50, help="row spacing of the grid for --save-frame")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video}")
    if a.save_frame is not None:
        save_grid_frame(cap, a.save_frame, a.output, a.grid_step)
        return

    lo = tuple(int(v) for v in a.hsv_lo.split(","))
    hi = tuple(int(v) for v in a.hsv_hi.split(","))
    asp_lo, asp_hi = (float(v) for v in a.aspect.split(","))
    fps = cap.get(cv2.CAP_PROP_FPS)

    rows, i, size = [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = frame.shape[1], frame.shape[0]
        t = i / a.fps if a.fps else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        b = find_ball(frame, lo, hi, a.min_area, asp_lo, asp_hi)
        rows.append([i, f"{t:.6f}"] + (list(b) if b else [""] * 6))
        i += 1
    if size is None:
        sys.exit("no frames read")

    t0 = float(rows[0][1])
    with open(a.output, "w", newline="") as f:
        f.write(f"# video={a.video}\n# width={size[0]}\n# height={size[1]}\n# fps={fps}\n")
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "x_px", "top_px", "bottom_px", "width_px", "height_px", "area_px"])
        for r in rows:
            r[1] = f"{float(r[1]) - t0:.6f}"
            w.writerow(r)
    found = sum(1 for r in rows if r[2] != "")
    print(f"{a.video}: {len(rows)} frames, ball found in {found}; written to {a.output}")


if __name__ == "__main__":
    main()
