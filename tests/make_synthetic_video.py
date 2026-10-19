#!/usr/bin/env python3
"""
Make a test video with known answers, to check track.py and analyse.py.

A yellow ball 1.00 m from the camera is dropped five times from 2.000 m onto
a 12 mm layer and caught just after the top of its rebound.  The tape measure
is 1.05 m from the camera (5 % further away than the ball, so there is
parallax), and the lens is 0.500 m above the floor.  The ball is drawn
stretched along its motion, like the motion blur in a real phone video.

    cd tests
    python make_synthetic_video.py
    python ../track.py synthetic.mp4 -o synthetic_track.csv
    python ../analyse.py synthetic_config.json --out synthetic_results

The corrected heights printed by analyse.py should match the true heights
printed here to about 1 mm, and the tape readings should match too.
"""
import json

import cv2
import numpy as np

W, H, FPS = 1080, 1920, 60
FOCAL = 1300          # px
D_BALL, D_TAPE = 1.00, 1.05   # distances from the camera (m)
LENS = 0.500          # lens height (m)
R, D, G = 0.0335, 0.012, 9.81
EXPOSURE = 1 / 200    # s, sets the length of the motion blur
E_TRUE = [0.70, 0.72, 0.69, 0.71, 0.70]


def row(h, dist):
    """Image row of a point at height h and distance dist (pinhole camera, level)."""
    return H / 2 - FOCAL * (h - LENS) / dist


def ball_state(t):
    """(height of the ball's bottom, velocity) at time t, or None if not in view."""
    for n, t0 in enumerate(range(1, 26, 5)):
        tau = t - t0
        if tau >= 0:
            t_fall = np.sqrt(2 * (2.0 - D) / G)
            if tau < t_fall:
                return 2.0 - 0.5 * G * tau ** 2, -G * tau
            v0 = E_TRUE[n] * G * t_fall
            s = tau - t_fall
            if s < v0 / G + 0.35:                      # caught 0.35 s after the top
                return D + v0 * s - 0.5 * G * s * s, v0 - G * s
    return None


def main():
    vw = cv2.VideoWriter("synthetic.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(27 * FPS):
        t = i / FPS
        img = np.full((H, W, 3), 150, np.uint8)
        for cm in range(0, 130, 10):                   # tape marks every 10 cm
            y = int(round(row(cm / 100, D_TAPE)))
            cv2.line(img, (600, y), (640, y), (0, 0, 0), 3)
        st = ball_state(t)
        if st:
            z, v = st
            yc = row(z + R, D_BALL)
            rad = FOCAL * R / D_BALL
            blur = abs(v) * EXPOSURE * FOCAL / D_BALL
            cv2.ellipse(img, (500, int(round(yc))), (int(rad), int(rad + blur / 2)), 0, 0, 360,
                        (40, 230, 230), -1)
        vw.write(img)
    vw.release()

    cal = [[round(row(cm / 100, D_TAPE), 1), cm / 100] for cm in range(10, 130, 10)]
    cfg = {"defaults": {"ball_radius_m": R, "drop_height_m": 2.0},
           "videos": [{"name": "synthetic", "tracking": "synthetic_track.csv",
                       "thickness_mm": D * 1000, "calibration": cal, "lens_height_m": LENS}]}
    json.dump(cfg, open("synthetic_config.json", "w"), indent=1)

    true_h = [e * e * (2.0 - D) for e in E_TRUE]
    tape = [LENS + (h + D + R - LENS) * D_TAPE / D_BALL - R - D for h in true_h]
    print("wrote synthetic.mp4 and synthetic_config.json")
    print("true rebound heights above the surface (m):", [round(h, 3) for h in true_h])
    print("what the tape would read (m):           ", [round(u, 3) for u in tape])
    print(f"true k = D_tape/D_ball - 1 = {D_TAPE / D_BALL - 1:.3f}")


if __name__ == "__main__":
    main()
