# Rebound height analysis

Two small Python scripts that measure how high a dropped ball rebounds, from a
phone video of the ball in front of a vertical tape measure.

* `track.py` finds the yellow ball in every frame and saves its position.
* `analyse.py` finds each rebound, fits a parabola to the top of it, converts
  pixels to metres with the tape measure, and works out the **parallax error**
  between the ball and the tape from the free-fall acceleration *g*.

Everything that depends on the set-up (lens height, tape calibration, ball
size, thickness of the surface layer, drop height, colour of the ball, fit window, rejection
rules) is a setting in a JSON config file, so the scripts can be used for any
similar drop experiment.

## Install

Python 3.8 or newer:

```
pip install -r requirements.txt
```

## Workflow

**1. Track the ball**

```
python track.py my_video.MOV -o tracking/run01.csv
```

This writes one line per frame (time, centre column, top and bottom row,
width, height and area of the ball). Times come from the video file itself,
so variable-frame-rate phone videos are timed correctly. If the ball is not
found, change the colour range with `--hsv-lo` / `--hsv-hi`.

**2. Read the tape calibration**

```
python track.py my_video.MOV --save-frame 100 -o frame100.png
```

This saves frame 100 with a labelled line every 50 pixel rows. Read off the
pixel row of several tape marks (e.g. every 10 cm) and list them in the
config as `[row, reading in metres]`.

**3. Write a config file** (see `config_example.json`)

```json
{
  "defaults": { "drop_height_m": 2.000, "ball_radius_m": 0.0335, "lens_height_m": 0.770 },
  "videos": [
    { "name": "run01", "tracking": "tracking/run01.csv", "thickness_mm": 0,
      "calibration": [[12, 1.25], [203, 1.15], [397, 1.05], [592, 0.95]] }
  ]
}
```

Anything in `defaults` applies to every video; anything inside a video entry
overrides it for that video only.

**4. Analyse**

```
python analyse.py config.json            # every video
python analyse.py config.json --only run01 # one video
```

For each drop it prints the tape reading of the rebound height, the
parallax-corrected height and the parallax error, and says why a drop was
rejected. It writes `results/<name>_drops.csv` for every video and
`results/summary.csv` with the mean rebound height, its uncertainty and √(*h*₂/*h*₁) for each.

## Readable report and plots

```
python report.py config.json
```

Runs the same analysis as `analyse.py`, but shows each run as a table in the
terminal (tape reading, parallax correction, corrected height and status of
every drop, then e with its uncertainty) and saves a plot of every rebound
that was used to `results/plots/`. Each plot shows the tracked points read
against the tape, the same points corrected for parallax, and the fitted
parabola whose curvature is g.

## How the parallax is found

The ball falls about 0.3 m in front of the tape, so it is nearer the camera
and looks bigger: one metre of height covers more pixels in the ball's plane
(scale *S*_b) than on the tape (scale *S*_t). Only on the image row level with
the lens (the line of sight, height *L*) do the two agree.

* *S*_t comes from the tape marks (straight-line fit of reading against row).
* *S*_b comes from the motion of the ball. Near the top of a rebound the
  ball is in free fall, so its centre row follows *y* = *at*² + *bt* + *c*
  with 2*a* = *g* · *S*_b, so *S*_b = 2*a*/*g*. The median over the clean
  rebounds of a video is used.
* With *k* = *S*_b/*S*_t − 1, a point whose true height is *H* reads
  *H* + (*H* − *L*)·*k* on the tape. The difference (*H* − *L*)·*k* is the
  parallax error Δp.

The script reports both the tape reading and the corrected height. The
summary uses the corrected heights; their uncertainty is the half-range of
the trials (or the reading uncertainty, if that is larger).

## Settings

| key | default | meaning |
|---|---|---|
| `tracking` | — | CSV file made by `track.py` (path relative to the config) |
| `calibration` | — | `[[pixel_row, tape_reading_m], …]`, at least two marks |
| `thickness_mm` | 0 | thickness of any layer the ball lands on; heights are measured from its top surface |
| `drop_height_m` | 2.000 | release height of the bottom of the ball |
| `drop_height_unc_m` | 0.002 | uncertainty in the release height |
| `ball_radius_m` | 0.0335 | radius of the ball |
| `g` | 9.81 | m s⁻² |
| `lens_height_m` | none | measured lens height. The line of sight is then the image row where the tape reads this height |
| `level_row_px` | middle row | image row of the line of sight, used when `lens_height_m` is not given |
| `centre_x_px` | middle column | column of the optical axis |
| `tape_line` | none | `{"x_px", "y_px", "slope"}`: where the tape is at one row and how far it drifts sideways per row, to correct a slightly rotated phone |
| `fit_window_s` | 0.21 | parabola fitted to points within this time of the top |
| `max_extrapolation_s` | 0.034 | reject if the top of the rebound is further than this past the last tracked frame |
| `max_sideways_m_s` | 0.30 | reject if the ball moves sideways faster than this |
| `max_rms_m` | 0.003 | reject if the parabola fit residual is larger |
| `off_line_widths` | 1.2 | reject if the ball is more than this many ball widths off the usual drop line |
| `min_width_frac` | 0.75 | a frame counts as "whole ball visible" when its width is at least this × the ball width |
| `require_fall` | true | only accept a rebound if the ball was seen falling just before it |
| `scale_min_width_frac`, `scale_require_fall` | 0.75, true | the same two settings for the rebounds used to measure *S*_b; they stay strict when the two above are relaxed |
| `skip_duplicate_frames` | false | ignore repeated identical frames (for videos re-encoded at a higher frame rate) |
| `width_from_top_blobs` | false | also estimate the ball width from the largest blobs (helps when the ball is often partly hidden) |
| `group_gap_s`, `link_gap_s`, `min_run_frames`, `start_row_frac`, `fall_rows_frac`, `fall_window_s` | 1.3, 0.075, 5, 0.573, 0.15625, 0.6 | how frames are split into drops and linked into a rebound |
| `n_trials` | 5 | accepted drops used per video |
| `anomaly_factor` | 3 | a drop further than this × the half-range from the mean of the others is anomalous and replaced by the next drop |
| `round_heights_m` | 0.001 | per-drop heights are reported to this resolution |
| `reading_unc_m` | 0.005 | smallest uncertainty used for the mean rebound height |

## Uncertainties in `summary.csv`

* Δ*h*₂ = half-range of the corrected heights (at least `reading_unc_m`)
* ratio = √(*h*₂/*h*₁), with Δratio/ratio = ½ (Δ*h*₂/*h*₂ + Δ*h*₁/*h*₁)

## Check it on a video with known answers

```
cd tests
python make_synthetic_video.py
python ../track.py synthetic.mp4 -o synthetic_track.csv
python ../analyse.py synthetic_config.json --out synthetic_results
```

The synthetic video has a tape 5 % further from the camera than the ball
(*k* = 0.050). The script finds *k* = 0.049 and gets every corrected rebound
height within 1 mm of the true value.

## Example data

`tracking/` holds example tracking files and `data_config.json` holds the
calibration and settings used for each, so

```
python analyse.py data_config.json
```

repeats the analysis without needing the videos. Per-video settings that
differ from the defaults: run05 uses relaxed detection
(`min_width_frac` 0.6, `require_fall` false) because too few of its
rebounds passed the standard detection; four runs had been re-encoded at 60 fps and use
`skip_duplicate_frames`; and several use `tape_line` because the phone was
slightly rotated.

## Licence

MIT, see `LICENSE`.
