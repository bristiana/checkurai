# Offline check: smolvla-place-both-v5

2026-11-14 20:37:26 (laptop clock) · smolvla.orig.xml (untouched) · CPU · 64 samples · infer 5694 ms median · run 969 s

## Open-loop fit: |predicted - recorded| per joint (deg; grip in Studio units 0..100)

| | pan | lift | elbow | wflex | wroll | grip |
|---|---|---|---|---|---|---|
| first 10 steps, median / p90 | 1.43 / 5.59 | 2.45 / 6.47 | 1.88 / 4.87 | 1.44 / 3.64 | 0.88 / 2.62 | 0.54 / 5.04 |
| first 50 steps, median / p90 | 3.02 / 14.94 | 4.04 / 36.02 | 3.14 / 15.91 | 2.52 / 9.55 | 1.49 / 10.72 | 0.99 / 5.09 |

## Instruction use

Release pose (pan, lift, elbow, wrist flex) from the demos: lilac [10.4, 44.1, -61.4, 60.0], orange [27.7, 0.4, -1.5, 70.5]. Chunk end = predicted pose at the end of the chunk (same span as the demo's future). 'Swap toward new bowl' = how far swapping the instruction moves that pose along the lilac->orange axis, towards the newly instructed bowl.

| moment | n | end nearer instructed bowl | same, instruction swapped | demo itself | swap toward new bowl (median deg / share > 0) | Δ actions, swap | Δ actions, same input |
|---|---|---|---|---|---|---|---|
| 0.0 | 16 | 0.5 | 0.5 | 0.44 | 1.72 / 0.5 | 9.014 | 7.211 |
| 0.3 | 16 | 0.38 | 0.56 | 0.38 | -1.28 / 0.44 | 3.401 | None |
| 0.5 | 16 | 0.81 | 0.56 | 0.81 | 20.4 / 0.75 | 8.625 | 1.369 |
| 0.7 | 16 | 0.81 | 0.69 | 0.81 | 48.69 / 0.94 | 15.619 | None |

## Dataset

```
{
 "episodes": {
  "lilac": 44,
  "orange": 38
 },
 "release_pan": {
  "lilac": {
   "median": 10.4,
   "p10": 3.2,
   "p90": 14.9
  },
  "orange": {
   "median": 27.7,
   "p10": 23.5,
   "p90": 34.9
  }
 },
 "release_at": {
  "lilac": 0.72,
  "orange": 0.73
 },
 "length_s": {
  "lilac": {
   "median": 13.6,
   "min": 8.6,
   "max": 42.4
  },
  "orange": {
   "median": 10.8,
   "min": 8.0,
   "max": 18.4
  }
 },
 "odd": [
  "episode 38 (lilac): 42.4 s vs median 13.6 s"
 ],
 "release_pose_pan_lift_elbow_wflex": {
  "lilac": [
   10.4,
   44.1,
   -61.4,
   60.0
  ],
  "orange": [
   27.7,
   0.4,
   -1.5,
   70.5
  ]
 },
 "idle_at_start_s": {
  "lilac": {
   "median": 1.07,
   "p90": 1.43,
   "max": 1.8
  },
  "orange": {
   "median": 1.07,
   "p90": 1.39,
   "max": 1.57
  }
 },
 "split_frame": 176,
 "split_s": 5.9,
 "split_fraction_of_median_episode": 0.49
}
```
